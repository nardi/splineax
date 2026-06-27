from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum, auto
from typing import Any, NamedTuple, NewType, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Inexact, Integer, PyTree
from klujax import KLUHandleManager
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver
from lineax._solver.misc import (
    PackedStructures,
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator

# `Ai` (row indices), `Aj` (column indices), `Ax` (values): the matrix in COO form.
_COO = tuple[Integer[Array, " a"], Integer[Array, " b"], Inexact[Array, " nse"]]

# Row and column indices only, without values.
_COOIndices = tuple[Integer[Array, " a"], Integer[Array, " b"]]

_KLUHandle = NewType("_KLUHandle", Array)


class _KLUFactorization(NamedTuple):
    symbolic: _KLUHandle
    numeric: _KLUHandle


class _KLUHandleType(Enum):
    SYMBOLIC = auto()
    NUMERIC = auto()


PyTreeT = TypeVar("PyTreeT", bound=PyTree)


class _KLUHandleAllocationScopeManager:
    _handles_allocated_in_scope: ContextVar[
        list[tuple[_KLUHandleType, KLUHandleManager]] | None
    ] = ContextVar("_klu_handles_in_use", default=None)

    _handle_dependencies: dict[int, list[Any]] = {}

    @classmethod
    @contextmanager
    def begin_scope(cls):
        klujax = _klujax()
        handles = []
        token = cls._handles_allocated_in_scope.set(handles)

        yield

        for handle_type, manager in reversed(handles):
            deps = cls._handle_dependencies.pop(id(manager.handle), [])
            match handle_type:
                case _KLUHandleType.SYMBOLIC:
                    klujax.free_symbolic(manager, deps)
                case _KLUHandleType.NUMERIC:
                    klujax.free_numeric(manager, deps)

        cls._handles_allocated_in_scope.reset(token)

    @classmethod
    def register_handle(
        cls, handle_type: _KLUHandleType, manager: KLUHandleManager
    ) -> KLUHandleManager:
        handles = cls._handles_allocated_in_scope.get()
        assert handles is not None
        handles.append((handle_type, manager))
        return manager

    @classmethod
    def register_dependency(cls, handle: Array, dependency: PyTreeT) -> PyTreeT:
        cls._handle_dependencies.setdefault(id(handle), []).append(dependency)
        return dependency


class _KLUBasicState(NamedTuple):
    coo: _COO
    shape: tuple[int, ...]
    packed_structures: PackedStructures

    @contextmanager
    def factorize(self):
        klujax = _klujax()
        Ai, Aj, Ax = self.coo

        with _KLUHandleAllocationScopeManager.begin_scope():
            symbolic = _KLUHandleAllocationScopeManager.register_handle(
                _KLUHandleType.SYMBOLIC, klujax.analyze(Ai, Aj, self.shape[1])
            )
            numeric = _KLUHandleAllocationScopeManager.register_handle(
                _KLUHandleType.NUMERIC, klujax.factor(Ai, Aj, Ax, symbolic)
            )

            yield _KLUNumericState(
                self.coo,
                _KLUFactorization(
                    _KLUHandle(symbolic.handle), _KLUHandle(numeric.handle)
                ),
                self.packed_structures,
                self.shape,
            )


class _KLUSymbolicState(NamedTuple):
    coo: _COO
    shape: tuple[int, ...]
    packed_structures: PackedStructures
    symbolic: _KLUHandle
    transposed: bool = False

    @contextmanager
    def factorize(self):
        Ai, Aj, Ax = self.coo
        with _KLUHandleAllocationScopeManager.begin_scope():
            # Only the numeric handle is registered here. The symbolic handle is
            # owned and freed by the outer factorize_symbolic() scope.
            numeric_manager = _KLUHandleAllocationScopeManager.register_handle(
                _KLUHandleType.NUMERIC,
                _klujax().factor(Ai, Aj, Ax, self.symbolic),
            )
            yield _KLUNumericState(
                self.coo,
                _KLUFactorization(self.symbolic, _KLUHandle(numeric_manager.handle)),
                self.packed_structures,
                self.shape,
            )


class _KLUNumericState(eqx.Module):
    coo: _COO
    factorization: _KLUFactorization
    packed_structures: PackedStructures
    # `shape` and `transposed` are static metadata, not traced leaves: `compute` branches on
    # `transposed` under AD tracing, where a traced leaf could not be used in `if`.
    shape: tuple[int, ...] = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)


class _KLUSymbolicScope(NamedTuple):
    indices: _COOIndices
    """Row and column indices of the analyzed matrix, without values."""
    shape: tuple[int, ...]
    symbolic: _KLUHandle

    def init(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _KLUSymbolicState:
        del options

        Ai, Aj = self.indices
        match operator:
            case BCSRLinearOperator(matrix):
                bcoo = matrix.to_bcoo()
            case BCOOLinearOperator(matrix):
                bcoo = matrix
            case _:
                raise TypeError(
                    "`_KLUSymbolicScope.init` requires a `BCOOLinearOperator` or "
                    f"`BCSRLinearOperator`; got {type(operator).__name__}."
                )
        Ax = bcoo.data

        # Import klujax to enable x64.
        _klujax()

        if Ax.dtype in COMPLEX_DTYPES:
            Ax = Ax.astype(jnp.complex128)
        else:
            Ax = Ax.astype(jnp.float64)

        return _KLUSymbolicState(
            (Ai, Aj, Ax),
            self.shape,
            pack_structures(operator),
            self.symbolic,
        )

    @contextmanager
    def factorize(self, operator: AbstractLinearOperator):
        with self.init(operator).factorize() as state:
            yield state


_KLUState = _KLUBasicState | _KLUSymbolicState | _KLUNumericState


def _klujax():
    # Lazy import: importing klujax enables jax x64 and forces the CPU platform globally,
    # so we defer it until a KLU solve actually runs (keeping it out of `import splineax`).
    import klujax

    return klujax


T = TypeVar("T")


def _ensure_cpu(args: T) -> T:
    """Return `x` unchanged, raising if the current platform is not CPU.

    Uses `jax.lax.platform_dependent` to produce a traced boolean and
    `equinox.error_if` to raise.
    """
    on_cpu = jax.lax.platform_dependent(
        args,
        default=lambda _: jnp.bool_(False),
        cpu=lambda _: jnp.bool_(True),
    )
    return eqx.error_if(
        args,
        ~on_cpu,
        "`KLU` can only solve on CPU; klujax wraps the CPU-only SuiteSparse KLU library.",
    )


COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    jnp.complex64,
    jnp.complex128,
)


class KLU(AbstractLinearSolver[_KLUState]):
    """Sparse direct solver wrapping the `klujax` (SuiteSparse KLU) library.

    This solver keeps the operator in its native sparse (COO) storage rather than
    densifying it, and so is intended for use with the sparse operators in this package
    (`BCOOLinearOperator` and `BCSRLinearOperator`).

    `klujax` is **CPU and double-precision only**: `float32`/`complex64` inputs are
    upcast to `float64`/`complex128`, and importing it enables JAX's x64 mode and forces
    the CPU platform globally (this happens lazily, on the first solve).

    This solver can only handle square nonsingular operators.
    """

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _KLUBasicState:
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`KLU` may only be used for linear solves with square matrices"
            )

        # `klujax.solve` consumes a COO triple. We assume the matrix is coalesced (no
        # duplicate indices); KLU builds CSC internally, so the index order is irrelevant.
        match operator:
            case BCSRLinearOperator(matrix):
                matrix = matrix.to_bcoo()
            case BCOOLinearOperator(matrix):
                pass
            case _:
                raise TypeError(
                    "`KLU` requires a sparse operator backed by a `BCOO` or `BCSR` "
                    "matrix (e.g. `splineax.BCOOLinearOperator` or "
                    f"`splineax.BCSRLinearOperator`); got {type(operator).__name__}."
                )

        # Import klujax to enable x64.
        _klujax()

        Ai = matrix.indices[:, 0].astype(jnp.int32)
        Aj = matrix.indices[:, 1].astype(jnp.int32)
        Ax = matrix.data

        # Upcast matrix data if necessary.
        if Ax.dtype in COMPLEX_DTYPES:
            Ax = Ax.astype(jnp.complex128)
        else:
            Ax = Ax.astype(jnp.float64)

        return _KLUBasicState(
            (Ai, Aj, Ax),
            matrix.shape,
            pack_structures(operator),
        )

    @classmethod
    @contextmanager
    def factorize(cls, operator: AbstractLinearOperator, options: dict[str, Any] = {}):
        """Convenience method: equivalent to `KLU().init(operator, options).factorize()`."""
        with cls().init(operator, options).factorize() as state:
            yield state

    @contextmanager
    def factorize_symbolic(
        self,
        sparsity: BCOO | BCSR | BCOOLinearOperator | BCSRLinearOperator,
    ):
        """Open a scope with a pre-computed KLU symbolic factorization.

        Yields a `_KLUSymbolicScope`. Inside the block, call:
        - `.init(operator)` to create a `_KLUSymbolicState` for `lx.linear_solve`
          (uses `solve_with_symbol`: factors numerically on each call, symbolic reused).
        - `.init(operator).factorize()` or equivalently `.factorize(operator)` to also
          pre-compute the numeric factorization (uses `solve_with_numeric`).

        The symbolic handle is freed when the `with` block exits, after all
        registered solve-result dependencies have been consumed.

        Args:
            sparsity: Sparse matrix whose sparsity pattern to pre-analyze. Accepts
                      `BCOO`, `BCSR`, `BCOOLinearOperator`, or `BCSRLinearOperator`.
        """
        match sparsity:
            case BCSRLinearOperator():
                bcoo = sparsity.matrix.to_bcoo()
            case BCOOLinearOperator():
                bcoo = sparsity.matrix
            case BCSR():
                bcoo = sparsity.to_bcoo()
            case BCOO():
                bcoo = sparsity
            case _:
                raise TypeError(
                    "`KLU.factorize_symbolic` requires a `BCOO`, `BCSR`, "
                    "`BCOOLinearOperator`, or `BCSRLinearOperator`; "
                    f"got {type(sparsity).__name__}."
                )

        if bcoo.shape[0] != bcoo.shape[1]:
            raise ValueError(
                "`KLU.factorize_symbolic` requires a square matrix; "
                f"got shape {bcoo.shape}."
            )

        Ai = bcoo.indices[:, 0].astype(jnp.int32)
        Aj = bcoo.indices[:, 1].astype(jnp.int32)

        with _KLUHandleAllocationScopeManager.begin_scope():
            symbolic = _KLUHandleAllocationScopeManager.register_handle(
                _KLUHandleType.SYMBOLIC, _klujax().analyze(Ai, Aj, bcoo.shape[1])
            )
            yield _KLUSymbolicScope(
                (Ai, Aj),
                bcoo.shape,
                _KLUHandle(symbolic.handle),
            )

    def compute(
        self,
        state: _KLUState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options

        Ai, Aj, Ax = state.coo
        b = ravel_vector(vector, state.packed_structures)
        Ai, Aj, Ax, b = _ensure_cpu((Ai, Aj, Ax, b))

        # Import klujax to enable x64.
        _klujax()

        # Upcast right-hand side vector if necessary.
        if b.dtype in COMPLEX_DTYPES:
            b = b.astype(jnp.complex128)
        else:
            b = b.astype(jnp.float64)

        klujax = _klujax()
        match state:
            case _KLUNumericState(
                factorization=_KLUFactorization(symbolic=symbolic, numeric=numeric),
                transposed=transposed,
            ):
                # `solve_with_numeric` and `tsolve_with_numeric` both expect the
                # non-swapped Ai/Aj index arrays. The transposed flag is used to
                # select `tsolve`.
                solve = (
                    klujax.tsolve_with_numeric
                    if transposed
                    else klujax.solve_with_numeric
                )
                x = solve(numeric, b, symbolic)
                _KLUHandleAllocationScopeManager.register_dependency(symbolic, x)
                _KLUHandleAllocationScopeManager.register_dependency(numeric, x)
            case _KLUSymbolicState(symbolic=symbolic, transposed=transposed):
                # `solve_with_symbol` and `tsolve_with_symbol` both expect the
                # non-swapped Ai/Aj index arrays. The transposed flag is used to
                # select `tsolve`.
                solve = (
                    klujax.tsolve_with_symbol
                    if transposed
                    else klujax.solve_with_symbol
                )
                x = solve(Ai, Aj, Ax, b, symbolic)
                _KLUHandleAllocationScopeManager.register_dependency(symbolic, x)
            case _KLUBasicState():
                x = klujax.solve(Ai, Aj, Ax, b)

        solution = unravel_solution(x, state.packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(
        self, state: _KLUState, options: dict[str, Any]
    ) -> tuple[_KLUState, dict[str, Any]]:
        del options
        Ai, Aj, Ax = state.coo
        packed_structures = transpose_packed_structures(state.packed_structures)

        match state:
            case _KLUNumericState(
                factorization=factorization, transposed=transposed, shape=shape
            ):
                # Reuse the existing factorization unchanged; `compute` will
                # `tsolve` against it.
                return _KLUNumericState(
                    (Aj, Ai, Ax),
                    factorization,
                    packed_structures,
                    shape[::-1],
                    not transposed,
                ), {}
            case _KLUSymbolicState(
                symbolic=symbolic, transposed=transposed, shape=shape
            ):
                # Keep original Ai/Aj (not swapped). The transposed flag governs
                # which solve function is selected; the indices of A remain
                # unchanged.
                return _KLUSymbolicState(
                    (Ai, Aj, Ax),
                    shape[::-1],
                    packed_structures,
                    symbolic,
                    not transposed,
                ), {}
            case _KLUBasicState(shape=shape):
                return _KLUBasicState((Aj, Ai, Ax), shape[::-1], packed_structures), {}

    def conj(
        self, state: _KLUState, options: dict[str, Any]
    ) -> tuple[_KLUState, dict[str, Any]]:
        del options
        Ai, Aj, Ax = state.coo

        if Ax.dtype not in COMPLEX_DTYPES:
            # Real: conj is a no-op for all state types.
            return state, {}

        match state:
            case _KLUNumericState(
                factorization=_KLUFactorization(symbolic=symbolic),
                packed_structures=packed_structures,
                shape=shape,
                transposed=transposed,
            ):
                # Complex numeric: re-factor conj(A) reusing the existing symbolic handle.
                # The sparsity is unchanged, so the symbolic analysis remains valid.
                # We register the new handle within the current allocation
                # scope, so that it gets freed eventually.
                # TODO: should we instead explicitly assign it to the scope of
                # the previous numeric handle? Does it matter?
                numeric = _KLUHandleAllocationScopeManager.register_handle(
                    _KLUHandleType.NUMERIC,
                    _klujax().factor(Ai, Aj, Ax.conj(), symbolic),
                )
                return _KLUNumericState(
                    (Ai, Aj, Ax.conj()),
                    _KLUFactorization(symbolic, _KLUHandle(numeric.handle)),
                    packed_structures,
                    shape,
                    transposed,
                ), {}
            case _KLUSymbolicState(
                symbolic=symbolic,
                transposed=transposed,
                shape=shape,
                packed_structures=packed_structures,
            ):
                # Complex symbolic: `(t)solve_with_symbol` re-factors
                # numerically per call, so conjugating the values is enough. The
                # symbolic factorization remains valid.
                return _KLUSymbolicState(
                    (Ai, Aj, Ax.conj()),
                    shape,
                    packed_structures,
                    symbolic,
                    transposed,
                ), {}
            case _KLUBasicState(shape=shape, packed_structures=packed_structures):
                return _KLUBasicState((Ai, Aj, Ax.conj()), shape, packed_structures), {}

    def assume_full_rank(self) -> bool:
        return True


KLU.__init__.__doc__ = """**Arguments:**

Nothing.
"""
