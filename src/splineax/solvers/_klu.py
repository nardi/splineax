from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum, auto
from typing import Any, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
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


class _KLUFactorization(NamedTuple):
    symbolic: Array
    numeric: Array


class KLUHandleType(Enum):
    SYMBOLIC = auto()
    NUMERIC = auto()


PyTreeT = TypeVar("PyTreeT", bound=PyTree)


class KLUHandleAllocationScopeManager:
    _handles_allocated_in_scope: ContextVar[
        list[tuple[KLUHandleType, KLUHandleManager]] | None
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
                case KLUHandleType.SYMBOLIC:
                    klujax.free_symbolic(manager, deps)
                case KLUHandleType.NUMERIC:
                    klujax.free_numeric(manager, deps)

        cls._handles_allocated_in_scope.reset(token)

    @classmethod
    def register_handle(
        cls, handle_type: KLUHandleType, manager: KLUHandleManager
    ) -> KLUHandleManager:
        handles = cls._handles_allocated_in_scope.get()
        assert handles is not None
        handles.append((handle_type, manager))
        return manager

    @classmethod
    def register_dependency(cls, handle: Array, dependency: PyTreeT) -> PyTreeT:
        cls._handle_dependencies.setdefault(id(handle), []).append(dependency)
        return dependency


class _KLUState(NamedTuple):
    coo: _COO
    shape: tuple[int, ...]
    packed_structures: PackedStructures

    @contextmanager
    def factorize(self):
        klujax = _klujax()
        Ai, Aj, Ax = self.coo

        with KLUHandleAllocationScopeManager.begin_scope():
            symbolic = KLUHandleAllocationScopeManager.register_handle(
                KLUHandleType.SYMBOLIC, klujax.analyze(Ai, Aj, self.shape[1])
            )
            numeric = KLUHandleAllocationScopeManager.register_handle(
                KLUHandleType.NUMERIC, klujax.factor(Ai, Aj, Ax, symbolic)
            )

            managed_state = _ManagedKLUState(
                self.coo,
                _KLUFactorization(symbolic.handle, numeric.handle),
                self.packed_structures,
                self.shape,
            )

            yield managed_state


class _ManagedKLUState(eqx.Module):
    coo: _COO
    factorization: _KLUFactorization
    packed_structures: PackedStructures
    # `shape` and `transposed` are static metadata, not traced leaves: `compute` branches on
    # `transposed` under AD tracing, where a traced leaf could not be used in `if`.
    shape: tuple[int, ...] = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)


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


class KLU(AbstractLinearSolver[_KLUState | _ManagedKLUState]):
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
    ) -> _KLUState:
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

        return _KLUState(
            (Ai, Aj, Ax),
            matrix.shape,
            pack_structures(operator),
        )

    def compute(
        self,
        state: _KLUState | _ManagedKLUState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        (Ai, Aj, Ax) = state.coo
        b = ravel_vector(vector, state.packed_structures)
        Ai, Aj, Ax, b = _ensure_cpu((Ai, Aj, Ax, b))

        # Upcast right-hand side vector if necessary.
        if b.dtype in COMPLEX_DTYPES:
            b = b.astype(jnp.complex128)
        else:
            b = b.astype(jnp.float64)

        if isinstance(state, _ManagedKLUState):
            klujax = _klujax()

            # A managed transpose reuses A's numeric factorization: `tsolve_with_numeric`
            # solves `Aᵀx=b` from it directly (KLU transposes during the triangular solve).
            solve = (
                klujax.tsolve_with_numeric
                if state.transposed
                else klujax.solve_with_numeric
            )

            # Perform the solve.
            x = solve(state.factorization.numeric, b, state.factorization.symbolic)

            # Register the solution as a dependency of the factorization handles.
            KLUHandleAllocationScopeManager.register_dependency(
                state.factorization.symbolic, x
            )
            KLUHandleAllocationScopeManager.register_dependency(
                state.factorization.numeric, x
            )
        else:
            x = _klujax().solve(Ai, Aj, Ax, b)

        solution = unravel_solution(x, state.packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(
        self, state: _KLUState | _ManagedKLUState, options: dict[str, Any]
    ) -> tuple[_KLUState | _ManagedKLUState, dict[str, Any]]:
        del options
        Ai, Aj, Ax = state.coo
        packed_structures = transpose_packed_structures(state.packed_structures)
        # The transpose of a COO matrix swaps its row and column indices.
        if isinstance(state, _ManagedKLUState):
            # Reuse the existing factorization unchanged; `compute` will `tsolve` against it.
            transpose_state = _ManagedKLUState(
                (Aj, Ai, Ax),
                state.factorization,
                packed_structures,
                state.shape[::-1],
                not state.transposed,
            )
        else:
            transpose_state = _KLUState(
                (Aj, Ai, Ax), state.shape[::-1], packed_structures
            )
        return transpose_state, {}

    def conj(
        self, state: _KLUState | _ManagedKLUState, options: dict[str, Any]
    ) -> tuple[_KLUState | _ManagedKLUState, dict[str, Any]]:
        del options
        Ai, Aj, Ax = state.coo
        if isinstance(state, _ManagedKLUState):
            if Ax.dtype not in COMPLEX_DTYPES:
                # Real values: conjugation is a no-op, so reuse the whole factorization.
                return state, {}
            # Complex values: the sparsity is unchanged, so reuse the symbolic analysis and
            # factor a fresh numeric for `conj(A)`, owned by the active `factorize` block.
            # We register the handle within the current allocation scope, so
            # that it gets freed eventually.
            # TODO: should we instead assign it to the scope of the symbolic
            # handle? Is that ever necessary/a good idea?
            numeric = KLUHandleAllocationScopeManager.register_handle(
                KLUHandleType.NUMERIC,
                _klujax().factor(Ai, Aj, Ax.conj(), state.factorization.symbolic),
            )
            conj_state = _ManagedKLUState(
                (Ai, Aj, Ax.conj()),
                _KLUFactorization(state.factorization.symbolic, numeric.handle),
                state.packed_structures,
                state.shape,
                state.transposed,
            )
            return conj_state, {}
        return _KLUState((Ai, Aj, Ax.conj()), state.shape, state.packed_structures), {}

    def assume_full_rank(self) -> bool:
        return True


KLU.__init__.__doc__ = """**Arguments:**

Nothing.
"""
