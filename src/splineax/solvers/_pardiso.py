import importlib.util
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Inexact, Integer, PyTree
from lineax import AbstractLinearOperator, materialise
from lineax._solution import RESULTS
from lineax._solver.misc import (
    PackedStructures,
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
)
from splineax.solvers._handle import (
    HandleDependencies,
    _HandleToken,
    handle_value,
    wrap_handle,
)
from splineax.solvers._handle import (
    rebind_handle as _rebind_token,
)
from splineax.solvers._klu import COMPLEX_DTYPES
from splineax.solvers._sparse import (
    AbstractSparseLinearSolver,
    SparseNumericState,
    factorize_through_init,
)

# `indptr`, `indices`, `values`: the matrix in CSR form.
_CSR = tuple[Integer[Array, " n+1"], Integer[Array, " nse"], Inexact[Array, " nse"]]

# A Pardiso factorization handle: either the raw int64 array
# `pardiso_mkl_jax.primitive.analyze` returns, or (when created while tracing) a
# `_HandleToken` wrapping it, see `_handle.py`. Being an ordinary JAX value rather
# than a Python-side id is what lets XLA order the whole analyze-factor-solve-release
# lifecycle by data dependency, so it composes inside a jitted function.
_PardisoHandle = Array | _HandleToken

PyTreeT = TypeVar("PyTreeT", bound=PyTree)


def _pardiso_available() -> bool:
    """Whether `pardiso_mkl_jax` is importable, without actually importing it.

    Checked with `importlib.util.find_spec` (no execution) rather than a real import.
    That way, probing availability from `Pardiso.__init__` and `AutoSparseLinearSolver`
    never pays for `pardiso_mkl_jax`'s import-time MKL runtime load. Unlike `klujax`,
    `pardiso-mkl-jax` is an optional dependency, so this check (with no equivalent in
    `_klu.py`) is what makes `Pardiso` unconstructible, and `AutoSparseLinearSolver`
    fall back to `KLU`, when it isn't installed.
    """
    return importlib.util.find_spec("pardiso_mkl_jax") is not None


def _pardiso_mkl_jax():
    # Lazy import: deferred until a Pardiso solve actually runs. Importing splineax,
    # or even constructing a `Pardiso` instance, never loads the MKL runtime unless
    # the solver is actually used (mirrors `_klu.py`'s `_klujax()`).
    import pardiso_mkl_jax

    return pardiso_mkl_jax


class _PardisoHandleAllocationScopeManager:
    """Frees Pardiso factorization handles when the scope that allocated them exits.

    Structurally identical to `_klu.py`'s `_KLUHandleAllocationScopeManager`, against
    `pardiso_mkl_jax.primitive` instead of `klujax`. Unlike KLU, which has genuinely
    separate symbolic and numeric native handles, Pardiso's `factor` mutates the same
    registry entry `analyze` allocated and passes the same handle value back through,
    so there is only ever one handle to free per scope, not two: `register_handle` is
    called once, when the handle is first allocated, and `rebind_handle` updates that
    registration in place whenever `factor` advances the handle further, so the scope
    always frees the latest handle in the chain rather than a stale one a later call
    still depends on.
    """

    _handles_allocated_in_scope: ContextVar[list[_PardisoHandle] | None] = ContextVar(
        "_pardiso_handles_in_use", default=None
    )

    _handle_dependencies = HandleDependencies()

    @classmethod
    @contextmanager
    def begin_scope(cls) -> Iterator[None]:
        primitive = _pardiso_mkl_jax().primitive
        handles: list[_PardisoHandle] = []
        token = cls._handles_allocated_in_scope.set(handles)

        yield

        for handle in reversed(handles):
            deps = cls._handle_dependencies.pop(handle)
            value = handle_value(handle)
            if deps:
                # Forces release to run after everything that used handle: release and
                # a solve both merely consume handle, so without this they share no
                # ordering XLA must respect, and could run in either order.
                value, _ = jax.lax.optimization_barrier((value, deps))
            primitive.release(value)

        cls._handles_allocated_in_scope.reset(token)

    @classmethod
    def register_handle(cls, value: Array) -> _PardisoHandle:
        handles = cls._handles_allocated_in_scope.get()
        assert handles is not None
        handle: _PardisoHandle = wrap_handle(value)
        cls._handle_dependencies.record_allocation(handle)
        handles.append(handle)
        return handle

    @classmethod
    def rebind_handle(
        cls, old_handle: _PardisoHandle, new_value: Array
    ) -> _PardisoHandle:
        """Replace a registered handle with the value `factor` advanced it to.

        Reuses `old_handle`'s token-or-not kind, and its stable id if it is one, so
        dependencies already registered against it are still found and the scope still
        frees the right, current resource rather than a stale one.
        """
        handles = cls._handles_allocated_in_scope.get()
        assert handles is not None
        new_handle: _PardisoHandle = _rebind_token(old_handle, new_value)
        handles[:] = [
            new_handle if handle is old_handle else handle for handle in handles
        ]
        return new_handle

    @classmethod
    def register_dependency(
        cls, handle: _PardisoHandle, dependency: PyTreeT
    ) -> PyTreeT:
        cls._handle_dependencies.register(handle, dependency)
        return dependency


T = TypeVar("T")


def _ensure_cpu(args: T) -> T:
    """Return `args` unchanged, raising if the current platform is not CPU.

    A local copy of `_klu.py`'s helper of the same name, with a Pardiso-specific
    message: the two solvers wrap different CPU-only native libraries.
    """
    on_cpu = jax.lax.platform_dependent(
        args,
        default=lambda _: jnp.bool_(False),
        cpu=lambda _: jnp.bool_(True),
    )
    return eqx.error_if(
        args,
        ~on_cpu,
        "`Pardiso` can only solve on CPU; it wraps the CPU-only Intel oneMKL Pardiso "
        "solver.",
    )


def _csr_from_coo_pattern(
    rows: Integer[Array, " nse"],
    cols: Integer[Array, " nse"],
    shape: tuple[int, ...],
    values: Inexact[Array, " nse"] | None = None,
) -> tuple[Integer[Array, " n+1"], Integer[Array, " nse"], Inexact[Array, " nse"]]:
    """Convert a COO `(row, col)` sparsity pattern to sorted CSR `(indptr, indices, values)`.

    `values` is optional because some `factorize_symbolic` inputs (a bare sparsity
    pattern, with no associated matrix) carry no numeric data. When omitted, a dummy
    `1.0` is used instead, exactly as for the pattern-only conversion this replaces:
    the symbolic analysis this feeds only needs *some* representative values to run,
    not necessarily meaningful ones, and every later solve refactors with the real
    values from the operator being solved.
    """
    if values is None:
        values = jnp.ones(rows.shape[0], dtype=jnp.float64)
    else:
        values = values.astype(jnp.float64)
    bcsr = BCSR.from_bcoo(
        BCOO((values, jnp.stack([rows, cols], axis=1)), shape=tuple(shape))
    )
    return bcsr.indptr.astype(jnp.int32), bcsr.indices.astype(jnp.int32), bcsr.data


class _PardisoBasicState(NamedTuple):
    csr: _CSR
    shape: tuple[int, ...]
    packed_structures: PackedStructures
    transposed: bool = False

    @contextmanager
    def factorize(self) -> Iterator["_PardisoNumericState"]:
        primitive = _pardiso_mkl_jax().primitive
        pmj = _pardiso_mkl_jax()
        indptr, indices, values = self.csr

        with _PardisoHandleAllocationScopeManager.begin_scope():
            value = primitive.analyze(
                indptr, indices, values, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
            )
            value = primitive.factor(
                value,
                indptr,
                indices,
                values,
                matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
            )
            handle = _PardisoHandleAllocationScopeManager.register_handle(value)
            yield _PardisoNumericState(
                self.csr, handle, self.packed_structures, self.shape, self.transposed
            )


class _PardisoSymbolicScope(NamedTuple):
    shape: tuple[int, ...]
    handle: _PardisoHandle

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> "_PardisoSymbolicState":
        match operator:
            case SparseJacobianLinearOperator():
                # Materialise the Jacobian into a `BCOOLinearOperator` and reuse the
                # BCOO path below.
                return self.init(materialise(operator), options)
            case BCSRLinearOperator(matrix):
                bcoo = matrix.to_bcoo()
            case BCOOLinearOperator(matrix):
                bcoo = matrix
            case _:
                raise TypeError(
                    "`Pardiso.factorize_symbolic` scope's `.init` requires a "
                    "`BCOOLinearOperator`, `BCSRLinearOperator`, or "
                    "`SparseJacobianLinearOperator`; got "
                    f"{type(operator).__name__}."
                )

        if bcoo.data.dtype in COMPLEX_DTYPES:
            raise TypeError(
                "`Pardiso` only supports real-valued matrices; `pardiso_mkl_jax` does "
                f"not support complex matrix types yet. Got dtype {bcoo.data.dtype}."
            )

        # `BCSR.from_bcoo` sorts into the same canonical (row, then column) order used
        # to build the scope's pattern in `Pardiso.factorize_symbolic`, so the reordered
        # values line up with the indices the stored `handle` was analyzed against.
        matrix_bcsr = BCSR.from_bcoo(bcoo)
        indptr = matrix_bcsr.indptr.astype(jnp.int32)
        indices = matrix_bcsr.indices.astype(jnp.int32)
        values = matrix_bcsr.data.astype(jnp.float64)
        packed_structures = pack_structures(operator)

        return _PardisoSymbolicState(
            (indptr, indices, values), packed_structures, self.handle, self.shape
        )

    @contextmanager
    def factorize(
        self, operator: AbstractLinearOperator
    ) -> Iterator["_PardisoNumericState"]:
        with self.init(operator).factorize() as state:
            yield state


class _PardisoSymbolicState(eqx.Module):
    """A solvable state that reuses a `factorize_symbolic` scope's symbolic analysis.

    The analysis was run once, when the scope was opened. Each `compute` reuses it and
    refactors numerically for `csr`'s values in one fused jit-safe call, so `handle` and
    `csr` are carried as dynamic pytree leaves and may be tracers, which is what lets the
    whole scope, not just this state, compose inside a jitted function. `.factorize()`
    promotes this to a `_PardisoNumericState` by running the numeric factorization once,
    to reuse it across many solves; it does not open its own handle-freeing scope, since
    the resulting numeric state shares the same handle the outer `factorize_symbolic`
    scope already owns and will free.
    """

    csr: _CSR
    packed_structures: PackedStructures
    handle: _PardisoHandle
    # `shape` and `transposed` are static metadata, not traced leaves: `compute` and
    # `transpose` branch on `transposed` under AD tracing, where a traced leaf could not
    # be used in `if`, and `shape` is a tuple of plain Python ints throughout this module.
    shape: tuple[int, ...] = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)

    @contextmanager
    def factorize(self) -> Iterator["_PardisoNumericState"]:
        primitive = _pardiso_mkl_jax().primitive
        pmj = _pardiso_mkl_jax()
        indptr, indices, values = self.csr
        new_value = primitive.factor(
            handle_value(self.handle),
            indptr,
            indices,
            values,
            matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        )
        handle = _PardisoHandleAllocationScopeManager.rebind_handle(
            self.handle, new_value
        )
        yield _PardisoNumericState(
            self.csr, handle, self.packed_structures, self.shape, self.transposed
        )

    def _register_solve_dependency(self, value: Array) -> None:
        # Lets `splineax.linear_solve` order this state's handle free after `value`,
        # picked up by `_sparse.linear_solve` via duck typing, see `_handle.py`.
        _PardisoHandleAllocationScopeManager.register_dependency(self.handle, value)


class _PardisoNumericState(eqx.Module):
    csr: _CSR
    handle: _PardisoHandle
    packed_structures: PackedStructures
    shape: tuple[int, ...] = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)

    def _register_solve_dependency(self, value: Array) -> None:
        # Same hook as `_PardisoSymbolicState`, see its comment.
        _PardisoHandleAllocationScopeManager.register_dependency(self.handle, value)


_PardisoState = _PardisoBasicState | _PardisoSymbolicState | _PardisoNumericState


class Pardiso(AbstractSparseLinearSolver[_PardisoState]):
    """Sparse direct solver wrapping `pardiso_mkl_jax` (Intel oneMKL Pardiso).

    This solver keeps the operator in its native sparse (CSR) storage rather than
    densifying it, and so is intended for use with the sparse operators in this package
    (`BCOOLinearOperator` and `BCSRLinearOperator`).

    `pardiso_mkl_jax` is **CPU, real-valued, and double-precision only**: `float32`
    inputs are upcast to `float64`, and complex operators raise `TypeError` (Pardiso's
    complex matrix types aren't supported by `pardiso_mkl_jax` yet). It does not enable
    JAX's x64 mode or force the CPU platform on import, so `jax_enable_x64` must already
    be on before this solver runs.

    This solver can only handle square nonsingular operators.

    Requires the optional `pardiso-mkl-jax` dependency (`pip install
    splineax[pardiso]`). Constructing `Pardiso()` raises `ImportError` if it isn't
    installed. `AutoSparseLinearSolver` prefers `Pardiso` over `KLU` on CPU with x64
    enabled, falling back to `KLU` automatically when `pardiso-mkl-jax` is missing.
    """

    def __init__(self) -> None:
        """**Arguments:**

        Nothing.
        """
        if not _pardiso_available():
            raise ImportError(
                "`Pardiso` requires the optional `pardiso-mkl-jax` dependency, which "
                "is not installed. Install it with `pip install splineax[pardiso]` "
                "(or `pip install pardiso-mkl-jax` directly)."
            )

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _PardisoBasicState:
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Pardiso` may only be used for linear solves with square matrices"
            )

        # `pardiso_mkl_jax` consumes a CSR triple with int32 indptr/indices, sorted
        # within each row. We assume the matrix is coalesced (no duplicate indices),
        # matching `KLU`/`Spsolve`.
        match operator:
            case SparseJacobianLinearOperator():
                # Materialise the Jacobian into a `BCOOLinearOperator` and reuse the
                # BCOO path below.
                return self.init(materialise(operator), options)
            case BCSRLinearOperator(matrix):
                # Round-trip an unsorted `BCSR` through `BCOO`, since
                # `BCSR.from_bcoo` sorts.
                matrix_bcsr = (
                    matrix
                    if matrix.indices_sorted
                    else BCSR.from_bcoo(matrix.to_bcoo())
                )
            case BCOOLinearOperator(matrix):
                # `BCSR.from_bcoo` sorts the indices itself when they are not
                # already sorted.
                matrix_bcsr = BCSR.from_bcoo(matrix)
            case _:
                raise TypeError(
                    "`Pardiso` requires a sparse operator backed by a `BCOO` or `BCSR` "
                    "matrix (e.g. `splineax.BCOOLinearOperator` or "
                    "`splineax.BCSRLinearOperator`), or a "
                    f"`splineax.SparseJacobianLinearOperator`; "
                    f"got {type(operator).__name__}."
                )

        if matrix_bcsr.dtype in COMPLEX_DTYPES:
            raise TypeError(
                "`Pardiso` only supports real-valued matrices; `pardiso_mkl_jax` does "
                f"not support complex matrix types yet. Got dtype {matrix_bcsr.dtype}."
            )

        indptr = matrix_bcsr.indptr.astype(jnp.int32)
        indices = matrix_bcsr.indices.astype(jnp.int32)
        values = matrix_bcsr.data.astype(jnp.float64)

        return _PardisoBasicState(
            (indptr, indices, values), matrix_bcsr.shape, pack_structures(operator)
        )

    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        """Pre-compute a full (analysis + numeric) factorization for reuse.

        Equivalent to `self.init(operator, options).factorize()`.
        """
        return factorize_through_init(self, operator, options)

    @contextmanager
    def factorize_symbolic(
        self,
        sparsity: BCOO
        | BCSR
        | BCOOLinearOperator
        | BCSRLinearOperator
        | SparseJacobianLinearOperator
        | SparseJacobianLinearOperatorColoring
        | JacobianColoring,
    ) -> Iterator[_PardisoSymbolicScope]:
        """Open a scope with a pre-computed Pardiso sparsity pattern.

        Yields a `_PardisoSymbolicScope`. Inside the block, call:
        - `.init(operator)` to create a `_PardisoSymbolicState` for `lx.linear_solve`.
          Every solve reuses the analysis performed when this scope was opened and only
          re-runs the numeric phase.
        - `.init(operator).factorize()` or equivalently `.factorize(operator)` to also
          pre-compute the numeric factorization.

        The symbolic analysis runs once, as this scope is opened, using representative
        values from `sparsity` itself where it carries any (a `BCOO`, `BCSR`,
        `BCOOLinearOperator`, or `BCSRLinearOperator`), or a placeholder otherwise (a
        bare sparsity pattern from a coloring). Because the resulting handle is an
        ordinary JAX array value, not a native object, this whole scope, including the
        analysis, composes inside a jitted function: `with solver.factorize_symbolic(...)
        as scope:` may be written directly inside `@jax.jit`, or the scope may be built
        eagerly and passed into one, either way is safe to reuse across solves.

        The handle is released when the `with` block exits, after every solve that used
        it.

        Args:
            sparsity: Sparse matrix whose sparsity pattern to pre-analyze. Accepts the
                      same types as `KLU.factorize_symbolic`: `BCOO`, `BCSR`,
                      `BCOOLinearOperator`, `BCSRLinearOperator`,
                      `SparseJacobianLinearOperator`,
                      `SparseJacobianLinearOperatorColoring`, or `JacobianColoring`.
        """
        values = None
        match sparsity:
            case SparseJacobianLinearOperator(transposed=True):
                # See `KLU.factorize_symbolic`'s matching case for why rows/columns
                # are swapped here.
                pattern = sparsity.coloring.sparsity
                rows = jnp.asarray(pattern.cols, dtype=jnp.int32)
                cols = jnp.asarray(pattern.rows, dtype=jnp.int32)
                shape = pattern.shape[::-1]
            case (
                SparseJacobianLinearOperator() | SparseJacobianLinearOperatorColoring()
            ):
                pattern = sparsity.coloring.sparsity
                rows = jnp.asarray(pattern.rows, dtype=jnp.int32)
                cols = jnp.asarray(pattern.cols, dtype=jnp.int32)
                shape = pattern.shape
            case JacobianColoring():
                pattern = sparsity.sparsity
                rows = jnp.asarray(pattern.rows, dtype=jnp.int32)
                cols = jnp.asarray(pattern.cols, dtype=jnp.int32)
                shape = pattern.shape
            case BCSRLinearOperator():
                bcoo = sparsity.matrix.to_bcoo()
                rows = bcoo.indices[:, 0].astype(jnp.int32)
                cols = bcoo.indices[:, 1].astype(jnp.int32)
                shape = bcoo.shape
                values = bcoo.data
            case BCOOLinearOperator():
                bcoo = sparsity.matrix
                rows = bcoo.indices[:, 0].astype(jnp.int32)
                cols = bcoo.indices[:, 1].astype(jnp.int32)
                shape = bcoo.shape
                values = bcoo.data
            case BCSR():
                bcoo = sparsity.to_bcoo()
                rows = bcoo.indices[:, 0].astype(jnp.int32)
                cols = bcoo.indices[:, 1].astype(jnp.int32)
                shape = bcoo.shape
                values = bcoo.data
            case BCOO():
                rows = sparsity.indices[:, 0].astype(jnp.int32)
                cols = sparsity.indices[:, 1].astype(jnp.int32)
                shape = sparsity.shape
                values = sparsity.data
            case _:
                raise TypeError(
                    "`Pardiso.factorize_symbolic` requires a `BCOO`, `BCSR`, "
                    "`BCOOLinearOperator`, `BCSRLinearOperator`, "
                    "`SparseJacobianLinearOperator`, "
                    "`SparseJacobianLinearOperatorColoring`, or `JacobianColoring`; "
                    f"got {type(sparsity).__name__}."
                )

        if shape[0] != shape[1]:
            raise ValueError(
                f"`Pardiso.factorize_symbolic` requires a square matrix; got shape "
                f"{shape}."
            )

        if values is not None and values.dtype in COMPLEX_DTYPES:
            raise TypeError(
                "`Pardiso` only supports real-valued matrices; `pardiso_mkl_jax` does "
                f"not support complex matrix types yet. Got dtype {values.dtype}."
            )

        indptr, indices, analyze_values = _csr_from_coo_pattern(
            rows, cols, shape, values
        )

        pmj = _pardiso_mkl_jax()
        with _PardisoHandleAllocationScopeManager.begin_scope():
            value = pmj.primitive.analyze(
                indptr,
                indices,
                analyze_values,
                matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
            )
            handle = _PardisoHandleAllocationScopeManager.register_handle(value)
            yield _PardisoSymbolicScope(tuple(shape), handle)

    def compute(
        self,
        state: _PardisoState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options

        b = ravel_vector(vector, state.packed_structures)
        b = _ensure_cpu(b)
        b = b.astype(jnp.float64)

        pmj = _pardiso_mkl_jax()
        primitive = pmj.primitive
        stacked_b = b[None, :]

        match state:
            case _PardisoNumericState(
                csr=(indptr, indices, values), handle=handle, transposed=transposed
            ):
                # Numeric factorization already done eagerly; just solve against it.
                solution = primitive.solve_stateful(
                    handle_value(handle),
                    indptr,
                    indices,
                    values,
                    stacked_b,
                    matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
                    transpose=transposed,
                )
                x = solution[0]
                _PardisoHandleAllocationScopeManager.register_dependency(handle, x)
            case _PardisoSymbolicState(
                csr=(indptr, indices, values), handle=handle, transposed=transposed
            ):
                # Reuse the symbolic analysis, refactor numerically for these values, and
                # solve in one fused call. Safe under jit: the values are passed
                # explicitly rather than stored on any native object.
                solution = primitive.factor_and_solve_stateful(
                    handle_value(handle),
                    indptr,
                    indices,
                    values,
                    stacked_b,
                    matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
                    transpose=transposed,
                )
                x = solution[0]
                _PardisoHandleAllocationScopeManager.register_dependency(handle, x)
            case _PardisoBasicState(
                csr=(indptr, indices, values), transposed=transposed
            ):
                x = pmj.solve(
                    indptr,
                    indices,
                    values,
                    b,
                    matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
                    transpose=transposed,
                )

        solution = unravel_solution(x, state.packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(
        self, state: _PardisoState, options: dict[str, Any]
    ) -> tuple[_PardisoState, dict[str, Any]]:
        del options
        # `pardiso_mkl_jax` solves against A^T natively, reusing whatever
        # factorization was built for A, so unlike `KLU` (which swaps COO row/column
        # arrays) and `Spsolve` (which rebuilds a transposed CSR matrix), transposing
        # here is pure metadata: flip `transposed`, transpose the packed structures,
        # and swap `shape`. Any existing factorization carries over unchanged.
        packed_structures = transpose_packed_structures(state.packed_structures)

        match state:
            case _PardisoNumericState(
                csr=csr, handle=handle, shape=shape, transposed=transposed
            ):
                return _PardisoNumericState(
                    csr, handle, packed_structures, shape[::-1], not transposed
                ), {}
            case _PardisoSymbolicState(
                csr=csr, handle=handle, shape=shape, transposed=transposed
            ):
                return _PardisoSymbolicState(
                    csr, packed_structures, handle, shape[::-1], not transposed
                ), {}
            case _PardisoBasicState(csr=csr, shape=shape, transposed=transposed):
                return _PardisoBasicState(
                    csr, shape[::-1], packed_structures, not transposed
                ), {}

    def conj(
        self, state: _PardisoState, options: dict[str, Any]
    ) -> tuple[_PardisoState, dict[str, Any]]:
        del options
        # Real-only solver (see the class docstring): conjugation is always a no-op.
        return state, {}

    def assume_full_rank(self) -> bool:
        return True
