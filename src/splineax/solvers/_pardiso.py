import importlib.util
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Iterator, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.core
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
from splineax.solvers._klu import COMPLEX_DTYPES
from splineax.solvers._sparse import (
    AbstractSparseLinearSolver,
    SparseNumericState,
    factorize_through_init,
)

# `indptr`, `indices`, `values`: the matrix in CSR form.
_CSR = tuple[Integer[Array, " n+1"], Integer[Array, " nse"], Inexact[Array, " nse"]]


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
    rows: Integer[Array, " nse"], cols: Integer[Array, " nse"], shape: tuple[int, ...]
) -> tuple[Integer[Array, " n+1"], Integer[Array, " nse"]]:
    """Convert a COO `(row, col)` sparsity pattern to sorted CSR `(indptr, indices)`.

    Values are irrelevant for the pattern alone, so a dummy `1.0` is used. `.init()`
    on the resulting `_PardisoSymbolicScope` later reorders each operator's real
    values through the same `BCSR.from_bcoo` round-trip, so they end up in this same
    canonical (row, then column) order and stay aligned with these `indices`.
    """
    dummy_data = jnp.ones(rows.shape[0], dtype=jnp.float64)
    bcsr = BCSR.from_bcoo(
        BCOO((dummy_data, jnp.stack([rows, cols], axis=1)), shape=tuple(shape))
    )
    return bcsr.indptr.astype(jnp.int32), bcsr.indices.astype(jnp.int32)


class _PardisoHandle:
    """Mutable wrapper around one open `pardiso_mkl_jax.PardisoSolver`.

    Tracks whether the symbolic analysis has run, so it is done exactly once. The
    analysis needs a representative values array (`pardiso_mkl_jax`'s non-symmetric
    heuristics look at numeric values for scaling and matching), so unlike `klujax`'s
    purely structural analysis it cannot run from sparsity alone. `refactor_and_solve`
    reuses that analysis in a single jit-safe call, and is what the symbolic tier uses;
    `factorize`/`solve` back the numeric tier, whose numeric factorization is done eagerly
    and reused across solves.
    """

    def __init__(self, solver: Any) -> None:
        self.solver = solver
        self._analyzed = False
        self._factorized = False

    def analyze(self, values: Inexact[Array, " nse"]) -> None:
        if not self._analyzed:
            self.solver.analyze(values)
            self._analyzed = True

    def factorize(self, values: Inexact[Array, " nse"]) -> None:
        self.analyze(values)
        if self._factorized:
            self.solver.refactorize(values)
        else:
            self.solver.factorize(values)
            self._factorized = True

    def refactor_and_solve(
        self,
        values: Inexact[Array, " nse"],
        right_hand_side: Inexact[Array, " n"],
        *,
        transpose: bool,
    ) -> Inexact[Array, " n"]:
        return self.solver.refactor_and_solve(
            values, right_hand_side, transpose=transpose
        )

    def solve(
        self, right_hand_side: Inexact[Array, " n"], *, transpose: bool
    ) -> Inexact[Array, " n"]:
        return self.solver.solve(right_hand_side, transpose=transpose)


class _PardisoBasicState(NamedTuple):
    csr: _CSR
    shape: tuple[int, ...]
    packed_structures: PackedStructures
    transposed: bool = False

    @contextmanager
    def factorize(self) -> Iterator["_PardisoNumericState"]:
        pmj = _pardiso_mkl_jax()
        indptr, indices, values = self.csr

        with pmj.PardisoSolver(
            indptr, indices, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
        ) as solver:
            handle = _PardisoHandle(solver)
            handle.factorize(values)
            yield _PardisoNumericState(
                handle, self.packed_structures, self.shape, self.transposed
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
        # to build `self.handle`'s pattern in `Pardiso.factorize_symbolic`, so the
        # reordered values line up with that pattern's indices.
        matrix_bcsr = BCSR.from_bcoo(bcoo)
        indptr = matrix_bcsr.indptr.astype(jnp.int32)
        indices = matrix_bcsr.indices.astype(jnp.int32)
        values = matrix_bcsr.data.astype(jnp.float64)
        packed_structures = pack_structures(operator)

        if not self.handle._analyzed:
            # The symbolic analysis needs concrete representative values and must run
            # exactly once, eagerly. Running it under a trace would bake it into every
            # solve. Callers reuse the analysis under jit by doing an eager `init` first,
            # after which this is skipped and only `compute` (a single fused
            # factor-and-solve call) runs inside the trace.
            if isinstance(values, jax.core.Tracer):
                raise RuntimeError(
                    "`Pardiso.factorize_symbolic` needs an eager `scope.init(...)` before "
                    "the scope is reused inside a jitted function, so the symbolic "
                    "analysis runs once outside the trace."
                )
            self.handle.analyze(values)

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

    The analysis was run once, eagerly, in `_PardisoSymbolicScope.init`. Each `compute`
    reuses it and refactors numerically for `csr`'s values in one fused jit-safe call, so
    the values are carried as dynamic data and may be tracers. `.factorize()` promotes
    this to a `_PardisoNumericState` by running the numeric factorization eagerly once, to
    reuse it across many solves.
    """

    csr: _CSR
    packed_structures: PackedStructures
    # `handle`, `shape`, and `transposed` are static metadata, not traced leaves:
    # `handle` wraps a native `PardisoSolver` (not a JAX array) and so cannot be a
    # dynamic pytree leaf at all, and `compute` branches on `transposed` under AD
    # tracing, where a traced leaf could not be used in `if`.
    handle: _PardisoHandle = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)

    @contextmanager
    def factorize(self) -> Iterator["_PardisoNumericState"]:
        # The analysis is already done; this runs the numeric factorization eagerly so it
        # can be reused across the numeric state's solves.
        self.handle.factorize(self.csr[2])
        yield _PardisoNumericState(
            self.handle, self.packed_structures, self.shape, self.transposed
        )


class _PardisoNumericState(eqx.Module):
    handle: _PardisoHandle = eqx.field(static=True)
    packed_structures: PackedStructures
    shape: tuple[int, ...] = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)


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
          The first, eager `.init(operator)` runs Pardiso's symbolic analysis. Unlike
          `klujax`, `pardiso_mkl_jax` needs representative values for analysis, not just
          the pattern, so this cannot happen at scope creation. Every solve then reuses
          the analysis and only re-runs the numeric phase.
        - `.init(operator).factorize()` or equivalently `.factorize(operator)` to also
          pre-compute the numeric factorization.

        Once the analysis has run (an eager `.init`), the scope can be passed into a
        jitted function that builds the operator inside and calls `.init` again, so the
        analysis is reused across solves whose values are only known under the trace.
        Reusing the scope under a trace before that first eager `.init` raises, since the
        analysis cannot run inside a trace.

        The underlying `PardisoSolver` is closed when the `with` block exits.

        Args:
            sparsity: Sparse matrix whose sparsity pattern to pre-analyze. Accepts the
                      same types as `KLU.factorize_symbolic`: `BCOO`, `BCSR`,
                      `BCOOLinearOperator`, `BCSRLinearOperator`,
                      `SparseJacobianLinearOperator`,
                      `SparseJacobianLinearOperatorColoring`, or `JacobianColoring`.
        """
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
            case BCOOLinearOperator():
                bcoo = sparsity.matrix
                rows = bcoo.indices[:, 0].astype(jnp.int32)
                cols = bcoo.indices[:, 1].astype(jnp.int32)
                shape = bcoo.shape
            case BCSR():
                bcoo = sparsity.to_bcoo()
                rows = bcoo.indices[:, 0].astype(jnp.int32)
                cols = bcoo.indices[:, 1].astype(jnp.int32)
                shape = bcoo.shape
            case BCOO():
                rows = sparsity.indices[:, 0].astype(jnp.int32)
                cols = sparsity.indices[:, 1].astype(jnp.int32)
                shape = sparsity.shape
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

        indptr, indices = _csr_from_coo_pattern(rows, cols, shape)

        pmj = _pardiso_mkl_jax()
        with pmj.PardisoSolver(
            indptr, indices, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
        ) as solver:
            yield _PardisoSymbolicScope(tuple(shape), _PardisoHandle(solver))

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

        match state:
            case _PardisoNumericState(handle=handle, transposed=transposed):
                # Numeric factorization already done eagerly; just solve against it.
                x = handle.solve(b, transpose=transposed)
            case _PardisoSymbolicState(
                csr=(_, _, values), handle=handle, transposed=transposed
            ):
                # Reuse the symbolic analysis, refactor numerically for these values, and
                # solve in one fused call. Safe under jit: the values are passed
                # explicitly rather than stored on the native solver.
                x = handle.refactor_and_solve(values, b, transpose=transposed)
            case _PardisoBasicState(
                csr=(indptr, indices, values), transposed=transposed
            ):
                pmj = _pardiso_mkl_jax()
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
                handle=handle, shape=shape, transposed=transposed
            ):
                return _PardisoNumericState(
                    handle, packed_structures, shape[::-1], not transposed
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
