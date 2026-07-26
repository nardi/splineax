import importlib.util
from contextlib import AbstractContextManager, contextmanager
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Iterator, Literal, NamedTuple, overload

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Inexact, Integer, PyTree
from lineax import AbstractLinearOperator
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
from splineax.operators._jacobian import SparseJacobianLinearOperator
from splineax.solvers._conversion import (
    csr_from_coo_pattern,
    operator_to_sparse_matrix,
    sparsity_to_coo_pattern,
)
from splineax.solvers._klu import COMPLEX_DTYPES
from splineax.solvers._sparse import (
    AbstractSparseLinearSolver,
    SparseNumericState,
    SymbolicScopedSparseLinearSolver,
    _Sparsity,
    as_scoped_solver,
    factorize_through_init,
)

if TYPE_CHECKING:
    # Only for type annotations: the real import stays lazy, see `_spineax_cudss`.
    from spineax.cudss import FactorToken

# Row and column indices plus values, unsorted: the matrix as read off the operator,
# before it has been turned into a real (sorted) CSR triple. Kept in this loose COO
# shape rather than eagerly sorted, so `_CuDSSBasicState.transpose` can swap rows and
# columns directly (mirrors `_klu.py`'s `_COO`); `csr_from_coo_pattern` does the actual
# CSR conversion right before a solve needs it.
_COO = tuple[Integer[Array, " nse"], Integer[Array, " nse"], Inexact[Array, " nse"]]

# cuDSS always sees the full stored matrix (both triangles), never just one, because
# `BCOO`/`BCSR` operators always store both: see `_mtype_id`.
_MVIEW_FULL = 0


def _cudss_available() -> bool:
    """Whether the optional cuDSS binding is importable, without importing it.

    Checked with `importlib.util.find_spec` (no execution), which only imports the
    binding's lightweight top-level package, not its `cudss` submodule: that
    submodule's real import calls `jax.devices()` and dlopens the CUDA extension, and
    must stay deferred until a solve actually runs (mirrors `_pardiso.py`'s
    `_pardiso_available`).

    `find_spec` on a dotted name only returns `None` for a missing *submodule*; if
    the top-level package itself isn't installed at all (the common case here, since
    the binding is an optional dependency), it raises `ModuleNotFoundError` instead,
    which is caught here alongside the "not installed" case it otherwise signals.
    """
    try:
        return importlib.util.find_spec("spineax.cudss") is not None
    except ModuleNotFoundError:
        return False


def _spineax_cudss():
    # Lazy import: deferred until a CuDSS solve actually runs, so importing splineax,
    # or even constructing a `CuDSS` instance, never touches CUDA unless the solver is
    # actually used (mirrors `_klu.py`'s `_klujax()`).
    from spineax import cudss

    return cudss


def _ensure_gpu(args: PyTree[Array]) -> PyTree[Array]:
    """Return `args` unchanged, raising if the current platform is not a CUDA GPU.

    The same `jax.lax.platform_dependent` trick `_klu.py`/`_pardiso.py` use for their
    CPU-only guard, checking the "cuda" backend specifically. `jax.default_backend()`
    reports "gpu" for both CUDA and ROCm, but cuDSS is CUDA-only, so the generic "gpu"
    key would wrongly let ROCm through here.
    """
    on_cuda = jax.lax.platform_dependent(
        args,
        default=lambda _: jnp.bool_(False),
        cuda=lambda _: jnp.bool_(True),
    )
    return eqx.error_if(
        args,
        ~on_cuda,
        "`CuDSS` can only solve on a CUDA GPU; it wraps NVIDIA's CUDA-only cuDSS "
        "library.",
    )


def _mtype_id(operator: AbstractLinearOperator) -> int:
    """The cuDSS matrix-type id for `operator`, read off its lineax tags.

    `0` general (LU), `1` symmetric (LDL^T), `3` symmetric positive semidefinite
    (Cholesky). cuDSS also has Hermitian ids (`2`, `4`), but lineax has no
    `is_hermitian` check to drive them from here, so they never come out of this
    function; `CuDSS.transpose`'s Hermitian handling is written to support them
    anyway, for when that tag exists to select one.
    """
    if lx.is_symmetric(operator):
        return 3 if lx.is_positive_semidefinite(operator) else 1
    return 0


def _mtype_id_for_sparsity(sparsity: _Sparsity) -> int:
    """The cuDSS matrix-type id for a bare `factorize_symbolic` sparsity pattern.

    Only three of the seven `_Sparsity` types carry lineax tags at all
    (`BCOOLinearOperator`, `BCSRLinearOperator`, `SparseJacobianLinearOperator`); a
    bare `BCOO`/`BCSR` or coloring carries no tags, so there is nothing to read and
    this falls back to `0` (general), the always-correct choice.
    """
    match sparsity:
        case (
            BCOOLinearOperator() | BCSRLinearOperator() | SparseJacobianLinearOperator()
        ):
            return _mtype_id(sparsity)
        case _:
            return 0


def _maybe_release(cudss: Any, token: "FactorToken") -> None:
    """Release `token` eagerly, unless its id is still a tracer (running under jit).

    `spineax.cudss.release` calls `jax.device_get` on the token id, so it can only run
    eagerly, never traced (there is no traced release primitive for cuDSS, unlike
    `KLU`'s/`Pardiso`'s native handles, so `_handle.py`'s machinery is not needed
    here at all). Skipping the release under jit is safe: the cache bounds memory on
    its own, evicting old factorizations as needed and transparently rebuilding one
    that is still referenced but was evicted (`spineax.cudss.rebuild_count()` counts
    this). A skipped release is therefore only ever a missed optimization: correct
    but slower, never wrong.
    """
    if not isinstance(token.id, jax.core.Tracer):
        cudss.release(token)


def _transpose_csr(
    offsets: Integer[Array, " n+1"],
    columns: Integer[Array, " nse"],
    values: Inexact[Array, " nse"],
    shape: tuple[int, ...],
) -> tuple[Integer[Array, " n+1"], Integer[Array, " nse"], Inexact[Array, " nse"]]:
    """Transpose a CSR triple by rebuilding it through `BCOO`/`BCSR`.

    cuDSS has no native transpose solve (see the `CuDSS` class docstring), so a
    general (non-symmetric) matrix needs a genuinely re-analyzed and re-factorized
    `A^T`, not just metadata. `BCSR` has no `.T`, so this round-trips through `BCOO`,
    the same way `BCSRLinearOperator.transpose` does.
    """
    bcsr = BCSR((values, columns, offsets), shape=shape)
    bcoo_T = bcsr.to_bcoo().T
    bcsr_T = BCSR.from_bcoo(bcoo_T)
    return (
        bcsr_T.indptr.astype(jnp.int32),
        bcsr_T.indices.astype(jnp.int32),
        bcsr_T.data,
    )


def _cudss_solve(
    cudss: Any,
    token: "FactorToken",
    b: Inexact[Array, " n"],
    ir_nsteps: int | None,
    conjugate_solve: bool,
) -> Inexact[Array, " n"]:
    """Run `spineax.cudss.solve`, conjugating in and out for the Hermitian case.

    cuDSS has no native transpose solve; a Hermitian-family state instead reuses A's
    own factors and solves `conj(A) conj(x) = conj(b)` in place of `A^T x = b` (see
    `CuDSS.transpose`), which only needs the right-hand side conjugated going in and
    the solution conjugated coming out. `conjugate_solve` is `False` everywhere else,
    where this is a no-op.
    """
    if conjugate_solve:
        b = jnp.conj(b)
    x = cudss.solve(token, b, ir_nsteps=ir_nsteps)
    return jnp.conj(x) if conjugate_solve else x


class _CuDSSBasicState(NamedTuple):
    coo: _COO
    shape: tuple[int, ...]
    packed_structures: PackedStructures
    mtype_id: int
    mview_id: int
    device_id: int
    reordering: int
    memory: int
    conjugate_solve: bool = False

    @contextmanager
    def factorize(self) -> Iterator["_CuDSSNumericState"]:
        cudss = _spineax_cudss()
        rows, cols, values = self.coo
        offsets, columns, csr_values = csr_from_coo_pattern(
            rows, cols, self.shape, values, dtype=values.dtype
        )
        token = cudss.analyze(
            csr_values,
            offsets,
            columns,
            mtype_id=self.mtype_id,
            mview_id=self.mview_id,
            device_id=self.device_id,
            reordering=self.reordering,
            memory=self.memory,
        )
        token = cudss.factorize(token, csr_values)
        try:
            yield _CuDSSNumericState(
                token, self.packed_structures, self.shape, self.conjugate_solve
            )
        finally:
            _maybe_release(cudss, token)


class _CuDSSSymbolicScope(NamedTuple):
    shape: tuple[int, ...]
    token: "FactorToken"
    """The analyzed-phase token, shared by every state built from this scope."""

    def init(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> "_CuDSSSymbolicState":
        matrix = operator_to_sparse_matrix(
            operator, error_prefix="`CuDSS.factorize_symbolic` scope's `.init`"
        )
        bcoo = matrix if isinstance(matrix, BCOO) else matrix.to_bcoo()
        matrix_bcsr = (
            matrix
            if isinstance(matrix, BCSR) and matrix.indices_sorted
            else BCSR.from_bcoo(bcoo)
        )
        # The token's own dtype was fixed when the scope was opened; every operator
        # solved through it must match, so cast here rather than let a confusing
        # dtype error surface from inside the binding at `compute` time.
        values = matrix_bcsr.data.astype(self.token.dtype)

        return _CuDSSSymbolicState(
            self.token, values, pack_structures(operator), self.shape
        )

    @contextmanager
    def factorize(
        self, operator: AbstractLinearOperator
    ) -> Iterator["_CuDSSNumericState"]:
        with self.init(operator).factorize() as state:
            yield state


class _CuDSSSymbolicState(eqx.Module):
    """A solvable state that reuses a `factorize_symbolic` scope's symbolic analysis.

    The analysis ran once, when the scope was opened. Each `compute` reuses `token`
    and refactors numerically for `values` (this state's own operator's values), the
    symbolic reuse the scope exists for; see `CuDSS.compute`. `.factorize()` promotes
    this to a `_CuDSSNumericState` by running that numeric factorization once, to
    reuse it across many solves; it does not release anything itself, since the
    resulting token is (or shares the id of) the one the outer `factorize_symbolic`
    scope already owns and will release when it closes.
    """

    token: "FactorToken"
    values: Inexact[Array, " nse"]
    packed_structures: PackedStructures
    # `shape` and `conjugate_solve` are static metadata, not traced leaves: `compute`
    # and `transpose` branch on them under AD tracing, where a traced leaf could not
    # be used in `if`.
    shape: tuple[int, ...] = eqx.field(static=True)
    conjugate_solve: bool = eqx.field(static=True, default=False)

    @contextmanager
    def factorize(self) -> Iterator["_CuDSSNumericState"]:
        cudss = _spineax_cudss()
        token = cudss.factorize(self.token, self.values)
        yield _CuDSSNumericState(
            token, self.packed_structures, self.shape, self.conjugate_solve
        )


class _CuDSSNumericState(eqx.Module):
    token: "FactorToken"
    packed_structures: PackedStructures
    shape: tuple[int, ...] = eqx.field(static=True)
    conjugate_solve: bool = eqx.field(static=True, default=False)

    # No `_register_solve_dependency`, unlike `_KLUSymbolicState`/`_PardisoSymbolicState`
    # and their numeric counterparts. cuDSS's `release` is eager-only (see
    # `_maybe_release`), so there is no traced release primitive for a solve's result
    # to be ordered against, and `splineax.linear_solve`'s `isinstance(state,
    # _HandleOwningState)` check simply finds neither `_CuDSSSymbolicState` nor
    # `_CuDSSNumericState` a match and does nothing, which is exactly right here.


_CuDSSState = _CuDSSBasicState | _CuDSSSymbolicState | _CuDSSNumericState


class CuDSSReordering(IntEnum):
    """Fill-reducing reordering passed to `analyze`. Distinct from `ReorderingScheme`
    (`Spsolve`'s cuSOLVER reordering, an unrelated enum despite the similar name)."""

    DEFAULT = 0
    BTF_COLAMD = 1
    COLAMD = 2
    AMD = 3
    NESTED_DISSECTION = 4
    NONE = 5


class CuDSSMemory(IntEnum):
    """Where cuDSS keeps the numeric factors."""

    DEVICE = 0
    """Factors live entirely in device (GPU) memory."""
    HYBRID = 1
    """Host and device factors, for problems whose factors don't fit on the device."""


class CuDSS(AbstractSparseLinearSolver[_CuDSSState]):
    """Sparse direct solver wrapping NVIDIA's cuDSS library.

    Unlike `KLU`/`Pardiso` (CPU-only) or `Spsolve` (no factorization reuse), this
    solver runs on a CUDA GPU and supports the full three-tier factorization-reuse
    API (`factorize`, `factorize_symbolic`) with real numeric refactorization. It
    keeps the operator in its native sparse (CSR) storage rather than densifying it,
    and so is intended for use with the sparse operators in this package
    (`BCOOLinearOperator` and `BCSRLinearOperator`).

    Supports `float32`, `float64`, `complex64`, and `complex128` directly, with no
    upcasting, unlike `KLU`/`Pardiso`.

    This solver can only handle square nonsingular operators, and only runs on a
    CUDA GPU (not ROCm, not CPU, not TPU): an error is raised at trace time
    otherwise.

    Every factorization lives in a size-bounded cache rather than behind an explicit
    handle, so `factorize`/`factorize_symbolic` scopes opened under `jax.jit` skip
    the eager release `KLU`/`Pardiso` perform when they close: the cache evicts old
    factorizations on its own, transparently (and correctly, if more slowly)
    rebuilding one that is still referenced but was evicted. See the "Advanced
    usage" guide for details.

    Requires the optional cuDSS dependency (`pip install splineax[cudss]`; CUDA 13,
    Python >=3.12, x86_64 Linux only). Constructing `CuDSS()` raises `ImportError`
    if it isn't installed. `AutoSparseLinearSolver` prefers `CuDSS` on a CUDA GPU
    when it is.

    A plain, un-stated solve (`lx.linear_solve(op, b, solver=CuDSS())` with no
    `state=`) re-runs the analysis on every call, minting a fresh cache entry each
    time. That's correct but wasteful: prefer `factorize`/`factorize_symbolic` for
    anything solved more than once, exactly as recommended for `KLU`.
    """

    reordering: CuDSSReordering = eqx.field(static=True)
    memory: CuDSSMemory = eqx.field(static=True)
    device_id: int = eqx.field(static=True)

    def __init__(
        self,
        reordering: CuDSSReordering = CuDSSReordering.DEFAULT,
        memory: CuDSSMemory = CuDSSMemory.DEVICE,
        device_id: int = 0,
    ) -> None:
        """**Arguments:**

        - `reordering`: fill-reducing reordering scheme passed to `analyze`.
            Defaults to `CuDSSReordering.DEFAULT`.
        - `memory`: where cuDSS keeps the numeric factors. Defaults to
            `CuDSSMemory.DEVICE`.
        - `device_id`: CUDA device index to run on. Defaults to `0`.
        """
        if not _cudss_available():
            raise ImportError(
                "`CuDSS` requires the optional cuDSS dependency, which is not "
                "installed. Install it with `pip install splineax[cudss]`."
            )
        self.reordering = reordering
        self.memory = memory
        self.device_id = device_id

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _CuDSSBasicState:
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`CuDSS` may only be used for linear solves with square matrices"
            )

        matrix = operator_to_sparse_matrix(operator, error_prefix="`CuDSS`")
        bcoo = matrix if isinstance(matrix, BCOO) else matrix.to_bcoo()
        rows = bcoo.indices[:, 0].astype(jnp.int32)
        cols = bcoo.indices[:, 1].astype(jnp.int32)

        return _CuDSSBasicState(
            (rows, cols, bcoo.data),
            bcoo.shape,
            pack_structures(operator),
            _mtype_id(operator),
            _MVIEW_FULL,
            self.device_id,
            int(self.reordering),
            int(self.memory),
        )

    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        """Pre-compute a full (analysis + numeric) factorization for reuse.

        Equivalent to `self.init(operator, options).factorize()`.
        """
        return factorize_through_init(self, operator, options)

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[False] = False
    ) -> AbstractContextManager[_CuDSSSymbolicScope]: ...

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[True]
    ) -> AbstractContextManager[SymbolicScopedSparseLinearSolver]: ...

    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: bool = False
    ) -> AbstractContextManager[_CuDSSSymbolicScope | SymbolicScopedSparseLinearSolver]:
        """Open a scope with a pre-computed cuDSS symbolic analysis.

        Yields a `_CuDSSSymbolicScope`. Inside the block, call:
        - `.init(operator)` to create a `_CuDSSSymbolicState` for `lx.linear_solve`.
          Every solve reuses the analysis performed when this scope was opened and
          only re-runs the numeric phase.
        - `.init(operator).factorize()` or equivalently `.factorize(operator)` to
          also pre-compute the numeric factorization.

        The symbolic analysis runs once, as this scope is opened, using
        representative values from `sparsity` itself where it carries any (a `BCOO`,
        `BCSR`, `BCOOLinearOperator`, or `BCSRLinearOperator`), or a placeholder
        otherwise (a bare sparsity pattern from a coloring), matching every value
        solved through this scope to the resulting token's dtype.

        Args:
            sparsity: Sparse matrix whose sparsity pattern to pre-analyze. Accepts
                      the same types as `KLU.factorize_symbolic`: `BCOO`, `BCSR`,
                      `BCOOLinearOperator`, `BCSRLinearOperator`,
                      `SparseJacobianLinearOperator`,
                      `SparseJacobianLinearOperatorColoring`, or `JacobianColoring`.
            as_solver: Yield a `SymbolicScopedSparseLinearSolver` pairing the scope
                       with this solver, instead of the bare scope, so that the two
                       need not be passed around together.
        """
        scope = self._factorize_symbolic(sparsity)
        return as_scoped_solver(self, scope) if as_solver else scope

    @contextmanager
    def _factorize_symbolic(self, sparsity: _Sparsity) -> Iterator[_CuDSSSymbolicScope]:
        # The scope itself, kept separate from `factorize_symbolic` above so that the
        # public method can be overloaded on `as_solver` (`@contextmanager` and
        # `@overload` do not compose).
        rows, cols, shape, values = sparsity_to_coo_pattern(
            sparsity, error_prefix="`CuDSS.factorize_symbolic`"
        )

        if shape[0] != shape[1]:
            raise ValueError(
                f"`CuDSS.factorize_symbolic` requires a square matrix; got shape "
                f"{shape}."
            )

        # A bare coloring carries no representative values, so the analysis (and so
        # every operator later solved through this scope) falls back to float64.
        # Pass a `BCOO`/`BCSR`/operator with real values instead for control over the
        # dtype this scope commits to.
        dtype = values.dtype if values is not None else jnp.float64
        offsets, columns, analyze_values = csr_from_coo_pattern(
            rows, cols, shape, values, dtype=dtype
        )

        cudss = _spineax_cudss()
        token = cudss.analyze(
            analyze_values,
            offsets,
            columns,
            mtype_id=_mtype_id_for_sparsity(sparsity),
            mview_id=_MVIEW_FULL,
            device_id=self.device_id,
            reordering=int(self.reordering),
            memory=int(self.memory),
        )
        try:
            yield _CuDSSSymbolicScope(shape, token)
        finally:
            _maybe_release(cudss, token)

    def refactorize(
        self, numeric_state: _CuDSSNumericState, operator: AbstractLinearOperator
    ) -> _CuDSSNumericState:
        """Refactor numerically for `operator`'s values, reusing `numeric_state`'s
        pivots, for the case where the values changed but the sparsity pattern and
        the pivoting choice made for it are still trustworthy.

        An explicit escape hatch outside the `SparseLinearSolver` protocol (every
        other solver's `compute` always re-selects pivots): skips pivot re-selection,
        so it is cheaper than solving through `factorize`, but can lose accuracy if
        the values changed by enough that the old pivots are no longer good ones.
        `numeric_state` must share `operator`'s sparsity pattern; nothing here checks
        that, so a mismatched pattern is undefined behaviour.
        """
        cudss = _spineax_cudss()
        matrix = operator_to_sparse_matrix(operator, error_prefix="`CuDSS.refactorize`")
        bcoo = matrix if isinstance(matrix, BCOO) else matrix.to_bcoo()
        matrix_bcsr = (
            matrix
            if isinstance(matrix, BCSR) and matrix.indices_sorted
            else BCSR.from_bcoo(bcoo)
        )
        values = matrix_bcsr.data.astype(numeric_state.token.dtype)
        token = cudss.refactorize(numeric_state.token, values)
        return _CuDSSNumericState(
            token,
            pack_structures(operator),
            numeric_state.shape,
            numeric_state.conjugate_solve,
        )

    def compute(
        self,
        state: _CuDSSState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        cudss = _spineax_cudss()
        b = ravel_vector(vector, state.packed_structures)
        b = _ensure_gpu(b)
        ir_nsteps = options.get("ir_nsteps")

        match state:
            case _CuDSSNumericState(token=token, conjugate_solve=conjugate_solve):
                # Numeric factorization already done eagerly; just solve against it.
                b = b.astype(token.dtype)
                x = _cudss_solve(cudss, token, b, ir_nsteps, conjugate_solve)
            case _CuDSSSymbolicState(
                token=token, values=values, conjugate_solve=conjugate_solve
            ):
                # Reuse the symbolic analysis and refactor numerically for these
                # values; this is the symbolic reuse the scope exists for.
                token = cudss.factorize(token, values)
                b = b.astype(token.dtype)
                x = _cudss_solve(cudss, token, b, ir_nsteps, conjugate_solve)
            case _CuDSSBasicState(
                coo=(rows, cols, values),
                shape=shape,
                mtype_id=mtype_id,
                mview_id=mview_id,
                device_id=device_id,
                reordering=reordering,
                memory=memory,
                conjugate_solve=conjugate_solve,
            ):
                # One-shot solve: analyze, factorize, solve, then release right
                # away, since nothing else can reuse this token afterwards.
                offsets, columns, csr_values = csr_from_coo_pattern(
                    rows, cols, shape, values, dtype=values.dtype
                )
                token = cudss.analyze(
                    csr_values,
                    offsets,
                    columns,
                    mtype_id=mtype_id,
                    mview_id=mview_id,
                    device_id=device_id,
                    reordering=reordering,
                    memory=memory,
                )
                token = cudss.factorize(token, csr_values)
                b = b.astype(token.dtype)
                x = _cudss_solve(cudss, token, b, ir_nsteps, conjugate_solve)
                _maybe_release(cudss, token)

        solution = unravel_solution(x, state.packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(
        self, state: _CuDSSState, options: dict[str, Any]
    ) -> tuple[_CuDSSState, dict[str, Any]]:
        del options
        packed_structures = transpose_packed_structures(state.packed_structures)

        match state:
            case _CuDSSNumericState(token=token, shape=shape):
                if token.mtype_id == 0:
                    cudss = _spineax_cudss()
                    offsets_T, columns_T, values_T = _transpose_csr(
                        token.offsets, token.columns, token.values, shape
                    )
                    new_token = cudss.analyze(
                        values_T,
                        offsets_T,
                        columns_T,
                        mtype_id=token.mtype_id,
                        mview_id=token.mview_id,
                        device_id=token.device_id,
                        reordering=token.reordering_id,
                        memory=token.memory_id,
                    )
                    new_token = cudss.factorize(new_token, values_T)
                    return _CuDSSNumericState(
                        new_token, packed_structures, shape[::-1], False
                    ), {}
                # Symmetric/Hermitian/SPD/HPD: A^T shares the same factors, no new
                # factorization needed (mtype 2/4, Hermitian, need `conj` around the
                # solve; see `_cudss_solve`, and the `_mtype_id` docstring for why
                # they are currently unreachable here).
                conjugate_solve = token.mtype_id in (2, 4)
                return _CuDSSNumericState(
                    token, packed_structures, shape[::-1], conjugate_solve
                ), {}
            case _CuDSSSymbolicState(token=token, values=values, shape=shape):
                if token.mtype_id == 0:
                    cudss = _spineax_cudss()
                    offsets_T, columns_T, values_T = _transpose_csr(
                        token.offsets, token.columns, values, shape
                    )
                    new_token = cudss.analyze(
                        values_T,
                        offsets_T,
                        columns_T,
                        mtype_id=token.mtype_id,
                        mview_id=token.mview_id,
                        device_id=token.device_id,
                        reordering=token.reordering_id,
                        memory=token.memory_id,
                    )
                    return _CuDSSSymbolicState(
                        new_token, values_T, packed_structures, shape[::-1], False
                    ), {}
                conjugate_solve = token.mtype_id in (2, 4)
                return _CuDSSSymbolicState(
                    token, values, packed_structures, shape[::-1], conjugate_solve
                ), {}
            case _CuDSSBasicState(
                coo=(rows, cols, values),
                shape=shape,
                mtype_id=mtype_id,
                mview_id=mview_id,
                device_id=device_id,
                reordering=reordering,
                memory=memory,
            ):
                # Nothing factorized yet: swap the row/column indices (a COO-style
                # transpose, same trick `_klu.py` uses) and defer everything else to
                # whenever `.factorize()`/`compute` actually runs.
                return _CuDSSBasicState(
                    (cols, rows, values),
                    shape[::-1],
                    packed_structures,
                    mtype_id,
                    mview_id,
                    device_id,
                    reordering,
                    memory,
                ), {}

    def conj(
        self, state: _CuDSSState, options: dict[str, Any]
    ) -> tuple[_CuDSSState, dict[str, Any]]:
        del options

        match state:
            case _CuDSSNumericState(token=token):
                dtype = token.dtype
            case _CuDSSSymbolicState(values=values):
                dtype = values.dtype
            case _CuDSSBasicState(coo=(_, _, values)):
                dtype = values.dtype

        if dtype not in COMPLEX_DTYPES:
            # Real: conj is a no-op for every state type.
            return state, {}

        match state:
            case _CuDSSNumericState(
                token=token, packed_structures=packed_structures, shape=shape
            ):
                # Conjugating values doesn't change their magnitudes, so the
                # existing pivots stay numerically valid: `refactorize` (pivot
                # reuse) is exactly right here, and cheaper than a fresh `factorize`.
                cudss = _spineax_cudss()
                conj_values = jnp.conj(token.values)
                new_token = cudss.refactorize(token, conj_values)
                return _CuDSSNumericState(
                    new_token, packed_structures, shape, False
                ), {}
            case _CuDSSSymbolicState(
                token=token,
                values=values,
                packed_structures=packed_structures,
                shape=shape,
            ):
                # The analyzed pattern is unchanged; only the values `compute`
                # refactors with each call need conjugating.
                return _CuDSSSymbolicState(
                    token, jnp.conj(values), packed_structures, shape, False
                ), {}
            case _CuDSSBasicState(
                coo=(rows, cols, values),
                shape=shape,
                packed_structures=packed_structures,
                mtype_id=mtype_id,
                mview_id=mview_id,
                device_id=device_id,
                reordering=reordering,
                memory=memory,
            ):
                return _CuDSSBasicState(
                    (rows, cols, jnp.conj(values)),
                    shape,
                    packed_structures,
                    mtype_id,
                    mview_id,
                    device_id,
                    reordering,
                    memory,
                ), {}

    def assume_full_rank(self) -> bool:
        return True
