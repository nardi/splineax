import importlib.util
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import lineax as lx
from entangle_jax import entangle
from jax.experimental.sparse import BCOO, BCSR
from jax.typing import DTypeLike
from jaxtyping import Array, Inexact, Integer, PyTree
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

from splineax._profile import compute_scope, record_operation
from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.solvers._klu import _REFACTOR_RCOND_FLOOR, COMPLEX_DTYPES
from splineax.solvers._sparse import (
    _Sparsity,
    operator_pattern_tag,
    pattern_coordinates,
    profile_inputs,
    sparse_operator,
    sparsity_pattern_tag,
    sparsity_reuse_block,
    update_then_compute,
)

if TYPE_CHECKING:
    # Only for type annotations: the real import stays lazy, see `_spineax_cudss`.
    from spineax.cudss import FactorToken

# CSR triple plus shape: offsets (n+1), column indices (nse), values (nse). This is the
# form cuDSS's `analyze`/`factorize` take, so the state carries it ready to hand over.
_CSR = tuple[Integer[Array, " n+1"], Integer[Array, " nse"], Inexact[Array, " nse"]]

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

    `find_spec` on a dotted name only returns `None` for a missing submodule. If the
    top-level package itself isn't installed at all (the common case here, since the
    binding is an optional dependency), it raises `ModuleNotFoundError` instead, which
    is caught here alongside the "not installed" case it otherwise signals.
    """
    try:
        return importlib.util.find_spec("spineax.cudss") is not None
    except ModuleNotFoundError:
        return False


def _spineax_cudss():
    # Lazy import: deferred until a CuDSS solve actually runs, so importing splineax, or
    # even constructing a `CuDSS` instance, never touches CUDA unless the solver is
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
    function. `CuDSS.transpose`'s Hermitian handling is written to support them anyway,
    for when that tag exists to select one.
    """
    if lx.is_symmetric(operator):
        return 3 if lx.is_positive_semidefinite(operator) else 1
    return 0


def _mtype_id_for_sparsity(sparsity: _Sparsity) -> int:
    """The cuDSS matrix-type id for a bare `init_symbolic` sparsity pattern.

    Only the operator forms of `_Sparsity` carry lineax tags. A bare `BCOO`/`BCSR`, tag
    or coloring carries none, so there is nothing to read and this falls back to `0`
    (general), the always-correct choice.
    """
    match sparsity:
        case lx.AbstractLinearOperator():
            return _mtype_id(sparsity)
        case _:
            return 0


def _operator_to_bcsr(
    operator: AbstractLinearOperator,
) -> tuple[BCSR, tuple[int, ...]]:
    """Unwrap a sparse operator into a row-major sorted `BCSR` matrix and its shape.

    A tagged `lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator` is
    materialised as a `BCOO` first. An already-sorted `BCSR` is used as-is. Everything
    else round-trips through `BCOO` so the CSR arrays come out row-major sorted, which
    cuDSS needs.
    """
    match sparse_operator(operator, "CuDSS"):
        case BCSRLinearOperator(matrix=BCSR() as matrix):
            if matrix.indices_sorted:
                return matrix, matrix.shape
            return BCSR.from_bcoo(matrix.to_bcoo()), matrix.shape
        case BCOOLinearOperator(matrix=BCOO() as matrix):
            return BCSR.from_bcoo(matrix), matrix.shape


def _extract_csr(operator: AbstractLinearOperator) -> tuple[_CSR, tuple[int, ...]]:
    """Read an operator's sorted CSR triple and shape.

    The dtype is left as the operator's own: cuDSS supports `float32`, `float64`,
    `complex64`, and `complex128` directly, so there is no upcast, unlike `KLU`/`Pardiso`.
    """
    bcsr, shape = _operator_to_bcsr(operator)
    offsets = bcsr.indptr.astype(jnp.int32)
    columns = bcsr.indices.astype(jnp.int32)
    # Stop gradients on the values before they reach `analyze`/`factorize`. The gradient
    # with respect to the matrix flows through the operator lineax carries, not through
    # the factorization, which is a constant.
    values = jax.lax.stop_gradient(bcsr.data)
    return (offsets, columns, values), tuple(shape)


def _pattern_to_csr(
    sparsity: _Sparsity,
) -> tuple[_CSR, tuple[int, ...]]:
    """Read a sparsity pattern's sorted CSR triple and shape, filling in dummy values.

    A tag, a tagged lineax operator and the coloring forms carry the pattern host-side,
    so the indices are read there rather than materialising anything. Forms that carry
    real values (a `BCOO`, `BCSR`, or their operators) pass those through, so
    `init_symbolic` can analyze with representative numbers and pin the dtype. A pattern
    without values gets `1.0` filled in and the dtype defaults to `float64`. Either way
    every later solve refactors with the operator's real values.
    """
    coordinates = pattern_coordinates(sparsity, "CuDSS")
    values = coordinates.values
    dtype = values.dtype if values is not None else jnp.float64
    csr = _coo_to_csr(
        coordinates.rows, coordinates.columns, coordinates.shape, values, dtype=dtype
    )
    return csr, coordinates.shape


def _coo_to_csr(
    rows: Integer[Array, " nse"],
    cols: Integer[Array, " nse"],
    shape: tuple[int, ...],
    values: Inexact[Array, " nse"] | None,
    *,
    dtype: DTypeLike,
) -> _CSR:
    """Convert a COO `(row, col)` pattern to a sorted CSR `(offsets, columns, values)`.

    `values` is optional because a bare sparsity pattern carries no numbers. When
    omitted, `1.0` fills in: the symbolic analysis this feeds only needs some
    representative values to run, not meaningful ones.
    """
    if values is None:
        values = jnp.ones(rows.shape[0], dtype=dtype)
    else:
        values = values.astype(dtype)
    bcsr = BCSR.from_bcoo(BCOO((values, jnp.stack([rows, cols], axis=1)), shape=shape))
    return bcsr.indptr.astype(jnp.int32), bcsr.indices.astype(jnp.int32), bcsr.data


def _maybe_release(cudss: Any, token: "FactorToken") -> bool:
    """Release `token` eagerly, unless its id is still a tracer (running under jit).

    Returns whether the release ran. `spineax.cudss.release` calls `jax.device_get` on
    the token id, so it can only run eagerly, never traced (there is no traced release
    primitive for cuDSS, unlike klujax's/Pardiso's native handles). Skipping the release
    under jit is safe: the cache bounds memory on its own, evicting old factorizations as
    needed and transparently rebuilding one that is still referenced but was evicted
    (`spineax.cudss.rebuild_count()` counts this). A skipped release is therefore only
    ever a missed optimization: correct but slower, never wrong.
    """
    if isinstance(token.id, jax.core.Tracer):
        return False
    cudss.release(token)
    return True


def _refactorize_or_factorize(
    cudss: Any, token: "FactorToken", values: Inexact[Array, " nse"]
) -> "FactorToken":
    """Refactorize reusing the previous pivots, falling back to a fresh factorize.

    `refactorize` is cheaper than `factorize` under the COLAMD reorderings because it
    reuses the pivots the last factorization chose, but those pivots can be a poor fit for
    new values. The factor's diagonal from `query` gives the same cheap estimate as
    `klu_rcond`, the ratio of the smallest to the largest pivot magnitude. A pivot that
    cuDSS had to perturb to its epsilon also drives the ratio down. Below
    `_REFACTOR_RCOND_FLOOR`, a fresh `factorize` keeps the analysis and picks new pivots.
    Falling back is always correct, only slower.
    """
    refreshed = cudss.refactorize(token, values)
    pivots = jnp.abs(cudss.query(refreshed)["diag"])
    # A zero largest pivot gives NaN, which fails the comparison and so falls back.
    reciprocal_condition = jnp.min(pivots) / jnp.max(pivots)
    reuse_is_safe = reciprocal_condition > _REFACTOR_RCOND_FLOOR
    floor = _REFACTOR_RCOND_FLOOR

    # The cond picks only the token, so no record callback sits inside it (an IO effect in
    # a cond breaks `vmap`-of-cond under a jitted forward-mode derivative). The branch flag
    # rides out as a dynamic value, and the host callback picks the reason below.
    chosen = jax.lax.cond(
        reuse_is_safe,
        lambda: refreshed,
        lambda: cudss.factorize(refreshed, values),
    )
    record_operation(
        "refactorize",
        "CuDSS",
        dynamic={"reused": reuse_is_safe, "rcond": reciprocal_condition},
        outputs=lambda values: {
            "reason": (
                f"Pivots stable: rcond > {floor:g}"
                if values["reused"]
                else f"Pivots unstable: rcond <= {floor:g}, factorized fresh"
            )
        },
    )
    return chosen


def _transpose_csr(
    offsets: Integer[Array, " n+1"],
    columns: Integer[Array, " nse"],
    values: Inexact[Array, " nse"],
    shape: tuple[int, ...],
) -> _CSR:
    """Transpose a CSR triple by rebuilding it through `BCOO`/`BCSR`.

    cuDSS has no native transpose solve (see the `CuDSS` class docstring), so a general
    (non-symmetric) matrix needs a genuinely re-analyzed and re-factorized `A^T`, not
    just metadata. `BCSR` has no `.T`, so this round-trips through `BCOO`, the same way
    `BCSRLinearOperator.transpose` does.
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

    cuDSS has no native transpose solve. A Hermitian-family state instead reuses A's own
    factors and solves `conj(A) conj(x) = conj(b)` in place of `A^T x = b` (see
    `CuDSS.transpose`), which only needs the right-hand side conjugated going in and the
    solution conjugated coming out. `conjugate_solve` is `False` everywhere else, where
    this is a no-op.
    """
    if conjugate_solve:
        b = jnp.conj(b)
    x = cudss.solve(token, b, ir_nsteps=ir_nsteps)
    return jnp.conj(x) if conjugate_solve else x


class _CuDSSState(eqx.Module):
    """A cuDSS solver state, carrying its factorization token.

    The state has two shapes. Straight from `init_symbolic` it holds an analyzed-only
    token and no `packed_structures`, so it is not solvable until `update` folds in an
    operator. After `init` or `update` it holds a factorized token and the structures a
    solve needs.

    Unlike `KLU`, there is no separate symbolic handle to keep alive. cuDSS's numeric
    phases thread the analysis forward through one token: `factorize` renames the
    registry entry rather than dropping the analysis, so `update` reuses the analysis by
    handing the stored token straight back to `factorize`. That rename is why `track`
    matters here. A solve still holding the old token id must run before the `factorize`
    that renames it, or the solve rebuilds the factorization from the token's arrays.
    """

    operator: AbstractLinearOperator | None
    """The operator this state was built on. Compared by identity in `update`."""
    token: "FactorToken"
    """The cuDSS token, analyzed-only or factorized. Also carries the CSR arrays and the
    analyze parameters, which `transpose` reads back."""
    packed_structures: PackedStructures | None
    """The lineax structure for ravel and unravel, None for a symbolic-only state."""
    shape: tuple[int, ...] = eqx.field(static=True)
    conjugate_solve: bool = eqx.field(static=True, default=False)
    sparsity_tag: object | None = eqx.field(static=True, default=None)

    def track(self, solution: Any) -> "_CuDSSState":
        """Return a copy of the state dependent on `solution`.

        Accepts the lineax `Solution` or a bare value pytree. The state becomes dependent
        on the solution values, so that further operations on it, like the `factorize`
        in a later `update`, will be ordered after this call.
        """
        record_operation("track")
        value = getattr(solution, "value", solution)
        # The witness only establishes an execution-order dependency, so stop its
        # gradient: a tracked state must stay usable inside `grad` of the solve.
        witness = jax.lax.stop_gradient(value)
        return _CuDSSState(
            self.operator,
            entangle(self.token, witness),
            self.packed_structures,
            self.shape,
            self.conjugate_solve,
            self.sparsity_tag,
        )

    def release(self) -> None:
        """Release the state's cache entry, unless traced (see `_maybe_release`)."""
        record_operation("release")
        released = _maybe_release(_spineax_cudss(), self.token)
        record_operation(
            "release",
            "CuDSS",
            outputs=None if released else {"note": "skipped under jit"},
        )


class CuDSSReordering(IntEnum):
    """Fill-reducing reordering passed to `analyze`. Distinct from `ReorderingScheme`
    (`Spsolve`'s cuSOLVER reordering, an unrelated enum despite the similar name)."""

    DEFAULT = 0
    BTF_COLAMD = 1
    """Block triangular form plus COLAMD, with global pivoting that `update` can reuse."""
    COLAMD = 2
    """COLAMD, with global pivoting that `update` can reuse."""
    AMD = 3
    NESTED_DISSECTION = 4
    NONE = 5


_REFACTORIZING_REORDERINGS = frozenset(
    {CuDSSReordering.BTF_COLAMD, CuDSSReordering.COLAMD}
)
"""The reorderings under which cuDSS's refactorization phase reuses the previous pivots.
Under every other reordering it runs the same phase as a fresh factorization."""


class CuDSSMemory(IntEnum):
    """Where cuDSS keeps the numeric factors."""

    DEVICE = 0
    """Factors live entirely in device (GPU) memory."""
    HYBRID = 1
    """Host and device factors, for problems whose factors don't fit on the device."""


class CuDSS(AbstractLinearSolver[_CuDSSState]):
    """Sparse direct solver wrapping NVIDIA's cuDSS library.

    Unlike `KLU`/`Pardiso` (CPU-only) or `Spsolve` (no factorization reuse), this solver
    runs on a CUDA GPU and reuses factorizations through the stateful solve API
    (`init`, `init_symbolic`, `update`). It keeps the operator in its native sparse (CSR)
    storage rather than densifying it, and so is intended for use with the sparse
    operators in this package (`BCOOLinearOperator` and `BCSRLinearOperator`).

    Supports `float32`, `float64`, `complex64`, and `complex128` directly, with no
    upcasting, unlike `KLU`/`Pardiso`.

    This solver can only handle square nonsingular operators, and only runs on a CUDA GPU
    (not ROCm, not CPU, not TPU): an error is raised at trace time otherwise.

    Every factorization lives in a size-bounded cache rather than behind an explicit
    handle, so a `release` traced inside `jax.jit` is skipped: the cache evicts old
    factorizations on its own, transparently (and correctly, if more slowly) rebuilding
    one that is still referenced but was evicted. See [Stateful solves](../guide/stateful.md)
    for details.

    `update` on a shared sparsity pattern keeps the analysis and redoes only the numeric
    phase. Under `CuDSSReordering.COLAMD` and `CuDSSReordering.BTF_COLAMD` it refactorizes
    with the previous pivots, and factorizes fresh if the refactorized pivots come out
    badly scaled, the same way `KLU` does. Under every other reordering cuDSS has no
    cheaper refactorization, so `update` always factorizes.

    Requires the optional cuDSS dependency, `pip install splineax[cudss]`, which needs
    CUDA 13, Python >=3.12, and x86_64 Linux. Constructing `CuDSS()` raises `ImportError`
    if it isn't installed. `AutoSparseLinearSolver` prefers `CuDSS` on a CUDA GPU when it
    is.

    A plain, stateless solve (`lx.linear_solve(op, b, solver=CuDSS())` with no `state=`)
    re-runs the analysis on every call, minting a fresh cache entry each time. That is
    correct but wasteful: thread a state through `splineax.linear_solve` for anything
    solved more than once, exactly as recommended for `KLU`.
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
        if not _cudss_available():
            raise ImportError(
                "`CuDSS` requires the optional cuDSS dependency, which is not "
                "installed. Install it with `pip install splineax[cudss]`."
            )
        self.reordering = reordering
        self.memory = memory
        self.device_id = device_id

    def _analyze(self, csr: _CSR, mtype_id: int) -> "FactorToken":
        """Run cuDSS's analysis phase for a CSR pattern, returning an analyzed token."""
        offsets, columns, values = csr
        record_operation("analyze", "CuDSS")
        return _spineax_cudss().analyze(
            values,
            offsets,
            columns,
            mtype_id=mtype_id,
            mview_id=_MVIEW_FULL,
            device_id=self.device_id,
            reordering=int(self.reordering),
            memory=int(self.memory),
        )

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _CuDSSState:
        record_operation(
            "init",
            inputs=lambda: profile_inputs(
                operator,
                operator_pattern_tag(operator),
                (operator.out_size(), operator.in_size()),
            ),
        )
        return self._analyze_and_factor(operator, options)

    def _analyze_and_factor(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _CuDSSState:
        """Analyze and factorize `operator` into a ready-to-solve state.

        Shared by `init` and by `update`'s rebuild path, so the profile records the
        analyze and factorize without a second `init` boundary when a changed pattern
        forces a rebuild.
        """
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`CuDSS` may only be used for linear solves with square matrices"
            )
        csr, shape = _extract_csr(operator)
        # Analyze then factorize right away, so the state is ready to solve and reusable
        # across right-hand sides.
        cudss = _spineax_cudss()
        token = self._analyze(csr, _mtype_id(operator))
        record_operation("factorize", "CuDSS")
        token = cudss.factorize(token, csr[2])
        return _CuDSSState(
            operator,
            token,
            pack_structures(operator),
            shape,
            False,
            operator_pattern_tag(operator),
        )

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> _CuDSSState:
        """Analyze a sparsity pattern into an analyzed-only state, no values folded in.

        Accepts a `BCOO`, `BCSR`, `BCOOLinearOperator`, `BCSRLinearOperator`, a
        sparsity-pattern tag, a tagged `lineax.JacobianLinearOperator` or
        `lineax.FunctionLinearOperator`, or an `asdex.ColoredPattern`. A later `update`
        folds in an operator sharing the pattern and reuses this analysis. The pattern
        must be concrete here, not a traced value.
        """
        del options
        csr, shape = _pattern_to_csr(sparsity)
        if shape[0] != shape[1]:
            raise ValueError(
                f"`CuDSS.init_symbolic` requires a square matrix; got shape {shape}."
            )
        record_operation(
            "init_symbolic",
            inputs=lambda: profile_inputs(
                sparsity, sparsity_pattern_tag(sparsity), shape
            ),
        )
        token = self._analyze(csr, _mtype_id_for_sparsity(sparsity))
        return _CuDSSState(
            None, token, None, shape, False, sparsity_pattern_tag(sparsity)
        )

    def update_and_compute(
        self,
        state: _CuDSSState,
        operator: AbstractLinearOperator,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any], _CuDSSState]:
        """Update `state` for `operator`, then solve. See `update_then_compute`."""
        return update_then_compute(self, state, operator, vector, options)

    def update(
        self,
        state: _CuDSSState,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _CuDSSState:
        """Fold a new operator into `state`, reusing the analysis where the pattern holds.

        Repeated calls with the same operator object are a no-op. When the operator
        shares the state's sparsity tag, the stored token's analysis is reused and only
        the numeric factorization is redone. Otherwise the operator is analyzed afresh.
        """
        if operator is state.operator:
            # Nothing changed, so this is a no-op.
            record_operation(
                "update",
                inputs=lambda: profile_inputs(
                    operator, operator_pattern_tag(operator), state.shape
                ),
                outputs={"outcome": "noop", "reason": "Same operator"},
            )
            return state
        tag = operator_pattern_tag(operator)
        reuse_block = sparsity_reuse_block(state.sparsity_tag, tag)
        if reuse_block is None:
            # Same pattern, new values. Reuse the analysis the stored token carries.
            record_operation(
                "update",
                inputs=lambda: profile_inputs(operator, tag, state.shape),
                outputs={"outcome": "reused", "reason": "Identical sparsity tag"},
            )
            return self._refactor(state, operator, tag)
        # Cannot reuse the analysis, so analyze from scratch. Recorded as a rebuild, with
        # the reason, so the analyze below is attributed to it rather than read as a
        # first init.
        record_operation(
            "update",
            inputs=lambda: profile_inputs(operator, tag, state.shape),
            outputs={"outcome": "rebuilt", "reason": reuse_block},
        )
        return self._analyze_and_factor(operator, options)

    def _refactor(
        self, state: _CuDSSState, operator: AbstractLinearOperator, tag: object
    ) -> _CuDSSState:
        """Refactor `state` for `operator`'s values, keeping the stored analysis.

        Both numeric phases rerun from the stored token, which keeps its analysis (the
        registry entry is renamed, not dropped). cuDSS only has a distinct refactorization
        under the COLAMD reorderings, where it reuses the previous pivots, so those go
        through `_refactorize_or_factorize`. Under every other reordering cuDSS runs the
        same phase for both, so a plain `factorize` costs no more.
        """
        (_, _, values), shape = _extract_csr(operator)
        values = values.astype(state.token.dtype)
        cudss = _spineax_cudss()
        if state.token.phase != "factorized":
            # A symbolic-only state from `init_symbolic` has no pivots to reuse yet.
            record_operation(
                "factorize", "CuDSS", outputs={"reason": "No prior factorization"}
            )
            token = cudss.factorize(state.token, values)
        elif self.reordering in _REFACTORIZING_REORDERINGS:
            token = _refactorize_or_factorize(cudss, state.token, values)
        else:
            record_operation(
                "factorize", "CuDSS", outputs={"reason": "Reused analysis"}
            )
            token = cudss.factorize(state.token, values)
        return _CuDSSState(
            operator,
            token,
            pack_structures(operator),
            shape,
            False,
            tag,
        )

    def compute(
        self,
        state: _CuDSSState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        if state.packed_structures is None:
            raise ValueError(
                "`CuDSS` cannot solve with a symbolic-only state; call `update` with an "
                "operator first."
            )
        with compute_scope():
            cudss = _spineax_cudss()
            b = ravel_vector(vector, state.packed_structures)
            b = _ensure_gpu(b)
            b = b.astype(state.token.dtype)
            x = _cudss_solve(
                cudss, state.token, b, options.get("ir_nsteps"), state.conjugate_solve
            )
            # cuDSS reports no per-solve rebuild status, unlike klujax and
            # pardiso_mkl_jax. A rebuild of an evicted entry only shows in the global
            # `spineax.cudss.rebuild_count()`.
            record_operation(
                "solve",
                "CuDSS",
                inputs={"conjugated": True} if state.conjugate_solve else None,
            )
            solution = unravel_solution(x, state.packed_structures)
            return solution, RESULTS.successful, {}

    def transpose(
        self, state: _CuDSSState, options: dict[str, Any]
    ) -> tuple[_CuDSSState, dict[str, Any]]:
        del options
        packed_structures = (
            None
            if state.packed_structures is None
            else transpose_packed_structures(state.packed_structures)
        )
        token = state.token
        if token.mtype_id == 0:
            # General matrix: A^T needs a genuinely re-analyzed, re-factorized token,
            # since cuDSS has no native transpose solve.
            cudss = _spineax_cudss()
            csr_T = _transpose_csr(
                token.offsets, token.columns, token.values, state.shape
            )
            record_operation(
                "analyze", "CuDSS", outputs={"reason": "Transposed general matrix"}
            )
            new_token = cudss.analyze(
                csr_T[2],
                csr_T[0],
                csr_T[1],
                mtype_id=token.mtype_id,
                mview_id=token.mview_id,
                device_id=token.device_id,
                reordering=token.reordering_id,
                memory=token.memory_id,
            )
            if state.packed_structures is not None:
                record_operation("factorize", "CuDSS")
                new_token = cudss.factorize(new_token, csr_T[2])
            return _CuDSSState(
                state.operator,
                new_token,
                packed_structures,
                state.shape[::-1],
                False,
                state.sparsity_tag,
            ), {}
        # Symmetric/Hermitian/SPD/HPD: A^T shares the same factors, no new factorization
        # needed. Mtype 2/4 (Hermitian) need `conj` around the solve, see `_cudss_solve`,
        # and the `_mtype_id` docstring for why they are currently unreachable here.
        conjugate_solve = token.mtype_id in (2, 4)
        return _CuDSSState(
            state.operator,
            token,
            packed_structures,
            state.shape[::-1],
            conjugate_solve,
            state.sparsity_tag,
        ), {}

    def conj(
        self, state: _CuDSSState, options: dict[str, Any]
    ) -> tuple[_CuDSSState, dict[str, Any]]:
        del options
        if state.packed_structures is None or state.token.dtype not in COMPLEX_DTYPES:
            # Symbolic-only, or real values, so conj is a no-op. A symbolic-only state is
            # never handed to lineax to solve, so its conjugation is deferred to the
            # `update` that folds in the operator's real values.
            return state, {}
        # Conjugating values does not change their magnitudes, so the existing pivots stay
        # numerically valid: `refactorize` (pivot reuse) is right here, and cheaper than a
        # fresh `factorize`.
        cudss = _spineax_cudss()
        record_operation(
            "refactorize", "CuDSS", outputs={"reason": "Conjugated values"}
        )
        new_token = cudss.refactorize(state.token, jnp.conj(state.token.values))
        return _CuDSSState(
            state.operator,
            new_token,
            state.packed_structures,
            state.shape,
            False,
            state.sparsity_tag,
        ), {}

    def assume_full_rank(self) -> bool:
        return True


CuDSS.__init__.__doc__ = """**Arguments:**

- `reordering`: fill-reducing reordering scheme passed to `analyze`. Defaults to
    `CuDSSReordering.DEFAULT`.
- `memory`: where cuDSS keeps the numeric factors. Defaults to `CuDSSMemory.DEVICE`.
- `device_id`: CUDA device index to run on. Defaults to `0`.
"""
