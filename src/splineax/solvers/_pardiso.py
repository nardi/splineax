import importlib.util
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCSR
from jaxtyping import Array, Inexact, Integer, PyTree
from lineax import AbstractLinearOperator, materialise
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver
from lineax._solver.misc import (
    PackedStructures,
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)

from splineax._trace import compute_scope, record_event
from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    SparseJacobianLinearOperator,
)
from splineax.solvers._klu import COMPLEX_DTYPES, _extract_pattern
from splineax.solvers._sparse import (
    _Sparsity,
    operator_pattern_tag,
    sparse_indices_sorted,
    sparsity_pattern_tag,
    sparsity_reuse_block,
    trace_inputs,
    warn_if_unsorted,
)

# `indptr`, `indices`, `values`: the matrix in CSR form.
_CSR = tuple[Integer[Array, " n+1"], Integer[Array, " nse"], Inexact[Array, " nse"]]


def _pardiso_available() -> bool:
    """Whether `pardiso_mkl_jax` is importable, without actually importing it.

    Checked with `importlib.util.find_spec` (no execution) rather than a real import.
    That way, probing availability from `Pardiso.__init__` and `AutoSparseLinearSolver`
    never pays for `pardiso_mkl_jax`'s import-time MKL runtime load. Unlike `klujax`,
    `pardiso-mkl-jax` is an optional dependency, so this check is what makes `Pardiso`
    unconstructible, and `AutoSparseLinearSolver` fall back to `KLU`, when it is missing.
    """
    return importlib.util.find_spec("pardiso_mkl_jax") is not None


def _pardiso_mkl_jax():
    # Lazy import: deferred until a Pardiso solve actually runs. Importing splineax, or
    # even constructing a `Pardiso`, never loads the MKL runtime unless the solver is
    # used (mirrors `_klu.py`'s `_klujax()`).
    import pardiso_mkl_jax

    return pardiso_mkl_jax


T = Any


def _ensure_cpu(args: T) -> T:
    """Return `args` unchanged, raising if the current platform is not CPU.

    A local copy of `_klu.py`'s helper of the same name, with a Pardiso-specific message:
    the two solvers wrap different CPU-only native libraries.
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


def _reject_complex(dtype: jnp.dtype) -> None:
    """Raise if `dtype` is complex, since `pardiso_mkl_jax` is real-only."""
    if dtype in COMPLEX_DTYPES:
        raise TypeError(
            "`Pardiso` only supports real-valued matrices; `pardiso_mkl_jax` does "
            f"not support complex matrix types yet. Got dtype {dtype}."
        )


def _extract_csr(
    operator: AbstractLinearOperator,
) -> tuple[Array, Array, Array, tuple[int, ...]]:
    """Read an operator as a sorted CSR triple with int32 indices and float64 values.

    A `SparseJacobianLinearOperator` is materialised first. An operator tagged
    `sparse_indices_sorted` asserts its indices need no sort.
    """
    sorted_asserted = sparse_indices_sorted in getattr(operator, "tags", ())
    match operator:
        case SparseJacobianLinearOperator():
            return _extract_csr(materialise(operator))
        case BCSRLinearOperator(matrix):
            _reject_complex(matrix.dtype)
            if matrix.indices_sorted or sorted_asserted:
                matrix_bcsr = matrix
            else:
                warn_if_unsorted(matrix, "Pardiso")
                matrix_bcsr = BCSR.from_bcoo(matrix.to_bcoo())
        case BCOOLinearOperator(matrix):
            _reject_complex(matrix.dtype)
            if not sorted_asserted:
                warn_if_unsorted(matrix, "Pardiso")
            matrix_bcsr = BCSR.from_bcoo(matrix)
        case _:
            raise TypeError(
                "`Pardiso` requires a sparse operator backed by a `BCOO` or `BCSR` "
                "matrix (e.g. `splineax.BCOOLinearOperator` or "
                "`splineax.BCSRLinearOperator`), or a "
                f"`splineax.SparseJacobianLinearOperator`; "
                f"got {type(operator).__name__}."
            )
    indptr = matrix_bcsr.indptr.astype(jnp.int32)
    indices = matrix_bcsr.indices.astype(jnp.int32)
    # Stop gradients on the values before they reach `analyze`/`factor`, which have no
    # differentiation rule. The gradient with respect to the matrix flows through the
    # operator lineax carries, not through the factorization, which is a constant.
    values = jax.lax.stop_gradient(matrix_bcsr.data.astype(jnp.float64))
    return indptr, indices, values, matrix_bcsr.shape


def _reanalyze_if_unstable(
    pmj: Any,
    token: Any,
    iparm: Array,
    indptr: Array,
    indices: Array,
    values: Array,
) -> Any:
    """Redo the analysis for these values if the reused matching factored badly.

    A `factor` that reuses a matching tuned for older values can perturb tiny pivots or
    hit a zero pivot, which the returned iparm records (`perturbed_pivot_count` at index
    13, `zero_or_negative_pivot_position` at index 29). When it does, `reanalyze` rebuilds
    the analysis and matching for these values in place, then factor again. Falling back is
    always correct, only slower, and equivalent to a fresh `init`.
    """
    primitive = pmj.primitive
    unstable = (iparm[13] > 0) | (iparm[29] != 0)

    def refresh() -> Any:
        # Record the branch actually taken: the reused matching was unstable, so the analysis
        # is rebuilt for these values and factored again.
        reason = "Unstable pivots"
        record_event("reanalyze", "Pardiso", outputs={"reason": reason})
        reanalyzed, _ = primitive.reanalyze(
            token,
            indptr,
            indices,
            values,
            matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        )
        record_event("factor", "Pardiso", outputs={"reason": reason})
        refactored, _ = primitive.factor(
            reanalyzed,
            indptr,
            indices,
            values,
            matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        )
        return refactored

    return jax.lax.cond(unstable, refresh, lambda: token)


class _PardisoState(eqx.Module):
    """A Pardiso solver state, carrying its factorization token.

    A state from `init_symbolic` holds only the shape and tag, with `token` None, since
    Pardiso defers analysis under weighted matching (see `Pardiso.init_symbolic`). After
    `init` or `update` it holds the CSR matrix and a factorized token, ready to solve.
    """

    operator: AbstractLinearOperator | None
    """The operator this state was built on. Compared by identity in `update`."""
    csr: _CSR | None
    """The sorted CSR triple, or None for a symbolic-only state."""
    token: Any
    """The `pardiso_mkl_jax` FactorizationToken, or None before any analysis."""
    packed_structures: PackedStructures | None
    """The lineax structure for ravel and unravel, None for a symbolic-only state."""
    shape: tuple[int, ...] = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)
    sparsity_tag: object | None = eqx.field(static=True, default=None)

    def track(self, solution: Any) -> "_PardisoState":
        """Return a state whose `release` is ordered after `solution`.

        Accepts the lineax `Solution` or a bare value pytree. A no-op when no analysis
        has run yet, see `pardiso_mkl_jax` FactorizationToken.track.
        """
        if self.token is None:
            return self
        record_event("track")
        value = getattr(solution, "value", solution)
        leaves = tuple(jax.tree_util.tree_leaves(value))
        return _PardisoState(
            self.operator,
            self.csr,
            self.token.track(*leaves),
            self.packed_structures,
            self.shape,
            self.transposed,
            self.sparsity_tag,
        )

    def release(self) -> None:
        """Free the native factorization, ordered after any tracked solves."""
        if self.token is None:
            return
        record_event("release")
        record_event("release", "Pardiso")
        _pardiso_mkl_jax().primitive.release(self.token)


class Pardiso(AbstractLinearSolver[_PardisoState]):
    """Sparse direct solver wrapping `pardiso_mkl_jax` (Intel oneMKL Pardiso).

    This solver keeps the operator in its native sparse (CSR) storage rather than
    densifying it, and so is intended for use with the sparse operators in this package
    (`BCOOLinearOperator` and `BCSRLinearOperator`).

    `pardiso_mkl_jax` is **CPU, real-valued, and double-precision only**: `float32`
    inputs are upcast to `float64`, and complex operators raise `TypeError` (Pardiso's
    complex matrix types are not supported by `pardiso_mkl_jax` yet). It does not enable
    JAX's x64 mode or force the CPU platform on import, so `jax_enable_x64` must already
    be on before this solver runs.

    This solver can only handle square nonsingular operators.

    Requires the optional `pardiso-mkl-jax` dependency (`pip install
    splineax[pardiso]`). Constructing `Pardiso()` raises `ImportError` if it is not
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

    def _analyze_and_factor(
        self,
        operator: AbstractLinearOperator,
        tag: object | None,
    ) -> _PardisoState:
        indptr, indices, values, shape = _extract_csr(operator)
        pmj = _pardiso_mkl_jax()
        primitive = pmj.primitive
        # `analyze` and `factor` return `(token, final_iparm)`. Only the token is kept;
        # the diagnostics iparm is dropped.
        record_event("analyze", "Pardiso")
        token, _ = primitive.analyze(
            indptr, indices, values, matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC
        )
        record_event("factor", "Pardiso")
        token, _ = primitive.factor(
            token,
            indptr,
            indices,
            values,
            matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        )
        return _PardisoState(
            operator,
            (indptr, indices, values),
            token,
            pack_structures(operator),
            shape,
            False,
            tag,
        )

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _PardisoState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Pardiso` may only be used for linear solves with square matrices"
            )
        record_event(
            "init",
            inputs=lambda: trace_inputs(
                operator,
                operator_pattern_tag(operator),
                (operator.out_size(), operator.in_size()),
            ),
        )
        return self._analyze_and_factor(operator, operator_pattern_tag(operator))

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> _PardisoState:
        """Record the pattern for reuse, deferring analysis to the first `update`.

        Under Pardiso's default weighted matching the analysis depends on the matrix
        values, so a values-independent symbolic phase is not sound. This keeps only the
        shape and a pattern tag. The first `update` with real values runs analyze and
        factor.
        """
        del options
        _, _, shape = _extract_pattern(sparsity)
        if shape[0] != shape[1]:
            raise ValueError(
                f"`Pardiso.init_symbolic` requires a square matrix; got shape {shape}."
            )
        # Deferred: no analyze runs here, only the pattern is recorded (see the docstring).
        record_event(
            "init_symbolic",
            inputs=lambda: trace_inputs(
                sparsity, sparsity_pattern_tag(sparsity), shape
            ),
            outputs={"note": "deferred"},
        )
        return _PardisoState(
            None,
            None,
            None,
            None,
            shape,
            False,
            sparsity_pattern_tag(sparsity),
        )

    def update(
        self,
        state: _PardisoState,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _PardisoState:
        """Fold a new operator into `state`, reusing the analysis where the pattern holds.

        Repeated calls with the same operator object are a no-op. When the operator shares
        the state's pattern and an analysis already exists, only the numeric factorization
        is redone. Otherwise the operator is analyzed from scratch.
        """
        del options
        if operator is state.operator:
            # Nothing changed, so this is a no-op.
            record_event(
                "update",
                inputs=lambda: trace_inputs(
                    operator, operator_pattern_tag(operator), state.shape
                ),
                outputs={"outcome": "noop", "reason": "Same operator"},
            )
            return state
        tag = operator_pattern_tag(operator)
        if state.token is None:
            # A symbolic-only state from `init_symbolic` deferred the analysis, so the first
            # update must analyze from scratch regardless of the tag.
            reuse_block: str | None = "Symbolic-only state"
        else:
            reuse_block = sparsity_reuse_block(state.sparsity_tag, tag)
        if reuse_block is None:
            # Same pattern, new values. Refactor against the stored analysis.
            record_event(
                "update",
                inputs=lambda: trace_inputs(operator, tag, state.shape),
                outputs={"outcome": "reused", "reason": "Identical sparsity tag"},
            )
            return self._refactor(state, operator, tag)
        # Cannot reuse the analysis, so analyze from scratch. Recorded as a rebuild, with the
        # reason, so the analyze below is attributed to it rather than read as a first init.
        record_event(
            "update",
            inputs=lambda: trace_inputs(operator, tag, state.shape),
            outputs={"outcome": "rebuilt", "reason": reuse_block},
        )
        return self._analyze_and_factor(operator, tag)

    def _refactor(
        self,
        state: _PardisoState,
        operator: AbstractLinearOperator,
        tag: object,
    ) -> _PardisoState:
        indptr, indices, values, shape = _extract_csr(operator)
        pmj = _pardiso_mkl_jax()
        # `factor` reuses the analysis stored under the token's id, including the weighted
        # matching, and returns a fresh token for the new values. That matching was tuned
        # for the previous values, so it can be a poor fit for these ones.
        token, iparm = pmj.primitive.factor(
            state.token,
            indptr,
            indices,
            values,
            matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
        )
        # `perturbed_pivots`/`zero_pivot` (iparm[13]/iparm[29]) drive the reanalyze fallback;
        # `reused` is True when the reused matching factored stably. See
        # `_reanalyze_if_unstable`.
        unstable = (iparm[13] > 0) | (iparm[29] != 0)
        record_event(
            "refactor",
            "Pardiso",
            outputs={"reason": "Reused matching"},
            dynamic={
                "reused": ~unstable,
                "perturbed_pivots": iparm[13],
                "zero_pivot": iparm[29] != 0,
            },
        )
        token = _reanalyze_if_unstable(pmj, token, iparm, indptr, indices, values)
        return _PardisoState(
            operator,
            (indptr, indices, values),
            token,
            pack_structures(operator),
            shape,
            False,
            tag,
        )

    def compute(
        self,
        state: _PardisoState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        if state.csr is None or state.token is None or state.packed_structures is None:
            raise ValueError(
                "`Pardiso` cannot solve with a symbolic-only state; call `update` with "
                "an operator first."
            )
        with compute_scope():
            b = ravel_vector(vector, state.packed_structures)
            b = _ensure_cpu(b)
            b = b.astype(jnp.float64)
            pmj = _pardiso_mkl_jax()
            primitive = pmj.primitive
            indptr, indices, values = state.csr
            record_event(
                "solve_stateful",
                "Pardiso",
                inputs={"transposed": state.transposed} if state.transposed else None,
            )
            # `solve_stateful` reuses the stored factorization, solving A^T when transposed.
            solution, _ = primitive.solve_stateful(
                state.token,
                indptr,
                indices,
                values,
                b[None, :],
                matrix_type=pmj.MatrixType.REAL_NONSYMMETRIC,
                transpose=state.transposed,
            )
            solution = unravel_solution(solution[0], state.packed_structures)
            return solution, RESULTS.successful, {}

    def transpose(
        self, state: _PardisoState, options: dict[str, Any]
    ) -> tuple[_PardisoState, dict[str, Any]]:
        del options
        # `pardiso_mkl_jax` solves against A^T natively with the same factorization, so
        # transposing is pure metadata: flip `transposed`, transpose the packed
        # structures, and swap `shape`. The token carries over unchanged.
        transposed_state = _PardisoState(
            state.operator,
            state.csr,
            state.token,
            transpose_packed_structures(state.packed_structures)
            if state.packed_structures is not None
            else None,
            state.shape[::-1],
            not state.transposed,
            state.sparsity_tag,
        )
        return transposed_state, {}

    def conj(
        self, state: _PardisoState, options: dict[str, Any]
    ) -> tuple[_PardisoState, dict[str, Any]]:
        del options
        # Real-only solver (see the class docstring), so conjugation is a no-op.
        return state, {}

    def assume_full_rank(self) -> bool:
        return True
