from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from asdex import ColoredPattern
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Inexact, Integer, PyTree
from klujax import NumericToken, SymbolToken
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

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
)
from splineax.solvers._sparse import (
    _Sparsity,
    operator_pattern_tag,
    sparsity_pattern_tag,
)

# `Ai` (row indices), `Aj` (column indices), `Ax` (values): the matrix in COO form.
_COO = tuple[Integer[Array, " a"], Integer[Array, " b"], Inexact[Array, " nse"]]

COMPLEX_DTYPES = (
    np.complex64,
    np.complex128,
    jnp.complex64,
    jnp.complex128,
)


def _klujax():
    # Lazy import: deferred until a KLU solve actually runs, so importing splineax never
    # pays for loading klujax's compiled extension unless the solver is used.
    import klujax

    return klujax


def _upcast(values: Array) -> Array:
    """Upcast values to the double precision klujax needs, complex or real."""
    if values.dtype in COMPLEX_DTYPES:
        return values.astype(jnp.complex128)
    return values.astype(jnp.float64)


def _extract_coo(
    operator: AbstractLinearOperator,
) -> tuple[Array, Array, Array, tuple[int, ...]]:
    """Read an operator's COO triple and shape, upcasting the values.

    A `SparseJacobianLinearOperator` is materialised first, then handled like a BCOO.
    """
    match operator:
        case SparseJacobianLinearOperator():
            return _extract_coo(materialise(operator))
        case BCSRLinearOperator(matrix):
            bcoo = matrix.to_bcoo()
        case BCOOLinearOperator(matrix):
            bcoo = matrix
        case _:
            raise TypeError(
                "`KLU` requires a sparse operator backed by a `BCOO` or `BCSR` "
                "matrix (e.g. `splineax.BCOOLinearOperator` or "
                "`splineax.BCSRLinearOperator`), or a "
                f"`splineax.SparseJacobianLinearOperator`; "
                f"got {type(operator).__name__}."
            )
    row = bcoo.indices[:, 0].astype(jnp.int32)
    col = bcoo.indices[:, 1].astype(jnp.int32)
    # Stop gradients on the values before they reach `analyze`/`factor`, which have no
    # differentiation rule. The gradient with respect to the matrix flows through the
    # operator lineax carries, not through the factorization, which is a constant.
    values = _upcast(jax.lax.stop_gradient(bcoo.data))
    return row, col, values, bcoo.shape


def _extract_pattern(sparsity: _Sparsity) -> tuple[Array, Array, tuple[int, ...]]:
    """Read a sparsity pattern's COO indices and shape, without any values.

    The Jacobian and coloring forms carry the pattern in their precomputed asdex
    coloring, so the indices are read there rather than materialising the Jacobian.
    """
    match sparsity:
        case SparseJacobianLinearOperator(transposed=True):
            # The stored pattern describes the forward Jacobian. asdex emits `BCOO`
            # values in the pattern's index order and `BCOO.T` swaps the index columns
            # without reordering entries, so swapping rows and columns here keeps the
            # indices aligned with the values a later solve pairs them with.
            pattern = sparsity.coloring.sparsity
            row = jnp.asarray(pattern.cols, dtype=jnp.int32)
            col = jnp.asarray(pattern.rows, dtype=jnp.int32)
            shape = pattern.shape[::-1]
        case SparseJacobianLinearOperator() | SparseJacobianLinearOperatorColoring():
            # Both hold the coloring one level in: the operator stores an
            # `asdex.ColoredPattern` whose `.sparsity` is the pattern, and the operator
            # coloring stores a `JacobianColoring` whose `.sparsity` property returns it.
            pattern = sparsity.coloring.sparsity
            row = jnp.asarray(pattern.rows, dtype=jnp.int32)
            col = jnp.asarray(pattern.cols, dtype=jnp.int32)
            shape = pattern.shape
        case JacobianColoring() | ColoredPattern():
            pattern = sparsity.sparsity
            row = jnp.asarray(pattern.rows, dtype=jnp.int32)
            col = jnp.asarray(pattern.cols, dtype=jnp.int32)
            shape = pattern.shape
        case BCSRLinearOperator():
            bcoo = sparsity.matrix.to_bcoo()
            row = bcoo.indices[:, 0].astype(jnp.int32)
            col = bcoo.indices[:, 1].astype(jnp.int32)
            shape = bcoo.shape
        case BCOOLinearOperator():
            bcoo = sparsity.matrix
            row = bcoo.indices[:, 0].astype(jnp.int32)
            col = bcoo.indices[:, 1].astype(jnp.int32)
            shape = bcoo.shape
        case BCSR():
            bcoo = sparsity.to_bcoo()
            row = bcoo.indices[:, 0].astype(jnp.int32)
            col = bcoo.indices[:, 1].astype(jnp.int32)
            shape = bcoo.shape
        case BCOO():
            row = sparsity.indices[:, 0].astype(jnp.int32)
            col = sparsity.indices[:, 1].astype(jnp.int32)
            shape = sparsity.shape
        case _:
            raise TypeError(
                "`KLU.init_symbolic` requires a `BCOO`, `BCSR`, `BCOOLinearOperator`, "
                "`BCSRLinearOperator`, `SparseJacobianLinearOperator`, "
                "`SparseJacobianLinearOperatorColoring`, `JacobianColoring`, or "
                f"`asdex.ColoredPattern`; got {type(sparsity).__name__}."
            )
    return row, col, tuple(shape)


class _KLUState(eqx.Module):
    """A KLU solver state, carrying its factorization tokens.

    The state has three shapes. Straight from `init_symbolic` it holds only the symbolic
    token, with no values yet, so it is not solvable until `update` gives it an operator.
    After `init` or `update` it also holds the values and a numeric token, ready to solve.
    """

    operator: AbstractLinearOperator | None
    """The operator this state was built on. Compared by identity in `update`."""
    coo: _COO | None
    """The extracted (Ai, Aj, Ax) triple, or None for a symbolic-only state."""
    symbol: SymbolToken
    """The symbolic analysis token, always present."""
    numeric: NumericToken | None
    """The numeric factorization token, None before any values are known."""
    packed_structures: PackedStructures | None
    """The lineax structure for ravel and unravel, None for a symbolic-only state."""
    shape: tuple[int, ...] = eqx.field(static=True)
    transposed: bool = eqx.field(static=True, default=False)
    sparsity_tag: object | None = eqx.field(static=True, default=None)

    def track(self, solution: Any) -> "_KLUState":
        """Return a state whose `release` is ordered after `solution`.

        Accepts the lineax `Solution` or a bare value pytree. The solution arrays become
        ordering dependencies on the tokens, see klujax `SymbolToken.track`.
        """
        value = getattr(solution, "value", solution)
        leaves = tuple(jax.tree_util.tree_leaves(value))
        symbol = self.symbol.track(*leaves)
        numeric = None if self.numeric is None else self.numeric.track(*leaves)
        return _KLUState(
            self.operator,
            self.coo,
            symbol,
            numeric,
            self.packed_structures,
            self.shape,
            self.transposed,
            self.sparsity_tag,
        )

    def release(self) -> None:
        """Free the cache slots this state owns, ordered after any tracked solves."""
        klujax = _klujax()
        if self.numeric is not None:
            klujax.free_numeric(self.numeric)
        klujax.free_symbolic(self.symbol)


T = TypeVar("T")


def _ensure_cpu(args: T) -> T:
    """Return `args` unchanged, raising if the current platform is not CPU.

    Uses `jax.lax.platform_dependent` to produce a traced boolean and `equinox.error_if`
    to raise.
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


_REFACTOR_RCOND_FLOOR = 1e-8
"""Reciprocal condition estimate below which a reused factorization is refreshed.

`klujax.rcond` returns `min|Uii| / max|Uii|`, the ratio of the smallest to
largest diagonal entry of `U` (`klu_rcond`). It is a cheap lower bound on the
reciprocal condition number of `U`. `refactor` keeps the pivots the previous
`factor` chose, so shifted values can leave `U` badly scaled, which shows up as
a small ratio here. The floor is roughly the square root of double-precision
machine epsilon. A solve loses about `-log10(rcond)` decimal digits, so `1e-8`
means about 8/16 digits for a double-precision solve are meaningful. Below this
rcond value we factor afresh, which is always correct, just slower.

The estimate only reads the diagonal of `U`, so it can miss growth in the
off-diagonal entries, and an ill-conditioned matrix gives a large residual even
from a fresh factor. This floor is a conservative guard against reused-pivot
degradation, not a residual guarantee. A caller that needs a tight residual
should check it rather than rely on this floor alone."""


def _reuse_or_refresh_numeric(
    klujax: Any,
    row: Array,
    col: Array,
    values: Array,
    symbol: SymbolToken,
    numeric: NumericToken,
) -> NumericToken:
    """Refactor reusing the previous pivot order, falling back to a fresh factor.

    `refactor` is cheaper than `factor` because it reuses the pivots the last
    factorization chose, but those pivots can be a poor fit for new values.
    `refactor_with_status` catches an outright failure without raising, and `rcond` catches
    pivots that survived but left `U` close to singular (see `_REFACTOR_RCOND_FLOOR`). On
    either, `factor` from the symbolic analysis instead. Falling back is always correct,
    only slower.
    """
    refreshed, status = klujax.refactor_with_status(row, col, values, numeric, symbol)
    dtype = jnp.complex128 if values.dtype in COMPLEX_DTYPES else jnp.float64
    reciprocal_condition = klujax.rcond(symbol, refreshed, dtype=dtype)
    reuse_is_safe = jnp.all(status == klujax.KLUStatus.OK) & jnp.all(
        reciprocal_condition > _REFACTOR_RCOND_FLOOR
    )
    return jax.lax.cond(
        reuse_is_safe,
        lambda: refreshed,
        lambda: klujax.factor(row, col, values, symbol),
    )


class KLU(AbstractLinearSolver[_KLUState]):
    """Sparse direct solver wrapping the `klujax` (SuiteSparse KLU) library.

    This solver keeps the operator in its native sparse (COO) storage rather than
    densifying it, and so is intended for use with the sparse operators in this package
    (`BCOOLinearOperator` and `BCSRLinearOperator`).

    `klujax` is **CPU and double-precision only**: `float32`/`complex64` inputs are
    upcast to `float64`/`complex128`. It does not enable JAX's x64 mode or force the CPU
    platform on import, so `jax_enable_x64` must already be on before this solver runs.
    `klujax` raises a clear error otherwise.

    This solver can only handle square nonsingular operators.
    """

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _KLUState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`KLU` may only be used for linear solves with square matrices"
            )
        row, col, values, shape = _extract_coo(operator)
        klujax = _klujax()
        # `init` analyzes and factorizes right away, so the state is ready to solve and
        # reusable across right-hand sides. `factor` needs the real `SymbolToken`, which
        # `analyze` returns and the state then carries.
        symbol = klujax.analyze(row, col, shape[1])
        numeric = klujax.factor(row, col, values, symbol)
        return _KLUState(
            operator,
            (row, col, values),
            symbol,
            numeric,
            pack_structures(operator),
            shape,
            False,
            operator_pattern_tag(operator),
        )

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> _KLUState:
        """Analyze a sparsity pattern into a symbolic-only state, no values yet.

        Accepts a `BCOO`, `BCSR`, `BCOOLinearOperator`, `BCSRLinearOperator`,
        `SparseJacobianLinearOperator`, `SparseJacobianLinearOperatorColoring`, or
        `JacobianColoring`. `update` then folds in an operator sharing the pattern and
        reuses this analysis. The pattern must be concrete here, not a traced value.
        """
        del options
        row, col, shape = _extract_pattern(sparsity)
        if shape[0] != shape[1]:
            raise ValueError(
                f"`KLU.init_symbolic` requires a square matrix; got shape {shape}."
            )
        symbol = _klujax().analyze(row, col, shape[1])
        return _KLUState(
            None,
            None,
            symbol,
            None,
            None,
            shape,
            False,
            sparsity_pattern_tag(sparsity),
        )

    def update(
        self,
        state: _KLUState,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _KLUState:
        """Fold a new operator into `state`, reusing the analysis where the pattern holds.

        Repeated calls with the same operator object are a no-op. When the operator shares
        the state's sparsity tag, the symbolic analysis is reused and only the numeric
        factorization is redone. Otherwise the operator is analyzed from scratch.
        """
        if operator is state.operator:
            # Nothing changed, so this is a no-op.
            return state
        tag = operator_pattern_tag(operator)
        if (
            state.sparsity_tag is not None
            and tag is not None
            and state.sparsity_tag == tag
        ):
            # Same pattern, new values. Reuse the symbolic analysis.
            return self._refactor(state, operator, tag, options)
        # New pattern, so analyze from scratch.
        return self.init(operator, options)

    def _refactor(
        self,
        state: _KLUState,
        operator: AbstractLinearOperator,
        tag: object,
        options: dict[str, Any],
    ) -> _KLUState:
        del options
        row, col, values, shape = _extract_coo(operator)
        klujax = _klujax()
        # Reuse the stored symbolic analysis. The tag asserts the indices match the ones
        # `symbol` was analyzed with.
        if state.numeric is None:
            # No previous numeric factorization to reuse, so build one fresh.
            numeric = klujax.factor(row, col, values, state.symbol)
        else:
            numeric = _reuse_or_refresh_numeric(
                klujax, row, col, values, state.symbol, state.numeric
            )
        return _KLUState(
            operator,
            (row, col, values),
            state.symbol,
            numeric,
            pack_structures(operator),
            shape,
            False,
            tag,
        )

    def compute(
        self,
        state: _KLUState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        if state.coo is None or state.packed_structures is None:
            raise ValueError(
                "`KLU` cannot solve with a symbolic-only state; call `update` with an "
                "operator first."
            )
        row, col, values = state.coo
        b = ravel_vector(vector, state.packed_structures)
        row, col, values, b = _ensure_cpu((row, col, values, b))
        klujax = _klujax()
        b = _upcast(b)
        # A symbolic-only tier is possible if `update` reused an analysis but the numeric
        # token was dropped, so factor here as a fallback. Normally `numeric` is present.
        numeric = state.numeric
        if numeric is None:
            numeric = klujax.factor(row, col, values, state.symbol)
        solve = (
            klujax.tsolve_with_numeric
            if state.transposed
            else klujax.solve_with_numeric
        )
        x = solve(numeric, b, state.symbol)
        solution = unravel_solution(x, state.packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(
        self, state: _KLUState, options: dict[str, Any]
    ) -> tuple[_KLUState, dict[str, Any]]:
        del options
        # Reuse the factorization unchanged and let `tsolve` handle the transposed
        # direction. `coo` stays A's own arrays, which `tsolve` needs.
        transposed_state = _KLUState(
            state.operator,
            state.coo,
            state.symbol,
            state.numeric,
            transpose_packed_structures(state.packed_structures)
            if state.packed_structures is not None
            else None,
            state.shape[::-1],
            not state.transposed,
            state.sparsity_tag,
        )
        return transposed_state, {}

    def conj(
        self, state: _KLUState, options: dict[str, Any]
    ) -> tuple[_KLUState, dict[str, Any]]:
        del options
        if state.coo is None:
            return state, {}
        row, col, values = state.coo
        if values.dtype not in COMPLEX_DTYPES:
            # Real values, so conj is a no-op.
            return state, {}
        # Complex: conjugate the values and refactor, reusing the symbolic analysis since
        # the sparsity is unchanged.
        conjugated = values.conj()
        numeric = (
            None
            if state.numeric is None
            else _klujax().factor(row, col, conjugated, state.symbol)
        )
        conjugated_state = _KLUState(
            state.operator,
            (row, col, conjugated),
            state.symbol,
            numeric,
            state.packed_structures,
            state.shape,
            state.transposed,
            state.sparsity_tag,
        )
        return conjugated_state, {}

    def assume_full_rank(self) -> bool:
        return True


KLU.__init__.__doc__ = """**Arguments:**

Nothing.
"""
