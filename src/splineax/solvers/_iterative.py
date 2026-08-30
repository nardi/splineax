"""Iterative refinement wrapped around any stateful sparse solver.

A direct solve returns `x0 = solve(b)`, accurate to the backend's working precision. When
that is not enough, iterative refinement improves it: form the residual `r = b - A x`,
solve `A dx = r` with the same factorization, and add the correction `x = x + dx`. Each
step reuses the inner solver's factorization, so a step costs one matrix-vector product
and one back-substitution, not a new factorization.

`IterativeRefinement` wraps any solver satisfying the stateful API and drives this loop in
`compute`. It targets a residual tolerance and gives up after a maximum number of steps,
returning NaN so a caller can see the solve did not reach the tolerance.
"""

from typing import Any, Callable, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, conj
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver

from splineax.solvers._sparse import SparseLinearSolver, _Sparsity
from splineax.solvers._stateful import TrackingState

_StateT = TypeVar("_StateT")

_CONVERGENCE_FLOOR_ULPS = 100.0
"""How close to machine precision the residual is allowed to demand.

Iterative refinement in a fixed working precision cannot push the relative residual below
a small multiple of that precision's machine epsilon, since the residual itself is formed
with rounding error of that size. The convergence threshold is floored at this many ulps
times the working epsilon, so a healthy solve at a tolerance tighter than the precision
can reach still reports success instead of exhausting its steps and returning NaN. A
genuinely ill-conditioned solve that cannot reach the floored tolerance still fails."""


def _tree_norm(tree: PyTree[Array]) -> Array:
    """Euclidean norm over all leaves of a pytree, treating them as one flat vector."""
    squared = sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in jtu.tree_leaves(tree))
    return jnp.sqrt(squared)


def _tree_add(x: PyTree[Array], y: PyTree[Array]) -> PyTree[Array]:
    return jtu.tree_map(lambda a, b: a + b, x, y)


def _tree_sub(x: PyTree[Array], y: PyTree[Array]) -> PyTree[Array]:
    return jtu.tree_map(lambda a, b: a - b, x, y)


def iterative_refinement(
    solve: Callable[[PyTree[Array]], PyTree[Array]],
    operator: AbstractLinearOperator,
    vector: PyTree[Array],
    tol: float,
    max_steps: int,
) -> tuple[PyTree[Array], RESULTS]:
    """Refine `solve(vector)` until its relative residual is within `tol`.

    `solve` maps a right-hand side to an approximate solution of `operator @ x = vector`,
    reusing a factorization across calls. Refinement stops once
    `||vector - operator @ x|| <= tol * ||vector||`, floored at machine precision (see
    `_CONVERGENCE_FLOOR_ULPS`), or after `max_steps` corrections. On failure to reach the
    tolerance the returned solution is all NaN and the result is `max_steps_reached`.
    """

    def residual(x: PyTree[Array]) -> PyTree[Array]:
        return _tree_sub(vector, operator.mv(x))

    x0 = solve(vector)
    r0 = residual(x0)

    # Float the threshold at the precision the residual is actually computed in, so a
    # tolerance tighter than that precision can reach does not force a NaN.
    residual_dtype = jnp.result_type(*jtu.tree_leaves(r0))
    floor = _CONVERGENCE_FLOOR_ULPS * jnp.finfo(residual_dtype).eps
    threshold = jnp.maximum(tol, floor) * _tree_norm(vector)

    def cond(carry: tuple[PyTree[Array], PyTree[Array], Array]) -> Array:
        _, residual_value, step = carry
        return (step < max_steps) & (_tree_norm(residual_value) > threshold)

    def body(
        carry: tuple[PyTree[Array], PyTree[Array], Array],
    ) -> tuple[PyTree[Array], PyTree[Array], Array]:
        x, residual_value, step = carry
        correction = solve(residual_value)
        x = _tree_add(x, correction)
        return x, residual(x), step + 1

    x, final_residual, _ = jax.lax.while_loop(cond, body, (x0, r0, jnp.array(0)))
    converged = _tree_norm(final_residual) <= threshold
    # NaN out a solution that never met the tolerance, so the caller sees the failure.
    solution = jtu.tree_map(lambda leaf: jnp.where(converged, leaf, jnp.nan), x)
    result = RESULTS.where(converged, RESULTS.successful, RESULTS.max_steps_reached)
    return solution, result


class _IterativeRefinementState(eqx.Module, Generic[_StateT]):
    """State of an `IterativeRefinement` solve.

    Wraps the inner solver's state together with the operator being solved. The operator
    is what the refinement loop forms residuals against, so it is transposed and
    conjugated alongside the inner state to keep the residual using the right matrix. A
    state straight from `init_symbolic` carries no operator and is not solvable until
    `update` gives it one.
    """

    inner_state: _StateT
    operator: AbstractLinearOperator | None
    """The operator this state represents, or None for a symbolic-only state."""

    def track(self, solution: Any) -> "_IterativeRefinementState[_StateT]":
        """Order a later `release` after `solution`, delegating to the inner state.

        A no-op for an inner state that owns nothing, matching `TrackingState`.
        """
        inner = self.inner_state
        # Only a `TrackingState` has memory to order a release against; others are a no-op.
        tracked = inner.track(solution) if isinstance(inner, TrackingState) else inner
        return _IterativeRefinementState(tracked, self.operator)

    def release(self) -> None:
        """Release the wrapped inner state, which owns any memory this state holds."""
        inner = self.inner_state
        if isinstance(inner, TrackingState):
            inner.release()


class IterativeRefinementSettings(eqx.Module):
    """The `tol` and `max_steps` of an iterative refinement, without a solver bound yet.

    A solver that offers refinement as an option, such as `AutoSparseLinearSolver`, takes
    one of these instead of separate tolerance and step-cap arguments, so the two settings
    travel together. See `IterativeRefinement` for what each does.
    """

    tol: float = eqx.field(default=1e-10, static=True)
    """Target relative residual, `||b - A x|| <= tol * ||b||`."""
    max_steps: int = eqx.field(default=10, static=True)
    """Maximum correction steps before returning NaN."""


class IterativeRefinement(AbstractLinearSolver[_IterativeRefinementState]):
    """Wraps a stateful solver and refines each solve with iterative refinement.

    The wrapped `solver` supplies the factorization and the per-step solves. `compute`
    runs the refinement loop (see `iterative_refinement`), reusing that factorization for
    both the initial solve and every correction. Every other method delegates to the
    wrapped solver, so `IterativeRefinement` exposes the same stateful API (`init`,
    `init_symbolic`, `update`, `transpose`, `conj`) and can stand in for the solver it
    wraps. Its state releases through the inner state, so `state.release()` still frees it.

    The wrapped solver must be square and nonsingular, since refinement assumes the
    correction solve returns a genuine approximate inverse.
    """

    solver: SparseLinearSolver[Any]
    tol: float = eqx.field(default=1e-10, static=True)
    max_steps: int = eqx.field(default=10, static=True)

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _IterativeRefinementState:
        return _IterativeRefinementState(self.solver.init(operator, options), operator)

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> _IterativeRefinementState:
        """Analyze a sparsity pattern, deferring to the wrapped solver's `init_symbolic`.

        The resulting state has no operator yet, so `update` must fold one in before a
        solve. Raises `AttributeError` if the wrapped solver has no symbolic phase.
        """
        inner = self.solver.init_symbolic(sparsity, options)
        return _IterativeRefinementState(inner, None)

    def update(
        self,
        state: _IterativeRefinementState,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _IterativeRefinementState:
        """Fold a new operator into `state` through the wrapped solver.

        Returns the same state object when the wrapped `update` did, so an update with an
        unchanged operator stays a no-op.
        """
        inner = self.solver.update(state.inner_state, operator, options)
        if inner is state.inner_state:
            return state
        return _IterativeRefinementState(inner, operator)

    def compute(
        self,
        state: _IterativeRefinementState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        if state.operator is None:
            raise ValueError(
                "`IterativeRefinement` cannot solve with a symbolic-only state; call "
                "`update` with an operator first."
            )
        operator = state.operator

        def solve(right_hand_side: PyTree[Array]) -> PyTree[Array]:
            solution, _, _ = self.solver.compute(
                state.inner_state, right_hand_side, options
            )
            return solution

        solution, result = iterative_refinement(
            solve, operator, vector, self.tol, self.max_steps
        )
        return solution, result, {}

    def transpose(
        self, state: _IterativeRefinementState, options: dict[str, Any]
    ) -> tuple[_IterativeRefinementState, dict[str, Any]]:
        inner_transpose, transpose_options = self.solver.transpose(
            state.inner_state, options
        )
        # Transpose the stored operator too, so the residual uses A^T on this state.
        operator = None if state.operator is None else state.operator.transpose()
        return (
            _IterativeRefinementState(inner_transpose, operator),
            transpose_options,
        )

    def conj(
        self, state: _IterativeRefinementState, options: dict[str, Any]
    ) -> tuple[_IterativeRefinementState, dict[str, Any]]:
        inner_conj, conj_options = self.solver.conj(state.inner_state, options)
        operator = None if state.operator is None else conj(state.operator)
        return _IterativeRefinementState(inner_conj, operator), conj_options

    def assume_full_rank(self) -> bool:
        return self.solver.assume_full_rank()


IterativeRefinement.__init__.__doc__ = """**Arguments:**

- `solver`: the stateful solver to wrap. It supplies the factorization and per-step
    solves, and must be square and nonsingular.
- `tol`: the target relative residual, `||b - A x|| <= tol * ||b||`. The threshold is
    floored at machine precision, so a tolerance tighter than the working precision can
    reach still succeeds rather than returning NaN. Defaults to `1e-10`.
- `max_steps`: the maximum number of correction steps before giving up and returning NaN.
    Defaults to `10`.
"""
