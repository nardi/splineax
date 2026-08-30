"""Tests for `IterativeRefinement`, the solver that wraps another and refines its solves.

The wrapper drives the classic refinement loop: solve, form the residual, solve for a
correction, repeat until the relative residual is within `tol` or `max_steps` steps are
spent, in which case the solution is NaN. These tests use two kinds of inner solver. A
direct solver (`KLU`, `Spsolve`) already lands within tolerance, so refinement should be a
transparent pass-through. A deliberately weak `_JacobiSolver` (one Jacobi sweep per solve)
turns the same loop into a stationary iteration, which lets a test watch refinement
actually reduce the residual step by step, and drive it into the NaN path on purpose.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS

import splineax as splx
from splineax import KLU, IterativeRefinement, Spsolve
from splineax.solvers._iterative import _IterativeRefinementState

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory

_EXPECTED = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))


class _JacobiState(eqx.Module):
    """State of `_JacobiSolver`: the operator's diagonal and the operator itself."""

    diagonal: Array
    operator: AbstractLinearOperator

    def release(self) -> None:
        """No-op, since a Jacobi state owns nothing to free."""


class _JacobiSolver(lx.AbstractLinearSolver[_JacobiState]):
    """A weak stateful solver: one Jacobi sweep, `x = b / diag(A)`.

    On its own this approximates `A^-1 b` only when `A` is strongly diagonally dominant.
    Wrapped in `IterativeRefinement`, the correction loop becomes the Jacobi iteration, so
    it is a controllable stand-in for a solver that needs several refinement steps to
    converge. It exists only for these tests.
    """

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _JacobiState:
        del options
        return _JacobiState(jnp.diag(operator.as_matrix()), operator)

    def init_symbolic(
        self, sparsity: Any, options: dict[str, Any] = {}
    ) -> _JacobiState:
        # A Jacobi sweep needs the diagonal values, which a bare pattern does not carry,
        # so there is no symbolic-only phase. Present only to satisfy `SparseLinearSolver`.
        raise NotImplementedError("`_JacobiSolver` has no symbolic phase.")

    def update(
        self,
        state: _JacobiState,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _JacobiState:
        return self.init(operator, options)

    def compute(
        self, state: _JacobiState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        return vector / state.diagonal, RESULTS.successful, {}

    def transpose(
        self, state: _JacobiState, options: dict[str, Any]
    ) -> tuple[_JacobiState, dict[str, Any]]:
        del options
        transposed = state.operator.transpose()
        return _JacobiState(jnp.diag(transposed.as_matrix()), transposed), {}

    def conj(
        self, state: _JacobiState, options: dict[str, Any]
    ) -> tuple[_JacobiState, dict[str, Any]]:
        del options
        return state, {}

    def assume_full_rank(self) -> bool:
        return True


def _relative_residual(
    operator: AbstractLinearOperator, solution: Array, vector: Array
) -> Array:
    return jnp.linalg.norm(vector - operator.mv(solution)) / jnp.linalg.norm(vector)


def test_wraps_direct_solver_matches_numpy(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """Refining a direct solve gives the same answer as `numpy.linalg.solve`; the
    refinement is a transparent pass-through when the inner solve is already accurate."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert jnp.allclose(solution, _EXPECTED, atol=1e-8)


def test_result_successful_and_residual_within_tol(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """A converged refinement reports `successful` and leaves a residual within `tol`.

    Run in float64 so the tolerance sits above the machine-precision floor and is the
    binding stop condition.
    """
    operator = make_operator(SQUARE_MATRIX.astype(jnp.float64))
    right_hand_side = RIGHT_HAND_SIDE.astype(jnp.float64)
    tol = 1e-10
    solver = IterativeRefinement(KLU(), tol=tol)
    solution = lx.linear_solve(operator, right_hand_side, solver=solver)
    assert solution.result == RESULTS.successful
    assert _relative_residual(operator, solution.value, right_hand_side) <= tol


def test_refinement_converges_a_weak_solver(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """The Jacobi sweep alone does not solve `SQUARE_MATRIX`, but enough refinement steps
    drive it to the true solution, so the loop is doing real work across steps."""
    operator = make_operator(SQUARE_MATRIX.astype(jnp.float64))
    right_hand_side = RIGHT_HAND_SIDE.astype(jnp.float64)
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX, dtype=np.float64), np.asarray(right_hand_side)
    )
    solver = IterativeRefinement(_JacobiSolver(), tol=1e-10, max_steps=200)
    solution = lx.linear_solve(operator, right_hand_side, solver=solver)
    assert solution.result == RESULTS.successful
    assert jnp.allclose(solution.value, expected, atol=1e-8)


def test_returns_nan_when_max_steps_exhausted(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """With too few steps to converge, the weak solver's refinement hits the step cap and
    returns NaN with `max_steps_reached` rather than a wrong answer reported as success."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(_JacobiSolver(), tol=1e-12, max_steps=1)
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver, throw=False)
    assert solution.result == RESULTS.max_steps_reached
    assert jnp.all(jnp.isnan(solution.value))


def test_tighter_tol_reduces_the_residual(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """A tighter tolerance runs more Jacobi corrections and leaves a smaller residual,
    confirming each step corrects the solution and that `tol` controls where it stops."""
    operator = make_operator(SQUARE_MATRIX)
    coarse = IterativeRefinement(_JacobiSolver(), tol=1e-2, max_steps=50)
    fine = IterativeRefinement(_JacobiSolver(), tol=1e-8, max_steps=50)
    coarse_solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=coarse).value
    fine_solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=fine).value
    coarse_residual = _relative_residual(operator, coarse_solution, RIGHT_HAND_SIDE)
    fine_residual = _relative_residual(operator, fine_solution, RIGHT_HAND_SIDE)
    assert fine_residual < coarse_residual


def test_symbolic_state_cannot_solve(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """A state from `init_symbolic` has no operator, so `compute` cannot form a residual
    and must raise until `update` folds one in."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())
    from jax.experimental.sparse import BCOO

    symbolic = solver.init_symbolic(BCOO.fromdense(SQUARE_MATRIX))
    with pytest.raises(ValueError, match="symbolic-only"):
        solver.compute(symbolic, RIGHT_HAND_SIDE, {})
    updated = solver.update(symbolic, operator)
    solution = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=solver, state=updated
    ).value
    updated.release()
    assert jnp.allclose(solution, _EXPECTED, atol=1e-8)


def test_transpose_solves_transposed_system(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """Transposing the state solves `A^T x = b`, so the refinement forms its residual
    against `A^T` rather than `A`."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )
    state = solver.init(operator, {})
    transposed, _ = solver.transpose(state, {})
    solution = np.asarray(solver.compute(transposed, RIGHT_HAND_SIDE, {})[0])
    state.release()
    assert jnp.allclose(solution, expected, atol=1e-8)


def test_update_no_op_returns_same_state(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """An `update` with the same operator is a no-op through the wrapper, matching the
    inner solver's own no-op so repeated updates cost nothing."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())
    state = solver.init(operator, {})
    again = solver.update(state, operator)
    assert again is state
    state.release()


def test_solve_under_jit(make_operator: OperatorFactory, enable_x64: None) -> None:
    """The refinement loop is traceable, so a solve wrapped in `jax.jit` runs and gives
    the right answer."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())

    @jax.jit
    def solve(b: Array) -> Array:
        return lx.linear_solve(operator, b, solver=solver).value

    assert jnp.allclose(solve(RIGHT_HAND_SIDE), _EXPECTED, atol=1e-8)


def test_differentiable_wrt_vector(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """Forward- and reverse-mode AD through the refined solve w.r.t. the right-hand side
    give `A^-1`, matching the wrapped solver. The reverse path exercises `transpose`."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())

    def solve(b: Array) -> Array:
        return lx.linear_solve(operator, b, solver=solver).value

    expected_jacobian = jnp.linalg.inv(SQUARE_MATRIX)
    assert jnp.allclose(
        jax.jacfwd(solve)(RIGHT_HAND_SIDE), expected_jacobian, atol=1e-8
    )
    assert jnp.allclose(
        jax.jacrev(solve)(RIGHT_HAND_SIDE), expected_jacobian, atol=1e-8
    )


def test_stateful_linear_solve_returns_tuple(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`splineax.linear_solve` drives the wrapper's init/track/release and returns a
    `(solution, state)` tuple, and threading the state reuses the inner factorization."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())
    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
    assert isinstance(state, _IterativeRefinementState)
    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver, state=state)
    state.release()
    assert jnp.allclose(solution.value, _EXPECTED, atol=1e-8)


def test_refinement_fixes_a_stale_factorization(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """A factorization of a nearby matrix is a poor solver on its own, but refinement uses
    it as the correction step and still converges to the true solution.

    Factorize a base matrix, perturb its values a little, then solve the perturbed system
    against the stale factorization without refactoring. The stale factorization alone
    leaves a large residual. Refinement drives the same factorization to the true solution.
    """
    base_matrix = SQUARE_MATRIX.astype(jnp.float64)
    right_hand_side = RIGHT_HAND_SIDE.astype(jnp.float64)

    # Scale the nonzero values by a few percent. Multiplying keeps the zeros zero, so the
    # perturbed matrix shares the base's sparsity pattern.
    rng = np.random.default_rng(0)
    perturbation = 1.0 + 0.05 * rng.uniform(-1.0, 1.0, size=base_matrix.shape)
    perturbed_matrix = base_matrix * jnp.asarray(perturbation)
    base_operator = make_operator(base_matrix)
    perturbed_operator = make_operator(perturbed_matrix)
    expected = jnp.linalg.solve(
        np.asarray(perturbed_matrix), np.asarray(right_hand_side)
    )

    klu = KLU()
    solver = IterativeRefinement(klu, tol=1e-10, max_steps=50)
    # Factorize the base matrix, then pair that factorization with the perturbed operator,
    # so the stored factorization is stale relative to the system actually solved.
    base_state = solver.init(base_operator, {})
    stale_state = _IterativeRefinementState(base_state.inner_state, perturbed_operator)

    # The stale factorization alone solves the base system, so its residual against the
    # perturbed operator is large.
    stale_solution, _, _ = klu.compute(base_state.inner_state, right_hand_side, {})
    stale_residual = _relative_residual(
        perturbed_operator, stale_solution, right_hand_side
    )
    assert stale_residual > 1e-3

    # Refinement uses the same stale factorization as its correction step and converges.
    refined_solution, result, _ = solver.compute(stale_state, right_hand_side, {})
    base_state.release()
    assert result == RESULTS.successful
    assert (
        _relative_residual(perturbed_operator, refined_solution, right_hand_side)
        <= 1e-9
    )
    assert jnp.allclose(refined_solution, expected, atol=1e-8)


def test_wraps_spsolve_without_x64(make_operator: OperatorFactory) -> None:
    """Wrapping `Spsolve` (single precision, no x64) still refines correctly, and the
    machine-precision floor keeps a healthy float32 solve from falsely failing."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(Spsolve())
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver)
    assert solution.result == RESULTS.successful
    assert jnp.allclose(solution.value, _EXPECTED, atol=1e-4)
