"""Tests for iterative refinement and for `HybridDirectIterative`.

`IterativeRefinement` wraps another solver and refines its solves.

The wrapper drives the classic refinement loop: solve, form the residual, solve for a
correction, repeat until the relative residual is within `tol` or `max_steps` steps are
spent, in which case the solution is NaN. These tests use two kinds of inner solver. A
direct solver (`KLU`, `Spsolve`) already lands within tolerance, so refinement should be a
transparent pass-through. A deliberately weak `_JacobiSolver` (one Jacobi sweep per solve)
turns the same loop into a stationary iteration, which lets a test watch refinement
actually reduce the residual step by step, and drive it into the NaN path on purpose.

The second half of this file tests `HybridDirectIterative`, which tries the factorization
of an earlier matrix as a preconditioner before it makes a new one. Those tests check
when the factorization is kept, when it is made again, and that the solution always meets
the tolerance.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS

import splineax as splx
from splineax import (
    KLU,
    BCOOLinearOperator,
    BiCGStabOptions,
    CGOptions,
    GMRESOptions,
    HybridDirectIterative,
    IterativeRefinement,
    ReuseOptions,
    RichardsonOptions,
    Spsolve,
)
from splineax.solvers._iterative import HybridState
from splineax.solvers._pardiso import _pardiso_available

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


@pytest.mark.cpu_only
def test_wraps_direct_solver_matches_numpy(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """Refining a direct solve gives the same answer as `numpy.linalg.solve`; the
    refinement is a transparent pass-through when the inner solve is already accurate."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert jnp.allclose(solution, _EXPECTED, atol=1e-8)


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_solve_under_jit(make_operator: OperatorFactory, enable_x64: None) -> None:
    """The refinement loop is traceable, so a solve wrapped in `jax.jit` runs and gives
    the right answer."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())

    @jax.jit
    def solve(b: Array) -> Array:
        return lx.linear_solve(operator, b, solver=solver).value

    assert jnp.allclose(solve(RIGHT_HAND_SIDE), _EXPECTED, atol=1e-8)


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_stateful_linear_solve_returns_tuple(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`splineax.linear_solve` drives the wrapper's init/track/release and returns a
    `(solution, state)` tuple, and threading the state reuses the inner factorization."""
    operator = make_operator(SQUARE_MATRIX)
    solver = IterativeRefinement(KLU())
    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
    assert isinstance(state, HybridState)
    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver, state=state)
    state.release()
    assert jnp.allclose(solution.value, _EXPECTED, atol=1e-8)


@pytest.mark.cpu_only
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
    stale_state = HybridState(
        base_state.inner_state,
        perturbed_operator,
        base_state.stale,
        base_state.reuses,
        base_state.refactor_next,
    )

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


# ---------------------------------------------------------------------------
# HybridDirectIterative
# ---------------------------------------------------------------------------


_TOLERANCE = 1e-10


@pytest.fixture(params=["klu", "pardiso"])
def direct(request: pytest.FixtureRequest, enable_x64: None) -> splx.SparseLinearSolver:
    """The direct solver under test, with x64 enabled as `KLU` and `Pardiso` need."""
    if request.param == "pardiso":
        if not _pardiso_available():
            pytest.skip("The optional `pardiso-mkl-jax` dependency is not installed.")
        return splx.Pardiso()
    return KLU()


def _tagged_operators(
    scales: list[float], seed: int = 0
) -> tuple[list[BCOOLinearOperator], list[np.ndarray]]:
    """Operators with one sparsity pattern, whose values are perturbed by each scale.

    A scale of `0.01` multiplies every nonzero by a random factor within one percent of
    one, so the operator stays close to the unperturbed matrix.
    """
    base_matrix = np.asarray(SQUARE_MATRIX, dtype=np.float64)
    sparsity = BCOO.fromdense(base_matrix)
    tag = splx.sparsity_pattern_tag(sparsity)
    random_state = np.random.default_rng(seed)
    operators, matrices = [], []
    for scale in scales:
        factors = 1.0 + scale * random_state.uniform(-1.0, 1.0, size=base_matrix.shape)
        matrix = base_matrix * factors
        operators.append(BCOOLinearOperator(BCOO.fromdense(matrix), tags=tag))
        matrices.append(matrix)
    return operators, matrices


def _right_hand_side() -> np.ndarray:
    return np.asarray(RIGHT_HAND_SIDE, dtype=np.float64)


def _dense_relative_residual(matrix: np.ndarray, solution: jax.Array) -> float:
    right_hand_side = _right_hand_side()
    return float(
        np.linalg.norm(right_hand_side - matrix @ np.asarray(solution))
        / np.linalg.norm(right_hand_side)
    )


def _solve_sequence(
    solver: HybridDirectIterative, operators: list[BCOOLinearOperator]
) -> list[lx.Solution]:
    """Solve each operator in turn, threading the state through `linear_solve`."""
    solutions = []
    state = None
    for operator in operators:
        solution, state = splx.linear_solve(
            operator, _right_hand_side(), solver, state=state
        )
        solutions.append(solution)
    assert state is not None
    state.release()
    return solutions


@pytest.mark.cpu_only
def test_small_change_keeps_the_factorization(direct: splx.SparseLinearSolver) -> None:
    """A matrix close to the factored one is solved with the old factorization."""
    operators, matrices = _tagged_operators([0.0, 1e-3])
    solver = HybridDirectIterative(direct, RichardsonOptions(max_steps_stale=8))

    _, reused = _solve_sequence(solver, operators)

    assert reused.result == lx.RESULTS.successful
    assert not reused.stats["refactored"]
    assert reused.stats["reused"]
    assert _dense_relative_residual(matrices[1], reused.value) <= _TOLERANCE


@pytest.mark.cpu_only
def test_large_change_makes_a_new_factorization(
    direct: splx.SparseLinearSolver,
) -> None:
    """When the old factorization is too far off, the solve factors the new matrix."""
    operators, matrices = _tagged_operators([0.0, 0.9])
    solver = HybridDirectIterative(direct, RichardsonOptions(max_steps_stale=2))

    _, fallback = _solve_sequence(solver, operators)

    assert fallback.result == lx.RESULTS.successful
    assert fallback.stats["refactored"]
    assert not fallback.stats["reused"]
    assert _dense_relative_residual(matrices[1], fallback.value) <= _TOLERANCE


@pytest.mark.cpu_only
def test_reuse_off_always_makes_a_new_factorization(
    direct: splx.SparseLinearSolver,
) -> None:
    """With `reuse_direct=False` every new operator is factored before the solve."""
    operators, matrices = _tagged_operators([0.0, 1e-3])
    solver = HybridDirectIterative(direct, reuse_direct=False)

    _, factored = _solve_sequence(solver, operators)

    assert factored.stats["refactored"]
    assert _dense_relative_residual(matrices[1], factored.value) <= _TOLERANCE


@pytest.mark.cpu_only
def test_gmres_converges_on_a_stale_factorization(
    direct: splx.SparseLinearSolver,
) -> None:
    """GMRES reaches the tolerance with the old factorization after a larger change."""
    operators, matrices = _tagged_operators([0.0, 0.1])
    solver = HybridDirectIterative(direct, GMRESOptions(max_steps_stale=20))

    _, reused = _solve_sequence(solver, operators)

    assert reused.result == lx.RESULTS.successful
    assert reused.stats["reused"]
    assert _dense_relative_residual(matrices[1], reused.value) <= _TOLERANCE


@pytest.mark.cpu_only
def test_max_reuses_forces_a_new_factorization(direct: splx.SparseLinearSolver) -> None:
    """After `max_reuses` solves with the old factorization, the next one is made again."""
    operators, _ = _tagged_operators([0.0, 1e-3, 1e-3, 1e-3], seed=1)
    solver = HybridDirectIterative(
        direct,
        RichardsonOptions(max_steps_stale=8),
        reuse_direct=ReuseOptions(max_reuses=2),
    )

    solutions = _solve_sequence(solver, operators)

    refactored = [bool(solution.stats["refactored"]) for solution in solutions[1:]]
    assert refactored == [False, False, True]


@pytest.mark.cpu_only
def test_slow_solve_makes_a_new_factorization_next(
    direct: splx.SparseLinearSolver,
) -> None:
    """A solve that uses most of its steps schedules a new factorization for the next."""
    operators, _ = _tagged_operators([0.0, 0.05, 0.05], seed=2)
    solver = HybridDirectIterative(
        direct,
        RichardsonOptions(max_steps_stale=30),
        reuse_direct=ReuseOptions(slow_fraction=0.01),
    )

    _, first, second = _solve_sequence(solver, operators)

    assert first.stats["reused"]
    assert second.stats["refactored"]


@pytest.mark.parametrize("scale", [1e-4, 1e-2, 0.3, 1.0])
@pytest.mark.parametrize(
    "iterative",
    [RichardsonOptions(), GMRESOptions(), BiCGStabOptions()],
    ids=["richardson", "gmres", "bicgstab"],
)
@pytest.mark.cpu_only
def test_solution_meets_the_tolerance_for_any_change(
    direct: splx.SparseLinearSolver,
    scale: float,
    iterative: RichardsonOptions | GMRESOptions | BiCGStabOptions,
) -> None:
    """Whatever the size of the change, a successful solve is within the tolerance."""
    operators, matrices = _tagged_operators([0.0, scale], seed=3)
    solver = HybridDirectIterative(direct, iterative)

    _, solution = _solve_sequence(solver, operators)

    assert solution.result == lx.RESULTS.successful
    assert _dense_relative_residual(matrices[1], solution.value) <= _TOLERANCE


@pytest.mark.cpu_only
def test_pattern_change_rebuilds(direct: splx.SparseLinearSolver) -> None:
    """An operator with a different sparsity pattern is analyzed and factored again."""
    base_matrix = np.asarray(SQUARE_MATRIX, dtype=np.float64)
    first = BCOOLinearOperator(BCOO.fromdense(base_matrix))
    other_matrix = base_matrix + np.diag([1.0, 1.0, 1.0], k=-1)
    second = BCOOLinearOperator(BCOO.fromdense(other_matrix))
    solver = HybridDirectIterative(direct)

    _, solution = _solve_sequence(solver, [first, second])

    assert solution.stats["refactored"]
    assert _dense_relative_residual(other_matrix, solution.value) <= _TOLERANCE


@pytest.mark.cpu_only
def test_state_threads_through_scan(direct: splx.SparseLinearSolver) -> None:
    """A state carried through `lax.scan` keeps one structure across solves."""
    operators, matrices = _tagged_operators([0.0, 1e-3, 1e-3, 1e-3], seed=5)
    solver = HybridDirectIterative(direct, RichardsonOptions(max_steps_stale=8))
    tag = operators[0].tags
    all_values = jnp.stack([operator.matrix.data for operator in operators])
    indices = operators[0].matrix.indices
    shape = operators[0].matrix.shape

    def make_operator(values: jax.Array) -> BCOOLinearOperator:
        return BCOOLinearOperator(BCOO((values, indices), shape=shape), tags=tag)

    first_solution, state = splx.linear_solve(
        make_operator(all_values[0]), _right_hand_side(), solver
    )

    def step(carry, values):
        solution, new_state = splx.linear_solve(
            make_operator(values), _right_hand_side(), solver, state=carry
        )
        return new_state, solution.value

    state, solutions = jax.jit(lambda s: jax.lax.scan(step, s, all_values[1:]))(state)
    state.release()

    for matrix, solution in zip(matrices[1:], solutions):
        assert _dense_relative_residual(matrix, solution) <= _TOLERANCE
    assert _dense_relative_residual(matrices[0], first_solution.value) <= _TOLERANCE


@pytest.mark.cpu_only
def test_gradient_matches_dense_solve(direct: splx.SparseLinearSolver) -> None:
    """Derivatives of a reused solve match the derivatives of a dense solve."""
    operators, matrices = _tagged_operators([0.0, 1e-3])
    solver = HybridDirectIterative(direct, RichardsonOptions(max_steps_stale=8))
    operator = operators[1]
    _, state = splx.linear_solve(operators[0], _right_hand_side(), solver)

    def loss(values: jax.Array, right_hand_side: jax.Array) -> jax.Array:
        perturbed = BCOOLinearOperator(
            BCOO((values, operator.matrix.indices), shape=operator.matrix.shape),
            tags=operator.tags,
        )
        solution, _ = splx.linear_solve(perturbed, right_hand_side, solver, state=state)
        return jnp.sum(solution.value**2)

    def dense_loss(values: jax.Array, right_hand_side: jax.Array) -> jax.Array:
        dense = BCOO(
            (values, operator.matrix.indices), shape=operator.matrix.shape
        ).todense()
        return jnp.sum(jnp.linalg.solve(dense, right_hand_side) ** 2)

    values = operator.matrix.data
    right_hand_side = jnp.asarray(_right_hand_side())
    gradient = jax.grad(loss, argnums=(0, 1))(values, right_hand_side)
    expected = jax.grad(dense_loss, argnums=(0, 1))(values, right_hand_side)
    state.release()

    for actual, reference in zip(gradient, expected):
        assert jnp.allclose(actual, reference, rtol=1e-6, atol=1e-8)
    del matrices


@pytest.mark.cpu_only
def test_vmap_over_right_hand_sides(direct: splx.SparseLinearSolver) -> None:
    """A batch of right-hand sides solves against one state."""
    if isinstance(direct, splx.Pardiso):
        pytest.skip("`Pardiso` has no batching rule for its solve.")
    operators, matrices = _tagged_operators([0.0, 1e-3])
    solver = HybridDirectIterative(direct, RichardsonOptions(max_steps_stale=8))
    _, state = splx.linear_solve(operators[0], _right_hand_side(), solver)
    right_hand_sides = jnp.stack([_right_hand_side(), 2.0 * _right_hand_side()])

    def solve(right_hand_side: jax.Array) -> jax.Array:
        solution, _ = splx.linear_solve(
            operators[1], right_hand_side, solver, state=state
        )
        return solution.value

    solutions = jax.vmap(solve)(right_hand_sides)
    state.release()

    expected = np.linalg.solve(matrices[1], np.asarray(right_hand_sides).T).T
    assert jnp.allclose(solutions, expected, atol=1e-8)


def test_richardson_solves_with_a_preconditioner(enable_x64: None) -> None:
    """`Richardson` is a lineax solver that takes the preconditioner as an option."""
    matrix = jnp.asarray(SQUARE_MATRIX, dtype=jnp.float64)
    operator = lx.MatrixLinearOperator(matrix)
    approximate_inverse = lx.MatrixLinearOperator(
        jnp.linalg.inv(matrix * 1.05 + 0.01 * jnp.eye(4))
    )

    solution = lx.linear_solve(
        operator,
        jnp.asarray(_right_hand_side()),
        splx.Richardson(rtol=1e-10, max_steps=50),
        options={"preconditioner": approximate_inverse},
    )

    assert solution.stats["num_steps"] > 0
    assert jnp.allclose(
        solution.value, jnp.linalg.solve(matrix, _right_hand_side()), atol=1e-8
    )


@pytest.mark.cpu_only
def test_transposed_stale_state_solves_the_transposed_system(
    direct: splx.SparseLinearSolver,
) -> None:
    """Transposing a state that kept an old factorization gives the transposed solve."""
    operators, matrices = _tagged_operators([0.0, 1e-3])
    solver = HybridDirectIterative(direct, RichardsonOptions(max_steps_stale=8))
    state = solver.init(operators[0], {})
    _, _, _, stale_state = solver.update_and_compute(
        state, operators[1], _right_hand_side(), {}
    )
    transposed_state, options = solver.transpose(stale_state, {})
    solution, result, _ = solver.compute(transposed_state, _right_hand_side(), options)
    stale_state.release()

    assert result == lx.RESULTS.successful
    expected = np.linalg.solve(matrices[1].T, _right_hand_side())
    assert jnp.allclose(solution, expected, atol=1e-8)


@pytest.mark.cpu_only
def test_gradient_through_a_stale_state(direct: splx.SparseLinearSolver) -> None:
    """Differentiating a solve against a state that kept an old factorization is correct."""
    operators, _ = _tagged_operators([0.0, 1e-3, 2e-3])
    solver = HybridDirectIterative(direct, RichardsonOptions(max_steps_stale=8))
    state = solver.init(operators[0], {})
    _, _, _, stale_state = solver.update_and_compute(
        state, operators[1], _right_hand_side(), {}
    )
    operator = operators[2]

    def build(values: jax.Array) -> BCOOLinearOperator:
        return BCOOLinearOperator(
            BCOO((values, operator.matrix.indices), shape=operator.matrix.shape),
            tags=operator.tags,
        )

    def loss(values: jax.Array) -> jax.Array:
        solution, _ = splx.linear_solve(
            build(values), jnp.asarray(_right_hand_side()), solver, state=stale_state
        )
        return jnp.sum(solution.value**2)

    def dense_loss(values: jax.Array) -> jax.Array:
        dense = BCOO(
            (values, operator.matrix.indices), shape=operator.matrix.shape
        ).todense()
        return jnp.sum(jnp.linalg.solve(dense, _right_hand_side()) ** 2)

    gradient = jax.grad(loss)(operator.matrix.data)
    stale_state.release()

    assert jnp.allclose(
        gradient, jax.grad(dense_loss)(operator.matrix.data), rtol=1e-6, atol=1e-8
    )


def _symmetric_operators(
    scales: list[float],
) -> tuple[list[BCOOLinearOperator], list[np.ndarray]]:
    """Symmetric positive definite operators with one pattern and perturbed values."""
    base = np.asarray(SQUARE_MATRIX, dtype=np.float64)
    symmetric = base @ base.T + 10.0 * np.eye(4)
    sparsity = BCOO.fromdense(symmetric)
    tag = splx.sparsity_pattern_tag(sparsity)
    random_state = np.random.default_rng(6)
    operators, matrices = [], []
    for scale in scales:
        noise = 1.0 + scale * random_state.uniform(-1.0, 1.0, size=symmetric.shape)
        matrix = symmetric * (noise + noise.T) / 2.0
        operators.append(
            BCOOLinearOperator(
                BCOO.fromdense(matrix),
                tags=frozenset({tag, lx.positive_semidefinite_tag}),
            )
        )
        matrices.append(matrix)
    return operators, matrices


@pytest.mark.cpu_only
def test_cg_reuses_the_factorization_of_a_symmetric_operator(
    direct: splx.SparseLinearSolver,
) -> None:
    """Conjugate gradients reaches the tolerance with a stale factorization."""
    operators, matrices = _symmetric_operators([0.0, 1e-3])
    solver = HybridDirectIterative(direct, CGOptions(max_steps_stale=20))

    _, reused = _solve_sequence(solver, operators)

    assert reused.result == lx.RESULTS.successful
    assert reused.stats["reused"]
    assert _dense_relative_residual(matrices[1], reused.value) <= _TOLERANCE


@pytest.mark.cpu_only
def test_profile_records_the_reuse_attempt_and_the_refactor(
    direct: splx.SparseLinearSolver,
) -> None:
    """The solve profile shows the attempt with the old factorization, and whether the
    update that followed made a new one."""
    operators, _ = _tagged_operators([0.0, 1e-3, 0.9], seed=7)
    solver = HybridDirectIterative(
        direct,
        RichardsonOptions(max_steps_stale=8),
        reuse_direct=ReuseOptions(slow_fraction=None),
    )

    profile = splx.create_solve_profile()
    with profile:
        _solve_sequence(solver, operators)

    attempts = [
        record for record in profile.records if record.operation == "reuse_attempt"
    ]
    updates = [
        record
        for record in profile.records
        if record.operation == "update" and "refactored" in record.outputs
    ]
    assert [record.outputs["accepted"] for record in attempts] == [True, False]
    assert [record.outputs["refactored"] for record in updates][-2:] == [False, True]


@pytest.mark.cpu_only
def test_nonfinite_vector_gives_the_nonfinite_input_result(
    direct: splx.SparseLinearSolver,
) -> None:
    """A right-hand side with a NaN is reported as such, not as a failed iteration."""
    operators, _ = _tagged_operators([0.0])
    solver = HybridDirectIterative(direct)
    vector = _right_hand_side().copy()
    vector[1] = np.nan

    solution, state = splx.linear_solve(operators[0], vector, solver, throw=False)
    state.release()

    assert solution.result == lx.RESULTS.nonfinite_input


@pytest.mark.cpu_only
def test_mismatched_vector_structure_raises(direct: splx.SparseLinearSolver) -> None:
    """A vector of the wrong size is rejected before any solve, as in `lineax`."""
    operators, _ = _tagged_operators([0.0, 1e-3])
    solver = HybridDirectIterative(direct)
    state = solver.init(operators[0], {})

    with pytest.raises(ValueError, match="structures do not match"):
        solver.update_and_compute(state, operators[1], np.ones(3), {})
    state.release()
