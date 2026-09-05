"""Behavioural factorization-reuse suite, shared across all sparse solvers.

Every solver in this package exposes the same stateful API: `init` and `init_symbolic`
build a state, `update` folds in a new operator reusing prior work, `release` frees it,
and `state.track` orders that release after solves. `splineax.linear_solve` ties these
together and returns a `(solution, state)` tuple. This module checks that contract at the
public API level, parametrised over the `solver` fixture (spsolve, klu, pardiso, auto)
from [conftest.py](conftest.py).

Solver-internal lifecycle tests (which factorization each tier reuses, native handle
behaviour) stay solver-specific in [test_klu.py](test_klu.py) and
[test_pardiso.py](test_pardiso.py). The basic solve suite lives in
[test_solvers.py](test_solvers.py).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO

import splineax as splx
from splineax import BCOOLinearOperator

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory

_EXPECTED = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))


def test_linear_solve_returns_solution_and_state(
    make_operator: OperatorFactory, solver: splx.SparseLinearSolver
) -> None:
    """`splineax.linear_solve` returns a `(solution, state)` tuple that solves correctly,
    building a fresh state through `init` when none is passed."""
    operator = make_operator(SQUARE_MATRIX)
    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
    state.release()
    assert jnp.allclose(solution.value, _EXPECTED, atol=1e-5)


def test_reuse_across_vectors_via_update(
    make_operator: OperatorFactory, solver: splx.SparseLinearSolver
) -> None:
    """Threading the returned state back into `splineax.linear_solve` reuses the
    factorization across right-hand sides, and every solve is correct."""
    operator = make_operator(SQUARE_MATRIX)
    second_rhs = jnp.array([4.0, 3.0, 2.0, 1.0]).astype(RIGHT_HAND_SIDE.dtype)
    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
    for rhs in (RIGHT_HAND_SIDE, second_rhs):
        solution, state = splx.linear_solve(operator, rhs, solver, state=state)
        expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(rhs))
        assert jnp.allclose(solution.value, expected, atol=1e-5)
    state.release()


def test_init_update_solve(
    make_operator: OperatorFactory, solver: splx.SparseLinearSolver
) -> None:
    """The explicit `init`/`update` path solves correctly through `lineax.linear_solve`."""
    operator = make_operator(SQUARE_MATRIX)
    state = solver.init(operator, {})
    state = solver.update(state, operator)
    solution = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=solver, state=state
    ).value
    state.release()
    assert jnp.allclose(solution, _EXPECTED, atol=1e-5)


def test_init_symbolic_then_update_solves(
    make_operator: OperatorFactory, solver: splx.SparseLinearSolver
) -> None:
    """`init_symbolic` builds a state from a pattern alone; `update` then folds in the
    operator's values, and the solve is correct."""
    operator = make_operator(SQUARE_MATRIX)
    state = solver.init_symbolic(BCOO.fromdense(SQUARE_MATRIX))
    state = solver.update(state, operator)
    solution = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=solver, state=state
    ).value
    state.release()
    assert jnp.allclose(solution, _EXPECTED, atol=1e-5)


def test_transpose_of_state_solves_transposed(
    make_operator: OperatorFactory, solver: splx.SparseLinearSolver
) -> None:
    """Transposing a state and solving must recover the A^T solution, reusing the
    factorization."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )
    state = solver.init(operator, {})
    transposed_state, _ = solver.transpose(state, {})
    solution = np.asarray(solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0])
    state.release()
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_repeated_update_is_a_no_op(
    make_operator: OperatorFactory, solver: splx.SparseLinearSolver
) -> None:
    """`update` with the same operator object is a no-op: it returns the same state,
    so repeated calls pay nothing and build no new factorization."""
    operator = make_operator(SQUARE_MATRIX)
    state = solver.init(operator, {})
    again = solver.update(state, operator)
    twice = solver.update(again, operator)
    assert again is state
    assert twice is state
    state.release()


def test_state_solve_under_jit(
    make_operator: OperatorFactory, solver: splx.SparseLinearSolver
) -> None:
    """A state passes into a jitted function that solves and returns the result, so any
    tier survives tracing."""
    operator = make_operator(SQUARE_MATRIX)

    @eqx.filter_jit
    def run(state, b):
        return lx.linear_solve(operator, b, solver=solver, state=state).value

    state = solver.init(operator, {})
    solution = np.asarray(run(state, RIGHT_HAND_SIDE))
    state.release()
    assert jnp.allclose(solution, _EXPECTED, atol=1e-5)


def test_release_ordered_under_jit(solver: splx.SparseLinearSolver) -> None:
    """Opening a state, solving, tracking, and releasing all inside one jitted function
    must give the right answer, so the native release is ordered after the solve."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape

    @eqx.filter_jit
    def run(data, b):
        operator = BCOOLinearOperator(
            BCOO((data, indices), shape=shape, indices_sorted=True)
        )
        solution, state = splx.linear_solve(operator, b, solver)
        state.release()
        return solution.value

    solution = np.asarray(run(sparsity.data, RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, _EXPECTED, atol=1e-5)


def test_release_ordered_in_while_loop(solver: splx.SparseLinearSolver) -> None:
    """Solving repeatedly against one state inside a `lax.fori_loop`, then releasing after
    the loop, must give the right accumulated answer, so the release is ordered after
    every iteration's solve."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape
    iterations = 3

    @eqx.filter_jit
    def run(data, b):
        operator = BCOOLinearOperator(
            BCOO((data, indices), shape=shape, indices_sorted=True)
        )
        state = solver.init(operator, {})

        def body(_, carry):
            state, total = carry
            solution, state = splx.linear_solve(operator, b, solver, state=state)
            return state, total + solution.value

        state, total = jax.lax.fori_loop(
            0, iterations, body, (state, jnp.zeros_like(b))
        )
        state.release()
        return total

    total = np.asarray(run(sparsity.data, RIGHT_HAND_SIDE))
    assert jnp.allclose(total, iterations * _EXPECTED, atol=1e-5)


def test_reuse_under_autodiff_matches_dense(
    solver: splx.SparseLinearSolver,
) -> None:
    """Differentiating two tag-sharing solves of different-valued operators, threaded
    through one state, matches a dense reference in both forward and reverse mode.

    This is the reused-factorization gradient bug. The two operators share a cache slot, so
    the second solve refactors it in place. `splineax.linear_solve` factors an independent
    slot for each differentiated solve, so the tangent and adjoint solves read the right
    matrix even under `jit`, where XLA is free to reorder a pure read against an in-place
    refactor. `Spsolve` keeps no shared slot and is a control here.
    """
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape
    tag = splx.sparsity_pattern_tag(sparsity)
    first_rhs = RIGHT_HAND_SIDE
    second_rhs = RIGHT_HAND_SIDE[::-1]

    def _operator(values: jax.Array) -> BCOOLinearOperator:
        matrix = BCOO((values, indices), shape=shape, indices_sorted=True)
        return BCOOLinearOperator(matrix, tags=tag)

    def loss(scale_first: jax.Array, scale_second: jax.Array) -> jax.Array:
        first_solution, state = splx.linear_solve(
            _operator(sparsity.data * scale_first), first_rhs, solver
        )
        second_solution, _ = splx.linear_solve(
            _operator(sparsity.data * scale_second), second_rhs, solver, state=state
        )
        return jnp.sum(first_solution.value**2) + jnp.sum(second_solution.value**2)

    def dense_loss(scale_first: jax.Array, scale_second: jax.Array) -> jax.Array:
        first = jnp.linalg.solve(np.asarray(SQUARE_MATRIX) * scale_first, first_rhs)
        second = jnp.linalg.solve(np.asarray(SQUARE_MATRIX) * scale_second, second_rhs)
        return jnp.sum(first**2) + jnp.sum(second**2)

    point = (jnp.asarray(1.3), jnp.asarray(0.7))
    reverse = jax.jit(jax.grad(loss, argnums=(0, 1)))(*point)
    reverse_reference = jax.grad(dense_loss, argnums=(0, 1))(*point)
    assert jnp.allclose(reverse[0], reverse_reference[0], atol=1e-6)
    assert jnp.allclose(reverse[1], reverse_reference[1], atol=1e-6)

    # Forward mode reuses the same shared slot for its tangent solves, so it exercises the
    # same hazard along an independent path.
    tangents = (jnp.asarray(1.0), jnp.asarray(-0.5))
    _, forward = jax.jit(lambda p, t: jax.jvp(loss, p, t))(point, tangents)
    _, forward_reference = jax.jvp(dense_loss, point, tangents)
    assert jnp.allclose(forward, forward_reference, atol=1e-6)


def test_sparsity_tag_reuse_solves_new_values(
    solver: splx.SparseLinearSolver,
) -> None:
    """Two operators sharing a `sparsity_pattern_tag` let `update` reuse the analysis for
    a matrix with the same pattern but new values, and the solve stays correct."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    first = BCOOLinearOperator(sparsity, tags=tag)
    second_matrix = 2.0 * SQUARE_MATRIX
    second = BCOOLinearOperator(BCOO.fromdense(second_matrix), tags=tag)

    state = solver.init(first, {})
    state = solver.update(state, second)
    solution = lx.linear_solve(
        second, RIGHT_HAND_SIDE, solver=solver, state=state
    ).value
    state.release()

    expected = jnp.linalg.solve(np.asarray(second_matrix), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)
