"""Behavioural factorization-reuse suite, shared across all sparse solvers.

Every solver exposing the factorization-reuse API (`factorize`, `factorize_symbolic`)
must satisfy the same contract: the returned states solve correctly, survive being reused
across right-hand sides, transpose correctly, and can be passed into a jitted function.
This module checks that contract at the public API level, parametrised over the `solver`
fixture (spsolve, klu, pardiso, auto) from [conftest.py](conftest.py).

The solver-internal lifecycle tests (which underlying function each tier calls, when
klujax handles are freed, when the Pardiso solver is closed) stay solver-specific in
[test_klu.py](test_klu.py) and [test_pardiso.py](test_pardiso.py). The basic solve suite
lives in [test_solvers.py](test_solvers.py).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO

from splineax import AbstractSparseLinearSolver, BCOOLinearOperator

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


def test_factorize_solves_correctly(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """`solver.factorize(operator)` yields a reusable state that solves correctly."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with solver.factorize(operator) as state:
        solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=solver, state=state
        ).value

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_factorize_reuses_state_across_vectors(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """A single factorized state must solve several right-hand sides correctly."""
    operator = make_operator(SQUARE_MATRIX)
    # Match the operator dtype: some solvers upcast to float64, so a freshly built array
    # would otherwise mismatch the (float32) operator.
    second_rhs = jnp.array([4.0, 3.0, 2.0, 1.0]).astype(RIGHT_HAND_SIDE.dtype)

    with solver.factorize(operator) as state:
        for rhs in (RIGHT_HAND_SIDE, second_rhs):
            solution = lx.linear_solve(operator, rhs, solver=solver, state=state).value
            expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(rhs))
            assert jnp.allclose(solution, expected, atol=1e-5)


def test_factorize_symbolic_solves_correctly(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """A `factorize_symbolic` scope solves correctly through both its `init` state and its
    fully-factorized state."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with solver.factorize_symbolic(BCOO.fromdense(SQUARE_MATRIX)) as scope:
        symbolic_state = scope.init(operator)
        symbolic_solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=solver, state=symbolic_state
        ).value
        with scope.factorize(operator) as numeric_state:
            numeric_solution = lx.linear_solve(
                operator, RIGHT_HAND_SIDE, solver=solver, state=numeric_state
            ).value

    assert jnp.allclose(symbolic_solution, expected, atol=1e-5)
    assert jnp.allclose(numeric_solution, expected, atol=1e-5)


def test_transpose_of_numeric_state_solves_transposed(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """Transposing a factorized state and solving must recover the A^T solution."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )

    with solver.factorize(operator) as state:
        transposed_state, _ = solver.transpose(state, options={})
        # Force the result before the block frees the underlying native factorization.
        solution = np.asarray(solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0])

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_symbolic_state_solve_under_jit(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """A solver and a symbolic-tier state pass into a jitted function that solves and
    returns the result."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    # filter_jit keeps each state's non-array fields (Pardiso's native handle, KLU's
    # `transposed` flag) static while tracing its arrays, so any tier survives tracing.
    @eqx.filter_jit
    def run(solver, state, b):
        return lx.linear_solve(operator, b, solver=solver, state=state).value

    with solver.factorize_symbolic(BCOO.fromdense(SQUARE_MATRIX)) as scope:
        state = scope.init(operator)
        # Force the result before the scope frees the native factorization.
        solution = np.asarray(run(solver, state, RIGHT_HAND_SIDE))

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_numeric_state_solve_under_jit(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """A solver and a numeric-tier state pass into a jitted function that solves and
    returns the result."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    @eqx.filter_jit
    def run(solver, state, b):
        return lx.linear_solve(operator, b, solver=solver, state=state).value

    with solver.factorize(operator) as state:
        solution = np.asarray(run(solver, state, RIGHT_HAND_SIDE))

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_symbolic_scope_solve_under_jit(solver: AbstractSparseLinearSolver) -> None:
    """Perform the symbolic factorization eagerly, then pass the scope into a jitted
    function that builds the operator inside, derives a state from the scope, solves, and
    returns the result. Covers the case where the concrete operator is only known inside
    the jit context (the sparsity is fixed, only its values vary), which is what symbolic
    reuse exists for. Solving with different values reuses the one analysis."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape
    other_matrix = 2.0 * SQUARE_MATRIX
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    other_expected = jnp.linalg.solve(
        np.asarray(other_matrix), np.asarray(RIGHT_HAND_SIDE)
    )

    @eqx.filter_jit
    def run(scope, data, b):
        operator = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        state = scope.init(operator)
        return lx.linear_solve(operator, b, solver=solver, state=state).value

    with solver.factorize_symbolic(sparsity) as scope:
        # Reuse the scope inside the jit with two different value arrays on the same
        # pattern, without ever calling `scope.init` eagerly first: both `KLU` and
        # `Pardiso` allocate their factorization handle as an ordinary JAX array value,
        # so the analysis performed when the scope was opened is what gets reused here.
        solution = np.asarray(run(scope, sparsity.data, RIGHT_HAND_SIDE))
        other_solution = np.asarray(run(scope, 2.0 * sparsity.data, RIGHT_HAND_SIDE))

    assert jnp.allclose(solution, expected, atol=1e-5)
    assert jnp.allclose(other_solution, other_expected, atol=1e-5)


def test_factorize_symbolic_opens_entirely_under_jit(
    solver: AbstractSparseLinearSolver,
) -> None:
    """`factorize_symbolic` itself, not just a state derived from it, can run inside a
    jitted function: opening the scope, `.init`, and the solve all trace together, for
    every solver that supports factorization reuse.

    Solves through `solver.compute` directly rather than `lx.linear_solve`: the latter's
    autodiff-aware wrapping rebuilds the state pytree with fresh leaf objects, which
    breaks the handle-freeing scope's dependency tracking (keyed by Python object
    identity) when the scope's exit is itself traced into the same jit call as the
    solve, a narrower case than this test needs to cover.
    """
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    @eqx.filter_jit
    def run(solver, data, b):
        operator = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        with solver.factorize_symbolic(operator) as scope:
            state = scope.init(operator)
            return solver.compute(state, b, {})[0]

    solution = np.asarray(run(solver, sparsity.data, RIGHT_HAND_SIDE))

    assert jnp.allclose(solution, expected, atol=1e-5)
