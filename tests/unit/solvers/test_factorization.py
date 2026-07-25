"""Behavioural factorization-reuse suite, shared across all sparse solvers.

Every solver exposing the factorization-reuse API (`factorize`, `factorize_symbolic`)
must satisfy the same contract: the returned states solve correctly, survive being reused
across right-hand sides, transpose correctly, and can be passed into a jitted function.
The same holds for a scope bundled with its solver by
`factorize_symbolic(..., as_solver=True)`, covered by the `as_solver` tests at the end.
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
import pytest
from jax.experimental.sparse import BCOO

import splineax as splx
from splineax import (
    AbstractSparseLinearSolver,
    BCOOLinearOperator,
    SymbolicScopedSparseLinearSolver,
)

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
    jitted function: opening the scope, deriving a state, the solve, and the scope's
    handle free all trace together, for every solver that supports factorization reuse.

    Solves through `splineax.linear_solve` rather than `lineax.linear_solve` directly.
    `lineax.linear_solve` runs the solver's `compute` in a nested trace, so the solve
    result registered there belongs to that nested trace. The handle-freeing scope's
    dependency tracking is scoped to the trace the free runs in
    (`splineax/solvers/_handle.py`), so that leaked nested-trace result is dropped
    rather than fed to the free from the outer trace (which used to raise
    `UnexpectedTracerError`, and segfault Pardiso). Dropping it is enough for KLU, whose
    native free short-circuits under jit, but Pardiso emits a real `release` that must be
    ordered after the solve, and the only value in the free's own trace that depends on
    the solve is `lineax.linear_solve`'s return. `splineax.linear_solve` registers
    exactly that, so the free orders correctly for every solver. Ordered effects (JEP
    10657) would order the release without needing this, but do not survive lineax's
    staging: an ordered effect inside `compute` fails to lower because the token is not
    threaded into lineax's nested call.
    """
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    @eqx.filter_jit
    def run(solver, data, b):
        operator = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        with solver.factorize_symbolic(operator) as scope:
            state = scope.init(operator)
            return splx.linear_solve(operator, b, solver, state=state).value

    solution = np.asarray(run(solver, sparsity.data, RIGHT_HAND_SIDE))

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_symbolic_scope_full_jit_raw_linear_solve_raises_helpful_error(
    solver: AbstractSparseLinearSolver,
) -> None:
    """Solving via bare `lineax.linear_solve`, instead of `splineax.linear_solve`, inside
    a `factorize_symbolic` scope opened and closed entirely under one `jax.jit` call is
    unsafe for a solver that owns a native handle: see
    `test_factorize_symbolic_opens_entirely_under_jit` above for why. `HandleDependencies`
    now catches this at trace time, in `compute`, and raises a clear error pointing at
    `splineax.linear_solve` instead of letting it surface later as an opaque tracer error
    or a native use-after-free. `Spsolve` owns no handle, so nothing needs to order its
    (no-op) release, and it solves normally either way.
    """
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape

    @eqx.filter_jit
    def run(solver, data, b):
        operator = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        with solver.factorize_symbolic(operator) as scope:
            state = scope.init(operator)
            return lx.linear_solve(operator, b, solver=solver, state=state).value

    # `AutoSparseLinearSolver` may itself resolve to `Spsolve` on some platforms; ask it
    # what it would pick here rather than assuming, so this test holds on any backend.
    resolved = solver
    if isinstance(solver, splx.AutoSparseLinearSolver):
        resolved = solver.select_solver(BCOOLinearOperator(sparsity))

    if isinstance(resolved, splx.Spsolve):
        expected = jnp.linalg.solve(
            np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE)
        )
        solution = np.asarray(run(solver, sparsity.data, RIGHT_HAND_SIDE))
        assert jnp.allclose(solution, expected, atol=1e-5)
    else:
        with pytest.raises(RuntimeError, match="splineax.linear_solve"):
            run(solver, sparsity.data, RIGHT_HAND_SIDE)


def test_as_solver_yields_a_scoped_solver(solver: AbstractSparseLinearSolver) -> None:
    """`factorize_symbolic(..., as_solver=True)` yields the scope paired with the solver
    it was called on, rather than the bare scope."""
    with solver.factorize_symbolic(
        BCOO.fromdense(SQUARE_MATRIX), as_solver=True
    ) as scoped_solver:
        assert isinstance(scoped_solver, SymbolicScopedSparseLinearSolver)
        assert scoped_solver.solver is solver


def test_as_solver_solves_correctly(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """A scoped solver solves correctly when passed as `solver=` with no state: its
    `init` is the scope's, so `lineax.linear_solve` builds a symbolic-tier state itself.
    Its own `factorize` covers the numeric tier, exactly as the scope's does."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with solver.factorize_symbolic(
        BCOO.fromdense(SQUARE_MATRIX), as_solver=True
    ) as scoped_solver:
        symbolic_solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=scoped_solver
        ).value
        with scoped_solver.factorize(operator) as numeric_state:
            numeric_solution = lx.linear_solve(
                operator, RIGHT_HAND_SIDE, solver=scoped_solver, state=numeric_state
            ).value

    assert jnp.allclose(symbolic_solution, expected, atol=1e-5)
    assert jnp.allclose(numeric_solution, expected, atol=1e-5)


def test_as_solver_init_matches_scope_init(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """The scoped solver's `init` is the scope's `init`, so an explicitly built state
    solves identically to letting `lineax.linear_solve` build one."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with solver.factorize_symbolic(
        BCOO.fromdense(SQUARE_MATRIX), as_solver=True
    ) as scoped_solver:
        state = scoped_solver.init(operator)
        assert type(state) is type(scoped_solver.scope.init(operator))
        solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=scoped_solver, state=state
        ).value

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_as_solver_transposes_through_the_wrapped_solver(
    make_operator: OperatorFactory, solver: AbstractSparseLinearSolver
) -> None:
    """`transpose`/`compute` on a scoped solver delegate to the solver it wraps, so a
    transposed state still recovers the A^T solution."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )

    with solver.factorize_symbolic(
        BCOO.fromdense(SQUARE_MATRIX), as_solver=True
    ) as scoped_solver:
        with scoped_solver.factorize(operator) as state:
            transposed_state, _ = scoped_solver.transpose(state, options={})
            # Force the result before the scope frees the native factorization.
            solution = np.asarray(
                scoped_solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0]
            )

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_as_solver_solve_under_jit(solver: AbstractSparseLinearSolver) -> None:
    """A scoped solver built eagerly passes into a jitted function as the only solver
    argument: no separate scope or state has to be threaded alongside it. Two different
    value arrays on the same pattern reuse the one symbolic factorization."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape
    other_matrix = 2.0 * SQUARE_MATRIX
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    other_expected = jnp.linalg.solve(
        np.asarray(other_matrix), np.asarray(RIGHT_HAND_SIDE)
    )

    @eqx.filter_jit
    def run(scoped_solver, data, b):
        operator = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        return lx.linear_solve(operator, b, solver=scoped_solver).value

    with solver.factorize_symbolic(sparsity, as_solver=True) as scoped_solver:
        # Force the results before the scope frees the native factorization.
        solution = np.asarray(run(scoped_solver, sparsity.data, RIGHT_HAND_SIDE))
        other_solution = np.asarray(
            run(scoped_solver, 2.0 * sparsity.data, RIGHT_HAND_SIDE)
        )

    assert jnp.allclose(solution, expected, atol=1e-5)
    assert jnp.allclose(other_solution, other_expected, atol=1e-5)


def test_as_solver_opens_entirely_under_jit(
    solver: AbstractSparseLinearSolver,
) -> None:
    """A scoped solver can also be opened and closed inside one jitted function, like the
    bare scope in `test_factorize_symbolic_opens_entirely_under_jit`.

    Solving through `splineax.linear_solve` without a state is what makes this safe: it
    runs the scoped solver's `init` itself, in the outer trace, so the resulting state is
    there to register the solve against, which `lineax.linear_solve` (building that state
    internally and never handing it back) could not offer.
    """
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    @eqx.filter_jit
    def run(solver, data, b):
        operator = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        with solver.factorize_symbolic(operator, as_solver=True) as scoped_solver:
            return splx.linear_solve(operator, b, scoped_solver).value

    solution = np.asarray(run(solver, sparsity.data, RIGHT_HAND_SIDE))

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_as_solver_full_jit_raw_linear_solve_raises_helpful_error(
    solver: AbstractSparseLinearSolver,
) -> None:
    """Bare `lineax.linear_solve` on a scoped solver opened and closed inside one jitted
    function is unsafe for the same reason as its state-passing equivalent above, and
    raises the same error pointing at `splineax.linear_solve`."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape

    @eqx.filter_jit
    def run(solver, data, b):
        operator = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        with solver.factorize_symbolic(operator, as_solver=True) as scoped_solver:
            return lx.linear_solve(operator, b, solver=scoped_solver).value

    resolved = solver
    if isinstance(solver, splx.AutoSparseLinearSolver):
        resolved = solver.select_solver(BCOOLinearOperator(sparsity))

    if isinstance(resolved, splx.Spsolve):
        expected = jnp.linalg.solve(
            np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE)
        )
        solution = np.asarray(run(solver, sparsity.data, RIGHT_HAND_SIDE))
        assert jnp.allclose(solution, expected, atol=1e-5)
    else:
        with pytest.raises(RuntimeError, match="splineax.linear_solve"):
            run(solver, sparsity.data, RIGHT_HAND_SIDE)
