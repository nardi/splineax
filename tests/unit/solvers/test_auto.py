"""Tests for `AutoSparseLinearSolver`, the sparse-solver Protocols, and the
no-op factorization API on `Spsolve`.

`AutoSparseLinearSolver` selects `KLU` on CPU when x64 is enabled, since `klujax` is
double precision only, and `Spsolve` otherwise (CPU without x64, or any other
platform). It exposes the same factorization API as `KLU` so it can be substituted
verbatim. The generic solve-correctness suite (parametrised over all three solvers)
lives in `test_solvers.py`; this module covers Auto-specific dispatch, Protocol
conformance, and the Spsolve no-op factorization behaviour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO

from splineax import (
    KLU,
    AutoSparseLinearSolver,
    BCOOLinearOperator,
    Spsolve,
)
from splineax.solvers import (
    SparseBasicState,
    SparseLinearSolver,
    SparseSymbolicScope,
    SparseSymbolicState,
)

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory

# ---------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------


def test_select_solver_defaults_to_klu_on_cpu_with_x64(
    make_operator: OperatorFactory,
) -> None:
    """With no override, `AutoSparseLinearSolver` selects `KLU` on the CPU test platform
    when x64 is enabled."""
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), KLU)


def test_select_solver_falls_back_to_spsolve_on_cpu_without_x64(
    make_operator: OperatorFactory,
) -> None:
    """On CPU with x64 disabled, `AutoSparseLinearSolver` falls back to `Spsolve`, since
    `KLU` (via klujax) is double precision only."""
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(False):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), Spsolve)


def test_select_solver_platform_override(make_operator: OperatorFactory) -> None:
    """An explicit `platform` override forces the corresponding solver, without running
    a solve (so no real GPU is required to check the non-CPU branch)."""
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(
            AutoSparseLinearSolver(platform="cpu").select_solver(operator), KLU
        )
        assert isinstance(
            AutoSparseLinearSolver(platform="gpu").select_solver(operator), Spsolve
        )
    with jax.enable_x64(False):
        assert isinstance(
            AutoSparseLinearSolver(platform="cpu").select_solver(operator), Spsolve
        )
        assert isinstance(
            AutoSparseLinearSolver(platform="gpu").select_solver(operator), Spsolve
        )


# ---------------------------------------------------------------------------
# Solve correctness through Auto
# ---------------------------------------------------------------------------


def test_auto_solve_matches_numpy(make_operator: OperatorFactory) -> None:
    """`AutoSparseLinearSolver` produces the same solution as `numpy.linalg.solve`."""
    operator = make_operator(SQUARE_MATRIX)
    solution = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=AutoSparseLinearSolver()
    ).value
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_auto_factorize_gives_correct_solution(
    make_operator: OperatorFactory,
) -> None:
    """`AutoSparseLinearSolver().factorize(operator)` yields a reusable state that solves
    correctly (delegating to KLU on CPU)."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    solver = AutoSparseLinearSolver()
    with solver.factorize(operator) as state:
        solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=solver, state=state
        ).value

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_auto_factorize_symbolic_gives_correct_solution(
    make_operator: OperatorFactory,
) -> None:
    """`AutoSparseLinearSolver().factorize_symbolic(...)` yields a scope whose `init` and
    `factorize` both produce correct solutions (delegating to KLU on CPU)."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    solver = AutoSparseLinearSolver()
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


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_solvers_satisfy_sparse_linear_solver_protocol() -> None:
    """All three solvers implement the `SparseLinearSolver` Protocol."""
    assert isinstance(KLU(), SparseLinearSolver)
    assert isinstance(Spsolve(), SparseLinearSolver)
    assert isinstance(AutoSparseLinearSolver(), SparseLinearSolver)


def test_states_and_scopes_satisfy_protocols(
    make_operator: OperatorFactory,
) -> None:
    """Init states, symbolic scopes, and symbolic states satisfy their Protocols, for
    both KLU and Spsolve."""
    operator = make_operator(SQUARE_MATRIX)
    sparsity = BCOO.fromdense(SQUARE_MATRIX)

    for solver in (KLU(), Spsolve()):
        assert isinstance(solver.init(operator, {}), SparseBasicState)
        with solver.factorize_symbolic(sparsity) as scope:
            assert isinstance(scope, SparseSymbolicScope)
            assert isinstance(scope.init(operator), SparseSymbolicState)


# ---------------------------------------------------------------------------
# Spsolve no-op factorization
# ---------------------------------------------------------------------------


def test_spsolve_factorize_noop_solves_correctly(
    make_operator: OperatorFactory,
) -> None:
    """`Spsolve().factorize(operator)` is a no-op that still yields a solvable state."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    solver = Spsolve()
    with solver.factorize(operator) as state:
        solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=solver, state=state
        ).value

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_spsolve_factorize_symbolic_noop_solves_correctly() -> None:
    """`Spsolve().factorize_symbolic(...)` yields a no-op scope whose `init` and
    `factorize` both produce solvable states."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    solver = Spsolve()
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
