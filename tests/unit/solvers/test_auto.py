"""Tests for `AutoSparseLinearSolver`, the sparse-solver Protocols, and the
no-op factorization API on `Spsolve`.

`AutoSparseLinearSolver` selects `Pardiso` (if the optional `pardiso-mkl-jax`
dependency is installed) or otherwise `KLU` on CPU when x64 is enabled, since both are
double precision only, and `Spsolve` otherwise (CPU without x64, or any other
platform). It exposes the same factorization API as `Pardiso`/`KLU` so it can be
substituted verbatim. The generic solve-correctness suite (parametrised over all
solvers) lives in `test_solvers.py`. This module covers Auto-specific dispatch,
Protocol conformance, and the Spsolve no-op factorization behaviour.

The dispatch tests monkeypatch `splineax.solvers._auto._pardiso_available` rather than
relying on whether `pardiso-mkl-jax` actually happens to be installed, so both branches
are exercised deterministically regardless of the test environment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import splineax.solvers._auto as _auto_module
from splineax import (
    KLU,
    AutoSparseLinearSolver,
    BCOOLinearOperator,
    Pardiso,
    Spsolve,
)
from splineax.solvers import (
    SparseBasicState,
    SparseLinearSolver,
    SparseSymbolicScope,
    SparseSymbolicState,
)
from splineax.solvers._klu import _KLUBasicState, _KLUNumericState, _KLUSymbolicState
from splineax.solvers._pardiso import _pardiso_available

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory

KLU_STATE_TYPES = (_KLUBasicState, _KLUSymbolicState, _KLUNumericState)

# ---------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------


def test_select_solver_prefers_pardiso_on_cpu_with_x64(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no override, `AutoSparseLinearSolver` selects `Pardiso` on the CPU test
    platform when x64 is enabled and `pardiso-mkl-jax` is installed."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), Pardiso)


def test_select_solver_falls_back_to_klu_when_pardiso_unavailable(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `pardiso-mkl-jax` isn't installed, `AutoSparseLinearSolver` falls back to
    `KLU` on the CPU test platform when x64 is enabled."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: False)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), KLU)


def test_select_solver_falls_back_to_spsolve_on_cpu_without_x64(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On CPU with x64 disabled, `AutoSparseLinearSolver` falls back to `Spsolve`, since
    both `Pardiso` and `KLU` are double precision only."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(False):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), Spsolve)


def test_select_solver_platform_override(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit `platform` override forces the corresponding solver, without running
    a solve (so no real GPU is required to check the non-CPU branch)."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(
            AutoSparseLinearSolver(platform="cpu").select_solver(operator), Pardiso
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


def test_auto_solve_matches_numpy(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`AutoSparseLinearSolver` produces the same solution as `numpy.linalg.solve`."""
    operator = make_operator(SQUARE_MATRIX)
    solution = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=AutoSparseLinearSolver()
    ).value
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_auto_factorize_gives_correct_solution(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`AutoSparseLinearSolver().factorize(operator)` yields a reusable state that solves
    correctly (delegating to Pardiso or KLU on CPU)."""
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    solver = AutoSparseLinearSolver()
    with solver.factorize(operator) as state:
        solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=solver, state=state
        ).value

    assert jnp.allclose(solution, expected, atol=1e-5)


def test_auto_factorize_symbolic_gives_correct_solution(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`AutoSparseLinearSolver().factorize_symbolic(...)` yields a scope whose `init` and
    `factorize` both produce correct solutions (delegating to Pardiso or KLU on CPU)."""
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


def test_auto_falls_back_to_klu_for_complex_when_pardiso_chosen(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pardiso_mkl_jax` doesn't support complex matrices, so `init`/`factorize` must
    fall back to `KLU` for a complex operator even when `Pardiso` was otherwise
    selected, keeping `Auto` able to solve anything `KLU` can."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)

    with jax.enable_x64(True):
        # Built inside the block: `.astype(jnp.complex128)` outside it would silently
        # truncate to complex64, since x64 isn't enabled yet at that point.
        complex_matrix = SQUARE_MATRIX.astype(jnp.complex128) * (1 + 1j)
        right_hand_side = RIGHT_HAND_SIDE.astype(jnp.complex128)
        operator = make_operator(complex_matrix)
        expected = jnp.linalg.solve(
            np.asarray(complex_matrix), np.asarray(right_hand_side)
        )

        solver = AutoSparseLinearSolver()
        assert isinstance(solver.select_solver(operator), Pardiso)

        state = solver.init(operator, {})
        assert isinstance(state, KLU_STATE_TYPES)
        solution = lx.linear_solve(
            operator, right_hand_side, solver=solver, state=state
        ).value
        assert jnp.allclose(solution, expected, atol=1e-5)

        with solver.factorize(operator) as numeric_state:
            assert isinstance(numeric_state, KLU_STATE_TYPES)
            factorized_solution = lx.linear_solve(
                operator, right_hand_side, solver=solver, state=numeric_state
            ).value
        assert jnp.allclose(factorized_solution, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_solvers_satisfy_sparse_linear_solver_protocol() -> None:
    """All solvers implement the `SparseLinearSolver` Protocol."""
    assert isinstance(KLU(), SparseLinearSolver)
    assert isinstance(Spsolve(), SparseLinearSolver)
    assert isinstance(AutoSparseLinearSolver(), SparseLinearSolver)
    if _pardiso_available():
        assert isinstance(Pardiso(), SparseLinearSolver)


def test_states_and_scopes_satisfy_protocols(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """Init states, symbolic scopes, and symbolic states satisfy their Protocols, for
    KLU, Spsolve, and (if installed) Pardiso."""
    operator = make_operator(SQUARE_MATRIX)
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    solvers = [KLU(), Spsolve()]
    if _pardiso_available():
        solvers.append(Pardiso())

    for solver in solvers:
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
