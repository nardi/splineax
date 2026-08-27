"""Tests for `AutoSparseLinearSolver` dispatch and the sparse-solver Protocol.

`AutoSparseLinearSolver` selects `Pardiso` (if the optional `pardiso-mkl-jax` dependency
is installed) or otherwise `KLU` on CPU when x64 is enabled, since both are double
precision only, and `Spsolve` otherwise. It exposes the same stateful API as
`Pardiso`/`KLU` so it can be substituted verbatim. The generic solve suite lives in
test_solvers.py and the shared reuse contract in test_factorization.py. This module covers
Auto-specific dispatch and Protocol conformance.

The dispatch tests monkeypatch `splineax.solvers._auto._pardiso_available` rather than
relying on whether `pardiso-mkl-jax` is installed, so both branches are exercised
deterministically.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import splineax as splx
import splineax.solvers._auto as _auto_module
from splineax import (
    KLU,
    AutoSparseLinearSolver,
    Pardiso,
    Spsolve,
)
from splineax.solvers import SparseLinearSolver
from splineax.solvers._klu import _KLUState
from splineax.solvers._pardiso import _pardiso_available

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


def test_select_solver_prefers_pardiso_on_cpu_with_x64(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no override, `AutoSparseLinearSolver` selects `Pardiso` on CPU when x64 is
    enabled and `pardiso-mkl-jax` is installed."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), Pardiso)


def test_select_solver_falls_back_to_klu_when_pardiso_unavailable(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `pardiso-mkl-jax` is not installed, `AutoSparseLinearSolver` falls back to
    `KLU` on CPU when x64 is enabled."""
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
    """An explicit `platform` override forces the corresponding solver, without a solve
    (so no real GPU is required to check the non-CPU branch)."""
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


def test_auto_stateful_api_solves_and_releases(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`AutoSparseLinearSolver` routes the whole stateful API (`init_symbolic`, `update`,
    `release`, and `splineax.linear_solve`'s tuple return) through the chosen solver."""
    operator = make_operator(SQUARE_MATRIX)
    solver = AutoSparseLinearSolver()
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
    assert jnp.allclose(solution.value, expected, atol=1e-5)
    solver.release(state)

    symbolic_state = solver.update(
        solver.init_symbolic(BCOO.fromdense(SQUARE_MATRIX)), operator
    )
    reused = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=solver, state=symbolic_state
    ).value
    solver.release(symbolic_state)
    assert jnp.allclose(reused, expected, atol=1e-5)


def test_auto_falls_back_to_klu_for_complex_when_pardiso_chosen(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`pardiso_mkl_jax` does not support complex matrices, so `init` falls back to `KLU`
    for a complex operator even when `Pardiso` was otherwise selected, and every later
    call on that state keeps using `KLU`."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)

    with jax.enable_x64(True):
        # Built inside the block: `.astype(jnp.complex128)` outside it would truncate to
        # complex64, since x64 is not enabled yet at that point.
        complex_matrix = SQUARE_MATRIX.astype(jnp.complex128) * (1 + 1j)
        right_hand_side = RIGHT_HAND_SIDE.astype(jnp.complex128)
        operator = make_operator(complex_matrix)
        expected = jnp.linalg.solve(
            np.asarray(complex_matrix), np.asarray(right_hand_side)
        )

        solver = AutoSparseLinearSolver()
        assert isinstance(solver.select_solver(operator), Pardiso)

        state = solver.init(operator, {})
        assert isinstance(state, _KLUState)
        solution = lx.linear_solve(
            operator, right_hand_side, solver=solver, state=state
        ).value
        assert jnp.allclose(solution, expected, atol=1e-5)

        updated = solver.update(state, operator)
        assert isinstance(updated, _KLUState)
        reused = lx.linear_solve(
            operator, right_hand_side, solver=solver, state=updated
        ).value
        solver.release(updated)
        assert jnp.allclose(reused, expected, atol=1e-5)


def test_solvers_satisfy_sparse_linear_solver_protocol() -> None:
    """All solvers structurally satisfy the `SparseLinearSolver` Protocol."""
    assert isinstance(KLU(), SparseLinearSolver)
    assert isinstance(Spsolve(), SparseLinearSolver)
    assert isinstance(AutoSparseLinearSolver(), SparseLinearSolver)
    if _pardiso_available():
        assert isinstance(Pardiso(), SparseLinearSolver)
