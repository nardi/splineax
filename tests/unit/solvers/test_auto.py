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
    IterativeRefinement,
    IterativeRefinementSettings,
    Pardiso,
    Spsolve,
)
from splineax.solvers import SparseLinearSolver
from splineax.solvers._auto import _AutoDispatch
from splineax.solvers._iterative import _IterativeRefinementState
from splineax.solvers._klu import _KLUState
from splineax.solvers._pardiso import _pardiso_available

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


def test_dispatch_prefers_pardiso_on_cpu_with_x64(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no override, the platform dispatch selects `Pardiso` on CPU when x64 is
    enabled and `pardiso-mkl-jax` is installed."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(_AutoDispatch().select_solver(operator), Pardiso)


def test_dispatch_falls_back_to_klu_when_pardiso_unavailable(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `pardiso-mkl-jax` is not installed, the dispatch falls back to `KLU` on CPU
    when x64 is enabled."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: False)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(_AutoDispatch().select_solver(operator), KLU)


def test_dispatch_falls_back_to_spsolve_on_cpu_without_x64(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On CPU with x64 disabled, the dispatch falls back to `Spsolve`, since both
    `Pardiso` and `KLU` are double precision only."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(False):
        assert isinstance(_AutoDispatch().select_solver(operator), Spsolve)


def test_dispatch_platform_override(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit `platform` override forces the corresponding direct solver, without a
    solve (so no real GPU is required to check the non-CPU branch)."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(
            _AutoDispatch(platform="cpu").select_solver(operator), Pardiso
        )
        assert isinstance(
            _AutoDispatch(platform="gpu").select_solver(operator), Spsolve
        )
    with jax.enable_x64(False):
        assert isinstance(
            _AutoDispatch(platform="cpu").select_solver(operator), Spsolve
        )


def test_select_solver_returns_exact_solver_with_refinement(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`AutoSparseLinearSolver.select_solver` returns the exact solver it runs: an
    `IterativeRefinement` wrapping the chosen direct solver by default, and the direct
    dispatch itself when refinement is off."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        refined = AutoSparseLinearSolver().select_solver(operator)
        assert isinstance(refined, IterativeRefinement)
        assert isinstance(refined.solver, _AutoDispatch)

        plain = AutoSparseLinearSolver(iterative_refinement=False).select_solver(
            operator
        )
        assert isinstance(plain, _AutoDispatch)


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
    state.release()

    symbolic_state = solver.update(
        solver.init_symbolic(BCOO.fromdense(SQUARE_MATRIX)), operator
    )
    reused = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=solver, state=symbolic_state
    ).value
    symbolic_state.release()
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

        # Disable refinement so the state is the chosen direct solver's own, which this
        # test inspects to confirm the complex fallback landed on `KLU`.
        solver = AutoSparseLinearSolver(iterative_refinement=False)
        assert isinstance(_AutoDispatch().select_solver(operator), Pardiso)

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
        updated.release()
        assert jnp.allclose(reused, expected, atol=1e-5)


def test_auto_applies_iterative_refinement_by_default(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """By default `AutoSparseLinearSolver` wraps its chosen solver in iterative
    refinement, so its state is an `_IterativeRefinementState`. Disabling it returns the
    chosen solver's own state instead."""
    operator = make_operator(SQUARE_MATRIX)

    refined = AutoSparseLinearSolver().init(operator, {})
    assert isinstance(refined, _IterativeRefinementState)

    plain = AutoSparseLinearSolver(iterative_refinement=False).init(operator, {})
    assert not isinstance(plain, _IterativeRefinementState)


def test_auto_refinement_settings_are_forwarded(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """An `IterativeRefinementSettings` on `AutoSparseLinearSolver` reaches the wrapping
    `IterativeRefinement`, and the configured solve is still correct."""
    operator = make_operator(SQUARE_MATRIX)
    settings = IterativeRefinementSettings(tol=1e-8, max_steps=3)
    solver = AutoSparseLinearSolver(iterative_refinement=settings)
    wrapper = solver.select_solver(operator)
    assert isinstance(wrapper, IterativeRefinement)
    assert wrapper.tol == 1e-8
    assert wrapper.max_steps == 3

    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert jnp.allclose(solution, expected, atol=1e-6)


def test_solvers_satisfy_sparse_linear_solver_protocol() -> None:
    """All solvers structurally satisfy the `SparseLinearSolver` Protocol."""
    assert isinstance(KLU(), SparseLinearSolver)
    assert isinstance(Spsolve(), SparseLinearSolver)
    assert isinstance(AutoSparseLinearSolver(), SparseLinearSolver)
    assert isinstance(IterativeRefinement(KLU()), SparseLinearSolver)
    if _pardiso_available():
        assert isinstance(Pardiso(), SparseLinearSolver)
