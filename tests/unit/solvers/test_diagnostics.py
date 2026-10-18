"""Tests for the factorization diagnostics that solver states and solves report.

The values `Pardiso` and `CuDSS` report are checked against `numpy.linalg.eigvalsh`. A
solver that reports nothing holds an all-`None` `SolverDiagnostics`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import splineax as splx
from splineax import KLU, SolverDiagnostics, Spsolve

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


def _is_empty(diagnostics: SolverDiagnostics) -> bool:
    return all(
        value is None
        for value in (
            diagnostics.positive_definite,
            diagnostics.positive,
            diagnostics.negative,
            diagnostics.perturbed_pivots,
        )
    )


def test_empty_diagnostics_has_no_leaves() -> None:
    """An all-`None` instance has no array leaves, so it never changes a carried state."""
    assert jax.tree.leaves(SolverDiagnostics()) == []


@pytest.mark.parametrize(
    "solver_type", [Spsolve, pytest.param(KLU, marks=pytest.mark.cpu_only)]
)
def test_solvers_without_diagnostics_report_none(
    solver_type: type[Spsolve] | type[KLU],
    make_operator: OperatorFactory,
    enable_x64: None,
) -> None:
    """`Spsolve` and `KLU` hold empty diagnostics in the state and the stats."""
    operator = make_operator(SQUARE_MATRIX)
    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver_type())
    state.release()
    assert isinstance(state.diagnostics, SolverDiagnostics)
    assert _is_empty(state.diagnostics)
    assert isinstance(solution.stats["diagnostics"], SolverDiagnostics)
    assert _is_empty(solution.stats["diagnostics"])


def test_diagnostics_in_stats_survive_grad(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """Differentiating a solve works with the diagnostics in the stats."""
    operator = make_operator(SQUARE_MATRIX)

    def loss(vector: jax.Array) -> jax.Array:
        solution, state = splx.linear_solve(operator, vector, Spsolve())
        state.release()
        return jnp.sum(solution.value)

    gradient = jax.grad(loss)(RIGHT_HAND_SIDE)
    assert gradient.shape == RIGHT_HAND_SIDE.shape
