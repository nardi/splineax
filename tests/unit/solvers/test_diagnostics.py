"""Tests for the factorization diagnostics that solver states and solves report.

`Pardiso` and `CuDSS` report the inertia of a symmetric matrix, which these tests check
against `numpy.linalg.eigvalsh`. A solver that reports nothing holds an all-`None`
`SolverDiagnostics`. The tests that take the `diagnostics_solver` fixture run on both
solvers, and `CuDSS` runs only on a GPU. The `CuDSS` logic is also tested against the fake
in [test_cudss.py](test_cudss.py).
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy.sparse
from jax.experimental.sparse import BCOO

import splineax as splx
from splineax import KLU, Pardiso, SolverDiagnostics, Spsolve
from splineax.solvers._pardiso import upper_triangle_selection

from .conftest import (
    RIGHT_HAND_SIDE,
    SQUARE_MATRIX,
    OperatorFactory,
    _make_bcoo_operator,
    _make_bcsr_operator,
)


@pytest.fixture
def pardiso(enable_x64: None) -> Pardiso:
    """A `Pardiso` solver, with x64 enabled, skipped when it is not installed."""
    return Pardiso()


def _tridiagonal(diagonal: list[float]) -> np.ndarray:
    size = len(diagonal)
    return (
        np.diag(diagonal)
        + np.diag(np.ones(size - 1), 1)
        + np.diag(np.ones(size - 1), -1)
    )


DEFINITE = _tridiagonal([4.0, 5.0, 6.0, 7.0, 8.0])
INDEFINITE = _tridiagonal([2.0, -3.0, 1.0, 4.0, -1.0, 5.0])
SINGULAR = np.block(
    [[np.ones((2, 2)), np.zeros((2, 2))], [np.zeros((2, 2)), np.diag([3.0, 4.0])]]
)
MISSING_DIAGONAL = np.diag([1.0, 0.0, 2.0]) + np.eye(3, k=2) + np.eye(3, k=-2)


Operator = splx.BCOOLinearOperator | splx.BCSRLinearOperator


def _count(value: jax.Array | None) -> int:
    assert value is not None
    return int(value)


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


def _symmetric_operator(
    dense: np.ndarray,
    make: Callable[..., Operator] = _make_bcoo_operator,
    tags: frozenset[object] = frozenset({lx.symmetric_tag}),
) -> Operator:
    """An operator tagged symmetric, with a sparsity tag of its own pattern."""
    operator = make(jnp.asarray(dense), tags)
    pattern = splx.sparsity_pattern_tag(operator.matrix)
    return make(jnp.asarray(dense), tags | {pattern})


def _rhs(dense: np.ndarray) -> jax.Array:
    return jnp.arange(1.0, len(dense) + 1.0)


def _inertia(dense: np.ndarray) -> tuple[int, int]:
    eigenvalues = np.linalg.eigvalsh(dense)
    return int((eigenvalues > 0).sum()), int((eigenvalues < 0).sum())


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


@pytest.mark.parametrize("make", [_make_bcoo_operator, _make_bcsr_operator])
@pytest.mark.parametrize(
    "dense",
    [DEFINITE, INDEFINITE, MISSING_DIAGONAL],
    ids=["definite", "indefinite", "missing_diagonal"],
)
def test_upper_triangle_selection(
    make: Callable[..., Operator], dense: np.ndarray
) -> None:
    """The selection gathers the upper triangle of the sorted CSR values, with a zero
    for every diagonal entry that the pattern lacks."""
    operator = _symmetric_operator(dense, make)
    selection = upper_triangle_selection(splx.sparsity_pattern_tag(operator))
    size = len(dense)
    rows = np.repeat(np.arange(size), np.diff(selection.indptr))
    columns = selection.indices
    expected_entries = set(zip(*np.nonzero(np.triu(dense)))) | {
        (index, index) for index in range(size)
    }
    assert list(zip(rows, columns)) == sorted(expected_entries)
    full = scipy.sparse.csr_matrix(dense)
    full.sort_indices()
    gathered = np.append(full.data, 0.0)[selection.gather]
    np.testing.assert_array_equal(gathered, np.triu(dense)[rows, columns])


def test_upper_triangle_selection_is_cached() -> None:
    """Equal tags share one selection, so it is computed once per pattern."""
    first = splx.sparsity_pattern_tag(_symmetric_operator(DEFINITE))
    second = splx.sparsity_pattern_tag(_symmetric_operator(DEFINITE))
    assert upper_triangle_selection(first) is upper_triangle_selection(second)


@pytest.mark.parametrize(
    "dense", [DEFINITE, INDEFINITE], ids=["definite", "indefinite"]
)
def test_counts_match_eigenvalues(
    diagnostics_solver: splx.SparseLinearSolver, dense: np.ndarray
) -> None:
    """The counts of a symmetric solver equal the signs of the eigenvalues, and the solve
    is still correct."""
    operator = _symmetric_operator(dense)
    solution, state = splx.linear_solve(operator, _rhs(dense), diagnostics_solver)
    state.release()
    positive, negative = _inertia(dense)
    diagnostics = solution.stats["diagnostics"]
    assert int(diagnostics.positive) == positive
    assert int(diagnostics.negative) == negative
    assert _count(diagnostics.perturbed_pivots) == 0
    assert bool(diagnostics.positive_definite) == (negative == 0)
    np.testing.assert_allclose(
        solution.value, np.linalg.solve(dense, np.asarray(_rhs(dense))), atol=1e-10
    )
    assert _count(state.diagnostics.positive) == positive


def test_refactor_updates_the_counts(
    diagnostics_solver: splx.SparseLinearSolver,
) -> None:
    """A new operator on the same pattern gives the counts of its own matrix."""
    first = _symmetric_operator(DEFINITE)
    state = diagnostics_solver.init(first, {})
    assert bool(state.diagnostics.positive_definite)
    # Both matrices have the pattern of `DEFINITE`, with other values.
    for dense in (_tridiagonal([2.0, -3.0, 1.0, 4.0, -1.0]), DEFINITE):
        operator = _symmetric_operator(dense)
        state = diagnostics_solver.update(state, operator, {})
        positive, negative = _inertia(dense)
        assert _count(state.diagnostics.positive) == positive
        assert _count(state.diagnostics.negative) == negative
    state.release()


@pytest.mark.parametrize("make", [_make_bcoo_operator, _make_bcsr_operator])
@pytest.mark.cpu_only
def test_symmetric_pardiso_matches_nonsymmetric(make, pardiso: Pardiso) -> None:
    """Factoring the upper triangle solves the same system as the full matrix."""
    tagged = _symmetric_operator(INDEFINITE, make)
    plain = make(jnp.asarray(INDEFINITE))
    rhs = _rhs(INDEFINITE)
    symmetric, symmetric_state = splx.linear_solve(tagged, rhs, pardiso)
    nonsymmetric, nonsymmetric_state = splx.linear_solve(plain, rhs, pardiso)
    symmetric_state.release()
    nonsymmetric_state.release()
    assert symmetric_state.symmetric
    assert not nonsymmetric_state.symmetric
    assert _is_empty(nonsymmetric_state.diagnostics)
    np.testing.assert_allclose(symmetric.value, nonsymmetric.value, atol=1e-10)


@pytest.mark.cpu_only
def test_pardiso_flags_a_singular_matrix(pardiso: Pardiso) -> None:
    """A singular matrix is not positive definite, and Pardiso reports a perturbed pivot."""
    operator = _symmetric_operator(SINGULAR)
    state = pardiso.init(operator, {})
    state.release()
    diagnostics = state.diagnostics
    assert not bool(diagnostics.positive_definite)
    assert _count(diagnostics.perturbed_pivots) > 0


@pytest.mark.cpu_only
def test_pardiso_psd_tag_still_reports_counts(pardiso: Pardiso) -> None:
    """A positive semidefinite tag factors as indefinite, since only that type has counts."""
    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    state = pardiso.init(_symmetric_operator(DEFINITE, tags=tags), {})
    state.release()
    assert state.symmetric
    assert _count(state.diagnostics.positive) == len(DEFINITE)


@pytest.mark.cpu_only
def test_pardiso_without_a_sparsity_tag_is_nonsymmetric(pardiso: Pardiso) -> None:
    """A symmetric tag with traced indices and no sparsity tag solves, but reports nothing."""
    rhs = _rhs(DEFINITE)

    @jax.jit
    def solve(dense: jax.Array):
        operator = splx.BCOOLinearOperator(
            BCOO.fromdense(dense, nse=int(np.count_nonzero(DEFINITE))),
            {lx.symmetric_tag},
        )
        solution, state = splx.linear_solve(operator, rhs, pardiso)
        state.release()
        return solution.value, solution.stats["diagnostics"]

    value, diagnostics = solve(jnp.asarray(DEFINITE))
    np.testing.assert_allclose(
        value, np.linalg.solve(DEFINITE, np.asarray(rhs)), atol=1e-10
    )
    assert _is_empty(diagnostics)


@pytest.mark.cpu_only
def test_pardiso_nonsymmetric_with_a_tag_only_for_the_pattern(pardiso: Pardiso) -> None:
    """An operator without the symmetric tag keeps the nonsymmetric factorization."""
    operator = _symmetric_operator(DEFINITE, tags=frozenset())
    state = pardiso.init(operator, {})
    state.release()
    assert not state.symmetric
    assert _is_empty(state.diagnostics)


_SHIFTS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)


def _smallest_definite_shift(dense: np.ndarray) -> float:
    return next(
        shift
        for shift in _SHIFTS
        if np.linalg.eigvalsh(dense + shift * np.eye(len(dense))).min() > 0
    )


def _shifted_operator_factory(dense: np.ndarray):
    """A function from a shift to the operator of `dense + shift * I` on one pattern."""
    template = _symmetric_operator(dense)
    matrix = template.matrix
    pattern = splx.sparsity_pattern_tag(template)
    on_diagonal = jnp.asarray(matrix.indices[:, 0] == matrix.indices[:, 1])

    def operator_at(shift: jax.Array) -> splx.BCOOLinearOperator:
        data = matrix.data + jnp.where(on_diagonal, shift, 0.0)
        shifted = BCOO(
            (data, matrix.indices),
            shape=matrix.shape,
            indices_sorted=True,
            unique_indices=True,
        )
        return splx.BCOOLinearOperator(shifted, {lx.symmetric_tag, pattern})

    return operator_at


def test_shift_search_in_a_jitted_loop(
    diagnostics_solver: splx.SparseLinearSolver,
) -> None:
    """A `while_loop` under `jit` raises a shift until `positive_definite` holds, carrying
    the state and reading the flag from the stats."""
    dense = _tridiagonal([2.0, 1.0, 3.0, 0.5, 2.5, 1.5]) - 2.6 * np.eye(6)
    operator_at = _shifted_operator_factory(dense)
    shifts = jnp.asarray(_SHIFTS)
    rhs = _rhs(dense)

    @jax.jit
    def search() -> jax.Array:
        state = diagnostics_solver.init(operator_at(shifts[0]), {})

        def keep_going(carry):
            index, _, definite = carry
            return ~definite & (index + 1 < len(_SHIFTS))

        def raise_shift(carry):
            index, state, _ = carry
            index = index + 1
            solution, state = splx.linear_solve(
                operator_at(shifts[index]), rhs, diagnostics_solver, state=state
            )
            return index, state, solution.stats["diagnostics"].positive_definite

        index, state, _ = jax.lax.while_loop(
            keep_going,
            raise_shift,
            (jnp.int32(0), state, state.diagnostics.positive_definite),
        )
        state.release()
        return shifts[index]

    assert float(search()) == _smallest_definite_shift(dense)


def test_shift_search_through_the_transform(
    diagnostics_solver: splx.SparseLinearSolver,
) -> None:
    """The same search written with `lineax.linear_solve` reads the flag from the stats
    inside `stateful_solve_transform`."""
    dense = _tridiagonal([2.0, 1.0, 3.0, 0.5, 2.5, 1.5]) - 2.6 * np.eye(6)
    operator_at = _shifted_operator_factory(dense)
    shifts = jnp.asarray(_SHIFTS)
    rhs = _rhs(dense)

    def search() -> jax.Array:
        def keep_going(carry):
            index, definite = carry
            return ~definite & (index + 1 < len(_SHIFTS))

        def raise_shift(carry):
            index, _ = carry
            index = index + 1
            solution = lx.linear_solve(
                operator_at(shifts[index]), rhs, diagnostics_solver
            )
            return index, solution.stats["diagnostics"].positive_definite

        index, _ = jax.lax.while_loop(
            keep_going, raise_shift, (jnp.int32(0), jnp.bool_(False))
        )
        return shifts[index]

    threaded = jax.jit(splx.stateful_solve_transform(search))
    assert float(threaded()) == _smallest_definite_shift(dense)
