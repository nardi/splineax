"""Tests for the factorization diagnostics that solver states and solves report.

`Pardiso` and `CuDSS` report the inertia of a symmetric matrix, which these tests check
against `numpy.linalg.eigvalsh`. A solver that reports nothing holds an all-`None`
`SolverDiagnostics`. The tests that take the `diagnostics_solver` fixture run on both
solvers, and `CuDSS` runs only on a GPU. The `CuDSS` logic is also tested eagerly against
the fake `spineax.cudss` module, which runs on CPU.
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
from splineax import KLU, CuDSS, Pardiso, SolverDiagnostics, Spsolve
from splineax.solvers import CuDSSMatrixType
from splineax.solvers._pardiso import upper_triangle_selection

from ...fake_cudss import FakeCuDSS
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


@pytest.mark.cpu_only
def test_refinement_passes_the_inner_diagnostics(pardiso: Pardiso) -> None:
    """A refined solve reports the diagnostics of the wrapped solver, in its state and in
    its stats."""
    solver = splx.IterativeRefinement(pardiso)
    solution, state = splx.linear_solve(
        _symmetric_operator(INDEFINITE), _rhs(INDEFINITE), solver
    )
    state.release()
    positive, negative = _inertia(INDEFINITE)
    for diagnostics in (state.diagnostics, solution.stats["diagnostics"]):
        assert _count(diagnostics.positive) == positive
        assert _count(diagnostics.negative) == negative


@pytest.mark.cpu_only
def test_hybrid_reports_the_stale_factorization(pardiso: Pardiso) -> None:
    """While a nearby operator is solved with the old factorization, the diagnostics are
    the ones of that factorization, and a refactorization replaces them."""
    solver = splx.HybridDirectIterative(
        pardiso, splx.RichardsonOptions(max_steps_stale=8)
    )
    first = _symmetric_operator(DEFINITE)
    nearby = _symmetric_operator(DEFINITE + 1e-3 * np.eye(len(DEFINITE)))
    _, state = splx.linear_solve(first, _rhs(DEFINITE), solver)
    solution, state = splx.linear_solve(nearby, _rhs(DEFINITE), solver, state=state)
    state.release()
    assert solution.stats["reused"]
    assert bool(solution.stats["diagnostics"].positive_definite)
    assert _count(solution.stats["diagnostics"].positive) == len(DEFINITE)

    far = _tridiagonal([-2.0, 5.0, 6.0, 7.0, 8.0])
    state = solver.init(first, {})
    solution, _, stats, state = solver.update_and_compute(
        state, _symmetric_operator(far), _rhs(far), {}
    )
    state.release()
    positive, negative = _inertia(far)
    assert _count(stats["diagnostics"].positive) == positive
    assert _count(stats["diagnostics"].negative) == negative


@pytest.mark.cpu_only
def test_auto_passes_the_inner_diagnostics(pardiso: Pardiso) -> None:
    """`AutoSparseLinearSolver` reports the diagnostics of the solver it picked."""
    solution, state = splx.linear_solve(
        _symmetric_operator(INDEFINITE),
        _rhs(INDEFINITE),
        splx.AutoSparseLinearSolver(),
    )
    state.release()
    positive, negative = _inertia(INDEFINITE)
    assert _count(solution.stats["diagnostics"].positive) == positive
    assert _count(solution.stats["diagnostics"].negative) == negative


def _cudss_state(
    dense: np.ndarray, tags: frozenset[object] = frozenset({lx.symmetric_tag})
):
    return CuDSS(diagnostics=True).init(_symmetric_operator(dense, tags=tags), {})


@pytest.mark.parametrize(
    "dense", [DEFINITE, INDEFINITE], ids=["definite", "indefinite"]
)
def test_cudss_symmetric_reports_the_inertia(
    dense: np.ndarray, fake_cudss: FakeCuDSS
) -> None:
    """A symmetric matrix is factored as LDL^T, and the state and the stats of a solve
    both carry its inertia."""
    solver = CuDSS(diagnostics=True)
    state = solver.init(_symmetric_operator(dense), {})
    _, _, stats = solver.compute(state, _rhs(dense), {})
    state.release()
    positive, negative = _inertia(dense)
    for diagnostics in (state.diagnostics, stats["diagnostics"]):
        assert _count(diagnostics.positive) == positive
        assert _count(diagnostics.negative) == negative
        assert _count(diagnostics.perturbed_pivots) == 0
        assert bool(diagnostics.positive_definite) == (negative == 0)


def test_cudss_flags_a_singular_symmetric_matrix(fake_cudss: FakeCuDSS) -> None:
    """A zero eigenvalue is a perturbed pivot counted as positive, so the matrix is not
    definite."""
    state = _cudss_state(SINGULAR)
    state.release()
    assert not bool(state.diagnostics.positive_definite)
    assert _count(state.diagnostics.perturbed_pivots) == 1
    assert _count(state.diagnostics.positive) == len(SINGULAR)


@pytest.mark.parametrize(
    ("dense", "definite"),
    [(DEFINITE, True), (INDEFINITE, False), (SINGULAR, False)],
    ids=["definite", "indefinite", "singular"],
)
def test_cudss_psd_tag_is_factored_as_ldlt_to_report(
    dense: np.ndarray, definite: bool, fake_cudss: FakeCuDSS
) -> None:
    """Cholesky hides a singular matrix, so a positive semidefinite tag gets LDL^T when
    diagnostics are on, and reports the counts and definiteness like any symmetric one."""
    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    state = _cudss_state(dense, tags)
    state.release()
    assert state.token.mtype_id == 1
    assert bool(state.diagnostics.positive_definite) == definite
    assert state.diagnostics.positive is not None


def test_cudss_psd_tag_still_uses_cholesky_without_diagnostics(
    fake_cudss: FakeCuDSS,
) -> None:
    """With the flag off, a positive semidefinite tag keeps its cheaper Cholesky type."""
    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    state = CuDSS().init(_symmetric_operator(DEFINITE, tags=tags), {})
    state.release()
    assert state.token.mtype_id == 3
    assert _is_empty(state.diagnostics)


def test_cudss_matrix_type_overrides_the_tags(fake_cudss: FakeCuDSS) -> None:
    """An explicit type wins over the tags and over the diagnostics flag, and applies to
    an untagged operator and to a bare sparsity pattern."""
    psd = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    operator = _symmetric_operator(DEFINITE, tags=psd)
    for matrix_type, expected in [
        (CuDSSMatrixType.GENERAL, 0),
        (CuDSSMatrixType.SYMMETRIC, 1),
        (CuDSSMatrixType.POSITIVE_DEFINITE, 3),
    ]:
        solver = CuDSS(diagnostics=True, matrix_type=matrix_type)
        state = solver.init(operator, {})
        state.release()
        assert state.token.mtype_id == expected
    untagged = _make_bcoo_operator(jnp.asarray(DEFINITE))
    state = CuDSS(diagnostics=True, matrix_type=CuDSSMatrixType.SYMMETRIC).init(
        untagged, {}
    )
    state.release()
    assert _count(state.diagnostics.positive) == len(DEFINITE)
    sparsity = BCOO.fromdense(jnp.asarray(DEFINITE))
    state = CuDSS(matrix_type=CuDSSMatrixType.SYMMETRIC).init_symbolic(sparsity)
    state.release()
    assert state.token.mtype_id == 1


def test_cudss_forced_cholesky_reports_nothing(fake_cudss: FakeCuDSS) -> None:
    """Cholesky reports no usable diagnostics, even when they were asked for."""
    solver = CuDSS(diagnostics=True, matrix_type=CuDSSMatrixType.POSITIVE_DEFINITE)
    state = solver.init(_symmetric_operator(DEFINITE), {})
    state.release()
    assert _is_empty(state.diagnostics)


def test_cudss_general_matrix_reports_nothing(fake_cudss: FakeCuDSS) -> None:
    """Without a symmetric tag the matrix is factored as LU, which reports nothing."""
    state = CuDSS(diagnostics=True).init(_make_bcoo_operator(jnp.asarray(DEFINITE)), {})
    state.release()
    assert _is_empty(state.diagnostics)


def test_cudss_refactor_updates_the_counts(fake_cudss: FakeCuDSS) -> None:
    """A new operator on the same pattern replaces the counts, and `track` and a
    transpose keep them."""
    solver = CuDSS(diagnostics=True)
    state = solver.init(_symmetric_operator(DEFINITE), {})
    other = _tridiagonal([2.0, -3.0, 1.0, 4.0, -1.0])
    state = solver.update(state, _symmetric_operator(other), {})
    positive, negative = _inertia(other)
    assert _count(state.diagnostics.positive) == positive
    assert _count(state.diagnostics.negative) == negative
    tracked = state.track(jnp.zeros(1))
    transposed, _ = solver.transpose(state, {})
    state.release()
    assert tracked.diagnostics is state.diagnostics
    assert transposed.diagnostics is state.diagnostics


def test_cudss_reports_nothing_unless_asked(fake_cudss: FakeCuDSS) -> None:
    """The diagnostics are off by default, so a symmetric state skips the query and holds
    empty ones, and the solve returns them in the stats."""
    solver = CuDSS()
    state = solver.init(_symmetric_operator(INDEFINITE), {})
    _, _, stats = solver.compute(state, _rhs(INDEFINITE), {})
    state = solver.update(state, _symmetric_operator(DEFINITE), {})
    state.release()
    assert _is_empty(state.diagnostics)
    assert _is_empty(stats["diagnostics"])


def test_cudss_symbolic_state_reports_nothing(fake_cudss: FakeCuDSS) -> None:
    """A state from `init_symbolic` has no factorization to report."""
    sparsity = BCOO.fromdense(jnp.asarray(DEFINITE))
    state = CuDSS(diagnostics=True).init_symbolic(sparsity)
    state.release()
    assert _is_empty(state.diagnostics)


@pytest.mark.cudss_gpu
@pytest.mark.parametrize(
    ("dense", "definite"),
    [(DEFINITE, True), (INDEFINITE, False), (SINGULAR, False)],
    ids=["definite", "indefinite", "singular"],
)
def test_gpu_psd_tag_reports_the_inertia(
    dense: np.ndarray, definite: bool, enable_x64: None
) -> None:
    """On real cuDSS a positive semidefinite tag is factored as LDL^T under diagnostics.
    The counts match the eigenvalues, except that a zero eigenvalue is counted as a
    perturbed pivot and as positive."""
    tags = frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag})
    state = CuDSS(diagnostics=True).init(_symmetric_operator(dense, tags=tags), {})
    state.release()
    diagnostics = state.diagnostics
    assert bool(diagnostics.positive_definite) == definite
    eigenvalues = np.linalg.eigvalsh(dense)
    zero = int((np.abs(eigenvalues) < 1e-9).sum())
    assert _count(diagnostics.perturbed_pivots) == zero
    assert _count(diagnostics.positive) == int((eigenvalues > 1e-9).sum()) + zero
    assert _count(diagnostics.negative) == int((eigenvalues < -1e-9).sum())


@pytest.mark.cudss_gpu
def test_gpu_vmap_over_values_reports_the_whole_batch(enable_x64: None) -> None:
    """Batching the matrix values makes cuDSS factor one block-diagonal matrix, so each
    element reports the counts of the whole batch and definiteness of the whole batch."""
    pattern = BCOO.fromdense(jnp.asarray(INDEFINITE))
    on_diagonal = pattern.indices[:, 0] == pattern.indices[:, 1]
    tag = splx.sparsity_pattern_tag(pattern)
    solver = CuDSS(diagnostics=True)
    shifts = np.array([0.0, 2.0, 4.0, 6.0])

    def solve(shift: jax.Array) -> tuple[jax.Array, SolverDiagnostics]:
        matrix = BCOO(
            (pattern.data + shift * on_diagonal, pattern.indices),
            shape=pattern.shape,
            indices_sorted=True,
        )
        operator = splx.BCOOLinearOperator(matrix, tags={lx.symmetric_tag, tag})
        solution = lx.linear_solve(operator, _rhs(INDEFINITE), solver)
        return solution.value, solution.stats["diagnostics"]

    values, diagnostics = jax.vmap(solve)(jnp.asarray(shifts))
    counts = [
        _inertia(INDEFINITE + shift * np.eye(len(INDEFINITE))) for shift in shifts
    ]
    assert np.asarray(diagnostics.positive).tolist() == [sum(c[0] for c in counts)] * 4
    assert np.asarray(diagnostics.negative).tolist() == [sum(c[1] for c in counts)] * 4
    assert not np.asarray(diagnostics.positive_definite).any()
    for shift, value in zip(shifts, values, strict=True):
        dense = INDEFINITE + shift * np.eye(len(INDEFINITE))
        np.testing.assert_allclose(
            dense @ np.asarray(value), _rhs(INDEFINITE), atol=1e-10
        )


@pytest.mark.cudss_gpu
def test_gpu_singular_matrix_has_perturbed_pivots(enable_x64: None) -> None:
    """A singular matrix gives a perturbed pivot and is not reported as definite, which
    pins down what `npivots` counts."""
    state = CuDSS(diagnostics=True).init(_symmetric_operator(SINGULAR), {})
    state.release()
    assert _count(state.diagnostics.perturbed_pivots) > 0
    assert not bool(state.diagnostics.positive_definite)


@pytest.mark.cpu_only
def test_symmetric_pardiso_is_differentiable(pardiso: Pardiso) -> None:
    """Diagnostics in the stats do not get in the way of a gradient, which matches the
    gradient of a dense solve."""
    sparsity = BCOO.fromdense(jnp.asarray(DEFINITE))
    pattern = splx.sparsity_pattern_tag(sparsity)
    rhs = _rhs(DEFINITE)

    def sparse_loss(scale: jax.Array) -> jax.Array:
        matrix = BCOO(
            (sparsity.data * scale, sparsity.indices),
            shape=sparsity.shape,
            indices_sorted=True,
        )
        operator = splx.BCOOLinearOperator(matrix, tags={lx.symmetric_tag, pattern})
        return lx.linear_solve(operator, rhs, pardiso).value.sum()

    def dense_loss(scale: jax.Array) -> jax.Array:
        return jnp.linalg.solve(scale * jnp.asarray(DEFINITE), rhs).sum()

    np.testing.assert_allclose(
        jax.jit(jax.grad(sparse_loss))(1.5), jax.grad(dense_loss)(1.5), rtol=1e-8
    )
