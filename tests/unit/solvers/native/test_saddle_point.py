"""Tests for the matching-informed grouping `BlockJacobiGMRES` uses on saddle points.

`test_block_jacobi.py` already checks that a saddle-point matrix survives (stays finite, and a
reported success is backed by a small residual), using the structurally zero blocks it forces.
What it does not check is the point of this change: that those blocks stop being structurally
zero at all, because the grouping now pairs each constraint with a matched ordinary unknown
before the blocks are ever cut. That is what the tests here are for.
"""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from splineax import BCOOLinearOperator, BlockJacobiGMRES, Ordering
from splineax.solvers.native._matching import matching
from splineax.solvers.native._ordering import order

from ..conftest import ZERO_DIAGONAL_MATRIX, ZERO_DIAGONAL_RIGHT_HAND_SIDE
from .matrices import divergence_saddle_point


def _operator(matrix: np.ndarray) -> BCOOLinearOperator:
    return BCOOLinearOperator(BCOO.fromdense(jnp.asarray(matrix)))


def _rhs(size: int, seed: int = 1) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(seed).normal(size=size))


def _relative_residual(
    matrix: np.ndarray, solution: jnp.ndarray, vector: jnp.ndarray
) -> float:
    residual = matrix @ np.asarray(solution) - np.asarray(vector)
    return float(np.linalg.norm(residual) / np.linalg.norm(np.asarray(vector)))


def _grid_laplacian(side: int) -> np.ndarray:
    identity = np.eye(side)
    line = np.eye(side) * 4.0 - np.eye(side, k=1) - np.eye(side, k=-1)
    coupling = np.eye(side, k=1) + np.eye(side, k=-1)
    return np.kron(identity, line) - np.kron(coupling, identity)


def _shuffled_banded(size: int, half_width: int, seed: int = 0) -> np.ndarray:
    """A diagonally dominant banded matrix with only its rows permuted, so the
    numbering hides a perfectly good diagonal for almost every row without the matrix
    becoming a saddle point: the underlying problem is still ordinary, just relabelled.

    Permuting rows *and* columns together (`matrix[np.ix_(p, p)]`) would be a symmetric
    similarity transform, and a symmetric transform preserves which rows have a
    diagonal entry exactly, so it could never produce this case at all. Only a row-only
    permutation actually hides the diagonal, which is why one is used here.
    """
    generator = np.random.default_rng(seed)
    within = np.abs(np.subtract.outer(np.arange(size), np.arange(size))) <= half_width
    matrix = generator.normal(size=(size, size)) * within + np.eye(size) * (
        3 * half_width + 5
    )
    permutation = np.random.default_rng(seed + 1).permutation(size)
    return matrix[permutation]


def _mildly_shuffled_banded(
    size: int, half_width: int, swaps: int, seed: int = 0
) -> np.ndarray:
    """A diagonally dominant banded matrix with a handful of row pairs swapped, rather
    than every row permuted.

    This is what an accidental hidden diagonal looks like in practice: a few unknowns
    end up relabelled, not the whole system. A full random permutation of every row
    (as `_shuffled_banded` above produces) destroys the matrix's locality altogether,
    which is a harder problem than a hidden diagonal and not one a row permutation
    alone can be expected to fix, since bandwidth reduction still has to work with
    whatever locality survives it. Keeping most rows in place is what lets the row
    permutation repair the diagonal without that side effect.
    """
    generator = np.random.default_rng(seed)
    within = np.abs(np.subtract.outer(np.arange(size), np.arange(size))) <= half_width
    matrix = generator.normal(size=(size, size)) * within + np.eye(size) * (
        3 * half_width + 5
    )
    permutation = np.arange(size)
    swap_generator = np.random.default_rng(seed + 1)
    rows = swap_generator.choice(size, size=2 * swaps, replace=False)
    for first, second in zip(rows[0::2], rows[1::2]):
        permutation[first], permutation[second] = (
            permutation[second],
            permutation[first],
        )
    return matrix[permutation]


def test_detects_the_zero_diagonal_rows(enable_x64: None) -> None:
    """The constraint mask, read back through the permutation, must be exactly the rows
    `ZERO_DIAGONAL_MATRIX` has no diagonal entry for: its first half."""
    matrix = np.asarray(ZERO_DIAGONAL_MATRIX)
    half = matrix.shape[0] // 2
    solver = BlockJacobiGMRES()
    state = solver.init(_operator(matrix), {})

    inv_perm = np.asarray(state.analysis.inv_perm)
    constraint_by_original_index = np.asarray(state.analysis.constraint)[inv_perm]
    expected = np.zeros(matrix.shape[0], dtype=bool)
    expected[:half] = True
    assert np.array_equal(constraint_by_original_index, expected)
    assert state.analysis.structural_rank == matrix.shape[0]


def test_the_guard_declines_a_shuffled_matrix(enable_x64: None) -> None:
    """A matrix whose rows have merely been shuffled must not be treated as a saddle
    point, even though almost every row can end up looking like a constraint row.
    Grouping it as one would be actively harmful: the guard exists precisely to send
    this case to the row-permutation stage instead."""
    matrix = _shuffled_banded(200, half_width=3)
    solver = BlockJacobiGMRES()
    state = solver.init(_operator(matrix), {})
    assert not bool(jnp.any(state.analysis.constraint))


def test_row_permutation_repairs_a_mildly_hidden_diagonal(enable_x64: None) -> None:
    """A handful of relabelled rows, the realistic shape of an accidental hidden
    diagonal, must be repaired well enough to solve to a tight tolerance. `perm` and
    `inv_perm` must also have visibly stopped being mutual inverses, confirming the row
    permutation actually engaged rather than the solve succeeding some other way."""
    matrix = _mildly_shuffled_banded(200, half_width=1, swaps=8)
    size = matrix.shape[0]
    solver = BlockJacobiGMRES(rtol=1e-10, atol=1e-10, block_size=16)
    operator = _operator(matrix)
    state = solver.init(operator, {})

    assert not bool(jnp.any(state.analysis.constraint))
    perm, inv_perm = (
        np.asarray(state.analysis.perm),
        np.asarray(state.analysis.inv_perm),
    )
    assert not np.array_equal(perm[inv_perm], np.arange(size))

    vector = _rhs(size, seed=7)
    solution = lx.linear_solve(operator, vector, solver=solver)
    expected = np.linalg.solve(matrix, np.asarray(vector))
    assert np.allclose(np.asarray(solution.value), expected, atol=1e-6)


def test_row_permutation_survives_the_adversarial_case(enable_x64: None) -> None:
    """A matrix with almost every row relabelled destroys locality outright, which is a
    harder problem than a hidden diagonal and not one a row permutation alone fixes.
    Nothing here promises convergence, only that the solve stays finite and a reported
    success is backed by a small residual, the same bar `test_block_jacobi.py` sets for
    a genuine saddle point that is too hard to precondition well."""
    matrix = _shuffled_banded(200, half_width=1)
    vector = _rhs(matrix.shape[0], seed=8)
    solution = lx.linear_solve(
        _operator(matrix),
        vector,
        solver=BlockJacobiGMRES(rtol=1e-8, atol=1e-8, block_size=16),
        throw=False,
    )
    assert np.all(np.isfinite(np.asarray(solution.value)))
    if solution.result == lx.RESULTS.successful:
        assert _relative_residual(matrix, solution.value, vector) < 1e-6


def test_transpose_solves_the_transposed_system(enable_x64: None) -> None:
    """`perm` and `inv_perm` deliberately stop being mutual inverses under the row
    permutation, so `transpose` can no longer reuse the analysis unchanged the way it
    does for a symmetric reordering. This checks the replacement directly: solving the
    transposed state must solve `A^T x = b`, not `A x = b`.

    The reordering the row permutation reuses was chosen for `A`'s own locality, not
    `A^T`'s, since recomputing it from scratch for every transpose would give up the
    reuse the analysis exists for. The transposed solve is therefore exact but can need
    substantially more of the iteration budget than the forward solve does on the same
    matrix, which is why this asks for a generous one rather than the default: the
    property being pinned is that it eventually reaches machine precision, not that it
    does so quickly.
    """
    matrix = _mildly_shuffled_banded(150, half_width=1, swaps=6)
    size = matrix.shape[0]
    solver = BlockJacobiGMRES(
        rtol=1e-10, atol=1e-10, block_size=16, max_steps=2000, restart=100
    )
    state = solver.init(_operator(matrix), {})

    transposed_state, _ = solver.transpose(state, {})
    vector = _rhs(size, seed=9)
    value, result, _ = solver.compute(transposed_state, vector, {})

    assert result == lx.RESULTS.successful
    expected = np.linalg.solve(matrix.T, np.asarray(vector))
    assert np.allclose(np.asarray(value), expected, atol=1e-6)


def test_every_constraint_sits_beside_its_partner(enable_x64: None) -> None:
    """Each constraint row's matched ordinary partner must end up immediately before it
    in the reordered range, which is what lets a single block contain both and be
    invertible. Checked against the matching computed independently, not just against
    "some ordinary row precedes it": the specific partner matters, since a block would
    still be singular if a constraint ended up next to an unrelated ordinary row."""
    matrix = divergence_saddle_point(200)
    size = matrix.shape[0]
    sparsity = BCOO.fromdense(jnp.asarray(matrix))
    rows, cols = sparsity.indices[:, 0], sparsity.indices[:, 1]
    partner, _ = matching(rows, cols, size)
    partner = np.asarray(partner)

    solver = BlockJacobiGMRES()
    state = solver.init(_operator(matrix), {})
    perm = np.asarray(state.analysis.perm)
    constraint = np.asarray(state.analysis.constraint)
    assert constraint.any(), "the family should have produced constraint rows"

    for position in np.flatnonzero(constraint):
        assert position > 0
        original_row = perm[position]
        preceding_original_row = perm[position - 1]
        assert partner[original_row] == preceding_original_row


def test_a_structurally_singular_operator_is_rejected(enable_x64: None) -> None:
    """A pattern with no perfect matching can never be inverted, whatever values it is
    given, so the solver must say so during analysis rather than iterate uselessly."""
    size = 20
    matrix = np.eye(size)
    matrix[3, 3] = 0.0
    matrix[3, 4] = 1.0
    matrix[4, 4] = 0.0
    with pytest.raises(ValueError, match="structurally singular"):
        BlockJacobiGMRES().init(_operator(matrix), {})


def test_preconditioning_accelerates_the_iteration_on_a_saddle_point(
    enable_x64: None,
) -> None:
    """The grouping must earn its keep: at a budget where the preconditioned solve
    succeeds, plain unpreconditioned GMRES on the same saddle point must not, mirroring
    `test_block_jacobi.test_preconditioning_accelerates_the_iteration`."""
    matrix = divergence_saddle_point(150)
    size = matrix.shape[0]
    vector = _rhs(size, seed=2)
    operator = _operator(matrix)
    budget = 6

    preconditioned = lx.linear_solve(
        operator,
        vector,
        solver=BlockJacobiGMRES(rtol=1e-8, atol=1e-8, max_steps=budget, block_size=32),
        throw=False,
    )
    plain = lx.linear_solve(
        operator,
        vector,
        solver=lx.GMRES(rtol=1e-8, atol=1e-8, max_steps=budget),
        throw=False,
    )

    assert preconditioned.result == lx.RESULTS.successful
    assert plain.result != lx.RESULTS.successful
    expected = np.linalg.solve(matrix, np.asarray(vector))
    assert np.allclose(np.asarray(preconditioned.value), expected, atol=1e-6)


@pytest.mark.parametrize("size", [128, 512, 2048])
def test_converges_as_the_saddle_point_grows(size: int, enable_x64: None) -> None:
    """Convergence must not degrade as the problem grows, which is what distinguishes a
    preconditioner that actually captures the constraint coupling from one that merely
    tolerates it. Only convergence and a small residual are asserted, not an iteration
    count, matching this suite's convention of not pinning iteration counts."""
    # An explicit block size, rather than the automatic choice: `lx.linear_solve` stages
    # `init` into its own trace, which routes automatic selection through the shape-only
    # estimate in `_estimated_block_size` rather than the pattern-measured choice
    # `choose_block_size` would make eagerly (see
    # `test_a_traced_pattern_falls_back_to_an_estimated_block_size`). That estimate is
    # sized for the *average* row of a general pattern, and on this family it can land on
    # a block too small for the grouping's pairing guarantee to be realised within a
    # block's owned core rather than merely somewhere in its wider, overlapping window,
    # which is a real sensitivity worth a caveat on the theory page rather than something
    # to route around silently here.
    matrix = divergence_saddle_point(size)
    vector = _rhs(matrix.shape[0], seed=3)
    solution = lx.linear_solve(
        _operator(matrix),
        vector,
        solver=BlockJacobiGMRES(rtol=1e-8, atol=1e-8, block_size=32, max_steps=64),
        throw=False,
    )
    assert solution.result == lx.RESULTS.successful
    assert _relative_residual(matrix, solution.value, vector) < 1e-6


def test_is_an_exact_no_op_without_constraint_rows(enable_x64: None) -> None:
    """A matrix with a full diagonal must be reordered exactly as it was before this
    change, not merely equivalently: the constraint mask is all `False`, the regrouping
    key reduces to twice the bandwidth-reducing rank, and sorting by it must reproduce
    that ordering exactly rather than only matching its bandwidth."""
    matrix = _grid_laplacian(16)
    size = matrix.shape[0]
    solver = BlockJacobiGMRES(ordering=Ordering.RCM)
    operator = _operator(matrix)
    state = solver.init(operator, {})

    matrix_pattern = BCOO.fromdense(jnp.asarray(matrix))
    expected_perm = order(
        matrix_pattern.indices[:, 0], matrix_pattern.indices[:, 1], size, Ordering.RCM
    )
    assert jnp.array_equal(state.analysis.perm, expected_perm)
    assert not bool(jnp.any(state.analysis.constraint))

    vector = _rhs(size, seed=4)
    solution = lx.linear_solve(operator, vector, solver=solver)
    expected = np.linalg.solve(matrix, np.asarray(vector))
    assert np.allclose(np.asarray(solution.value), expected, atol=1e-6)


def test_survives_a_matrix_with_no_ordinary_rows(enable_x64: None) -> None:
    """A matrix with no diagonal entries at all is not a saddle point by this solver's
    definition, since it has no ordinary unknowns to pair constraints with, and the
    guard correctly declines it. It must still solve like any other structurally
    full-rank pattern: finiteness always, and a reported success backed by a small
    residual.

    An even-sized antidiagonal matrix is the simplest such case: a scaled permutation,
    so it is trivially invertible, and `i == size - 1 - i` has no integer solution when
    `size` is even, so the main diagonal is empty everywhere rather than almost
    everywhere.
    """
    size = 24
    matrix = np.fliplr(np.eye(size)) * 5.0
    vector = _rhs(size, seed=5)
    solution = lx.linear_solve(
        _operator(matrix),
        vector,
        solver=BlockJacobiGMRES(rtol=1e-8, atol=1e-8),
        throw=False,
    )
    assert np.all(np.isfinite(np.asarray(solution.value)))
    if solution.result == lx.RESULTS.successful:
        assert _relative_residual(matrix, solution.value, vector) < 1e-6


def test_traces_with_traced_indices_on_a_saddle_point(enable_x64: None) -> None:
    """An explicit block size makes the whole solve traceable even when the pattern's
    own indices are traced, as `test_block_jacobi.py` checks for a banded matrix. This
    exercises the same property with a saddle-point pattern, so the matching and the
    grouping it drives are covered under trace too."""
    matrix = divergence_saddle_point(90)
    size = matrix.shape[0]
    sparsity = BCOO.fromdense(jnp.asarray(matrix))
    shape = sparsity.shape

    @jax.jit
    def run(indices: jnp.ndarray, values: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        operator = BCOOLinearOperator(BCOO((values, indices), shape=shape))
        solver = BlockJacobiGMRES(rtol=1e-8, atol=1e-8, block_size=32)
        return lx.linear_solve(operator, b, solver=solver).value

    vector = _rhs(size, seed=6)
    solution = run(sparsity.indices, sparsity.data, vector)
    expected = np.linalg.solve(matrix, np.asarray(vector))
    assert np.allclose(np.asarray(solution), expected, atol=1e-6)


def test_zero_diagonal_matrix_family_now_converges(enable_x64: None) -> None:
    """`ZERO_DIAGONAL_MATRIX` is the shape a saddle point has, so with the grouping in
    place it should now converge rather than merely survive."""
    matrix = np.asarray(ZERO_DIAGONAL_MATRIX)
    vector = jnp.asarray(ZERO_DIAGONAL_RIGHT_HAND_SIDE)
    solution = lx.linear_solve(
        _operator(matrix),
        vector,
        solver=BlockJacobiGMRES(rtol=1e-8, atol=1e-8),
        throw=False,
    )
    assert solution.result == lx.RESULTS.successful
    assert _relative_residual(matrix, solution.value, vector) < 1e-6
