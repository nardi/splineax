"""Tests specific to `BlockJacobiGMRES`.

The shared suites in the parent directory already cover the solver interface, since
`BlockJacobiGMRES` is in the `solver` fixture. What they cannot cover is anything that only
shows up at a size where the preconditioner is an approximation: their reference matrices fit
inside one block, which makes the preconditioner an exact inverse and the solve immediate.

So the tests here work at sizes where the blocks really are partial, and check the three
things that then matter. That the block inverses are the inverses they claim to be, that
preconditioning actually accelerates the iteration, and that a solve which cannot converge
says so rather than returning a wrong answer quietly.
"""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from splineax import BCOOLinearOperator, BlockInverse, BlockJacobiGMRES, Ordering
from splineax.solvers.native._blocks import block_starts, choose_block_size, geometry

from ..conftest import ZERO_DIAGONAL_MATRIX, ZERO_DIAGONAL_RIGHT_HAND_SIDE

BLOCK_INVERSES = [BlockInverse.SVD, BlockInverse.QR]
ORDERINGS = [Ordering.NONE, Ordering.RCM, Ordering.SPECTRAL]


def _banded(size: int, half_width: int, seed: int = 0) -> np.ndarray:
    """A diagonally dominant banded matrix, so the solve is well posed."""
    generator = np.random.default_rng(seed)
    within = np.abs(np.subtract.outer(np.arange(size), np.arange(size))) <= half_width
    return generator.normal(size=(size, size)) * within + np.eye(size) * (
        3 * half_width + 5
    )


def _grid_laplacian(side: int) -> np.ndarray:
    """The five-point Laplacian, whose condition number grows with the grid.

    Unlike the banded matrices above this one is not diagonally dominant enough for an
    unpreconditioned iteration to make quick progress, which is what makes it useful for
    showing that the preconditioner does something.
    """
    identity = np.eye(side)
    line = np.eye(side) * 4.0 - np.eye(side, k=1) - np.eye(side, k=-1)
    coupling = np.eye(side, k=1) + np.eye(side, k=-1)
    return np.kron(identity, line) - np.kron(coupling, identity)


def _operator(matrix: np.ndarray) -> BCOOLinearOperator:
    return BCOOLinearOperator(BCOO.fromdense(jnp.asarray(matrix)))


def _rhs(size: int, seed: int = 1) -> jnp.ndarray:
    return jnp.asarray(np.random.default_rng(seed).normal(size=size))


def _relative_residual(
    matrix: np.ndarray, solution: jnp.ndarray, vector: jnp.ndarray
) -> float:
    residual = matrix @ np.asarray(solution) - np.asarray(vector)
    return float(np.linalg.norm(residual) / np.linalg.norm(np.asarray(vector)))


@pytest.mark.parametrize("block_inverse", BLOCK_INVERSES, ids=lambda m: m.name)
def test_a_single_block_inverts_the_whole_matrix(
    block_inverse: BlockInverse, enable_x64: None
) -> None:
    """When the matrix fits one block the preconditioner is its exact inverse. This is the
    base case both inversion routes must agree on, and it is what makes the solver degenerate
    into a dense direct method for small systems."""
    matrix = _banded(48, half_width=4, seed=2)
    solver = BlockJacobiGMRES(block_inverse=block_inverse, block_size=48)
    state = solver.init(_operator(matrix), {})

    assert state.analysis.num_blocks == 1
    assert state.analysis.captured == 1.0
    # The reordering is a similarity transform, so compare against the reordered matrix.
    perm = np.asarray(state.analysis.perm)
    reordered = matrix[np.ix_(perm, perm)]
    assert np.allclose(np.asarray(state.inv_blocks[0]), np.linalg.inv(reordered))


@pytest.mark.parametrize("block_inverse", BLOCK_INVERSES, ids=lambda m: m.name)
@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5])
def test_each_block_inverse_matches_its_dense_sub_block(
    overlap: float, block_inverse: BlockInverse, enable_x64: None
) -> None:
    """Every block inverse must be the inverse of the corresponding diagonal sub-block of the
    reordered matrix, including the overlapping ones and the pulled-back last one."""
    size = 96
    matrix = _banded(size, half_width=3, seed=3)
    solver = BlockJacobiGMRES(
        block_inverse=block_inverse, block_size=16, overlap_fraction=overlap
    )
    state = solver.init(_operator(matrix), {})

    perm = np.asarray(state.analysis.perm)
    reordered = matrix[np.ix_(perm, perm)]
    width = state.analysis.block_size
    starts = np.asarray(block_starts(size, 16, overlap))

    for index, start in enumerate(starts):
        block = reordered[start : start + width, start : start + width]
        assert np.allclose(
            np.asarray(state.inv_blocks[index]), np.linalg.inv(block), atol=1e-8
        )


def test_both_inversion_routes_agree(enable_x64: None) -> None:
    """The two routes must agree on blocks of full rank, which is what makes the cheaper one a
    drop-in replacement rather than a different solver."""
    matrix = _banded(96, half_width=3, seed=4)
    states = [
        BlockJacobiGMRES(block_inverse=mode, block_size=16).init(_operator(matrix), {})
        for mode in BLOCK_INVERSES
    ]
    assert np.allclose(
        np.asarray(states[0].inv_blocks), np.asarray(states[1].inv_blocks), atol=1e-8
    )


@pytest.mark.parametrize("block_inverse", BLOCK_INVERSES, ids=lambda m: m.name)
@pytest.mark.parametrize("ordering", ORDERINGS, ids=lambda o: o.name)
def test_solves_a_banded_system(
    ordering: Ordering, block_inverse: BlockInverse, enable_x64: None
) -> None:
    """A banded system larger than any single block must still solve to the requested
    tolerance, for every combination of ordering and inversion route."""
    size = 512
    matrix = _banded(size, half_width=3, seed=5)
    vector = _rhs(size)
    solver = BlockJacobiGMRES(
        rtol=1e-10, atol=1e-10, ordering=ordering, block_inverse=block_inverse
    )
    solution = lx.linear_solve(_operator(matrix), vector, solver=solver)

    assert solution.result == lx.RESULTS.successful
    expected = np.linalg.solve(matrix, np.asarray(vector))
    assert np.allclose(np.asarray(solution.value), expected, atol=1e-6)


def test_preconditioning_accelerates_the_iteration(enable_x64: None) -> None:
    """The point of the preconditioner is that it converges in fewer steps.

    Comparing step counts directly would be brittle, so this compares what a fixed budget
    buys: at a budget where the preconditioned solve succeeds, the same iteration without a
    preconditioner must still be short of the tolerance.
    """
    matrix = _grid_laplacian(32)
    size = matrix.shape[0]
    vector = _rhs(size, seed=6)
    operator = _operator(matrix)
    budget = 16

    preconditioned = lx.linear_solve(
        operator,
        vector,
        solver=BlockJacobiGMRES(rtol=1e-8, atol=1e-8, max_steps=budget),
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


def test_reports_failure_rather_than_a_wrong_answer(enable_x64: None) -> None:
    """A solve that runs out of budget must say so. An iterative solver that reported success
    regardless would be far worse than one that fails, since the caller has no other way to
    tell."""
    matrix = _grid_laplacian(24)
    solution = lx.linear_solve(
        _operator(matrix),
        _rhs(matrix.shape[0], seed=7),
        solver=BlockJacobiGMRES(rtol=1e-12, atol=1e-12, max_steps=1),
        throw=False,
    )
    assert solution.result != lx.RESULTS.successful


@pytest.mark.parametrize("block_inverse", BLOCK_INVERSES, ids=lambda m: m.name)
def test_survives_structurally_singular_blocks(
    block_inverse: BlockInverse, enable_x64: None
) -> None:
    """A saddle-point matrix has structurally zero diagonal entries, so blocks that lie over
    them are singular. The block size is forced small here, because the default would cover
    this small a matrix in one block and never exercise that.

    Two things are pinned. The output stays finite, where inverting a singular block directly
    would give arbitrarily large entries that destroy the iteration. And convergence is not
    claimed unless it happened: this class of problem is exactly where the preconditioner is
    weak, so a reported success has to be backed by the true residual.
    """
    matrix = np.asarray(ZERO_DIAGONAL_MATRIX)
    vector = jnp.asarray(ZERO_DIAGONAL_RIGHT_HAND_SIDE)
    solution = lx.linear_solve(
        _operator(matrix),
        vector,
        solver=BlockJacobiGMRES(
            rtol=1e-8, atol=1e-8, block_size=8, block_inverse=block_inverse
        ),
        throw=False,
    )
    assert np.all(np.isfinite(np.asarray(solution.value)))
    if solution.result == lx.RESULTS.successful:
        assert _relative_residual(matrix, solution.value, vector) < 1e-6


@pytest.mark.parametrize("block_inverse", BLOCK_INVERSES, ids=lambda m: m.name)
@pytest.mark.parametrize("block_size", [4, 8, 16, 24, 32])
def test_success_always_implies_a_small_residual(
    block_size: int, block_inverse: BlockInverse, enable_x64: None
) -> None:
    """A reported success must be backed by the unpreconditioned residual.

    GMRES measures convergence on the preconditioned residual, which only tracks the real one
    while the preconditioner is well conditioned. A weak block structure over a hard matrix
    breaks that, so the solver checks the real residual itself before agreeing that the solve
    converged. Sweeping the block size over a saddle point covers both sides of the boundary.
    """
    matrix = np.asarray(ZERO_DIAGONAL_MATRIX)
    vector = jnp.asarray(ZERO_DIAGONAL_RIGHT_HAND_SIDE)
    solution = lx.linear_solve(
        _operator(matrix),
        vector,
        solver=BlockJacobiGMRES(
            rtol=1e-8, atol=1e-8, block_size=block_size, block_inverse=block_inverse
        ),
        throw=False,
    )
    if solution.result == lx.RESULTS.successful:
        assert _relative_residual(matrix, solution.value, vector) < 1e-6


def test_capture_is_reported(enable_x64: None) -> None:
    """The fraction of the matrix the blocks cover is worth surfacing, since it is the one
    number that explains slow convergence. A narrow band must reach the target."""
    size = 512
    solver = BlockJacobiGMRES()
    with solver.factorize_symbolic(
        BCOO.fromdense(jnp.asarray(_banded(size, half_width=3, seed=8)))
    ) as scope:
        assert scope.analysis.captured >= solver.capture_target
        assert scope.analysis.num_blocks > 1


def test_symbolic_analysis_is_reused_across_values(enable_x64: None) -> None:
    """One analyzed pattern must serve different values, which is the whole point of
    separating the stages. Solving twice through the same scope must give both answers."""
    size = 256
    matrix = _banded(size, half_width=2, seed=9)
    sparsity = BCOO.fromdense(jnp.asarray(matrix))
    indices, shape = sparsity.indices, sparsity.shape
    vector = _rhs(size, seed=10)
    solver = BlockJacobiGMRES(rtol=1e-10, atol=1e-10)

    with solver.factorize_symbolic(sparsity, as_solver=True) as scoped:
        for scale in (1.0, 3.0):
            operator = BCOOLinearOperator(
                BCOO((sparsity.data * scale, indices), shape=shape)
            )
            solution = lx.linear_solve(operator, vector, solver=scoped).value
            expected = np.linalg.solve(matrix * scale, np.asarray(vector))
            assert np.allclose(np.asarray(solution), expected, atol=1e-6)


def test_an_explicit_block_size_traces_with_traced_indices(enable_x64: None) -> None:
    """Choosing a block size needs the indices as values, since it sets array shapes. Given
    one explicitly, nothing else does, so the whole solve traces even when the pattern itself
    is an argument."""
    size = 128
    matrix = _banded(size, half_width=2, seed=11)
    sparsity = BCOO.fromdense(jnp.asarray(matrix))
    shape = sparsity.shape

    @jax.jit
    def run(indices: jnp.ndarray, values: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        operator = BCOOLinearOperator(BCOO((values, indices), shape=shape))
        solver = BlockJacobiGMRES(rtol=1e-10, atol=1e-10, block_size=16)
        return lx.linear_solve(operator, b, solver=solver).value

    vector = _rhs(size, seed=12)
    solution = run(sparsity.indices, sparsity.data, vector)
    expected = np.linalg.solve(matrix, np.asarray(vector))
    assert np.allclose(np.asarray(solution), expected, atol=1e-6)


def test_a_traced_pattern_falls_back_to_an_estimated_block_size(
    enable_x64: None,
) -> None:
    """Without an explicit block size and without index values, the block size is estimated
    from the shape alone. The solve must still be correct, only less well tuned."""
    size = 128
    matrix = _banded(size, half_width=2, seed=13)
    sparsity = BCOO.fromdense(jnp.asarray(matrix))
    shape = sparsity.shape

    @jax.jit
    def run(indices: jnp.ndarray, values: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        operator = BCOOLinearOperator(BCOO((values, indices), shape=shape))
        return lx.linear_solve(
            operator, b, solver=BlockJacobiGMRES(rtol=1e-10, atol=1e-10)
        ).value

    vector = _rhs(size, seed=14)
    solution = run(sparsity.indices, sparsity.data, vector)
    expected = np.linalg.solve(matrix, np.asarray(vector))
    assert np.allclose(np.asarray(solution), expected, atol=1e-6)


def test_reject_estimated_block_size_declines_a_traced_pattern(
    enable_x64: None,
) -> None:
    """`reject_estimated_block_size=True` must refuse the same traced pattern the
    previous test accepts, rather than silently falling back to the shape-only
    estimate: a caller who asked for the choice to be measured, not guessed, should
    hear about it when that cannot be honoured."""
    size = 128
    matrix = _banded(size, half_width=2, seed=13)
    sparsity = BCOO.fromdense(jnp.asarray(matrix))
    shape = sparsity.shape

    @jax.jit
    def run(indices: jnp.ndarray, values: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        operator = BCOOLinearOperator(BCOO((values, indices), shape=shape))
        solver = BlockJacobiGMRES(
            rtol=1e-10, atol=1e-10, reject_estimated_block_size=True
        )
        return lx.linear_solve(operator, b, solver=solver).value

    vector = _rhs(size, seed=14)
    with pytest.raises(ValueError, match="reject_estimated_block_size"):
        run(sparsity.indices, sparsity.data, vector)


def test_reject_estimated_block_size_allows_an_explicit_size(
    enable_x64: None,
) -> None:
    """The setting only guards the estimate. An explicit `block_size` never estimates
    anything, so the same traced pattern must still solve when one is given."""
    size = 128
    matrix = _banded(size, half_width=2, seed=13)
    sparsity = BCOO.fromdense(jnp.asarray(matrix))
    shape = sparsity.shape

    @jax.jit
    def run(indices: jnp.ndarray, values: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        operator = BCOOLinearOperator(BCOO((values, indices), shape=shape))
        solver = BlockJacobiGMRES(
            rtol=1e-10,
            atol=1e-10,
            block_size=16,
            reject_estimated_block_size=True,
        )
        return lx.linear_solve(operator, b, solver=solver).value

    vector = _rhs(size, seed=14)
    solution = run(sparsity.indices, sparsity.data, vector)
    expected = np.linalg.solve(matrix, np.asarray(vector))
    assert np.allclose(np.asarray(solution), expected, atol=1e-6)


def test_choosing_a_block_size_rejects_a_traced_pattern() -> None:
    """The choice itself must fail with an explanation rather than an opaque tracer error,
    since the fix is to pass a block size and that is not otherwise obvious."""
    size = 32

    @jax.jit
    def run(rows: jnp.ndarray, cols: jnp.ndarray) -> int:
        return choose_block_size(rows, cols, size, 16, 0.25, 0.8)[0]

    rows = jnp.arange(size, dtype=jnp.int32)
    with pytest.raises(ValueError, match="block_size"):
        run(rows, rows)


def test_rejects_a_non_square_operator(enable_x64: None) -> None:
    """GMRES needs a square operator, so a non-square one must fail at `init` rather than
    producing nonsense."""
    wide = BCOOLinearOperator(BCOO.fromdense(jnp.zeros((3, 5))))
    with pytest.raises(ValueError, match="square"):
        BlockJacobiGMRES().init(wide, {})


def test_rejects_a_state_built_for_another_size(enable_x64: None) -> None:
    """A symbolic factorization is only valid for the pattern it was computed from. Reusing
    one at the wrong size would silently index out of range, so it is caught."""
    solver = BlockJacobiGMRES()
    with solver.factorize_symbolic(
        BCOO.fromdense(jnp.asarray(_banded(32, half_width=2, seed=15)))
    ) as scope:
        with pytest.raises(ValueError, match="symbolic factorization"):
            scope.init(_operator(_banded(64, half_width=2, seed=16)))


def test_overlap_is_reflected_in_the_partition(enable_x64: None) -> None:
    """More overlap must mean more blocks covering the same range, which is the cost side of
    the trade the overlap fraction controls."""
    size = 256
    counts = [geometry(size, 16, overlap)[2] for overlap in (0.0, 0.25, 0.5, 0.75)]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]
