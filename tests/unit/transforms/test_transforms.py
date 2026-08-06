"""Test suite for the system-transform framework: `AggregationClustering`,
`RuizEquilibration`, `BlockJacobi`, and `compose_transforms`.

Each transform is checked algebraically against a dense reference (`A x = b` and
`(L A R) y = L b`, `x = R y` must agree), plus whatever is specific to that transform
(permutation validity, row/column maxima, block inversion). `PreconditionedIterativeSolver`
itself is covered separately in `tests/unit/solvers/test_iterative.py`.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from lineax import is_positive_semidefinite
from lineax._tags import positive_semidefinite_tag

from splineax.operators._pattern import MatrixSparsity
from splineax.transforms import (
    AggregationClustering,
    AppliedPermutation,
    AppliedScaling,
    BlockJacobi,
    RuizEquilibration,
    compose_transforms,
)
from splineax.transforms._clustering import _aggregate_clusters, _default_block_size

# A 2D 5-point Laplacian (symmetric positive definite, sparse), used throughout as a
# stand-in for a realistic operator: enough structure for clustering and equilibration
# to do meaningful work, small enough to densify for a reference solve.
_NX, _NY = 6, 6


def _laplacian_2d(nx: int, ny: int) -> np.ndarray:
    n = nx * ny
    dense = np.zeros((n, n))

    def index(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = index(i, j)
            dense[k, k] = 4.0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    dense[k, index(ii, jj)] = -1.0
    return dense


LAPLACIAN: np.ndarray = _laplacian_2d(_NX, _NY)
N: int = LAPLACIAN.shape[0]
RIGHT_HAND_SIDE: jax.Array = jnp.arange(N, dtype=jnp.float32) + 1.0


def _pattern_of(matrix: BCOO) -> MatrixSparsity:
    return MatrixSparsity(
        matrix.indices[:, 0], matrix.indices[:, 1], (matrix.shape[0], matrix.shape[1])
    )


# ---------------------------------------------------------------------------
# Shared algebra: every transform (and their composition) must satisfy
# recover_solution(solve(A', transform_vector(b))) == solve(A, b).
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        "clustering",
        "equilibration",
        "equilibration_symmetric",
        "composed",
    ]
)
def transform(request: pytest.FixtureRequest):
    return {
        "clustering": AggregationClustering(block_size=4),
        "equilibration": RuizEquilibration(iterations=8),
        "equilibration_symmetric": RuizEquilibration(iterations=8, symmetric=True),
        "composed": compose_transforms(
            AggregationClustering(block_size=4), RuizEquilibration(iterations=8)
        ),
    }[request.param]


def test_transform_algebra_matches_dense_solve(transform) -> None:
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    plan = transform.analyze_symbolic(_pattern_of(matrix))
    transformed_matrix, applied = plan.analyze_numeric(matrix)

    transformed_b = applied.transform_vector(RIGHT_HAND_SIDE)
    y = jnp.linalg.solve(transformed_matrix.todense(), transformed_b)
    x = applied.recover_solution(y)

    expected = jnp.linalg.solve(
        jnp.array(LAPLACIAN, dtype=jnp.float32), RIGHT_HAND_SIDE
    )
    assert jnp.allclose(x, expected, atol=1e-3)


def test_transform_transpose_solves_transposed_system(transform) -> None:
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    plan = transform.analyze_symbolic(_pattern_of(matrix))
    transformed_matrix, applied = plan.analyze_numeric(matrix)
    transposed = applied.transpose()

    # The Laplacian is symmetric, so A^T = A and the transposed applied-transform must
    # recover the *same* solution as the untransposed one.
    transformed_b = transposed.transform_vector(RIGHT_HAND_SIDE)
    y = jnp.linalg.solve(transformed_matrix.todense().T, transformed_b)
    x = transposed.recover_solution(y)

    expected = jnp.linalg.solve(
        jnp.array(LAPLACIAN, dtype=jnp.float32), RIGHT_HAND_SIDE
    )
    assert jnp.allclose(x, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# AggregationClustering / AppliedPermutation
# ---------------------------------------------------------------------------


def test_clustering_permutation_matches_dense_permutation() -> None:
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    transform = AggregationClustering(block_size=4)
    plan = transform.analyze_symbolic(_pattern_of(matrix))
    assert plan.is_congruence

    transformed_matrix, applied = plan.analyze_numeric(matrix)
    assert isinstance(applied, AppliedPermutation)
    perm = np.array(applied.row_perm)
    assert np.array_equal(perm, np.array(applied.col_perm)), (
        "clustering must be symmetric"
    )
    assert sorted(perm.tolist()) == list(range(N)), "row_perm must be a permutation"

    expected = LAPLACIAN[perm][:, perm]
    assert np.allclose(np.array(transformed_matrix.todense()), expected, atol=1e-5)


def test_clustering_produces_bounded_clusters() -> None:
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    block_size = 4
    label = _aggregate_clusters(
        matrix.indices[:, 0].astype(jnp.int32),
        matrix.indices[:, 1].astype(jnp.int32),
        N,
        block_size,
    )
    _, counts = np.unique(np.array(label), return_counts=True)
    # A merge can overshoot the cap by at most one (both sides one below it).
    assert counts.max() <= block_size + 1


def test_clustering_runs_under_jit() -> None:
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))

    @jax.jit
    def run(rows, cols):
        pattern = MatrixSparsity(rows, cols, (N, N))
        plan = AggregationClustering(block_size=4).analyze_symbolic(pattern)
        return plan.pattern.rows, plan.pattern.cols

    rows, cols = run(matrix.indices[:, 0], matrix.indices[:, 1])
    assert rows.shape == matrix.indices[:, 0].shape


def test_default_block_size_is_a_clamped_power_of_two() -> None:
    assert _default_block_size(nse=1000, n=1000) == 4  # clamps to the minimum
    assert _default_block_size(nse=8000, n=1000) == 8
    assert _default_block_size(nse=10**7, n=1000) == 128  # clamps to the maximum


def test_asymmetric_permutation_is_not_a_congruence() -> None:
    row_perm = jnp.array([1, 0, 2])
    col_perm = jnp.array([0, 2, 1])
    applied = AppliedPermutation(row_perm, col_perm)

    dense = np.arange(9.0).reshape(3, 3) + np.eye(3) * 10
    permuted = dense[np.array(row_perm)][:, np.array(col_perm)]

    b = jnp.array([1.0, 2.0, 3.0])
    transformed_b = applied.transform_vector(b)
    assert jnp.allclose(transformed_b, b[row_perm])

    y = jnp.linalg.solve(jnp.array(permuted), np.array(transformed_b))
    x = applied.recover_solution(y)
    expected = jnp.linalg.solve(jnp.array(dense), b)
    assert jnp.allclose(x, expected, atol=1e-5)

    transposed = applied.transpose()
    assert jnp.array_equal(transposed.row_perm, applied.col_perm)
    assert jnp.array_equal(transposed.col_perm, applied.row_perm)


# ---------------------------------------------------------------------------
# RuizEquilibration / AppliedScaling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symmetric", [False, True])
def test_ruiz_equilibrates_row_and_column_maxima(symmetric: bool) -> None:
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    transform = RuizEquilibration(iterations=10, symmetric=symmetric)
    plan = transform.analyze_symbolic(_pattern_of(matrix))
    assert plan.is_congruence == symmetric

    transformed_matrix, applied = plan.analyze_numeric(matrix)
    assert isinstance(applied, AppliedScaling)
    dense = np.abs(np.array(transformed_matrix.todense()))
    assert np.allclose(dense.max(axis=1), 1.0, atol=0.05)
    assert np.allclose(dense.max(axis=0), 1.0, atol=0.05)
    if symmetric:
        assert jnp.allclose(applied.r, applied.c)


def test_ruiz_leaves_pattern_unchanged() -> None:
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    pattern = _pattern_of(matrix)
    plan = RuizEquilibration().analyze_symbolic(pattern)
    assert jnp.array_equal(plan.pattern.rows, pattern.rows)
    assert jnp.array_equal(plan.pattern.cols, pattern.cols)


# ---------------------------------------------------------------------------
# BlockJacobi
# ---------------------------------------------------------------------------


def test_block_jacobi_matches_dense_block_inverse() -> None:
    block_size = 4
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    preconditioner = BlockJacobi(block_size=block_size)
    plan = preconditioner.analyze_symbolic(_pattern_of(matrix))

    with plan.analyze_numeric(matrix, frozenset()) as operator:
        result = operator.mv(RIGHT_HAND_SIDE)

    expected = np.zeros(N)
    b_np = np.array(RIGHT_HAND_SIDE)
    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        block = LAPLACIAN[start:end, start:end]
        expected[start:end] = np.linalg.solve(block, b_np[start:end])
    assert np.allclose(np.array(result), expected, atol=1e-3)


def test_block_jacobi_singular_block_falls_back_to_identity() -> None:
    n = 4
    dense = np.zeros((n, n))  # fully singular: every block is the zero matrix
    matrix = BCOO.fromdense(jnp.array(dense))
    preconditioner = BlockJacobi(block_size=2)
    pattern = MatrixSparsity(
        jnp.array([], dtype=jnp.int32), jnp.array([], dtype=jnp.int32), (n, n)
    )
    plan = preconditioner.analyze_symbolic(pattern)

    with plan.analyze_numeric(matrix, frozenset()) as operator:
        result = operator.mv(jnp.ones(n))

    assert jnp.all(jnp.isfinite(result)), "a singular block must not produce NaN/Inf"
    assert jnp.allclose(result, jnp.ones(n)), "should fall back to the identity"


def test_block_jacobi_handles_tail_padding() -> None:
    # n not a multiple of block_size: the last block is partly padding.
    n = 7
    block_size = 3
    dense = np.eye(n) * 5.0
    matrix = BCOO.fromdense(jnp.array(dense, dtype=jnp.float32))
    preconditioner = BlockJacobi(block_size=block_size)
    plan = preconditioner.analyze_symbolic(_pattern_of(matrix))

    with plan.analyze_numeric(matrix, frozenset()) as operator:
        result = operator.mv(jnp.ones(n))

    assert jnp.all(jnp.isfinite(result))
    assert jnp.allclose(result, jnp.full(n, 0.2), atol=1e-5)


def test_block_jacobi_drops_off_block_entries() -> None:
    block_size = 2
    dense = np.array(
        [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 4.0, 0.0, 0.5],
            [0.5, 0.0, 4.0, 1.0],
            [0.0, 0.5, 1.0, 4.0],
        ]
    )
    matrix = BCOO.fromdense(jnp.array(dense, dtype=jnp.float32))
    preconditioner = BlockJacobi(block_size=block_size)
    plan = preconditioner.analyze_symbolic(_pattern_of(matrix))

    with plan.analyze_numeric(matrix, frozenset()) as operator:
        result = operator.mv(jnp.ones(4))

    block0 = dense[:2, :2]
    block1 = dense[2:, 2:]
    expected = np.concatenate(
        [np.linalg.solve(block0, np.ones(2)), np.linalg.solve(block1, np.ones(2))]
    )
    assert np.allclose(np.array(result), expected, atol=1e-4)


def test_block_jacobi_tags_positive_semidefinite_only_when_carried() -> None:
    matrix = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    preconditioner = BlockJacobi(block_size=4)
    plan = preconditioner.analyze_symbolic(_pattern_of(matrix))

    with plan.analyze_numeric(matrix, frozenset()) as operator:
        assert not is_positive_semidefinite(operator)

    with plan.analyze_numeric(
        matrix, frozenset({positive_semidefinite_tag})
    ) as operator:
        assert is_positive_semidefinite(operator)
