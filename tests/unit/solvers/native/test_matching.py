"""Tests for the maximum bipartite matching.

Two things are checked, mirroring [test_ordering.py](test_ordering.py). First that the result
is always a genuine matching, since a subtly wrong one would silently corrupt the grouping and
the singularity check both depend on. Second that its size agrees with SciPy's reference, which
pins maximality, the property Hopcroft-Karp exists to guarantee and a merely maximal matching
would not.

The graphs below are chosen for the ways they break a matching routine: one is already
zero-diagonal-free, one hides a perfectly good diagonal behind a shuffled numbering, one is a
genuine saddle point, one has no perfect matching at all, and one is disconnected.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse
from scipy.sparse.csgraph import maximum_bipartite_matching

from splineax.solvers.native._matching import matching


def _pattern(matrix: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """The COO index pair of a dense reference matrix, plus its size."""
    coo = scipy.sparse.coo_matrix(matrix)
    return (
        jnp.asarray(coo.row, dtype=jnp.int32),
        jnp.asarray(coo.col, dtype=jnp.int32),
        matrix.shape[0],
    )


def _scipy_matching_size(matrix: np.ndarray) -> int:
    pattern = scipy.sparse.csr_matrix(np.abs(matrix) > 1e-14)
    result = maximum_bipartite_matching(pattern, perm_type="row")
    return int(np.sum(result >= 0))


def _already_matched(size: int = 40) -> np.ndarray:
    """A banded matrix with a full diagonal, so the matching is trivially the identity."""
    generator = np.random.default_rng(0)
    within = np.abs(np.subtract.outer(np.arange(size), np.arange(size))) <= 2
    return generator.normal(size=(size, size)) * within + np.eye(size) * 5.0


def _shuffled_diagonal(size: int = 64) -> np.ndarray:
    """A diagonal matrix with its rows permuted, so every row looks like a constraint row
    even though the underlying problem has none."""
    matrix = np.diag(np.arange(1.0, size + 1.0))
    order = np.random.default_rng(1).permutation(size)
    return matrix[order]


def _saddle_point(size: int = 48) -> np.ndarray:
    """An `[[F, B^T], [B, 0]]` block system with a rectangular `B`, a genuine saddle point
    with a perfect matching between its rows and columns."""
    ordinary = (2 * size) // 3
    constraints = ordinary // 3
    stiffness = np.eye(ordinary) * 2.0 - np.eye(ordinary, k=1) - np.eye(ordinary, k=-1)
    divergence = np.zeros((constraints, ordinary))
    for row in range(constraints):
        divergence[row, 3 * row] = -1.0
        divergence[row, 3 * row + 1] = 1.0
        divergence[row, min(3 * row + 2, ordinary - 1)] += 0.5
    return np.block(
        [[stiffness, divergence.T], [divergence, np.zeros((constraints, constraints))]]
    )


def _structurally_singular(size: int = 30) -> np.ndarray:
    """Two rows sharing support with only one column between them, so no perfect matching
    exists whatever values the pattern is given."""
    matrix = np.eye(size)
    matrix[3, 3] = 0.0
    matrix[3, 4] = 1.0
    matrix[4, 4] = 0.0
    return matrix


def _disconnected(size: int = 24) -> np.ndarray:
    """Two uncoupled banded blocks, so the matching has to be found independently in each."""
    half = size // 2
    within = np.abs(np.subtract.outer(np.arange(half), np.arange(half))) <= 1
    generator = np.random.default_rng(2)
    block = generator.normal(size=(half, half)) * within + np.eye(half) * 5.0
    matrix = np.zeros((size, size))
    matrix[:half, :half] = block
    matrix[half:, half:] = block
    return matrix


CASES: dict[str, np.ndarray] = {
    "already_matched": _already_matched(),
    "shuffled_diagonal": _shuffled_diagonal(),
    "saddle_point": _saddle_point(),
    "structurally_singular": _structurally_singular(),
    "disconnected": _disconnected(),
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_returns_a_valid_matching(case: str) -> None:
    """Every matched pair must be a stored entry, and no row or column may be claimed
    twice. A subtly wrong matching is worse than none, since the grouping it feeds would
    silently pair a constraint with a row it is not actually coupled to."""
    matrix = CASES[case]
    rows, cols, size = _pattern(matrix)
    partner, matched = matching(rows, cols, size)

    partner_np = np.asarray(partner)
    stored_pairs = {
        (int(r), int(c)) for r, c in zip(*np.nonzero(np.abs(matrix) > 1e-14))
    }
    matched_rows = np.flatnonzero(partner_np >= 0)

    for row in matched_rows:
        assert (int(row), int(partner_np[row])) in stored_pairs

    matched_cols = partner_np[matched_rows]
    assert len(set(matched_cols.tolist())) == len(matched_rows), (
        "the same column was claimed by more than one row"
    )
    assert int(matched) == len(matched_rows)


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_matches_scipy_in_size(case: str) -> None:
    """The matching found must be *maximum*, not merely maximal: its size must equal
    SciPy's reference size on every pattern, including the one with no perfect matching."""
    matrix = CASES[case]
    rows, cols, size = _pattern(matrix)
    _, matched = matching(rows, cols, size)
    assert int(matched) == _scipy_matching_size(matrix)


def test_structurally_singular_pattern_is_reported_as_deficient() -> None:
    """The one pattern with no perfect matching must come back short of `size`, which is
    what the structural-rank check in `_block_jacobi.py` relies on to detect it."""
    matrix = _structurally_singular()
    rows, cols, size = _pattern(matrix)
    _, matched = matching(rows, cols, size)
    assert int(matched) < size


def test_an_empty_pattern_matches_nothing() -> None:
    """A pattern with no stored entries at all must terminate immediately rather than
    hang, and report a matching of size zero."""
    size = 10
    rows = jnp.zeros(0, dtype=jnp.int32)
    cols = jnp.zeros(0, dtype=jnp.int32)
    partner, matched = matching(rows, cols, size)
    assert int(matched) == 0
    assert np.all(np.asarray(partner) == -1)
