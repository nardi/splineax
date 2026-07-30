"""Tests for the bandwidth-reducing orderings.

These check two things. First that every ordering returns a genuine permutation, since a
subtly wrong one would silently corrupt every solve built on it. Second that the bandwidth
it achieves is competitive, measured against `scipy.sparse.csgraph` for the level-set
ordering and against leaving the pattern alone for the spectral one.

The graphs below are chosen for the ways they break these heuristics: a grid has a
degenerate second Laplacian eigenvalue, a disconnected graph has a degenerate first one, a
path has as many breadth-first levels as it has vertices, and a shuffled graph removes any
help the original numbering was accidentally providing.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse
import scipy.sparse.csgraph

from splineax.solvers.native import (
    Ordering,
    bandwidth,
    inverse_permutation,
    order,
)

REORDERINGS = [Ordering.RCM, Ordering.SPECTRAL]


def _pattern(matrix: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """The COO index pair of a dense reference matrix, plus its size."""
    coo = scipy.sparse.coo_matrix(matrix)
    return (
        jnp.asarray(coo.row, dtype=jnp.int32),
        jnp.asarray(coo.col, dtype=jnp.int32),
        matrix.shape[0],
    )


def _reordered_bandwidth(matrix: np.ndarray, ordering: Ordering) -> int:
    """The bandwidth `ordering` achieves on `matrix`, checking it is a permutation first."""
    rows, cols, size = _pattern(matrix)
    perm = order(rows, cols, size, ordering)
    assert sorted(np.asarray(perm).tolist()) == list(range(size)), (
        f"{ordering.name} did not return a permutation"
    )
    relabelled = inverse_permutation(perm)
    return int(bandwidth(relabelled[rows], relabelled[cols]))


def _scipy_bandwidth(matrix: np.ndarray) -> int:
    """The bandwidth SciPy's reverse Cuthill-McKee achieves, as a reference point."""
    rows, cols, _ = _pattern(matrix)
    perm = scipy.sparse.csgraph.reverse_cuthill_mckee(
        scipy.sparse.csr_matrix(matrix), symmetric_mode=True
    )
    relabelled = inverse_permutation(jnp.asarray(perm, dtype=jnp.int32))
    return int(bandwidth(relabelled[rows], relabelled[cols]))


def _grid_laplacian(side: int) -> np.ndarray:
    """The five-point Laplacian on a `side` by `side` grid.

    Its natural row-by-row numbering already has bandwidth `side`, and the symmetry between
    the two axes makes the second Laplacian eigenvalue exactly double, which is the case
    that defeats sorting by a single arbitrary Fiedler vector.
    """
    identity = np.eye(side)
    line = np.eye(side) * 4.0 - np.eye(side, k=1) - np.eye(side, k=-1)
    coupling = np.eye(side, k=1) + np.eye(side, k=-1)
    return np.kron(identity, line) - np.kron(coupling, identity)


def _path(size: int) -> np.ndarray:
    """A tridiagonal matrix, whose graph is a path with `size` breadth-first levels."""
    return np.eye(size) + np.eye(size, k=1) + np.eye(size, k=-1)


def _random_symmetric(size: int, density: float, seed: int) -> np.ndarray:
    """A diagonally dominant symmetric matrix with a scattered pattern."""
    sparse = scipy.sparse.random(size, size, density=density, random_state=seed)
    dense = sparse.toarray()
    return dense + dense.T + np.eye(size) * 5.0


def _shuffled(matrix: np.ndarray, seed: int) -> np.ndarray:
    """`matrix` under a random symmetric permutation, so its numbering carries no help."""
    order_ = np.random.default_rng(seed).permutation(matrix.shape[0])
    return matrix[np.ix_(order_, order_)]


def _disconnected() -> np.ndarray:
    """Two uncoupled components, so the Laplacian eigenvalue zero is double."""
    block = _grid_laplacian(5)
    size = block.shape[0]
    matrix = np.zeros((2 * size, 2 * size))
    matrix[:size, :size] = block
    matrix[size:, size:] = block
    return matrix


CASES: dict[str, np.ndarray] = {
    "grid": _grid_laplacian(12),
    "grid_shuffled": _shuffled(_grid_laplacian(12), seed=0),
    "path": _path(64),
    "random": _random_symmetric(200, density=0.02, seed=0),
    "random_shuffled": _shuffled(_random_symmetric(200, density=0.02, seed=0), seed=1),
    "disconnected": _disconnected(),
}


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@pytest.mark.parametrize("ordering", REORDERINGS, ids=lambda o: o.name)
def test_returns_a_permutation(case: str, ordering: Ordering) -> None:
    """Every ordering must return each index exactly once. `_reordered_bandwidth` asserts
    this, so calling it is the test."""
    _reordered_bandwidth(CASES[case], ordering)


def test_none_is_the_identity() -> None:
    """`Ordering.NONE` must leave the numbering alone, so that an operator already in a good
    order can skip the analysis entirely."""
    rows, cols, size = _pattern(CASES["random"])
    perm = order(rows, cols, size, Ordering.NONE)
    assert jnp.array_equal(perm, jnp.arange(size))


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
@pytest.mark.parametrize("ordering", REORDERINGS, ids=lambda o: o.name)
def test_never_worse_than_leaving_it_alone(case: str, ordering: Ordering) -> None:
    """No ordering may increase the bandwidth.

    For the spectral ordering this is the point of scoring the identity alongside the
    eigenvectors: a degenerate eigenvalue can make every candidate eigenvector a poor
    coordinate, and without the comparison the result could come out worse than doing
    nothing.
    """
    matrix = CASES[case]
    rows, cols, _ = _pattern(matrix)
    assert _reordered_bandwidth(matrix, ordering) <= int(bandwidth(rows, cols))


@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_rcm_matches_scipy(case: str) -> None:
    """The level-set ordering must be at least as good as SciPy's.

    Equality is not required, since the two break ties differently, but a gap would mean
    something is wrong: both number by breadth-first level and order within a level by the
    position of the earliest-numbered neighbour, so they should agree closely.
    """
    matrix = CASES[case]
    assert _reordered_bandwidth(matrix, Ordering.RCM) <= _scipy_bandwidth(matrix)


@pytest.mark.parametrize("side", [8, 12])
def test_rcm_reaches_the_grid_optimum(side: int) -> None:
    """On a grid the narrowest possible band is one grid line wide, and the level-set
    ordering must find it even when the original numbering is shuffled. This is what fails
    if a level is ordered by neighbour index rather than neighbour position."""
    shuffled = _shuffled(_grid_laplacian(side), seed=2)
    assert _reordered_bandwidth(shuffled, Ordering.RCM) == side


def test_rcm_separates_components() -> None:
    """A disconnected graph must be numbered one component at a time, so that the bandwidth
    reflects the components rather than the gap between them."""
    matrix = _disconnected()
    within_component = _reordered_bandwidth(_grid_laplacian(5), Ordering.RCM)
    assert _reordered_bandwidth(matrix, Ordering.RCM) == within_component


def test_path_is_already_optimal() -> None:
    """A path graph is already minimally banded, so no ordering may make it worse. It also
    exercises the level-set ordering's worst case, where there is one vertex per level."""
    for ordering in REORDERINGS:
        assert _reordered_bandwidth(_path(64), ordering) == 1


def test_handles_a_non_symmetric_pattern() -> None:
    """Reordering symmetrises the pattern, so an operator whose pattern is not symmetric
    must still be reordered rather than rejected."""
    dense = scipy.sparse.random(150, 150, density=0.02, random_state=7).toarray()
    matrix = dense + np.eye(150) * 5.0
    rows, cols, _ = _pattern(matrix)
    assert _reordered_bandwidth(matrix, Ordering.RCM) < int(bandwidth(rows, cols))


def test_spectral_uses_the_iterative_eigensolver_above_the_dense_limit() -> None:
    """Past the dense cut-off the Laplacian is only ever multiplied by a vector, never
    assembled. The ordering must still be valid and useful there, which the smaller cases
    above cannot show since they take the dense path."""
    matrix = _random_symmetric(600, density=0.006, seed=3)
    rows, cols, _ = _pattern(matrix)
    assert _reordered_bandwidth(matrix, Ordering.SPECTRAL) <= int(bandwidth(rows, cols))


def test_rejects_an_unknown_ordering() -> None:
    """An ordering outside the enumeration must fail loudly rather than silently returning
    an arbitrary permutation."""
    rows, cols, size = _pattern(CASES["path"])
    with pytest.raises(ValueError, match="Ordering.NONE"):
        order(rows, cols, size, "spectral")  # type: ignore[arg-type]
