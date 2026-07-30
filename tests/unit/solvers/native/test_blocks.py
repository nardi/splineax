"""Tests for the overlapping block partition and the choice of block size.

The partition fixes the shape of everything the solver computes later, so the properties
worth pinning are structural: every row is covered, exactly one block owns it, and the
scatter destinations reproduce the diagonal sub-blocks of the matrix they claim to. Those are
checked against brute-force equivalents rather than against expected values, since the
geometry has awkward cases at the tail that are easy to get subtly wrong.

The block size selection is checked mainly for the guarantee that matters: a matrix that fits
in a single block gets a single block, so the preconditioner is then an exact inverse.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse

from splineax.solvers.native._blocks import (
    block_destinations,
    block_starts,
    capture_fraction,
    choose_block_size,
    geometry,
    max_blocks_per_entry,
    partition,
)

# Sizes and block sizes chosen so that some combinations divide evenly and others leave the
# last block pulled back over its neighbour, which is where the geometry is trickiest.
SIZES = [7, 10, 11, 17, 33, 64]
BLOCK_SIZES = [1, 2, 3, 4, 8, 16]
OVERLAPS = [0.0, 0.25, 0.5, 0.75]


def _pattern(matrix: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The COO triple of a dense reference matrix."""
    coo = scipy.sparse.coo_matrix(matrix)
    return (
        jnp.asarray(coo.row, dtype=jnp.int32),
        jnp.asarray(coo.col, dtype=jnp.int32),
        jnp.asarray(coo.data),
    )


def _random_matrix(size: int, seed: int = 0, density: float = 0.35) -> np.ndarray:
    generator = np.random.default_rng(seed)
    mask = generator.random((size, size)) < density
    return mask * generator.normal(size=(size, size)) + np.eye(size) * 3.0


def _banded(size: int, half_width: int) -> np.ndarray:
    matrix = np.zeros((size, size))
    for offset in range(-half_width, half_width + 1):
        matrix += np.eye(size, k=offset)
    return matrix


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("size", SIZES)
def test_every_row_has_exactly_one_owner(
    size: int, block_size: int, overlap: float
) -> None:
    """The owned rows must partition the index range.

    This is what makes applying the block inverses well defined. If a row had two owners its
    contribution would be counted twice, and if it had none the preconditioner would zero it.
    """
    gather_index, core_mask = partition(size, block_size, overlap)
    owned = np.asarray(gather_index)[np.asarray(core_mask)]
    assert sorted(owned.tolist()) == list(range(size))


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("size", SIZES)
def test_blocks_stay_inside_the_matrix(
    size: int, block_size: int, overlap: float
) -> None:
    """Pulling the last block back is what removes padding, so no block may reach past the
    end and every block must be full width."""
    resolved, _, num_blocks = geometry(size, block_size, overlap)
    gather_index, core_mask = partition(size, block_size, overlap)
    assert gather_index.shape == (num_blocks, resolved) == core_mask.shape
    assert int(gather_index.min()) == 0
    assert int(gather_index.max()) == size - 1


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("size", SIZES)
def test_destinations_reproduce_the_diagonal_sub_blocks(
    size: int, block_size: int, overlap: float
) -> None:
    """Scattering the values to their destinations must give exactly the diagonal sub-blocks
    of the matrix, for every block, including the pulled-back last one."""
    matrix = _random_matrix(size)
    rows, cols, data = _pattern(matrix)
    resolved, _, num_blocks = geometry(size, block_size, overlap)

    destinations = block_destinations(rows, cols, size, block_size, overlap)
    # One slot past the end absorbs everything no block covers, then is discarded.
    flat = jnp.zeros(num_blocks * resolved * resolved + 1, data.dtype)
    blocks = flat.at[destinations].add(data[None, :])[:-1]
    blocks = np.asarray(blocks.reshape(num_blocks, resolved, resolved))

    for index, start in enumerate(np.asarray(block_starts(size, block_size, overlap))):
        expected = matrix[start : start + resolved, start : start + resolved]
        assert np.allclose(blocks[index], expected)


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("size", SIZES)
def test_each_entry_reaches_every_block_containing_it(
    size: int, block_size: int, overlap: float
) -> None:
    """An entry in an overlap region belongs to several blocks and must appear in each.

    Counting the valid destinations against a brute-force sweep over the blocks catches both
    a missing copy, which would leave a hole in some block, and a spurious one, which would
    double a value.
    """
    matrix = _random_matrix(size)
    rows, cols, _ = _pattern(matrix)
    resolved, _, num_blocks = geometry(size, block_size, overlap)
    starts = np.asarray(block_starts(size, block_size, overlap))

    row_index = np.asarray(rows)
    col_index = np.asarray(cols)
    expected = sum(
        int(
            (
                (row_index >= start)
                & (row_index < start + resolved)
                & (col_index >= start)
                & (col_index < start + resolved)
            ).sum()
        )
        for start in starts
    )

    destinations = block_destinations(rows, cols, size, block_size, overlap)
    discarded = num_blocks * resolved * resolved
    assert int((destinations < discarded).sum()) == expected


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("size", SIZES)
def test_capture_matches_a_brute_force_count(
    size: int, block_size: int, overlap: float
) -> None:
    """Capture is advertised as measured rather than estimated, so it must agree exactly with
    counting the entries some block covers."""
    matrix = _random_matrix(size)
    rows, cols, _ = _pattern(matrix)
    resolved, _, _ = geometry(size, block_size, overlap)

    row_index = np.asarray(rows)
    col_index = np.asarray(cols)
    covered = np.zeros(row_index.shape, dtype=bool)
    for start in np.asarray(block_starts(size, block_size, overlap)):
        covered |= (
            (row_index >= start)
            & (row_index < start + resolved)
            & (col_index >= start)
            & (col_index < start + resolved)
        )

    measured = float(capture_fraction(rows, cols, size, block_size, overlap))
    assert measured == pytest.approx(float(covered.mean()))


@pytest.mark.parametrize("size", [8, 64, 128])
def test_a_matrix_that_fits_one_block_gets_one_block(size: int) -> None:
    """The round-up guarantee. When the whole matrix fits inside the largest permitted block,
    that single block captures everything, and since its inverse is the inverse of the whole
    matrix the iteration converges at once. Choosing by cost rather than by smallest adequate
    block size is what produces this."""
    rows, cols, _ = _pattern(np.ones((size, size)))
    block_size, captured = choose_block_size(
        rows, cols, size, max_block_size=128, overlap_fraction=0.25, capture_target=0.8
    )
    assert block_size == size
    assert geometry(size, block_size, 0.25)[2] == 1
    assert captured == 1.0


@pytest.mark.parametrize("overlap", OVERLAPS)
def test_a_narrow_band_reaches_the_target_with_small_blocks(overlap: float) -> None:
    """A matrix already concentrated near the diagonal must not need large blocks."""
    size = 512
    rows, cols, _ = _pattern(_banded(size, half_width=3))
    block_size, captured = choose_block_size(
        rows,
        cols,
        size,
        max_block_size=128,
        overlap_fraction=overlap,
        capture_target=0.8,
    )
    assert captured >= 0.8
    assert block_size <= 32


def test_a_spread_pattern_clamps_to_the_maximum() -> None:
    """When no permitted block size reaches the target, the largest is taken and the capture
    it achieved is reported, rather than the choice failing."""
    size = 600
    rows, cols, _ = _pattern(_random_matrix(size, seed=1, density=0.02))
    block_size, captured = choose_block_size(
        rows, cols, size, max_block_size=128, overlap_fraction=0.25, capture_target=0.8
    )
    assert block_size == 128
    assert captured < 0.8


def test_overlap_does_not_reduce_capture() -> None:
    """Widening the blocks can only bring more entries inside them.

    Capture is not asserted to increase strictly, because it saturates: for a band of
    half-width 3, an overlap of a quarter of a 16-wide block already covers every offset, so
    there is nothing left for more overlap to gain.
    """
    size = 512
    rows, cols, _ = _pattern(_banded(size, half_width=3))
    measured = [
        float(capture_fraction(rows, cols, size, 16, overlap)) for overlap in OVERLAPS
    ]
    assert measured == sorted(measured)
    assert measured[0] < 1.0
    assert measured[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("size", SIZES)
def test_candidate_count_bounds_the_real_multiplicity(
    size: int, block_size: int, overlap: float
) -> None:
    """`max_blocks_per_entry` sets how many destinations each entry gets, so it must never
    undercount. The stride alone would: pulling the last block back leaves the final starts
    closer together than the stride, which lets one entry sit in more blocks than
    `ceil(b / stride)` allows for."""
    resolved, _, _ = geometry(size, block_size, overlap)
    starts = np.asarray(block_starts(size, block_size, overlap))
    actual = max(
        int(((starts > lower - resolved) & (starts <= lower)).sum())
        for lower in range(size)
    )
    assert max_blocks_per_entry(size, block_size, overlap) >= actual


def test_rejects_an_out_of_range_overlap() -> None:
    """An overlap of one would leave a zero stride and infinitely many blocks."""
    with pytest.raises(ValueError, match="overlap_fraction"):
        geometry(10, 4, 1.0)
    with pytest.raises(ValueError, match="overlap_fraction"):
        geometry(10, 4, -0.1)


def test_rejects_an_out_of_range_capture_target() -> None:
    """A target of zero would always be met by a single scalar block, and one above one could
    never be met."""
    rows, cols, _ = _pattern(_banded(32, half_width=1))
    with pytest.raises(ValueError, match="capture_target"):
        choose_block_size(rows, cols, 32, 16, 0.25, 0.0)
    with pytest.raises(ValueError, match="capture_target"):
        choose_block_size(rows, cols, 32, 16, 0.25, 1.5)
