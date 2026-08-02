"""Tests for block partitions and the partitioner that derives one from a pattern.

Pure NumPy and host-side throughout: the whole point of this layer is that it runs once,
in Python, before any values exist. [test_blockjacobi.py](test_blockjacobi.py) covers
what consumes the partition.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import splineax as splx
from splineax._partition import BlockPartition
from splineax._pattern import ConcreteCooPattern, as_coo_pattern
from splineax.preconditioners._partition import (
    _crossing_counts,
    _split_maximal_capture,
    captured_fraction,
)


def pattern_of(dense: np.ndarray) -> ConcreteCooPattern:
    """The concrete sparsity pattern of a dense reference matrix."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(jnp.asarray(dense)))
    return as_coo_pattern(operator, "test").concrete("test")


def partition_of(dense: np.ndarray, **kwargs) -> BlockPartition:
    """Run the shipped partitioner over a dense reference matrix."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(jnp.asarray(dense)))
    partitioner = splx.MaximalCaptureBlockPartitioner(**kwargs)
    return partitioner.partition(as_coo_pattern(operator, "test"))


def test_uniform_partition_splits_evenly():
    """A block size dividing the system gives equal blocks."""
    assert BlockPartition.uniform(12, 4).sizes == (4, 4, 4)


def test_uniform_partition_keeps_a_short_final_block():
    """A block size that does not divide leaves a shorter last block."""
    assert BlockPartition.uniform(10, 4).sizes == (4, 4, 2)


def test_from_sizes_rejects_a_wrong_total():
    """Block sizes must account for every index."""
    with pytest.raises(ValueError, match="sum to 7, but the system has size 8"):
        BlockPartition.from_sizes((3, 4), size=8)


def test_empty_partition_is_rejected():
    """A zero-sized system has nothing to precondition."""
    with pytest.raises(ValueError, match="at least one block"):
        BlockPartition(())


def test_detects_exact_uniform_blocks():
    """An exactly block-diagonal matrix is recovered block for block."""
    dense = np.zeros((12, 12))
    for start in (0, 4, 8):
        dense[start : start + 4, start : start + 4] = 1.0
    assert partition_of(dense).sizes == (4, 4, 4)


def test_detects_exact_ragged_blocks():
    """Detection does not assume the blocks are the same size."""
    dense = np.zeros((12, 12))
    start = 0
    for size in (3, 5, 4):
        dense[start : start + size, start : start + size] = 1.0
        start += size
    assert partition_of(dense).sizes == (3, 5, 4)


def test_empty_pattern_gives_singletons():
    """With no entries, nothing is coupled, so every index is its own block."""
    assert partition_of(np.zeros((5, 5))).sizes == (1,) * 5


def test_unsymmetric_entry_still_couples_its_indices():
    """A lone entry below the diagonal couples just as much as one above it.

    The min/max closure is what makes this hold; a scan over rows alone would miss it.
    """
    dense = np.eye(6)
    dense[5, 0] = 1.0
    with pytest.warns(UserWarning, match="single block spanning the whole"):
        assert partition_of(dense, max_block_size=None).sizes == (6,)


def test_one_far_entry_fuses_the_whole_matrix():
    """A single corner entry is enough to make the exact partition useless.

    This is exactly why `max_block_size` exists, and why its default is not `None`.
    """
    dense = np.eye(64)
    dense[0, 63] = 1.0
    with pytest.warns(UserWarning, match="single block spanning the whole"):
        assert partition_of(dense, max_block_size=None).sizes == (64,)


def test_the_cap_splits_an_oversized_component():
    """Past the cap, the component is cut rather than inverted whole."""
    dense = np.eye(64)
    dense[0, 63] = 1.0
    partition = partition_of(dense, max_block_size=16)
    assert partition.max_block_size <= 16
    assert partition.size == 64


def test_split_minimises_discarded_entries():
    """The cut lands where the matrix is weakest, not where the cap happens to fall.

    A dense clump at 0..5 and another at 6..11, joined by one entry: with a cap of 8 the
    component must be cut, and the only cut discarding a single entry is between the
    clumps. Splitting at the cap instead would slice through a clump.
    """
    dense = np.zeros((12, 12))
    dense[0:6, 0:6] = 1.0
    dense[6:12, 6:12] = 1.0
    dense[0, 11] = 1.0
    assert partition_of(dense, max_block_size=8).sizes == (6, 6)


@pytest.mark.parametrize("length,max_block_size", [(6, 3), (7, 4), (9, 4), (10, 5)])
def test_split_matches_brute_force(length: int, max_block_size: int):
    """The linear-time split agrees with enumerating every legal partition."""
    rng = np.random.default_rng(length * 100 + max_block_size)
    crossing = rng.integers(0, 20, size=length)
    crossing[0] = 0

    def cost(cuts: tuple[int, ...]) -> int:
        return sum(int(crossing[c]) for c in cuts if c != length)

    best = min(
        (
            cuts
            for count in range(length)
            for cuts in itertools.combinations(range(1, length), count)
            if all(b - a <= max_block_size for a, b in zip((0, *cuts), (*cuts, length)))
        ),
        key=cost,
    )
    chosen = _split_maximal_capture(crossing, length, max_block_size)
    assert cost(tuple(chosen)) == cost(best)
    assert chosen[-1] == length
    assert all(b - a <= max_block_size for a, b in zip([0, *chosen[:-1]], chosen))


def test_crossing_counts_count_straddling_entries():
    """`crossing[k]` counts exactly the entries a cut at `k` would discard."""
    dense = np.zeros((5, 5))
    dense[0, 3] = dense[1, 4] = dense[3, 0] = 1.0
    crossing = _crossing_counts(pattern_of(dense), 0, 5)
    # (0,3) and (3,0) both straddle cuts 1..3; (1,4) straddles cuts 2..4.
    assert crossing[1] == 2
    assert crossing[2] == 3
    assert crossing[4] == 1


def test_captured_fraction_is_one_for_an_exact_partition():
    """An exactly block-diagonal matrix loses nothing."""
    dense = np.zeros((8, 8))
    dense[0:4, 0:4] = dense[4:8, 4:8] = 1.0
    assert captured_fraction(pattern_of(dense), BlockPartition((4, 4))) == 1.0


def test_captured_fraction_reports_the_loss():
    """Out-of-block entries show up as a shortfall."""
    dense = np.zeros((4, 4))
    dense[0:2, 0:2] = dense[2:4, 2:4] = 1.0  # 8 entries captured
    dense[0, 3] = dense[3, 0] = 1.0  # 2 entries discarded
    assert captured_fraction(
        pattern_of(dense), BlockPartition((2, 2))
    ) == pytest.approx(0.8)


def test_coverage_accepts_an_operator():
    """The public helper takes the same sparsity types as everything else."""
    dense = np.zeros((4, 4))
    dense[0:2, 0:2] = dense[2:4, 2:4] = 1.0
    operator = splx.BCOOLinearOperator(BCOO.fromdense(jnp.asarray(dense)))
    assert splx.coverage(operator, BlockPartition((2, 2))) == 1.0


def test_partitioner_satisfies_its_protocol():
    """`MaximalCaptureBlockPartitioner` is usable through the protocol alone."""
    assert isinstance(splx.MaximalCaptureBlockPartitioner(), splx.BlockPartitioner)


def test_rejects_a_nonpositive_cap():
    """A cap of zero would admit no blocks at all."""
    with pytest.raises(ValueError, match="must be positive"):
        splx.MaximalCaptureBlockPartitioner(max_block_size=0)
