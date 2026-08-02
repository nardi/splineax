"""Providers for the block-partition parameter of a block preconditioner.

`BlockJacobi` is parametrized by a partition of the index range into contiguous
diagonal blocks. That parameter can be fulfilled by a fixed value -- a `BlockPartition`,
a uniform block size, or a sequence of sizes -- or by an object that derives one from
the sparsity pattern and is injected as a dependency. `BlockPartitioner` below is the
protocol such a provider satisfies, and `MaximalCaptureBlockPartitioner` is the one this
package ships.

The protocol is deliberately as small as it can be: a pattern in, a partition out. It
does not permute (that is what the solver's `transforms` are for), and it does not
report on conditioning -- `BlockJacobi` derives what it needs about singularity from
the pattern and the partition itself, rather than the provider having to anticipate
what will consume it.
"""

import warnings
from typing import Protocol, runtime_checkable

import equinox as eqx
import numpy as np

from splineax._partition import BlockPartition, _block_of_index
from splineax._pattern import ConcreteCooPattern, CooPattern


@runtime_checkable
class BlockPartitioner(Protocol):
    """Provides the `blocks` parameter of a block preconditioner.

    Anything with this one method can be injected wherever a partition is wanted; there
    is no base class to subclass.
    """

    def partition(self, pattern: CooPattern) -> BlockPartition:
        """Derive a block partition from a sparsity pattern."""
        ...


def captured_fraction(pattern: ConcreteCooPattern, partition: BlockPartition) -> float:
    """The fraction of stored entries that fall inside the partition's blocks.

    Everything else is discarded when the preconditioner is built -- which is what
    "block Jacobi" *means*, but is worth being able to see. `1.0` means the matrix is
    exactly block diagonal for this partition.
    """
    if pattern.nse == 0:
        return 1.0
    block_of, _ = _block_of_index(partition.sizes)
    inside = block_of[pattern.rows] == block_of[pattern.cols]
    return float(np.count_nonzero(inside) / pattern.nse)


def _connected_components(pattern: ConcreteCooPattern) -> np.ndarray:
    """The coarsest exactly-block-diagonal partition, as block boundaries.

    Two indices must share a block whenever an entry couples them, so the finest
    possible exact partition is the connected components of the interval graph the
    nonzeros induce. Because blocks are contiguous, a component is closed at position
    `i` exactly when nothing at or before `i` reaches past it -- one prefix-maximum
    scan, `O(nse + size)`, and no graph traversal.

    The min/max closure is what makes this correct for an unsymmetric pattern too: a
    lone entry at `(5, 0)` couples 0 and 5 just as much as one at `(0, 5)` would.
    """
    size = pattern.shape[0]
    reach = np.arange(size, dtype=np.int64)
    if pattern.nse:
        lo = np.minimum(pattern.rows, pattern.cols)
        hi = np.maximum(pattern.rows, pattern.cols)
        np.maximum.at(reach, lo, hi)
    running = np.maximum.accumulate(reach)
    closes = np.flatnonzero(running == np.arange(size, dtype=np.int64))
    return np.concatenate([[0], closes + 1])


def _crossing_counts(pattern: ConcreteCooPattern, start: int, stop: int) -> np.ndarray:
    """How many entries straddle each cut position inside `[start, stop)`.

    `result[k]` counts the entries that would be discarded by cutting between local
    positions `k - 1` and `k`. Built with a difference array, so `O(nse + length)`
    rather than a pass per candidate cut.
    """
    length = stop - start
    diff = np.zeros(length + 2, dtype=np.int64)
    lo = np.minimum(pattern.rows, pattern.cols)
    hi = np.maximum(pattern.rows, pattern.cols)
    inside = (lo >= start) & (hi < stop) & (lo != hi)
    lo, hi = lo[inside] - start, hi[inside] - start
    # An entry spanning local `lo..hi` crosses every cut at `lo < k <= hi`.
    np.add.at(diff, lo + 1, 1)
    np.add.at(diff, hi + 1, -1)
    return np.cumsum(diff)[:length]


def _split_maximal_capture(
    crossing: np.ndarray, length: int, max_block_size: int
) -> list[int]:
    """Cut positions within a component that discard the fewest entries.

    A component too long to invert as one block has to be cut somewhere, and cutting
    evenly is arbitrary: it can slice straight through the densest part of the matrix.
    Instead this minimises the total weight of entries crossing the chosen cuts, subject
    to no block exceeding `max_block_size`.

    That is a shortest-path over cut positions, `best[j] = crossing[j] + min(best[i])`
    for `i` in the window `[j - max_block_size, j)`. The window minimum comes from a
    monotonic deque, so the whole thing is `O(length)` rather than
    `O(length * max_block_size)`.
    """
    best = np.zeros(length + 1, dtype=np.int64)
    previous = np.zeros(length + 1, dtype=np.int64)
    window: list[int] = [0]  # indices of candidate predecessors, by increasing `best`
    head = 0
    for j in range(1, length + 1):
        while window[head] < j - max_block_size:
            head += 1
        source = window[head]
        cost = 0 if j == length else int(crossing[j])
        best[j] = best[source] + cost
        previous[j] = source
        while len(window) > head and best[window[-1]] >= best[j]:
            window.pop()
        window.append(j)
    cuts = []
    position = length
    while position > 0:
        cuts.append(position)
        position = int(previous[position])
    return sorted(cuts)


class MaximalCaptureBlockPartitioner(eqx.Module):
    """Chooses diagonal blocks capturing as many of the matrix's entries as possible.

    Two steps. First it finds the coarsest partition that is *exactly* block diagonal --
    the connected components of the pattern -- which by construction captures every
    entry. Then, because a single far-off-diagonal entry is enough to fuse the whole
    matrix into one component (and inverting an `n x n` "block" is exactly the cost
    block Jacobi exists to avoid), any component longer than `max_block_size` is cut.

    Where it is cut is the interesting part, and what the name refers to: the cuts are
    placed to discard the fewest entries, not to divide the component evenly. Splitting
    a 257-long component into 256 + 1 because that is where the cap fell would throw
    away whatever happens to sit at that boundary; minimising the crossing weight
    instead keeps the strongest coupling inside the blocks.

    Satisfies the `BlockPartitioner` protocol, so it can be injected as
    [`splineax.BlockJacobi`][]'s `blocks` parameter:

    ```python
    import splineax as splx

    preconditioner = splx.BlockJacobi(blocks=splx.MaximalCaptureBlockPartitioner())
    assert isinstance(preconditioner.blocks, splx.MaximalCaptureBlockPartitioner)
    ```
    """

    max_block_size: int | None = eqx.field(default=256, static=True)

    def __check_init__(self):
        if self.max_block_size is not None and self.max_block_size <= 0:
            raise ValueError(
                f"`max_block_size` must be positive or `None`; got "
                f"{self.max_block_size}."
            )

    def partition(self, pattern: CooPattern) -> BlockPartition:
        """Derive the partition from the pattern."""
        concrete = pattern.concrete("`MaximalCaptureBlockPartitioner`")
        size = concrete.shape[0]
        if concrete.shape[0] != concrete.shape[1]:
            raise ValueError(
                "`MaximalCaptureBlockPartitioner` requires a square pattern; got "
                f"shape {concrete.shape}."
            )
        boundaries = _connected_components(concrete)
        sizes: list[int] = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            length = int(stop - start)
            if self.max_block_size is None or length <= self.max_block_size:
                sizes.append(length)
                continue
            crossing = _crossing_counts(concrete, int(start), int(stop))
            cuts = _split_maximal_capture(crossing, length, self.max_block_size)
            previous = 0
            for cut in cuts:
                sizes.append(cut - previous)
                previous = cut
        partition = BlockPartition.from_sizes(sizes, size)
        if self.max_block_size is None and partition.num_blocks == 1 and size > 1:
            warnings.warn(
                "`MaximalCaptureBlockPartitioner(max_block_size=None)` found a single "
                f"block spanning the whole {size}x{size} system, so the preconditioner "
                "will invert it densely. Set `max_block_size` to cap the block size.",
                stacklevel=2,
            )
        return partition


MaximalCaptureBlockPartitioner.__init__.__doc__ = """**Arguments:**

- `max_block_size`: the largest block to produce. Components longer than this are cut
    at whichever positions discard the fewest entries. Defaults to `256`, which inverts
    cheaply while being wide enough to capture most locally-coupled structure. `None`
    removes the cap, which makes the partition exact but risks a single block spanning
    the whole system, and warns if that happens.
"""
