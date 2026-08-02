"""A partition of an index range into contiguous blocks.

Shared between `splineax.operators.BlockDiagonalLinearOperator`, which uses it as a
storage layout, and the preconditioners, which use it as a parameter. It lives here
rather than with either of them so that neither has to depend on the other.
"""

from functools import lru_cache
from typing import Sequence

import equinox as eqx
import numpy as np


class BlockPartition(eqx.Module):
    """A partition of `0..size` into contiguous blocks along the diagonal.

    Blocks are contiguous and in order, so the partition is fully described by its
    sizes. Held as a `tuple` rather than an array because it is static data: it decides
    array *shapes* in the numeric phase, and is hashed as part of a jit cache key.
    """

    sizes: tuple[int, ...] = eqx.field(static=True)
    """The size of each block, in order. Sums to the size of the system."""

    def __check_init__(self):
        if len(self.sizes) == 0:
            raise ValueError(
                "A `BlockPartition` must have at least one block; got none. (A "
                "zero-sized system cannot be preconditioned.)"
            )
        if any(size <= 0 for size in self.sizes):
            raise ValueError(
                f"Every block of a `BlockPartition` must be non-empty; got {self.sizes}."
            )

    @property
    def size(self) -> int:
        """The size of the system this partitions."""
        return sum(self.sizes)

    @property
    def num_blocks(self) -> int:
        return len(self.sizes)

    @property
    def max_block_size(self) -> int:
        """The largest block, which is the padded width every block is stored at."""
        return max(self.sizes)

    @property
    def is_uniform(self) -> bool:
        """Whether every block is the same size, so no padding is needed at all."""
        return len(set(self.sizes)) == 1

    @staticmethod
    def uniform(size: int, block_size: int) -> "BlockPartition":
        """Split `size` into blocks of `block_size`, with a shorter final block if it
        does not divide evenly."""
        if block_size <= 0:
            raise ValueError(f"`block_size` must be positive; got {block_size}.")
        if size <= 0:
            raise ValueError(f"`size` must be positive; got {size}.")
        whole, remainder = divmod(size, block_size)
        sizes = (block_size,) * whole + ((remainder,) if remainder else ())
        return BlockPartition(sizes)

    @staticmethod
    def from_sizes(sizes: Sequence[int], size: int | None = None) -> "BlockPartition":
        """Build from explicit block sizes, checking they sum to `size` if given."""
        partition = BlockPartition(tuple(int(s) for s in sizes))
        if size is not None and partition.size != size:
            raise ValueError(
                f"Block sizes {partition.sizes} sum to {partition.size}, but the "
                f"system has size {size}."
            )
        return partition

    @property
    def boundaries(self) -> np.ndarray:
        """The `num_blocks + 1` block start offsets, ending at `size`."""
        return _boundaries(self.sizes)


@lru_cache(maxsize=None)
def _boundaries(sizes: tuple[int, ...]) -> np.ndarray:
    boundaries = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=boundaries[1:])
    boundaries.flags.writeable = False
    return boundaries


@lru_cache(maxsize=None)
def _block_of_index(sizes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """For each index: which block it is in, and where that block starts."""
    boundaries = _boundaries(sizes)
    block_of = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    start_of = np.repeat(boundaries[:-1], sizes)
    return block_of, start_of
