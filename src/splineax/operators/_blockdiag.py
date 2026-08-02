"""A block-diagonal linear operator, stored as a stack of dense blocks."""

from functools import lru_cache

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Inexact
from lineax import AbstractLinearOperator, is_symmetric
from lineax._tags import transpose_tags

from splineax._partition import BlockPartition
from splineax.operators._operations import register_operator_tags


@lru_cache(maxsize=None)
def _padded_index(sizes: tuple[int, ...]) -> np.ndarray:
    """Where each global index sits in the padded `(num_blocks, max_block_size)` layout.

    Every block is stored at the same width so that the whole stack is one array and
    one batched matmul; the shorter blocks leave a gap, and this is the map that skips
    it. Only needed for a ragged partition -- when every block is the same size the
    layout is a plain reshape.
    """
    boundaries = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=boundaries[1:])
    width = max(sizes)
    block_of = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    start_of = np.repeat(boundaries[:-1], sizes)
    index = block_of * width + (np.arange(boundaries[-1], dtype=np.int64) - start_of)
    index.flags.writeable = False
    return index


@lru_cache(maxsize=None)
def _dense_scatter(sizes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global `(row, column)` coordinates of every stored block entry, and its offset
    into the flattened block stack. Used only by `as_matrix`."""
    boundaries = np.zeros(len(sizes) + 1, dtype=np.int64)
    np.cumsum(sizes, out=boundaries[1:])
    width = max(sizes)
    rows, cols, flat = [], [], []
    for block, (start, size) in enumerate(zip(boundaries[:-1], sizes)):
        local = np.arange(size, dtype=np.int64)
        row, col = np.meshgrid(local, local, indexing="ij")
        rows.append((row + start).ravel())
        cols.append((col + start).ravel())
        flat.append((block * width * width + row * width + col).ravel())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(flat)


class BlockDiagonalLinearOperator(AbstractLinearOperator):
    """A square operator that is zero outside contiguous diagonal blocks.

    The blocks are held as one `(num_blocks, width, width)` array, where `width` is the
    largest block: keeping them in a single stack is what makes a matrix-vector product
    one batched `einsum` regardless of how many blocks there are, rather than a Python
    loop over them. A partition whose blocks differ in size leaves unused space in the
    shorter ones, which the index maps above skip over.

    This is what [`splineax.BlockJacobi`][] hands to lineax as
    `options["preconditioner"]`, holding the *inverted* blocks. It is a normal
    `lineax.AbstractLinearOperator` and can be used on its own.
    """

    blocks: Inexact[Array, "num_blocks width width"]
    partition: BlockPartition = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        blocks: Inexact[Array, "num_blocks width width"],
        partition: BlockPartition,
        tags: object | frozenset[object] = (),
    ):
        """**Arguments:**

        - `blocks`: the diagonal blocks, stacked into one array of shape
            `(num_blocks, width, width)` with `width` the largest block size. A block
            shorter than `width` uses the leading rows and columns of its slice; the
            rest is ignored, though `BlockJacobi` keeps an identity there so that
            inverting the stack leaves it well posed.
        - `partition`: the block sizes, which say how much of each slice is real.
        - `tags`: any tags indicating properties of this matrix, as elsewhere in lineax.
            Unchecked.
        """
        expected = (
            partition.num_blocks,
            partition.max_block_size,
            partition.max_block_size,
        )
        if jnp.shape(blocks) != expected:
            raise ValueError(
                f"`blocks` should have shape {expected} for this partition; got "
                f"{jnp.shape(blocks)}."
            )
        self.blocks = blocks
        self.partition = partition
        self.tags = tags if isinstance(tags, frozenset) else frozenset([tags])

    def mv(self, vector: Inexact[Array, " size"]) -> Inexact[Array, " size"]:
        partition = self.partition
        if partition.is_uniform:
            stacked = vector.reshape(partition.num_blocks, partition.max_block_size)
            product = jnp.einsum("nij,nj->ni", self.blocks, stacked)
            return product.reshape(partition.size)
        index = _padded_index(partition.sizes)
        width = partition.max_block_size
        padded = jnp.zeros(partition.num_blocks * width, dtype=vector.dtype)
        padded = padded.at[index].set(vector)
        stacked = padded.reshape(partition.num_blocks, width)
        product = jnp.einsum("nij,nj->ni", self.blocks, stacked)
        return product.reshape(-1)[index]

    def as_matrix(self) -> Inexact[Array, "size size"]:
        rows, cols, flat = _dense_scatter(self.partition.sizes)
        size = self.partition.size
        dense = jnp.zeros((size, size), dtype=self.blocks.dtype)
        return dense.at[rows, cols].set(self.blocks.reshape(-1)[flat])

    def transpose(self) -> "BlockDiagonalLinearOperator":
        if is_symmetric(self):
            return self
        return BlockDiagonalLinearOperator(
            jnp.swapaxes(self.blocks, -1, -2),
            self.partition,
            transpose_tags(self.tags),
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(
            shape=(self.partition.size,), dtype=self.blocks.dtype
        )

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.in_structure()

    def _conj(self) -> "BlockDiagonalLinearOperator":
        return BlockDiagonalLinearOperator(
            self.blocks.conj(), self.partition, self.tags
        )


# Only the tag layer: a block-diagonal matrix has no `matrix` to extract bands from,
# and is *not* tridiagonal unless every block happens to be at most 2 wide, so
# `diagonal`/`tridiagonal` are deliberately left unregistered.
register_operator_tags(BlockDiagonalLinearOperator)
