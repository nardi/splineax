"""Tests for `BlockDiagonalLinearOperator`.

The dense block-diagonal matrix is the source of truth throughout: whatever the padded
stacked storage does internally, it has to agree with `scipy.linalg.block_diag` of the
same blocks.

The `linearise`/`materialise` tests here are not incidental. Lineax calls `linearise` on
whatever is passed as `options["preconditioner"]`, and its default rule densifies to an
`n x n` array --- so if those registrations were ever dropped, every preconditioned
solve would silently allocate the dense matrix this operator exists to avoid, with no
failure to show for it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy.linalg

from splineax import BlockDiagonalLinearOperator, BlockPartition

UNIFORM = BlockPartition((3, 3))
RAGGED = BlockPartition((3, 1, 2))


def stacked(partition: BlockPartition, seed: int = 0) -> jax.Array:
    """Random blocks in the padded `(num_blocks, width, width)` layout."""
    rng = np.random.default_rng(seed)
    width = partition.max_block_size
    blocks = np.zeros((partition.num_blocks, width, width))
    for index, size in enumerate(partition.sizes):
        blocks[index, :size, :size] = rng.uniform(-1.0, 1.0, (size, size))
    return jnp.asarray(blocks)


def reference(blocks: jax.Array, partition: BlockPartition) -> np.ndarray:
    """The same operator as a dense matrix, built independently."""
    return scipy.linalg.block_diag(
        *[np.asarray(blocks[i, :size, :size]) for i, size in enumerate(partition.sizes)]
    )


@pytest.fixture(params=[UNIFORM, RAGGED], ids=["uniform", "ragged"])
def partition(request: pytest.FixtureRequest) -> BlockPartition:
    """Both storage paths: a plain reshape, and one that skips the padding."""
    return request.param


def test_as_matrix_matches_block_diag(partition: BlockPartition):
    """`as_matrix` agrees with an independently built dense block diagonal."""
    blocks = stacked(partition)
    operator = BlockDiagonalLinearOperator(blocks, partition)
    assert np.allclose(np.asarray(operator.as_matrix()), reference(blocks, partition))


def test_mv_matches_the_dense_product(partition: BlockPartition):
    """The batched einsum agrees with a dense matrix-vector product."""
    blocks = stacked(partition)
    operator = BlockDiagonalLinearOperator(blocks, partition)
    vector = jnp.arange(1.0, partition.size + 1)
    assert np.allclose(
        np.asarray(operator.mv(vector)),
        reference(blocks, partition) @ np.asarray(vector),
    )


def test_transpose_transposes_each_block(partition: BlockPartition):
    """Transposing the operator transposes the dense matrix."""
    blocks = stacked(partition)
    operator = BlockDiagonalLinearOperator(blocks, partition)
    assert np.allclose(
        np.asarray(operator.transpose().as_matrix()),
        reference(blocks, partition).T,
    )


def test_conj_conjugates(partition: BlockPartition):
    """Conjugation reaches the stored blocks."""
    blocks = stacked(partition) * (1.0 + 1.0j)
    operator = BlockDiagonalLinearOperator(blocks, partition)
    assert np.allclose(
        np.asarray(lx.conj(operator).as_matrix()),
        np.conj(np.asarray(operator.as_matrix())),
    )


def test_linearise_and_materialise_keep_it_sparse(partition: BlockPartition):
    """Both must be the identity, or every preconditioned solve densifies silently."""
    operator = BlockDiagonalLinearOperator(stacked(partition), partition)
    assert lx.linearise(operator) is operator
    assert lx.materialise(operator) is operator


def test_structures_are_square_and_match_the_block_dtype(partition: BlockPartition):
    """In and out structures agree, which is what makes it usable as a preconditioner."""
    operator = BlockDiagonalLinearOperator(stacked(partition), partition)
    assert operator.in_structure() == operator.out_structure()
    assert operator.in_structure().shape == (partition.size,)


def test_tags_are_reported(partition: BlockPartition):
    """The tag layer is shared with the other operators in this package."""
    operator = BlockDiagonalLinearOperator(
        stacked(partition), partition, lx.positive_semidefinite_tag
    )
    assert lx.is_positive_semidefinite(operator)
    assert lx.is_symmetric(operator)


def test_not_reported_as_tridiagonal():
    """A block-diagonal matrix is not tridiagonal, and must not claim to be."""
    operator = BlockDiagonalLinearOperator(stacked(UNIFORM), UNIFORM)
    assert not lx.is_tridiagonal(operator)


def test_mismatched_block_shape_is_rejected():
    """The stack has to match the partition it claims to describe."""
    with pytest.raises(ValueError, match="should have shape"):
        BlockDiagonalLinearOperator(jnp.zeros((2, 2, 2)), BlockPartition((3, 3)))


def test_works_under_jit_and_vmap(partition: BlockPartition):
    """The partition is static, so it survives both transforms as a constant."""
    blocks = stacked(partition)
    vector = jnp.arange(1.0, partition.size + 1)

    @jax.jit
    def apply(blocks: jax.Array) -> jax.Array:
        return BlockDiagonalLinearOperator(blocks, partition).mv(vector)

    assert np.allclose(
        np.asarray(apply(blocks)), reference(blocks, partition) @ np.asarray(vector)
    )
    batched = jax.vmap(apply)(jnp.stack([blocks, 2.0 * blocks]))
    assert batched.shape == (2, partition.size)
