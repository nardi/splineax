"""Tests for `BlockJacobi`.

Two references, used deliberately. On an exactly block-diagonal matrix the
preconditioner *is* the inverse, so any error is a bug. On a matrix with coupling
outside the blocks it must equal the inverse of the block diagonal --- discarding that
coupling is what block Jacobi means, so the test pins the discarding rather than
treating it as a loss of accuracy.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from lineax._tags import positive_semidefinite_tag

import splineax as splx
from splineax._partition import BlockPartition
from splineax._pattern import as_coo_pattern
from splineax.preconditioners._blockjacobi import structurally_singular_blocks

from .conftest import block_diagonal_matrix, with_coupling


def operator_of(dense: np.ndarray, tags: object = ()) -> splx.BCOOLinearOperator:
    return splx.BCOOLinearOperator(BCOO.fromdense(jnp.asarray(dense)), tags)


def built(dense: np.ndarray, preconditioner: splx.BlockJacobi, tags: object = ()):
    """The `M` a preconditioner builds for a dense reference matrix."""
    operator = operator_of(dense, tags)
    return preconditioner.symbolic(operator).numeric(operator).left_operator()


def block_diagonal_of(dense: np.ndarray, sizes: tuple[int, ...]) -> np.ndarray:
    """Everything outside the blocks zeroed --- what block Jacobi actually inverts."""
    kept = np.zeros_like(dense)
    start = 0
    for size in sizes:
        kept[start : start + size, start : start + size] = dense[
            start : start + size, start : start + size
        ]
        start += size
    return kept


def test_is_the_exact_inverse_of_a_block_diagonal_matrix():
    """With nothing to discard, the preconditioner is the inverse."""
    dense = block_diagonal_matrix()
    operator = built(dense, splx.BlockJacobi(blocks=4))
    assert np.allclose(
        np.asarray(operator.as_matrix()), np.linalg.inv(dense), atol=1e-5
    )


def test_inverts_only_the_blocks_when_there_is_coupling():
    """The out-of-block entries are discarded, not approximated."""
    dense = with_coupling(block_diagonal_matrix())
    operator = built(dense, splx.BlockJacobi(blocks=4))
    expected = np.linalg.inv(block_diagonal_of(dense, (4, 4, 4)))
    assert np.allclose(np.asarray(operator.as_matrix()), expected, atol=1e-5)


def test_ragged_blocks_invert_independently():
    """A partition whose blocks differ in size still inverts each one exactly."""
    dense = block_diagonal_matrix((3, 5, 4))
    operator = built(dense, splx.BlockJacobi(blocks=(3, 5, 4)))
    assert np.allclose(
        np.asarray(operator.as_matrix()), np.linalg.inv(dense), atol=1e-5
    )


def test_a_fixed_partition_and_an_injected_provider_agree():
    """The two ways of fulfilling the `blocks` parameter are interchangeable.

    On a cleanly 4-blocked matrix the partitioner derives exactly the partition the
    fixed value states, so the resulting preconditioners must be identical --- which is
    the substitutability the injection design is for.
    """
    dense = block_diagonal_matrix()
    fixed = built(dense, splx.BlockJacobi(blocks=4))
    derived = built(
        dense, splx.BlockJacobi(blocks=splx.MaximalCaptureBlockPartitioner())
    )
    assert np.allclose(np.asarray(fixed.as_matrix()), np.asarray(derived.as_matrix()))


def test_accepts_a_block_partition_value():
    """A `BlockPartition` can be handed over directly."""
    dense = block_diagonal_matrix()
    operator = built(dense, splx.BlockJacobi(blocks=BlockPartition((4, 4, 4))))
    assert np.allclose(
        np.asarray(operator.as_matrix()), np.linalg.inv(dense), atol=1e-5
    )


def test_rejects_a_partition_of_the_wrong_size():
    """A partition has to cover the system it is used on."""
    with pytest.raises(ValueError, match="partitions 8 indices"):
        built(block_diagonal_matrix(), splx.BlockJacobi(blocks=BlockPartition((4, 4))))


def test_rejects_an_unusable_blocks_value():
    """`blocks` must be a value or a provider, and says so when it is neither."""
    with pytest.raises(TypeError, match="`BlockPartitioner` protocol"):
        built(
            block_diagonal_matrix(),
            splx.BlockJacobi(blocks=object()),  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("factorization", ["lu", "qr", "svd"])
def test_factorizations_agree_when_the_blocks_are_well_conditioned(
    factorization: Literal["lu", "qr", "svd"],
):
    """All three routes compute the same inverse for an invertible block."""
    dense = block_diagonal_matrix()
    operator = built(dense, splx.BlockJacobi(blocks=4, factorization=factorization))
    assert np.allclose(
        np.asarray(operator.as_matrix()), np.linalg.inv(dense), atol=1e-4
    )


def test_svd_stays_finite_on_a_singular_block():
    """The default survives a singular block, which is why it is the default."""
    dense = block_diagonal_matrix((2, 2))
    dense[0:2, 0:2] = 0.0  # structurally present, numerically singular
    dense[0, 0] = dense[0, 1] = dense[1, 0] = dense[1, 1] = 1.0
    operator = built(dense, splx.BlockJacobi(blocks=2, factorization="svd"))
    assert np.all(np.isfinite(np.asarray(operator.as_matrix())))


def test_qr_truncates_a_rank_deficient_block():
    """`'qr'` is rank-revealing: it drops the deficient direction rather than blowing up.

    It is not the Moore-Penrose pseudo-inverse `'svd'` computes --- it is a one-sided
    inverse on the subspace it keeps, which is all a preconditioner needs.
    """
    dense = np.zeros((4, 4))
    dense[0:2, 0:2] = 1.0  # rank 1
    dense[2:4, 2:4] = [[2.0, 0.0], [0.0, 4.0]]
    operator = built(dense, splx.BlockJacobi(blocks=2, factorization="qr"))
    matrix = np.asarray(operator.as_matrix())
    assert np.all(np.isfinite(matrix))
    # The invertible block is unaffected by its neighbour's deficiency.
    assert np.allclose(matrix[2:4, 2:4], np.diag([0.5, 0.25]), atol=1e-6)


def test_lu_does_not_stay_finite_on_a_singular_block():
    """Pinning the documented hazard: `'lu'` fails silently, with no error raised."""
    dense = block_diagonal_matrix((2, 2))
    dense[0:2, 0:2] = 1.0
    operator = built(dense, splx.BlockJacobi(blocks=2, factorization="lu"))
    assert not np.all(np.isfinite(np.asarray(operator.as_matrix())))


def test_structural_singularity_is_derived_from_the_pattern():
    """A block with an empty row is singular whatever its values are."""
    dense = np.zeros((4, 4))
    dense[0, 0] = dense[0, 1] = dense[1, 1] = (
        1.0  # row 1 of block 0 has no entry at 1,0
    )
    dense[1, 0] = 0.0
    dense[2, 2] = dense[3, 3] = 1.0
    pattern = as_coo_pattern(operator_of(dense), "test").concrete("test")
    singular = structurally_singular_blocks(pattern, BlockPartition((2, 2)))
    assert not singular[1]


def test_structurally_empty_row_is_detected():
    """An index with no in-block entry at all makes its block exactly singular."""
    dense = np.zeros((4, 4))
    dense[0, 0] = 1.0  # index 1 is entirely empty
    dense[2, 2] = dense[3, 3] = 1.0
    pattern = as_coo_pattern(operator_of(dense), "test").concrete("test")
    singular = structurally_singular_blocks(pattern, BlockPartition((2, 2)))
    assert singular[0] and not singular[1]


def test_lu_refuses_a_structurally_singular_partition():
    """A provably impossible request is refused up front, naming the blocks."""
    dense = np.zeros((4, 4))
    dense[0, 0] = 1.0
    dense[2, 2] = dense[3, 3] = 1.0
    with pytest.raises(ValueError, match=r"cannot invert blocks \[0\]"):
        built(dense, splx.BlockJacobi(blocks=2, factorization="lu"))


def test_assume_nonsingular_is_checked_against_the_pattern():
    """An assertion the pattern disproves is an error, not a hint."""
    dense = np.zeros((4, 4))
    dense[0, 0] = 1.0
    dense[2, 2] = dense[3, 3] = 1.0
    with pytest.raises(ValueError, match="assume_nonsingular=True"):
        built(dense, splx.BlockJacobi(blocks=2, assume_nonsingular=True))


@pytest.mark.parametrize("assume_nonsingular,expected", [(False, "svd"), (True, "lu")])
def test_auto_resolves_by_the_nonsingularity_assertion(
    assume_nonsingular: bool, expected: str
):
    """`'auto'` takes the cheap route only when told the blocks are invertible."""
    preconditioner = splx.BlockJacobi(
        blocks=4, factorization="auto", assume_nonsingular=assume_nonsingular
    )
    symbolic = preconditioner.symbolic(operator_of(block_diagonal_matrix()))
    assert symbolic.factorization == expected


def test_rejects_an_unknown_factorization():
    """The factorization choice is closed."""
    with pytest.raises(ValueError, match="must be one of"):
        splx.BlockJacobi(
            blocks=4,
            factorization="cholesky",  # ty: ignore[invalid-argument-type]
        )


def test_positive_definiteness_is_inferred():
    """A definite system yields a definite preconditioner, as `lineax.CG` requires."""
    dense = block_diagonal_matrix()
    dense = dense @ dense.T + np.eye(dense.shape[0])
    operator = built(dense, splx.BlockJacobi(blocks=4), lx.positive_semidefinite_tag)
    assert positive_semidefinite_tag in operator.tags
    assert lx.is_positive_semidefinite(operator)


def test_negative_definite_systems_give_a_positive_definite_preconditioner():
    """`lineax.CG` negates the operator but not the preconditioner, so we negate here."""
    dense = block_diagonal_matrix()
    dense = -(dense @ dense.T + np.eye(dense.shape[0]))
    operator = built(dense, splx.BlockJacobi(blocks=4), lx.negative_semidefinite_tag)
    assert lx.is_positive_semidefinite(operator)
    eigenvalues = np.linalg.eigvalsh(np.asarray(operator.as_matrix()))
    assert np.all(eigenvalues > 0)


@pytest.mark.parametrize("dtype", ["float32", "float64", "complex64"])
def test_preserves_the_operator_dtype(dtype: str):
    """Lineax compares preconditioner structures including dtype, so it must match."""
    with jax.enable_x64(True):
        dense = block_diagonal_matrix().astype(dtype)
        operator = operator_of(dense)
        preconditioner = splx.BlockJacobi(blocks=4)
        built_operator = (
            preconditioner.symbolic(operator).numeric(operator).left_operator()
        )
        assert built_operator.in_structure() == operator.in_structure()


def test_both_sides_give_the_same_operator():
    """Block Jacobi is one `M`, applicable either way round."""
    dense = block_diagonal_matrix()
    operator = operator_of(dense)
    numeric = splx.BlockJacobi(blocks=4).symbolic(operator).numeric(operator)
    assert numeric.left_operator() is numeric.right_operator()
    assert splx.Side.LEFT in splx.BlockJacobi(blocks=4).sides
    assert splx.Side.RIGHT in splx.BlockJacobi(blocks=4).sides


def test_symbolic_is_reusable_for_new_values():
    """The point of the two-phase split: same pattern, different numbers."""
    dense = block_diagonal_matrix()
    symbolic = splx.BlockJacobi(blocks=4).symbolic(operator_of(dense))
    for scale in (1.0, 3.0):
        operator = operator_of(scale * dense)
        rebuilt = symbolic.numeric(operator).left_operator()
        assert np.allclose(
            np.asarray(rebuilt.as_matrix()),
            np.linalg.inv(scale * dense),
            atol=1e-5,
        )


def test_symbolic_rejects_a_different_pattern():
    """Reuse is only valid for the pattern the symbolic phase analysed."""
    symbolic = splx.BlockJacobi(blocks=4).symbolic(operator_of(block_diagonal_matrix()))
    with pytest.raises(ValueError, match="stored entries"):
        symbolic.numeric(operator_of(with_coupling(block_diagonal_matrix())))


def test_rejects_a_non_square_pattern():
    """A preconditioner for a rectangular system is not defined."""
    with pytest.raises(ValueError, match="square pattern"):
        splx.BlockJacobi(blocks=2).symbolic(operator_of(np.ones((2, 3))))


def test_uncoalesced_duplicates_are_summed():
    """A repeated coordinate contributes the sum of its values, as `BCOO` defines."""
    indices = jnp.array([[0, 0], [0, 0], [1, 1]])
    data = jnp.array([1.0, 3.0, 2.0])
    operator = splx.BCOOLinearOperator(BCOO((data, indices), shape=(2, 2)))
    built_operator = (
        splx.BlockJacobi(blocks=2).symbolic(operator).numeric(operator).left_operator()
    )
    # The block is diag(4, 2), so its inverse is diag(0.25, 0.5).
    assert np.allclose(
        np.asarray(built_operator.as_matrix()), np.diag([0.25, 0.5]), atol=1e-6
    )
