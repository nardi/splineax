"""Tests for the sparsity-pattern tags, their colorings, and how they transpose."""

import asdex
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO, BCSR

from splineax import (
    BCOOLinearOperator,
    BCSRLinearOperator,
    JacobianDirection,
    sparsity_coloring_tag,
    sparsity_pattern_tag,
)
from splineax.operators._tags import _ContentPatternTag, find_pattern_tag

# A non-symmetric square matrix, so the transposed pattern differs from the original.
SQUARE_MATRIX = jnp.array(
    [
        [1.0, 2.0, 0.0, 7.0],
        [3.0, 4.0, 5.0, 0.0],
        [0.0, 6.0, 8.0, 9.0],
        [0.0, 0.0, 1.0, 2.0],
    ]
)
# A wide matrix, so the transposed tag also has a different shape.
WIDE_MATRIX = jnp.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])


def test_content_tag_transposes_twice_to_an_equal_tag() -> None:
    """Transposing a content tag swaps its pattern, and a second transpose undoes it."""
    tag = sparsity_pattern_tag(BCOO.fromdense(SQUARE_MATRIX))
    transposed = tag.transpose()
    assert transposed != tag
    assert transposed == sparsity_pattern_tag(BCOO.fromdense(SQUARE_MATRIX).T)
    assert transposed.transpose() == tag


def test_identity_tag_transposes_twice_to_an_equal_tag() -> None:
    """An identity tag has no indices, so its transpose is a distinct tag that compares
    equal to every other transpose of the same tag."""
    tag = sparsity_pattern_tag()
    transposed = tag.transpose()
    assert transposed != tag
    assert transposed == tag.transpose()
    assert hash(transposed) == hash(tag.transpose())
    assert transposed.transpose() == tag


def test_bcoo_transpose_carries_the_transposed_tag() -> None:
    """A transposed `BCOOLinearOperator` carries the tag of its own transposed matrix."""
    matrix = BCOO.fromdense(WIDE_MATRIX)
    operator = BCOOLinearOperator(matrix, sparsity_pattern_tag(matrix))
    transposed = operator.transpose()
    assert find_pattern_tag(transposed.tags) == sparsity_pattern_tag(transposed.matrix)
    assert find_pattern_tag(transposed.transpose().tags) == find_pattern_tag(
        operator.tags
    )


def test_bcsr_transpose_carries_a_row_major_tag() -> None:
    """A transposed `BCSRLinearOperator` sorts its entries, so its tag is sorted too and
    matches the tag of its own transposed matrix."""
    matrix = BCSR.fromdense(SQUARE_MATRIX)
    operator = BCSRLinearOperator(matrix, sparsity_pattern_tag(matrix))
    transposed = operator.transpose()
    assert find_pattern_tag(transposed.tags) == sparsity_pattern_tag(transposed.matrix)
    assert find_pattern_tag(transposed.transpose().tags) == find_pattern_tag(
        operator.tags
    )


def test_symmetric_operator_keeps_its_tag() -> None:
    """A symmetric operator is its own transpose, so its tag does not change."""
    symmetric = SQUARE_MATRIX + SQUARE_MATRIX.T
    matrix = BCOO.fromdense(symmetric)
    tag = sparsity_pattern_tag(matrix)
    operator = BCOOLinearOperator(matrix, (tag, lx.symmetric_tag))
    assert operator.transpose() is operator


def test_lineax_operators_carry_the_transposed_tag() -> None:
    """lineax's own operators transpose their tags through `lineax.transpose_tags`."""
    matrix = BCOO.fromdense(WIDE_MATRIX)
    tag = sparsity_pattern_tag(matrix)
    operator = lx.FunctionLinearOperator(
        lambda vector: WIDE_MATRIX @ vector,
        jax.ShapeDtypeStruct((3,), jnp.float32),
        tags=tag,
    )
    assert find_pattern_tag(operator.transpose().tags) == tag.transpose()
    tagged = lx.TaggedLinearOperator(lx.MatrixLinearOperator(WIDE_MATRIX), tag)
    assert find_pattern_tag(tagged.transpose().tags) == tag.transpose()


def test_transposed_tag_keeps_the_entry_order() -> None:
    """The transposed tag swaps the index columns without reordering the entries."""
    indices = np.array([[1, 0], [0, 2], [0, 0]])
    matrix = BCOO((jnp.ones(3), jnp.asarray(indices)), shape=(2, 3))
    transposed = sparsity_pattern_tag(matrix).transpose()
    assert isinstance(transposed, _ContentPatternTag)
    np.testing.assert_array_equal(transposed.indices, indices[:, ::-1])
    assert transposed.shape == (3, 2)


def banded_function(point: jax.Array, args: None) -> jax.Array:
    """A nonlinear function with a tridiagonal Jacobian."""
    del args
    return point**2 + 0.5 * jnp.roll(point, 1) * point + jnp.roll(point, -1)


def banded_pattern(size: int) -> BCOO:
    """The tridiagonal pattern of `banded_function`, with periodic corners."""
    dense = np.eye(size) + np.eye(size, k=1) + np.eye(size, k=-1)
    dense[0, -1] = dense[-1, 0] = 1.0
    return BCOO.fromdense(jnp.asarray(dense))


def test_coloring_tag_equals_the_plain_tag() -> None:
    """A tag with a coloring compares equal to the plain tag of the same pattern, so the
    two share a factorization."""
    pattern = banded_pattern(6)
    colored = sparsity_coloring_tag(pattern)
    plain = sparsity_pattern_tag(pattern)
    assert colored == plain
    assert hash(colored) == hash(plain)
    assert colored.coloring is not None


def test_detected_tag_matches_the_known_pattern() -> None:
    """Detecting the pattern of a function gives the same tag as the known pattern."""
    detected = sparsity_coloring_tag(banded_function, jnp.ones(6))
    assert detected == sparsity_pattern_tag(banded_pattern(6))
    abstract = sparsity_coloring_tag(
        banded_function, jax.ShapeDtypeStruct((6,), jnp.float32)
    )
    assert abstract == detected


def test_detection_ravels_pytrees() -> None:
    """A function of pytrees is detected on its raveled input and output."""

    def function(point: dict[str, jax.Array], args: None) -> tuple[jax.Array, ...]:
        del args
        return point["a"] * point["b"], point["a"] ** 2

    tag = sparsity_coloring_tag(function, {"a": jnp.ones(2), "b": jnp.ones(2)})
    expected = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    assert tag == sparsity_pattern_tag(expected.astype(bool))


@pytest.mark.parametrize(("jac", "mode"), [("fwd", "fwd"), ("bwd", "rev")])
def test_lazy_coloring_follows_jac_and_is_cached(
    jac: JacobianDirection, mode: str
) -> None:
    """A plain tag colors on first use in the direction asked for, then reuses it."""
    tag = sparsity_pattern_tag(banded_pattern(6))
    assert isinstance(tag, _ContentPatternTag)
    coloring = tag.jacobian_coloring(jac)
    assert coloring.mode == mode
    assert tag.jacobian_coloring(jac) is coloring


def test_fixed_coloring_wins_over_jac() -> None:
    """A coloring fixed up front is used whatever direction is asked for."""
    tag = sparsity_coloring_tag(banded_pattern(6), jac="fwd")
    assert tag.jacobian_coloring("bwd") is tag.coloring
    assert tag.jacobian_coloring("bwd").mode == "fwd"


def test_colored_pattern_with_another_direction_is_rejected() -> None:
    """A precomputed coloring cannot be relabelled with another direction."""
    coloring = asdex.jacobian_coloring_from_sparsity(banded_pattern(6), mode="fwd")
    with pytest.raises(ValueError, match="does not match"):
        sparsity_coloring_tag(coloring, jac="bwd")


def test_traced_pattern_is_rejected() -> None:
    """A coloring tag needs concrete indices."""

    @jax.jit
    def build_tag(matrix: BCOO) -> jax.Array:
        sparsity_coloring_tag(matrix)
        return matrix.data

    with pytest.raises(TypeError, match="concrete indices"):
        build_tag(banded_pattern(6))


@pytest.mark.parametrize("jac", ["fwd", "bwd"])
def test_transposed_coloring_reuses_the_colors(jac: JacobianDirection) -> None:
    """The transposed tag reuses the colors with the direction swapped, and the result
    computes the transposed Jacobian."""
    matrix = jnp.asarray(np.random.default_rng(0).normal(size=(4, 5)))
    matrix = matrix * (jnp.abs(matrix) > 0.5)
    tag = sparsity_coloring_tag(BCOO.fromdense(matrix), jac=jac)
    assert tag.coloring is not None
    transposed = tag.transpose()
    assert transposed.coloring is not None
    assert transposed.coloring.mode == ("rev" if jac == "fwd" else "fwd")
    np.testing.assert_array_equal(transposed.coloring.colors, tag.coloring.colors)
    jacobian = asdex.jacobian_from_coloring(
        lambda vector: matrix.T @ vector, transposed.coloring, "dense"
    )(jnp.zeros(4))
    np.testing.assert_allclose(jacobian, matrix.T)


def test_lazy_colorings_carry_over_to_the_transpose() -> None:
    """A coloring cached for one direction is cached for the other one on the transpose."""
    tag = sparsity_pattern_tag(banded_pattern(6))
    assert isinstance(tag, _ContentPatternTag)
    forward = tag.jacobian_coloring("fwd")
    transposed = tag.transpose()
    reverse = transposed.jacobian_coloring("bwd")
    assert reverse.mode == "rev"
    np.testing.assert_array_equal(reverse.colors, forward.colors)


def test_bcsr_transpose_keeps_a_valid_coloring() -> None:
    """Sorting the transposed tag of a `BCSR` operator keeps its coloring valid."""
    matrix = BCSR.fromdense(SQUARE_MATRIX)
    operator = BCSRLinearOperator(matrix, sparsity_coloring_tag(matrix, jac="fwd"))
    tag = find_pattern_tag(operator.transpose().tags)
    assert isinstance(tag, _ContentPatternTag)
    assert tag.coloring is not None
    jacobian = asdex.jacobian_from_coloring(
        lambda vector: SQUARE_MATRIX.T @ vector, tag.coloring, "bcoo"
    )(jnp.zeros(4))
    np.testing.assert_array_equal(np.asarray(jacobian.indices), tag.indices)
    np.testing.assert_allclose(jacobian.todense(), SQUARE_MATRIX.T)
