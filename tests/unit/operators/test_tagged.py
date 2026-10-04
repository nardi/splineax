"""Tests for the sparse materialisation of tagged lineax operators.

The reference for correctness is the dense Jacobian from `jax.jacfwd`. A
`lineax.JacobianLinearOperator` of a function and a `lineax.FunctionLinearOperator` of
its JVP must give the same `BCOO`, in indices and values, when they share a tag.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from splineax import (
    BCOOLinearOperator,
    JacobianDirection,
    materialise_as_bcoo,
    sparsity_coloring_tag,
    sparsity_pattern_tag,
)
from splineax.operators._tags import _ContentPatternTag, find_pattern_tag


def elementwise_function(point: jax.Array, args: object) -> jax.Array:
    """An elementwise map, whose Jacobian is diagonal."""
    del args
    return jnp.sin(point) + point**2


def banded_function(point: jax.Array, args: object) -> jax.Array:
    """A nearest-neighbour coupling, whose rectangular Jacobian is banded and needs more
    than one color."""
    del args
    return jnp.diff(point) * point[:-1] + jnp.cos(point[1:])


EVALUATION_POINT = jnp.linspace(1.0, 2.0, 6)


def dense_jacobian(function, point: jax.Array) -> jax.Array:
    return jax.jacfwd(lambda input_point: function(input_point, None))(point)


def jvp_operator(function, point: jax.Array, tag: object) -> lx.FunctionLinearOperator:
    """A `lineax.FunctionLinearOperator` of the JVP of `function` at `point`."""

    def jvp_of_function(tangent: jax.Array) -> jax.Array:
        _, output_tangent = jax.jvp(
            lambda input_point: function(input_point, None), (point,), (tangent,)
        )
        return output_tangent

    return lx.FunctionLinearOperator(
        jvp_of_function, jax.ShapeDtypeStruct(point.shape, point.dtype), tags=tag
    )


@pytest.mark.parametrize("function", [elementwise_function, banded_function])
def test_jacobian_operator_matches_the_dense_jacobian(function) -> None:
    """A tagged `lineax.JacobianLinearOperator` materialises to its dense Jacobian."""
    tag = sparsity_coloring_tag(function, EVALUATION_POINT)
    operator = lx.JacobianLinearOperator(function, EVALUATION_POINT, tags=tag)
    materialised = materialise_as_bcoo(operator)
    assert isinstance(materialised, BCOOLinearOperator)
    np.testing.assert_allclose(
        materialised.as_matrix(), dense_jacobian(function, EVALUATION_POINT)
    )
    assert tag in materialised.tags


@pytest.mark.parametrize("function", [elementwise_function, banded_function])
@pytest.mark.parametrize("jac", ["fwd", "bwd"])
def test_function_operator_of_a_jvp_matches_the_jacobian_operator(
    function, jac: JacobianDirection
) -> None:
    """A `lineax.FunctionLinearOperator` of `jax.jvp(f)` and a
    `lineax.JacobianLinearOperator` of `f` give the same indices and values."""
    tag = sparsity_coloring_tag(function, EVALUATION_POINT, jac=jac)
    jacobian = materialise_as_bcoo(
        lx.JacobianLinearOperator(function, EVALUATION_POINT, tags=tag)
    ).matrix
    function_matrix = materialise_as_bcoo(
        jvp_operator(function, EVALUATION_POINT, tag)
    ).matrix
    np.testing.assert_array_equal(jacobian.indices, function_matrix.indices)
    np.testing.assert_allclose(jacobian.data, function_matrix.data, rtol=1e-6)
    np.testing.assert_array_equal(jacobian.indices, tag.indices)


@pytest.mark.parametrize(("jac", "mode"), [("fwd", "fwd"), ("bwd", "rev")])
def test_plain_tag_colors_in_the_operator_direction(
    jac: JacobianDirection, mode: str
) -> None:
    """With no coloring on the tag, the `jac` of the operator picks the direction, and
    the coloring is cached on the tag."""
    tag = sparsity_pattern_tag(
        BCOO.fromdense(dense_jacobian(banded_function, EVALUATION_POINT))
    )
    assert isinstance(tag, _ContentPatternTag)
    operator = lx.JacobianLinearOperator(
        banded_function, EVALUATION_POINT, tags=tag, jac=jac
    )
    materialised = materialise_as_bcoo(operator)
    np.testing.assert_allclose(
        materialised.as_matrix(), dense_jacobian(banded_function, EVALUATION_POINT)
    )
    assert tag.jacobian_coloring(jac).mode == mode


def test_tag_coloring_wins_over_the_operator_direction() -> None:
    """A coloring fixed on the tag is used even when the operator asks for another
    direction."""
    tag = sparsity_coloring_tag(banded_function, EVALUATION_POINT, jac="fwd")
    operator = lx.JacobianLinearOperator(
        banded_function, EVALUATION_POINT, tags=tag, jac="bwd"
    )
    materialised = materialise_as_bcoo(operator)
    assert "bwd" not in tag._colorings
    np.testing.assert_array_equal(materialised.matrix.indices, tag.indices)


def test_custom_vjp_function_converts_with_bwd() -> None:
    """A function that only defines a custom VJP converts with `jac="bwd"`."""

    @jax.custom_vjp
    def scale(point: jax.Array) -> jax.Array:
        return 2.0 * point

    def scale_forward(point: jax.Array) -> tuple[jax.Array, None]:
        return scale(point), None

    def scale_backward(residual: None, cotangent: jax.Array) -> tuple[jax.Array]:
        del residual
        return (2.0 * cotangent,)

    scale.defvjp(scale_forward, scale_backward)

    def function(point: jax.Array, args: object) -> jax.Array:
        del args
        return scale(point) * point

    tag = sparsity_pattern_tag(BCOO.fromdense(jnp.eye(4)))
    operator = lx.JacobianLinearOperator(
        function, jnp.arange(1.0, 5.0), tags=tag, jac="bwd"
    )
    np.testing.assert_allclose(
        materialise_as_bcoo(operator).as_matrix(),
        jnp.diag(4.0 * jnp.arange(1.0, 5.0)),
    )


def test_pytree_inputs_and_outputs_are_raveled() -> None:
    """Pytree inputs and outputs are raveled in leaf order."""

    def function(point: dict[str, jax.Array], args: object) -> tuple[jax.Array, ...]:
        del args
        return point["a"] * point["b"], jnp.sin(point["a"])

    point = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    tag = sparsity_coloring_tag(function, point)
    operator = lx.JacobianLinearOperator(function, point, tags=tag)
    expected = jnp.array(
        [
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 4.0, 0.0, 2.0],
            [jnp.cos(1.0), 0.0, 0.0, 0.0],
            [0.0, jnp.cos(2.0), 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(materialise_as_bcoo(operator).as_matrix(), expected)


def test_tagged_linear_operator_adds_its_tags() -> None:
    """A `lineax.TaggedLinearOperator` may carry the pattern tag for the operator it
    wraps."""
    tag = sparsity_coloring_tag(banded_function, EVALUATION_POINT)
    inner = lx.JacobianLinearOperator(banded_function, EVALUATION_POINT)
    operator = lx.TaggedLinearOperator(inner, tag)
    materialised = materialise_as_bcoo(operator)
    assert tag in materialised.tags
    np.testing.assert_allclose(
        materialised.as_matrix(), dense_jacobian(banded_function, EVALUATION_POINT)
    )


def test_unsorted_tag_sets_the_entry_order() -> None:
    """The `BCOO` lists its entries in the order of the tag's indices."""
    indices = np.array([[2, 2], [0, 0], [1, 1], [3, 3]])
    pattern = BCOO((jnp.ones(4), jnp.asarray(indices)), shape=(4, 4))
    tag = sparsity_pattern_tag(pattern)
    point = jnp.arange(1.0, 5.0)
    operator = lx.JacobianLinearOperator(elementwise_function, point, tags=tag)
    matrix = materialise_as_bcoo(operator).matrix
    np.testing.assert_array_equal(matrix.indices, indices)
    np.testing.assert_allclose(matrix.data, (jnp.cos(point) + 2 * point)[indices[:, 0]])


@pytest.mark.parametrize("jac", ["fwd", "bwd"])
def test_transposes_materialise_to_the_transposed_matrix(
    jac: JacobianDirection,
) -> None:
    """The transpose of a tagged operator carries the transposed tag, and materialises
    to the transposed matrix with its entries in the same order."""
    tag = sparsity_coloring_tag(banded_function, EVALUATION_POINT, jac=jac)
    for operator in (
        lx.JacobianLinearOperator(banded_function, EVALUATION_POINT, tags=tag),
        jvp_operator(banded_function, EVALUATION_POINT, tag),
    ):
        forward = materialise_as_bcoo(operator).matrix
        transposed = materialise_as_bcoo(operator.transpose()).matrix
        assert isinstance(transposed, BCOO)
        np.testing.assert_array_equal(transposed.indices, forward.T.indices)
        np.testing.assert_allclose(transposed.data, forward.T.data, rtol=1e-6)


def test_missing_tag_is_rejected() -> None:
    operator = lx.JacobianLinearOperator(banded_function, EVALUATION_POINT)
    with pytest.raises(TypeError, match="needs a sparsity-pattern tag"):
        materialise_as_bcoo(operator)


def test_identity_tag_is_rejected() -> None:
    operator = lx.JacobianLinearOperator(
        banded_function, EVALUATION_POINT, tags=sparsity_pattern_tag()
    )
    with pytest.raises(TypeError, match="without concrete indices"):
        materialise_as_bcoo(operator)


def test_shape_mismatch_is_rejected() -> None:
    tag = sparsity_coloring_tag(elementwise_function, EVALUATION_POINT)
    operator = lx.JacobianLinearOperator(banded_function, EVALUATION_POINT, tags=tag)
    with pytest.raises(ValueError, match="has shape"):
        materialise_as_bcoo(operator)


def test_complex_dtype_is_rejected() -> None:
    tag = sparsity_pattern_tag(BCOO.fromdense(jnp.eye(2)))
    operator = lx.JacobianLinearOperator(
        elementwise_function, jnp.ones(2, dtype=jnp.complex64), tags=tag
    )
    with pytest.raises(TypeError, match="only supports real dtypes"):
        materialise_as_bcoo(operator)


def test_unsupported_operator_is_rejected() -> None:
    with pytest.raises(TypeError, match="requires a"):
        materialise_as_bcoo(lx.MatrixLinearOperator(jnp.eye(2)))


def test_jit_cache_is_stable_across_points() -> None:
    """Operators that share a tag at different points compile once."""
    tag = sparsity_coloring_tag(banded_function, EVALUATION_POINT)
    trace_log: list[bool] = []

    @eqx.filter_jit
    def materialise_data(operator: lx.AbstractLinearOperator) -> jax.Array:
        trace_log.append(True)
        return materialise_as_bcoo(operator).matrix.data

    for shift in (0.0, 1.0):
        point = EVALUATION_POINT + shift
        operator = lx.JacobianLinearOperator(banded_function, point, tags=tag)
        expected = BCOO.fromdense(dense_jacobian(banded_function, point)).data
        np.testing.assert_allclose(materialise_data(operator), expected, rtol=1e-6)
    assert len(trace_log) == 1


def test_linearised_operator_keeps_its_tag() -> None:
    """`lineax.linearise` turns a tagged Jacobian operator into a tagged function
    operator, which materialises to the same matrix."""
    tag = sparsity_coloring_tag(banded_function, EVALUATION_POINT)
    operator = lx.JacobianLinearOperator(banded_function, EVALUATION_POINT, tags=tag)
    linearised = lx.linearise(operator)
    assert isinstance(linearised, lx.FunctionLinearOperator)
    assert find_pattern_tag(linearised.tags) == tag
    np.testing.assert_allclose(
        materialise_as_bcoo(linearised).as_matrix(),
        dense_jacobian(banded_function, EVALUATION_POINT),
        rtol=1e-6,
    )
