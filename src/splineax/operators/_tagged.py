"""Sparse materialisation of tagged lineax operators.

A `lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator` that carries a
content sparsity-pattern tag can be turned into a `BCOO` with asdex. The tag holds the
pattern and a Jacobian coloring of it, so the conversion costs one JVP or VJP per color
instead of one per column or row.

A `lineax.FunctionLinearOperator` is treated as a linear function, so its matrix is the
Jacobian of its `mv`. A `FunctionLinearOperator` of `jax.jvp(f, (x,), (v,))` therefore
gives the same `BCOO` as a `JacobianLinearOperator` of `f` at `x`.
"""

from collections.abc import Callable
from typing import cast

import asdex
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PyTree
from lineax._operator import _NoAuxIn

from ._bcoo import BCOOLinearOperator
from ._bcsr import BCSRLinearOperator
from ._tags import (
    JacobianDirection,
    _ContentPatternTag,
    example_point,
    find_pattern_tag,
    flat_function,
)


def materialise_as_bcoo(operator: lx.AbstractLinearOperator) -> BCOOLinearOperator:
    """Materialise a sparse or tagged operator as a `BCOOLinearOperator`.

    A `BCOOLinearOperator` is returned as it is, and a `BCSRLinearOperator` is converted.
    A `lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator` must carry a tag
    from [`splineax.sparsity_pattern_tag`][] or [`splineax.sparsity_coloring_tag`][]
    built from concrete indices. Its matrix is then computed with asdex, one JVP or VJP
    per color of the tag's coloring. The entries of the result are in the order of the
    tag's indices. A `lineax.TaggedLinearOperator` around either of them adds its own
    tags.

    The direction of a tag's coloring is fixed if the tag holds one. Otherwise it is the
    `jac` of a `lineax.JacobianLinearOperator`, and asdex picks it when `jac` is None or
    for a `lineax.FunctionLinearOperator`.

    Inputs and outputs may be pytrees, which are raveled in leaf order. Only real dtypes
    are supported.
    """
    return _materialise_with_tags(operator, frozenset())


def _materialise_with_tags(
    operator: lx.AbstractLinearOperator, outer_tags: frozenset[object]
) -> BCOOLinearOperator:
    """Materialise `operator` as in `materialise_as_bcoo`, adding `outer_tags` to it.

    `outer_tags` collects the tags of the `lineax.TaggedLinearOperator`s around it.
    """
    function: Callable[[PyTree[Array]], PyTree[Array]]
    jac: JacobianDirection | None
    match operator:
        case BCOOLinearOperator(matrix=matrix, tags=tags):
            if not outer_tags:
                return operator
            return BCOOLinearOperator(matrix, tags | outer_tags)
        case BCSRLinearOperator(matrix=matrix, tags=tags):
            return BCOOLinearOperator(matrix.to_bcoo(), tags | outer_tags)
        case lx.TaggedLinearOperator(operator=inner_operator, tags=tags):
            return _materialise_with_tags(inner_operator, tags | outer_tags)
        case lx.JacobianLinearOperator(fn=fn, x=point, args=args, jac=jac):
            # The same function lineax differentiates in `mv`.
            function = _NoAuxIn(fn, args)
        case lx.FunctionLinearOperator():
            # A linear function has the same Jacobian everywhere, so zero will do.
            function = operator.mv
            point = example_point(operator.in_structure())
            jac = None
        case _:
            raise TypeError(
                "`materialise_as_bcoo` requires a `BCOOLinearOperator`, a "
                "`BCSRLinearOperator`, or a `lineax.JacobianLinearOperator` or "
                "`lineax.FunctionLinearOperator` carrying a sparsity-pattern tag, got "
                f"`{type(operator).__name__}`."
            )
    tags = operator.tags | outer_tags
    pattern_tag = _checked_pattern_tag(operator, tags)
    function_of_flat_point, flat_point = flat_function(function, point)
    coloring = pattern_tag.jacobian_coloring(jac)
    matrix = cast(
        BCOO,
        asdex.jacobian_from_coloring(function_of_flat_point, coloring, "bcoo")(
            flat_point
        ),
    )
    return BCOOLinearOperator(matrix, tags)


def _checked_pattern_tag(
    operator: lx.AbstractLinearOperator, tags: frozenset[object]
) -> _ContentPatternTag:
    """Return the content pattern tag among `tags`, checking it fits `operator`.

    Raises if there is no tag, if the tag carries no indices, if its shape does not
    match the operator, or if the operator has a complex dtype.
    """
    operator_name = type(operator).__name__
    pattern_tag = find_pattern_tag(tags)
    if pattern_tag is None:
        raise TypeError(
            f"A `lineax.{operator_name}` needs a sparsity-pattern tag to be "
            "materialised sparsely. Create one with `splineax.sparsity_pattern_tag` or "
            "`splineax.sparsity_coloring_tag`."
        )
    if not isinstance(pattern_tag, _ContentPatternTag):
        raise TypeError(
            f"The sparsity-pattern tag of this `lineax.{operator_name}` was created "
            "without concrete indices, so it cannot be used to materialise the "
            "operator. Create the tag from a concrete pattern, outside `jax.jit`."
        )
    expected_shape = (operator.out_size(), operator.in_size())
    if tuple(pattern_tag.shape) != expected_shape:
        raise ValueError(
            f"The sparsity-pattern tag has shape {tuple(pattern_tag.shape)}, but the "
            f"`lineax.{operator_name}` has shape {expected_shape}."
        )
    leaves = jax.tree.leaves((operator.in_structure(), operator.out_structure()))
    if any(jnp.issubdtype(leaf.dtype, jnp.complexfloating) for leaf in leaves):
        raise TypeError(
            f"Sparse materialisation of a `lineax.{operator_name}` only supports real "
            "dtypes."
        )
    return pattern_tag
