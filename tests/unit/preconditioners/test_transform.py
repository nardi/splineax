"""Tests for system transforms and the reordering that ships with them.

The invariants here are what let the solver hide a reordering completely: permuting the
system, solving, and un-permuting must reproduce the original answer, and the tags the
transformed operator carries must still be true --- `lineax.CG` refuses an operator that
is not tagged definite, so a tag dropped too eagerly breaks a solve outright.
"""

from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from lineax._tags import (
    diagonal_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    tridiagonal_tag,
)

import splineax as splx
from splineax._pattern import as_coo_pattern
from splineax.preconditioners._transform import (
    ComposedTransform,
    IdentityTransform,
    SymmetricPermutation,
    compose,
    operator_tags,
    preserves_symmetry,
    transform_operator,
    transform_pattern,
    transform_solution,
    transform_vector,
    transformed_tags,
    transpose_transform,
    untransform_solution,
)

ORDER = np.array([2, 0, 3, 1])
MATRIX = np.array(
    [
        [4.0, 1.0, 0.0, 2.0],
        [1.0, 5.0, 3.0, 0.0],
        [0.0, 3.0, 6.0, 1.0],
        [2.0, 0.0, 1.0, 7.0],
    ]
)


def permutation() -> SymmetricPermutation:
    return SymmetricPermutation.from_order(ORDER)


def operator_of(dense: np.ndarray, tags: object = ()) -> splx.BCOOLinearOperator:
    return splx.BCOOLinearOperator(BCOO.fromdense(jnp.asarray(dense)), tags)


def test_inverse_is_consistent():
    """`inverse` undoes `permutation`, by construction."""
    transform = permutation()
    assert np.array_equal(
        np.asarray(transform.inverse)[np.asarray(transform.permutation)], np.arange(4)
    )


def test_identity_leaves_everything_alone():
    """The no-transform path touches nothing, so it costs nothing."""
    transform = IdentityTransform()
    operator = operator_of(MATRIX)
    vector = jnp.arange(4.0)
    assert transform_operator(transform, operator) is operator
    assert transform_vector(transform, vector) is vector
    assert untransform_solution(transform, vector) is vector


def test_permuted_operator_matches_the_dense_permutation():
    """`A -> P A P^T` really is a symmetric row-and-column reordering."""
    permuted = transform_operator(permutation(), operator_of(MATRIX))
    expected = MATRIX[ORDER][:, ORDER]
    assert np.allclose(np.asarray(permuted.as_matrix()), expected)


def test_permuted_pattern_matches_the_permuted_operator():
    """The pattern the preconditioner sees is the transformed operator's pattern."""
    transform = permutation()
    operator = operator_of(MATRIX)
    pattern = transform_pattern(transform, as_coo_pattern(operator, "test"))
    permuted = transform_operator(transform, operator)
    dense = np.zeros((4, 4))
    dense[np.asarray(pattern.rows), np.asarray(pattern.cols)] = 1.0
    assert np.array_equal(dense != 0, np.asarray(permuted.as_matrix()) != 0)


def test_solution_round_trips():
    """Moving a solution into the transformed space and back is the identity."""
    transform = permutation()
    x = jnp.arange(4.0)
    assert np.allclose(
        np.asarray(untransform_solution(transform, transform_solution(transform, x))),
        np.asarray(x),
    )


def test_solving_the_permuted_system_gives_the_same_answer():
    """The whole point: a reordering is invisible from the outside."""
    transform = permutation()
    operator = operator_of(MATRIX)
    b = jnp.array([1.0, 2.0, 3.0, 4.0])
    permuted = transform_operator(transform, operator)
    y = jnp.linalg.solve(permuted.as_matrix(), transform_vector(transform, b))
    assert np.allclose(
        np.asarray(untransform_solution(transform, y)),
        np.asarray(jnp.linalg.solve(jnp.asarray(MATRIX), b)),
    )


def test_compose_flattens_and_drops_identities():
    """Composition keeps the common cases free of wrappers."""
    transform = permutation()
    assert isinstance(
        compose(IdentityTransform(), IdentityTransform()), IdentityTransform
    )
    assert compose(IdentityTransform(), transform) is transform
    composed = compose(compose(transform, transform), transform)
    assert isinstance(composed, ComposedTransform)
    assert len(composed.transforms) == 3


def test_composed_transform_round_trips():
    """Applying several transforms and undoing them returns the original."""
    transform = compose(permutation(), permutation())
    x = jnp.arange(4.0)
    assert np.allclose(
        np.asarray(untransform_solution(transform, transform_solution(transform, x))),
        np.asarray(x),
    )


def test_structural_tags_are_dropped():
    """Where the nonzeros are is exactly what a reordering changes."""
    kept = transformed_tags(permutation(), frozenset({diagonal_tag, tridiagonal_tag}))
    assert kept == frozenset()


def test_definiteness_survives_a_symmetric_permutation():
    """`P A P^T` is definite whenever `A` is --- and `lineax.CG` depends on the tag."""
    tags = frozenset({positive_semidefinite_tag, symmetric_tag})
    assert transformed_tags(permutation(), tags) == tags
    assert preserves_symmetry(permutation())


def test_a_permutation_is_its_own_transpose():
    """`(P A P^T)^T = P A^T P^T`."""
    transform = permutation()
    assert transpose_transform(transform) is transform


def test_operator_tags_reads_properties_back():
    """Tags are recovered through lineax's predicates, not an attribute."""
    operator = operator_of(MATRIX, lx.positive_semidefinite_tag)
    assert positive_semidefinite_tag in operator_tags(operator)


def test_reverse_cuthill_mckee_reduces_bandwidth():
    """The reordering does what it is for: it narrows the band."""
    size = 20
    dense = np.eye(size)
    # A deliberately bad ordering: index i couples to i + size // 2.
    for i in range(size // 2):
        dense[i, i + size // 2] = dense[i + size // 2, i] = 1.0
    operator = operator_of(dense)
    transform = splx.ReverseCuthillMcKee().symbolic(as_coo_pattern(operator, "test"))
    permuted = np.asarray(transform_operator(transform, operator).as_matrix())

    def bandwidth(matrix: np.ndarray) -> int:
        rows, cols = np.nonzero(matrix)
        return int(np.max(np.abs(rows - cols)))

    assert bandwidth(permuted) < bandwidth(dense)


def test_reverse_cuthill_mckee_is_a_permutation():
    """Every index appears exactly once, disconnected components included."""
    dense = np.eye(10)
    dense[0, 1] = dense[1, 0] = 1.0
    operator = operator_of(dense)
    transform = splx.ReverseCuthillMcKee().symbolic(as_coo_pattern(operator, "test"))
    assert isinstance(transform, SymmetricPermutation)
    assert sorted(np.asarray(transform.permutation).tolist()) == list(range(10))


def test_reverse_cuthill_mckee_preserves_the_solution():
    """Reordering and solving agrees with solving directly."""
    operator = operator_of(MATRIX)
    transform = splx.ReverseCuthillMcKee().symbolic(as_coo_pattern(operator, "test"))
    b = jnp.array([1.0, 2.0, 3.0, 4.0])
    permuted = transform_operator(transform, operator)
    y = jnp.linalg.solve(permuted.as_matrix(), transform_vector(transform, b))
    assert np.allclose(
        np.asarray(untransform_solution(transform, y)),
        np.asarray(jnp.linalg.solve(jnp.asarray(MATRIX), b)),
    )


def test_pytree_vectors_are_refused_clearly():
    """Permuting a PyTree-structured system is not supported, and says so."""
    with pytest.raises(NotImplementedError, match="PyTree-structured"):
        transform_vector(permutation(), {"a": jnp.arange(2.0)})
