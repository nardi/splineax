"""Test suite shared by every sparse operator.

The same behavioural checks must hold for `BCOOLinearOperator` and
`BCSRLinearOperator` (the only differences between them are internal storage details).
Rather than duplicating the suite per format, the `make_operator` fixture is
parametrised over the two formats, so every test below runs once for each. Each test
receives a factory that wraps a *dense* reference matrix in the operator under test;
this keeps the tests format-agnostic and lets us compare against the dense matrix as
the source of truth.
"""

from typing import Protocol

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO, BCSR

from splineax import BCOOLinearOperator, BCSRLinearOperator


class OperatorFactory(Protocol):
    """Builds the operator under test from a dense reference matrix."""

    def __call__(
        self, dense_matrix: jax.Array, tags: object = ()
    ) -> lx.AbstractLinearOperator: ...


# A non-symmetric square matrix whose tridiagonal band is fully populated *and* which
# carries off-band entries (the `7.` and `9.`), so that diagonal/band extraction is
# forced to filter rather than trivially return every stored value.
SQUARE_MATRIX: jax.Array = jnp.array(
    [
        [1.0, 2.0, 0.0, 7.0],
        [3.0, 4.0, 5.0, 0.0],
        [0.0, 6.0, 8.0, 9.0],
        [0.0, 0.0, 1.0, 2.0],
    ]
)
# A non-square (wide) matrix, to exercise the asymmetry between input/output structures.
WIDE_MATRIX: jax.Array = jnp.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
# A complex matrix, to exercise conjugation and complex dtype handling.
COMPLEX_MATRIX: jax.Array = jnp.array([[1.0 + 2.0j, 0.0], [4.0 - 1.0j, 3.0j]])


def _make_bcoo_operator(
    dense_matrix: jax.Array, tags: object = ()
) -> BCOOLinearOperator:
    return BCOOLinearOperator(BCOO.fromdense(dense_matrix), tags)


def _make_bcsr_operator(
    dense_matrix: jax.Array, tags: object = ()
) -> BCSRLinearOperator:
    return BCSRLinearOperator(BCSR.fromdense(dense_matrix), tags)


@pytest.fixture(params=["bcoo", "bcsr"])
def make_operator(request: pytest.FixtureRequest) -> OperatorFactory:
    """Yields a factory that creates a sparse linear operator from a dense array."""
    return {
        "bcoo": _make_bcoo_operator,
        "bcsr": _make_bcsr_operator,
    }[request.param]


def test_matrix_vector_product_matches_dense(make_operator: OperatorFactory) -> None:
    """`mv` must reproduce ordinary dense matrix-vector multiplication, for both square
    and non-square matrices (the latter confirms the shapes are not assumed square)."""
    square_operator = make_operator(SQUARE_MATRIX)
    square_vector = jnp.array([1.0, -2.0, 3.0, 0.5])
    assert jnp.allclose(
        square_operator.mv(square_vector), SQUARE_MATRIX @ square_vector
    )

    wide_operator = make_operator(WIDE_MATRIX)
    wide_vector = jnp.array([1.0, 2.0, 3.0])
    assert jnp.allclose(wide_operator.mv(wide_vector), WIDE_MATRIX @ wide_vector)


def test_as_matrix_round_trips_to_dense(make_operator: OperatorFactory) -> None:
    """`as_matrix` must materialise back to exactly the dense matrix we wrapped, so that
    densifying consumers (and our own band extraction tests) have a faithful reference."""
    assert jnp.allclose(make_operator(SQUARE_MATRIX).as_matrix(), SQUARE_MATRIX)
    assert jnp.allclose(make_operator(WIDE_MATRIX).as_matrix(), WIDE_MATRIX)


def test_transpose_matches_dense_transpose(make_operator: OperatorFactory) -> None:
    """`transpose()` and the `.T` property must both give the dense transpose. Uses the
    wide matrix so a wrong implementation that assumed squareness would fail on shape."""
    operator = make_operator(WIDE_MATRIX)
    assert jnp.allclose(operator.transpose().as_matrix(), WIDE_MATRIX.T)
    assert jnp.allclose(operator.T.as_matrix(), WIDE_MATRIX.T)


def test_in_and_out_structure_describe_shapes(make_operator: OperatorFactory) -> None:
    """A matrix of shape `(out, in)` must report `in_structure`/`out_structure` (and the
    derived sizes) consistent with mapping a vector of length `in` to one of length
    `out`. The wide matrix makes the in/out distinction observable."""
    operator = make_operator(WIDE_MATRIX)  # shape (2, 3): out_size=2, in_size=3
    assert operator.in_structure() == jax.ShapeDtypeStruct((3,), WIDE_MATRIX.dtype)
    assert operator.out_structure() == jax.ShapeDtypeStruct((2,), WIDE_MATRIX.dtype)
    assert operator.in_size() == 3
    assert operator.out_size() == 2


def test_integer_matrix_is_promoted_to_floating(
    make_operator: OperatorFactory,
) -> None:
    """Lineax operators work over inexact dtypes; an integer matrix must be promoted to
    floating point (mirroring `MatrixLinearOperator`) rather than rejected or kept
    integer, which would break downstream solves."""
    operator = make_operator(jnp.array([[1, 2], [3, 4]], dtype=jnp.int32))
    assert jnp.issubdtype(operator.as_matrix().dtype, jnp.floating)


def test_non_2d_matrix_is_rejected(make_operator: OperatorFactory) -> None:
    """The operators model 2-D linear maps, so constructing one from a 1-D array must
    fail loudly instead of silently producing nonsense."""
    with pytest.raises(ValueError):
        make_operator(jnp.array([1.0, 2.0, 3.0]))


def test_conjugate_matches_dense_conjugate(make_operator: OperatorFactory) -> None:
    """`lx.conj` must elementwise-conjugate the operator. Tested on a complex matrix,
    since for real matrices conjugation is a no-op and would not catch a broken
    implementation."""
    operator = make_operator(COMPLEX_MATRIX)
    assert jnp.allclose(lx.conj(operator).as_matrix(), COMPLEX_MATRIX.conj())


def test_untagged_operator_reports_no_properties(
    make_operator: OperatorFactory,
) -> None:
    """An operator created without tags must report `False` for every structural
    property. Lineax solvers branch on these predicates, so a spurious `True` could
    select an invalid (e.g. symmetric-only) solver path."""
    operator = make_operator(SQUARE_MATRIX)
    assert lx.is_symmetric(operator) is False
    assert lx.is_diagonal(operator) is False
    assert lx.is_lower_triangular(operator) is False
    assert lx.is_upper_triangular(operator) is False
    assert lx.is_positive_semidefinite(operator) is False
    assert lx.is_negative_semidefinite(operator) is False
    assert lx.has_unit_diagonal(operator) is False


def test_tags_drive_property_predicates(make_operator: OperatorFactory) -> None:
    """Passing a tag must make the corresponding predicate report `True`, confirming the
    `singledispatch` registrations actually consult the operator's tags (the mechanism
    by which a caller asserts known structure)."""
    operator = make_operator(SQUARE_MATRIX, tags=lx.symmetric_tag)
    assert lx.is_symmetric(operator) is True


def test_diagonal_extraction_matches_dense(make_operator: OperatorFactory) -> None:
    """`lx.diagonal` must equal the dense main diagonal, and must remain correct under
    `jax.jit` — the latter proves the sparse (non-densifying) extraction is traceable,
    not relying on Python-level data inspection."""
    operator = make_operator(SQUARE_MATRIX)
    expected_diagonal = jnp.diagonal(SQUARE_MATRIX, 0)
    assert jnp.allclose(lx.diagonal(operator), expected_diagonal)
    assert jnp.allclose(jax.jit(lx.diagonal)(operator), expected_diagonal)


def test_tridiagonal_extraction_matches_dense(make_operator: OperatorFactory) -> None:
    """`lx.tridiagonal` must return the (main, lower, upper) diagonals matching the dense
    matrix. The reference matrix has off-band entries, so this also confirms those are
    correctly excluded from the bands."""
    operator = make_operator(SQUARE_MATRIX)
    main_diagonal, lower_diagonal, upper_diagonal = lx.tridiagonal(operator)
    assert jnp.allclose(main_diagonal, jnp.diagonal(SQUARE_MATRIX, 0))
    assert jnp.allclose(lower_diagonal, jnp.diagonal(SQUARE_MATRIX, -1))
    assert jnp.allclose(upper_diagonal, jnp.diagonal(SQUARE_MATRIX, 1))


def test_linear_solve_matches_numpy(make_operator: OperatorFactory) -> None:
    """End-to-end integration proof: `lx.linear_solve` against the operator must recover
    the same solution as `numpy.linalg.solve`. This only succeeds if every dispatch
    registration the solver relies on is present, so it is the strongest single test
    that the operator is genuinely usable inside Lineax."""
    diagonally_dominant_matrix = SQUARE_MATRIX + 10.0 * jnp.eye(4)
    right_hand_side = jnp.array([1.0, 2.0, 3.0, 4.0])
    operator = make_operator(diagonally_dominant_matrix)
    solution = lx.linear_solve(operator, right_hand_side, solver=lx.LU()).value
    expected_solution = jnp.linalg.solve(
        np.asarray(diagonally_dominant_matrix), np.asarray(right_hand_side)
    )
    assert jnp.allclose(solution, expected_solution, atol=1e-5)
