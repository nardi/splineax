"""Shared fixtures and reference data for the sparse-solver test suites.

[test_solvers.py](test_solvers.py) runs the solver/format-agnostic suite (parametrised over both
solvers and both operator formats); [test_klu.py](test_klu.py) holds the KLU-specific
factorization tests. Both draw their operators and reference matrices from here.
"""

from typing import Protocol

import jax
import jax.numpy as jnp
import lineax as lx
import pytest
from jax.experimental.sparse import BCOO, BCSR

from splineax import (
    KLU,
    AutoSparseLinearSolver,
    BCOOLinearOperator,
    BCSRLinearOperator,
    Spsolve,
)


class OperatorFactory(Protocol):
    """Builds the operator under test from a dense reference matrix."""

    def __call__(
        self, dense_matrix: jax.Array, tags: object = ()
    ) -> lx.AbstractLinearOperator: ...


# A diagonally dominant (hence nonsingular, well-conditioned) square matrix, so the
# direct solve is well posed and comparable against a dense reference solver.
SQUARE_MATRIX: jax.Array = jnp.array(
    [
        [1.0, 2.0, 0.0, 7.0],
        [3.0, 4.0, 5.0, 0.0],
        [0.0, 6.0, 8.0, 9.0],
        [0.0, 0.0, 1.0, 2.0],
    ]
) + 10.0 * jnp.eye(4)
RIGHT_HAND_SIDE: jax.Array = jnp.array([1.0, 2.0, 3.0, 4.0])
# A non-square (wide) matrix, to confirm the square-only contract is enforced.
WIDE_MATRIX: jax.Array = jnp.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
# A diagonally dominant complex matrix, to exercise the complex128 solve path.
COMPLEX_MATRIX: jax.Array = jnp.array(
    [
        [10.0 + 2.0j, 1.0 + 0.0j, 0.0, 0.0],
        [0.0, 8.0 - 1.0j, 2.0 + 0.0j, 0.0],
        [0.0, 0.0, 9.0 + 0.0j, 1.0 + 1.0j],
        [1.0 + 0.0j, 0.0, 0.0, 7.0 + 0.0j],
    ]
)
COMPLEX_RIGHT_HAND_SIDE: jax.Array = jnp.array(
    [1.0 + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j, 2.0j]
)


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


@pytest.fixture(
    params=[Spsolve, KLU, AutoSparseLinearSolver],
    ids=["spsolve", "klu", "auto"],
)
def solver(request: pytest.FixtureRequest) -> lx.AbstractLinearSolver:
    """Yields an instance of each sparse direct solver under test.

    `AutoSparseLinearSolver` dispatches to `KLU` on the (CPU) test platform when x64 is
    enabled, otherwise to `Spsolve`.
    """
    return request.param()
