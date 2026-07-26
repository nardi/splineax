"""Shared fixtures and reference data for the sparse-solver test suites.

[test_solvers.py](test_solvers.py) runs the solver/format-agnostic suite (parametrised over both
solvers and both operator formats); [test_klu.py](test_klu.py) holds the KLU-specific
factorization tests. Both draw their operators and reference matrices from here.
"""

from typing import Protocol

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
import scipy.sparse
from jax.experimental.sparse import BCOO, BCSR

from splineax import (
    KLU,
    AutoSparseLinearSolver,
    BCOOLinearOperator,
    BCSRLinearOperator,
    CuDSS,
    Pardiso,
    Spsolve,
)
from splineax.solvers._auto import _cuda_backend_available
from splineax.solvers._cudss import _cudss_available
from splineax.solvers._pardiso import _pardiso_available


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


def _zero_diagonal_saddle_point(half: int, seed: int) -> np.ndarray:
    """A well-conditioned `[[0, B], [B.T, -C]]` matrix with an all-zero leading diagonal.

    Every other reference matrix here is diagonally dominant, which is the easiest case
    a sparse direct solver ever sees. This one is the opposite: half its diagonal
    entries are structurally zero, so the solver has to permute large entries onto the
    diagonal to factor it stably rather than perturbing the tiny pivots it finds there.
    A solver that gets that wrong still returns a solution and still reports success.
    Only the residual gives it away, so this shape is worth pinning explicitly.
    """
    random_state = np.random.default_rng(seed)
    block = scipy.sparse.random(
        half, half, density=0.3, random_state=random_state, format="csr"
    ).toarray()
    assert np.linalg.matrix_rank(block) == half, "off-diagonal block is singular"
    lower_right = np.diag(random_state.uniform(0.5, 1.5, size=half))
    return np.block([[np.zeros((half, half)), block], [block.T, -lower_right]])


# Kept as float64 NumPy arrays rather than `jax.Array`s like the constants above: this
# module is imported before any test enables x64, so a `jnp.asarray` here would round
# the matrix to float32 at import time and blunt the very distinction it exists to
# draw. Tests convert them under the `enable_x64` fixture instead.
ZERO_DIAGONAL_MATRIX: np.ndarray = _zero_diagonal_saddle_point(half=16, seed=0)
ZERO_DIAGONAL_RIGHT_HAND_SIDE: np.ndarray = np.random.default_rng(1).uniform(
    -1.0, 1.0, size=ZERO_DIAGONAL_MATRIX.shape[0]
)
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
    params=[
        Spsolve,
        KLU,
        pytest.param(
            Pardiso,
            marks=pytest.mark.skipif(
                not _pardiso_available(), reason="pardiso-mkl-jax is not installed"
            ),
        ),
        pytest.param(
            CuDSS,
            marks=pytest.mark.skipif(
                not (_cudss_available() and _cuda_backend_available()),
                reason="the optional cuDSS dependency is not installed, or no CUDA "
                "GPU is visible",
            ),
        ),
        AutoSparseLinearSolver,
    ],
    ids=["spsolve", "klu", "pardiso", "cudss", "auto"],
)
def solver(request: pytest.FixtureRequest, enable_x64: None) -> lx.AbstractLinearSolver:
    """Yields an instance of each sparse direct solver under test.

    `AutoSparseLinearSolver` dispatches to `Pardiso` (if installed) or `KLU` on the
    (CPU) test platform when x64 is enabled, otherwise to `Spsolve`. `Pardiso` itself is
    skipped when its optional dependency isn't installed, and `CuDSS` when it isn't
    installed or no CUDA GPU is visible. Depends on `enable_x64` (from the top-level
    conftest) so every test using this fixture runs with x64 enabled for its whole
    body, since `KLU`/`Pardiso` require it but no longer enable it themselves (`CuDSS`
    has no such requirement, but running under x64 doesn't hurt it either).
    """
    return request.param()
