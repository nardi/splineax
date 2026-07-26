"""Fixtures for the solver benchmarks.

x64 is enabled at **import** time, before any JAX array exists. `KLU` and `Pardiso`
require it and no longer enable it themselves, and the function-scoped `enable_x64`
fixture in `tests/conftest.py` is the wrong tool for a setting that has to hold for the
whole session.
"""

import jax

jax.config.update("jax_enable_x64", True)

import pytest  # noqa: E402
from jax import Array  # noqa: E402

from . import config  # noqa: E402
from .matrices import Matrix, MatrixSpec, generate, generate_rhs  # noqa: E402

# Matrix specs, ordered so pytest keeps consecutive benchmarks on the same matrix. The
# generated matrices are cached for the whole session, so ordering only affects locality,
# not how often generation runs.
MATRIX_SPECS: list[MatrixSpec] = [
    MatrixSpec(family=family, n=n, replicate=replicate)
    for family in config.FAMILIES
    for n in config.SIZES
    for replicate in range(config.REPLICATES)
]


@pytest.fixture(scope="session", params=MATRIX_SPECS, ids=lambda spec: spec.id)
def matrix(request: pytest.FixtureRequest) -> Matrix:
    """One generated matrix, built once per spec for the whole session.

    Session-scoped and parametrized, so pytest caches one instance per spec: the same
    matrix serves every (solver, mode, n_rhs) benchmark at that spec instead of being
    rebuilt for each. Total cached data is a few MB, since the largest matrix has about
    60000 nonzeros.
    """
    return generate(request.param)


@pytest.fixture(scope="session", params=config.N_RHS, ids=lambda k: f"rhs{k}")
def n_rhs(request: pytest.FixtureRequest) -> int:
    """Number of right-hand sides to solve for."""
    return request.param


@pytest.fixture(scope="session")
def rhs(matrix: Matrix, n_rhs: int) -> Array:
    """The right-hand side block, cached per (actual size, count).

    Keyed on size rather than on the matrix, and deliberately not on solver or mode, so
    every configuration at a given size solves the identical system and the comparisons
    are like for like. It depends on `matrix` only to learn the actual size, which
    `grid2d` rounds.
    """
    return generate_rhs(matrix.n, n_rhs)
