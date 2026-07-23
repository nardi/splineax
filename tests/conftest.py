import jax
import pytest


@pytest.fixture
def enable_x64():
    """Enable JAX's 64-bit mode for a test's duration.

    `KLU`/`Pardiso` (via klujax/pardiso_mkl_jax) require x64 but no longer enable it as
    an import side effect, so any test that solves through them must request this
    fixture (or otherwise scope `jax.enable_x64(True)` itself).
    """
    with jax.enable_x64(True):
        yield
