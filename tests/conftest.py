import jax
import pytest

from splineax.solvers._auto import _cuda_backend_available
from splineax.solvers._cudss import _cudss_available


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "cpu_only: skip unless JAX's default backend is CPU, for tests that solve "
        "through the CPU-only `KLU` or `Pardiso`",
    )
    config.addinivalue_line(
        "markers",
        "cudss_gpu: skip unless the optional cuDSS dependency is installed and a CUDA "
        "GPU is visible, for tests that solve through the real `CuDSS`",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a `cpu_only` or `cudss_gpu` test on a machine that cannot run it.

    `KLU`/`Pardiso` wrap CPU-only libraries and fail at trace time on any other platform,
    so on a GPU machine their tests skip. Nothing in the suite pins arrays to a device, so
    the default backend decides. `CuDSS` needs its optional dependency and a CUDA device,
    so its tests skip everywhere else.
    """
    if item.get_closest_marker("cpu_only") and jax.default_backend() != "cpu":
        pytest.skip("`KLU`/`Pardiso` are CPU-only and JAX's default backend is not CPU")
    if item.get_closest_marker("cudss_gpu") and not (
        _cudss_available() and _cuda_backend_available()
    ):
        pytest.skip(
            "the optional cuDSS dependency is not installed, or no CUDA GPU is visible"
        )


@pytest.fixture
def enable_x64():
    """Enable JAX's 64-bit mode for a test's duration.

    `KLU`/`Pardiso` (via klujax/pardiso_mkl_jax) require x64 but no longer enable it as
    an import side effect, so any test that solves through them must request this
    fixture (or otherwise scope `jax.enable_x64(True)` itself).
    """
    with jax.enable_x64(True):
        yield
