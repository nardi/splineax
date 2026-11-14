from collections.abc import Callable
from typing import Any

import jax
import pytest

import splineax as splx
from splineax.solvers._auto import _cuda_backend_available
from splineax.solvers._cudss import _cudss_available

from .fake_cudss import fake_cudss as fake_cudss


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


class SparseSolverCase:
    """A sparse stateful solver and the way to count its factorization rebuilds."""

    def __init__(self, solver_name: str) -> None:
        self.solver_name = solver_name
        """The name of the solver class, as a profile record names it."""

    def make_solver(self) -> Any:
        """Return a new solver of this kind."""
        return getattr(splx, self.solver_name)()

    def rebuild_total(self) -> int:
        """Return how many times the backing library rebuilt a factorization so far.

        A rebuild is the cost of a use of a factorization that was freed or evicted. The
        total counts since the library loaded, so a test compares two readings.
        """
        # Each library is optional, so a solver imports its own library when a test asks.
        counter: Callable[[], dict[Any, int]]
        match self.solver_name:
            case "KLU":
                import klujax

                counter = klujax.rebuild_stats
            case "Pardiso":
                from pardiso_mkl_jax import primitive

                counter = primitive.rebuild_stats
            case "CuDSS":
                import spineax.cudss as cudss

                return int(cudss.rebuild_count())
            case _:
                raise ValueError(f"Unknown sparse solver: {self.solver_name}.")
        return sum(count for reason, count in counter().items() if int(reason) != 0)


@pytest.fixture(
    params=[
        pytest.param("KLU", marks=pytest.mark.cpu_only),
        pytest.param("Pardiso", marks=pytest.mark.cpu_only),
        pytest.param("CuDSS", marks=pytest.mark.cudss_gpu),
    ]
)
def sparse_solver(request: pytest.FixtureRequest) -> SparseSolverCase:
    """Each sparse solver that keeps a factorization, with the marks that skip it.

    `KLU` and `Pardiso` run on the CPU and `CuDSS` runs on a CUDA GPU, see
    `pytest_runtest_setup`.
    """
    return SparseSolverCase(request.param)
