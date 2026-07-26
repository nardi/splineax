"""How each sparse solver scales, across matrix families, sizes and reuse tiers.

The suite measures the full cross product of solver, reuse mode, right-hand-side count,
matrix family, replicate and size. What lands inside the timer depends on the reuse mode:
whatever that tier says is already available is set up in a fixture instead.

Two results look like harness bugs and are not. Both are explained at length in
[README.md](README.md), and briefly here:

- `Spsolve`'s reuse tiers are no-ops, so its `none` and `symbolic` timings should agree.
  That agreement is a sanity check on the harness, not a redundancy.
- `symbolic` with 100 right-hand sides is 100 numeric factorizations, because that tier
  fuses refactorization into each solve. It answers "100 matrices sharing one pattern",
  not "one matrix with 100 right-hand sides", which is the `numeric` tier.
"""

from contextlib import ExitStack
from typing import Any, Iterator

import pytest

from . import cases, config
from .matrices import Matrix

pytestmark = pytest.mark.benchmark

# `Pardiso` is an optional extra, guarded the same way as in
# `tests/unit/solvers/conftest.py`.
try:
    from splineax.solvers._pardiso import _pardiso_available

    PARDISO_AVAILABLE = _pardiso_available()
except ImportError:  # pragma: no cover
    PARDISO_AVAILABLE = False

SOLVER_PARAMS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            name == "pardiso" and not PARDISO_AVAILABLE,
            reason="pardiso-mkl-jax is not installed",
        ),
        id=name,
    )
    for name in config.SOLVERS
]


@pytest.fixture
def case(
    request: pytest.FixtureRequest,
    matrix: Matrix,
    rhs: Any,
    n_rhs: int,
    solver_name: str,
    mode: str,
) -> Iterator[cases.Case]:
    """Everything the reuse mode provides, set up outside the timer.

    The `ExitStack` is the important part: exiting a `factorize` or `factorize_symbolic`
    block frees the native handle, so the scope has to stay open across every timed call
    and is only unwound at teardown.
    """
    with ExitStack() as stack:
        built = cases.build_case(
            stack=stack,
            solver_name=solver_name,
            mode=mode,
            n_rhs=n_rhs,
            matrix=matrix,
            rhs=rhs,
        )
        if built.residual > config.RESIDUAL_TOL:
            pytest.fail(
                f"relative residual {built.residual:.3e} exceeds "
                f"{config.RESIDUAL_TOL:.0e} for {solver_name}/{mode}/"
                f"{matrix.spec.id}, so this timing would describe a wrong answer"
            )
        yield built


@pytest.fixture(params=SOLVER_PARAMS)
def solver_name(request: pytest.FixtureRequest) -> str:
    """Which solver to measure. `auto` is excluded, since it only dispatches to these."""
    return request.param


@pytest.fixture(params=config.MODES, ids=lambda mode: mode)
def mode(request: pytest.FixtureRequest) -> str:
    """Factorization-reuse tier, which decides what is set up outside the timer."""
    return request.param


def test_solve(benchmark: Any, case: cases.Case) -> None:
    """Time one (solver, mode, n_rhs) configuration on one matrix.

    Grouped by (mode, n_rhs) so the terminal table breaks into comparable blocks rather
    than one flat list of several hundred rows.
    """
    benchmark.group = f"{case.mode}-rhs{case.n_rhs}"
    benchmark.extra_info.update(case.extra)
    benchmark.extra_info["config"] = config.resolved()

    # Warm-up in `build_case` already compiled the timed function. Recording the cache
    # size here and checking it afterwards turns a silent recompilation, which would
    # inflate the first round, into a failure.
    before = case.cache_size()
    benchmark(case.run)
    after = case.cache_size()

    assert after == before, (
        f"the timed function recompiled during measurement (compile cache went from "
        f"{before} to {after}), so these timings include compilation"
    )
