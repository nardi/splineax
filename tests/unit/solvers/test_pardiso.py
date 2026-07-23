"""Pardiso-specific tests for factorization reuse, availability, and handle lifecycle.

`Pardiso()` requires the optional `pardiso-mkl-jax` dependency: the availability check
itself is always exercised (via monkeypatching, independent of whether the real package
is installed), while the factorization-reuse tests are skipped when it isn't.

The generic solve-correctness suite (all solvers, both operator formats) lives in
test_solvers.py. `AutoSparseLinearSolver`'s dispatch (including the Pardiso/KLU choice)
lives in test_auto.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import splineax.solvers._pardiso as _pardiso_module
from splineax import BCOOLinearOperator, Pardiso
from splineax.solvers._pardiso import _PardisoNumericState

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory

# Pardiso requires 64-bit mode but does not enable it as an import side effect, so
# every test in this module gets it from the shared `enable_x64` fixture
# (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("enable_x64")


def test_pardiso_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Pardiso()` must raise `ImportError` when `pardiso_mkl_jax` isn't installed.

    Monkeypatches `_pardiso_available` directly, so this runs regardless of whether
    the real (optional) dependency happens to be installed in this environment.
    """
    monkeypatch.setattr(_pardiso_module, "_pardiso_available", lambda: False)
    with pytest.raises(ImportError, match="pardiso-mkl-jax"):
        Pardiso()


pytest.importorskip("pardiso_mkl_jax")

import pardiso_mkl_jax as pmj  # noqa: E402


@contextmanager
def _spy(name: str) -> Generator[list[bool], None, None]:
    """Intercept a `pardiso_mkl_jax.PardisoSolver` method and record every call.

    A `list[bool]` rather than a counter, so the log stays truthful under JIT tracing
    (each trace-time call appends one entry, mirroring `test_klu.py`'s `_spy_solve`).
    """
    call_log: list[bool] = []
    original = getattr(pmj.PardisoSolver, name)

    def spy(self, *args, **kwargs):
        call_log.append(True)
        return original(self, *args, **kwargs)

    setattr(pmj.PardisoSolver, name, spy)
    try:
        yield call_log
    finally:
        setattr(pmj.PardisoSolver, name, original)


def test_factorize_closes_solver_on_exit(make_operator: OperatorFactory) -> None:
    """`Pardiso().factorize(operator)` must call `.factorize()` (not just `.analyze()`)
    exactly once, and close the underlying `PardisoSolver` when the block exits."""
    operator = make_operator(SQUARE_MATRIX)
    solver = Pardiso()

    with (
        _spy("factorize") as factorize_calls,
        _spy("close") as close_calls,
    ):
        with solver.factorize(operator) as state:
            assert not close_calls, "solver was closed before the block exited"
            solution = lx.linear_solve(
                operator, RIGHT_HAND_SIDE, solver=solver, state=state
            ).value

    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)
    assert factorize_calls, "PardisoSolver.factorize was not called"
    assert close_calls, "PardisoSolver was not closed when the factorize block exited"


def test_factorize_symbolic_reuses_analysis_across_init_calls() -> None:
    """Inside a `factorize_symbolic()` block, `.analyze()` must run only once across
    multiple `.init()` calls sharing the scope, while `.factorize()`/`.refactorize()`
    run once each, mirroring `pardiso_mkl_jax`'s own "analysis reused, numeric phase
    redone per values" contract."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = Pardiso()
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with (
        _spy("analyze") as analyze_calls,
        _spy("factorize") as factorize_calls,
        _spy("refactorize") as refactorize_calls,
        _spy("close") as close_calls,
    ):
        with solver.factorize_symbolic(BCOO.fromdense(SQUARE_MATRIX)) as scope:
            first_state = scope.init(operator)
            first_solution = solver.compute(first_state, RIGHT_HAND_SIDE, {})[0]

            second_state = scope.init(operator)
            second_solution = solver.compute(second_state, RIGHT_HAND_SIDE, {})[0]

            assert not close_calls, "solver was closed before the block exited"

    assert len(analyze_calls) == 1, (
        f"expected exactly 1 analyze() call, got {len(analyze_calls)}"
    )
    assert len(factorize_calls) == 1, (
        f"expected exactly 1 factorize() call, got {len(factorize_calls)}"
    )
    assert len(refactorize_calls) == 1, (
        f"expected exactly 1 refactorize() call, got {len(refactorize_calls)}"
    )
    assert close_calls, "PardisoSolver was not closed when the scope exited"
    assert jnp.allclose(first_solution, expected, atol=1e-5)
    assert jnp.allclose(second_solution, expected, atol=1e-5)


def test_transpose_reuses_factorization() -> None:
    """`solver.transpose()` must reuse the existing factorization: no extra
    `.analyze()`/`.factorize()`/`.refactorize()` calls, just a `.solve(transpose=True)`
    against the same handle."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = Pardiso()
    expected_transpose = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )

    with solver.init(operator, {}).factorize() as state:
        with (
            _spy("analyze") as analyze_calls,
            _spy("factorize") as factorize_calls,
            _spy("refactorize") as refactorize_calls,
        ):
            transposed_state, _ = solver.transpose(state, options={})
            assert isinstance(transposed_state, _PardisoNumericState)
            assert transposed_state.handle is state.handle, (
                "transpose() should reuse the same handle, not build a new one"
            )
            solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, options={})[0]

    assert not analyze_calls, "transpose() should not re-analyze"
    assert not factorize_calls, "transpose() should not re-factorize"
    assert not refactorize_calls, "transpose() should not refactorize"
    assert jnp.allclose(solution, expected_transpose, atol=1e-5), (
        "transposed solve produced an incorrect solution"
    )


def test_conj_real_is_noop() -> None:
    """For a real matrix, `solver.conj()` must return the state unchanged: `Pardiso`
    only supports real matrices, so conjugation is always the identity."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = Pardiso()

    with solver.init(operator, {}).factorize() as state:
        conj_state, _ = solver.conj(state, options={})
        assert conj_state is state
