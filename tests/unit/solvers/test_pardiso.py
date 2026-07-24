"""Pardiso-specific tests for factorization reuse, availability, and handle lifecycle.

`Pardiso()` requires the optional `pardiso-mkl-jax` dependency: the availability check
itself is always exercised (via monkeypatching, independent of whether the real package
is installed), while the factorization-reuse tests are skipped when it isn't.

The solver-agnostic factorization-reuse contract (correctness, reuse, transpose, passing
states into a jitted function) lives in test_factorization.py, the generic solve suite in
test_solvers.py, and `AutoSparseLinearSolver`'s dispatch (including the Pardiso/KLU choice)
in test_auto.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import equinox as eqx
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

from pardiso_mkl_jax import _ffi, primitive  # noqa: E402


@contextmanager
def _spy(name: str) -> Generator[list[bool], None, None]:
    """Intercept a `pardiso_mkl_jax.primitive` function and record every call.

    A `list[bool]` rather than a counter, so the log stays truthful under JIT tracing
    (each trace-time call appends one entry, mirroring `test_klu.py`'s `_spy_solve`).
    """
    call_log: list[bool] = []
    original = getattr(primitive, name)

    def spy(*args, **kwargs):
        call_log.append(True)
        return original(*args, **kwargs)

    setattr(primitive, name, spy)
    # `_pardiso.py` looks these functions up on the `primitive` module at call time
    # (through `_pardiso_mkl_jax().primitive`), so patching the module attribute here
    # is enough to intercept every call.
    try:
        yield call_log
    finally:
        setattr(primitive, name, original)


def test_factorize_closes_solver_on_exit(make_operator: OperatorFactory) -> None:
    """`Pardiso().factorize(operator)` must call `.factor()` (not just `.analyze()`)
    exactly once, and release the underlying handle when the block exits."""
    operator = make_operator(SQUARE_MATRIX)
    solver = Pardiso()

    with (
        _spy("factor") as factor_calls,
        _spy("release") as release_calls,
    ):
        with solver.factorize(operator) as state:
            assert not release_calls, "handle was released before the block exited"
            solution = lx.linear_solve(
                operator, RIGHT_HAND_SIDE, solver=solver, state=state
            ).value

    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)
    assert factor_calls, "primitive.factor was not called"
    assert release_calls, "the handle was not released when the factorize block exited"


def test_factorize_symbolic_reuses_analysis_across_solves() -> None:
    """A `factorize_symbolic` scope analyses once and reuses it across solves, redoing the
    numeric phase per call through `factor_and_solve_stateful`, and releasing on scope
    exit."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = Pardiso()
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with (
        _spy("analyze") as analyze_calls,
        _spy("factor_and_solve_stateful") as factor_and_solve_calls,
        _spy("release") as release_calls,
    ):
        with solver.factorize_symbolic(BCOO.fromdense(SQUARE_MATRIX)) as scope:
            handle = scope.handle

            first_state = scope.init(operator)
            first_solution = solver.compute(first_state, RIGHT_HAND_SIDE, {})[0]

            second_state = scope.init(operator)
            second_solution = solver.compute(second_state, RIGHT_HAND_SIDE, {})[0]

            assert not release_calls, "handle was released before the block exited"
            # Ground truth from the native side: the symbolic phase ran once.
            assert _ffi.analysis_count(handle) == 1

    assert len(analyze_calls) == 1, (
        f"expected exactly 1 analyze() call, got {len(analyze_calls)}"
    )
    assert len(factor_and_solve_calls) == 2, (
        f"expected 2 factor_and_solve_stateful() calls, got {len(factor_and_solve_calls)}"
    )
    assert release_calls, "the handle was not released when the scope exited"
    assert jnp.allclose(first_solution, expected, atol=1e-5)
    assert jnp.allclose(second_solution, expected, atol=1e-5)


def test_symbolic_scope_reused_under_jit_analyses_once() -> None:
    """The core requirement: open the symbolic factorization scope eagerly, then reuse the
    scope inside a jitted function across different values. The analysis must run exactly
    once (analysis_count stays 1) and every solve must be correct."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape
    solver = Pardiso()
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    other_matrix = 2.0 * SQUARE_MATRIX
    other_expected = jnp.linalg.solve(
        np.asarray(other_matrix), np.asarray(RIGHT_HAND_SIDE)
    )

    @eqx.filter_jit
    def run(scope, data, b):
        operator = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        state = scope.init(operator)
        return lx.linear_solve(operator, b, solver=solver, state=state).value

    with solver.factorize_symbolic(sparsity) as scope:
        handle = scope.handle
        solution = np.asarray(run(scope, sparsity.data, RIGHT_HAND_SIDE))
        other_solution = np.asarray(run(scope, 2.0 * sparsity.data, RIGHT_HAND_SIDE))
        assert _ffi.analysis_count(handle) == 1

    assert jnp.allclose(solution, expected, atol=1e-5)
    assert jnp.allclose(other_solution, other_expected, atol=1e-5)


def test_symbolic_scope_handle_released_exactly_once() -> None:
    """The handle a `factorize_symbolic` scope allocates is released exactly once, on
    scope exit, whether or not the scope's state is ever used inside a jitted function."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    solver = Pardiso()

    with _spy("release") as release_calls:
        with solver.factorize_symbolic(sparsity) as scope:
            handle = scope.handle
            state = scope.init(BCOOLinearOperator(sparsity))
            solver.compute(state, RIGHT_HAND_SIDE, {})
            assert not release_calls

    assert len(release_calls) == 1, (
        f"expected exactly 1 release() call, got {len(release_calls)}"
    )
    # The handle's native registry entry is gone, so the analysis-count hook falls
    # back to reporting 0 for it.
    assert _ffi.analysis_count(handle) == 0


def test_transpose_reuses_factorization() -> None:
    """`solver.transpose()` must reuse the existing factorization: no extra
    `.analyze()`/`.factor()` calls, just a `solve_stateful(transpose=True)` against the
    same handle."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = Pardiso()
    expected_transpose = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )

    with solver.init(operator, {}).factorize() as state:
        with (
            _spy("analyze") as analyze_calls,
            _spy("factor") as factor_calls,
        ):
            transposed_state, _ = solver.transpose(state, options={})
            assert isinstance(transposed_state, _PardisoNumericState)
            assert transposed_state.handle is state.handle, (
                "transpose() should reuse the same handle, not build a new one"
            )
            solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, options={})[0]

    assert not analyze_calls, "transpose() should not re-analyze"
    assert not factor_calls, "transpose() should not re-factor"
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
