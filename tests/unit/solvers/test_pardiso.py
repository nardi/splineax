"""Pardiso-specific tests for factorization reuse, availability, and token lifecycle.

`Pardiso()` requires the optional `pardiso-mkl-jax` dependency: the availability check is
always exercised (via monkeypatching), while the reuse tests are skipped when it is not
installed.

The solver-agnostic contract lives in test_factorization.py, the generic solve suite in
test_solvers.py, and `AutoSparseLinearSolver`'s dispatch in test_auto.py.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import splineax as splx
import splineax.solvers._pardiso as _pardiso_module
from splineax import KLU, BCOOLinearOperator, Pardiso

from .conftest import (
    RIGHT_HAND_SIDE,
    SQUARE_MATRIX,
    ZERO_DIAGONAL_MATRIX,
    ZERO_DIAGONAL_RIGHT_HAND_SIDE,
    OperatorFactory,
)

# Pardiso requires 64-bit mode but does not enable it as an import side effect, so every
# test here gets it from the shared `enable_x64` fixture (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("enable_x64")


def test_pardiso_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Pardiso()` raises `ImportError` when `pardiso_mkl_jax` is not installed.

    Monkeypatches `_pardiso_available`, so this runs regardless of whether the real
    (optional) dependency is installed here.
    """
    monkeypatch.setattr(_pardiso_module, "_pardiso_available", lambda: False)
    with pytest.raises(ImportError, match="pardiso-mkl-jax"):
        Pardiso()


pytest.importorskip("pardiso_mkl_jax")

from pardiso_mkl_jax import _ffi, primitive  # noqa: E402


@contextmanager
def _spy(name: str) -> Generator[list[bool], None, None]:
    """Intercept a `pardiso_mkl_jax.primitive` function and record every call.

    `_pardiso.py` looks these up on the `primitive` module at call time, so patching the
    attribute intercepts every call. A `list[bool]` keeps the log truthful under jit.
    """
    call_log: list[bool] = []
    original = getattr(primitive, name)

    def spy(*args, **kwargs):
        call_log.append(True)
        return original(*args, **kwargs)

    setattr(primitive, name, spy)
    try:
        yield call_log
    finally:
        setattr(primitive, name, original)


def test_init_factors_then_solves_stateful(make_operator: OperatorFactory) -> None:
    """`init` analyzes and factorizes once, then `compute` reuses the factorization
    through `solve_stateful`."""
    operator = make_operator(SQUARE_MATRIX)
    solver = Pardiso()
    with (
        _spy("analyze") as analyze_calls,
        _spy("factor") as factor_calls,
        _spy("solve_stateful") as solve_calls,
    ):
        state = solver.init(operator, {})
        solver.compute(state, RIGHT_HAND_SIDE, {})
    assert len(analyze_calls) == 1
    assert len(factor_calls) == 1
    assert len(solve_calls) == 1


def test_update_same_pattern_reuses_analysis() -> None:
    """`update` on an operator sharing the sparsity tag refactors while reusing the
    analysis, so `analyze` runs once, `factor` runs per matrix, and the native
    analysis count stays at one."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(SQUARE_MATRIX))
    first = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX), tags=tag)
    second_matrix = 2.0 * SQUARE_MATRIX
    second = BCOOLinearOperator(BCOO.fromdense(second_matrix), tags=tag)
    solver = Pardiso()
    with _spy("analyze") as analyze_calls:
        state = solver.init(first, {})
        updated = solver.update(state, second)
    assert len(analyze_calls) == 1, "update re-ran the analysis for a matching pattern"
    # The native analysis count stays at one, so the well-conditioned refactor reused the
    # analysis rather than falling back to a reanalyze.
    assert _ffi.analysis_count(updated.token.id) == 1
    solution = lx.linear_solve(
        second, RIGHT_HAND_SIDE, solver=solver, state=updated
    ).value
    solver.release(updated)
    expected = jnp.linalg.solve(np.asarray(second_matrix), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_init_symbolic_defers_analysis() -> None:
    """Under Pardiso's default weighted matching, `init_symbolic` defers analysis: it
    carries no token, and the first `update` runs analyze and factor."""
    solver = Pardiso()
    state = solver.init_symbolic(BCOO.fromdense(SQUARE_MATRIX))
    assert state.token is None
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    with _spy("analyze") as analyze_calls, _spy("factor") as factor_calls:
        updated = solver.update(state, operator)
    assert len(analyze_calls) == 1
    assert len(factor_calls) == 1
    assert updated.token is not None


def test_transpose_reuses_factorization() -> None:
    """`transpose` reuses the same token and solves A^T through `solve_stateful`, with no
    extra analyze or factor."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = Pardiso()
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )
    state = solver.init(operator, {})
    with _spy("analyze") as analyze_calls, _spy("factor") as factor_calls:
        transposed, _ = solver.transpose(state, {})
        assert transposed.token is state.token
        solution = solver.compute(transposed, RIGHT_HAND_SIDE, {})[0]
    assert not analyze_calls, "transpose re-analyzed the pattern"
    assert not factor_calls, "transpose re-factored the matrix"
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_conj_real_is_a_no_op() -> None:
    """`Pardiso` is real-only, so `conj` returns the state unchanged."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = Pardiso()
    state = solver.init(operator, {})
    conjugated, _ = solver.conj(state, {})
    assert conjugated is state


def test_release_frees_the_handle() -> None:
    """`release` calls `primitive.release` once, freeing the native factorization."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = Pardiso()
    with _spy("release") as release_calls:
        state = solver.init(operator, {})
        solver.release(state)
    assert len(release_calls) == 1


def test_zero_diagonal_matrix_solves_accurately() -> None:
    """A matrix with zeros on its diagonal must solve to a small residual.

    A regression test for pardiso-mkl-jax's `iparm` defaults, exercised from the splineax
    side. Pardiso needs weighted matching to factor a matrix like this stably. Without it,
    it perturbs the tiny pivots on the zero diagonal and returns a solution whose residual
    is many orders of magnitude too large, while still reporting success, so the residual
    is the only thing that catches it. Both the direct `init` path and the deferred
    `init_symbolic` then `update` path are covered.
    """
    matrix = jnp.asarray(ZERO_DIAGONAL_MATRIX)
    vector = jnp.asarray(ZERO_DIAGONAL_RIGHT_HAND_SIDE)
    operator = BCOOLinearOperator(BCOO.fromdense(matrix))
    solver = Pardiso()

    def residual(solution: jnp.ndarray) -> float:
        return float(jnp.abs(matrix @ solution - vector).max())

    # KLU first, as an independent check that the system itself is well posed.
    assert residual(lx.linear_solve(operator, vector, solver=KLU()).value) < 1e-10

    direct, state = splx.linear_solve(operator, vector, solver)
    solver.release(state)
    assert residual(direct.value) < 1e-10, "init path perturbed its pivots"

    symbolic_state = solver.update(solver.init_symbolic(operator), operator)
    symbolic = lx.linear_solve(
        operator, vector, solver=solver, state=symbolic_state
    ).value
    solver.release(symbolic_state)
    assert residual(symbolic) < 1e-10, "deferred symbolic path perturbed its pivots"


def test_update_reuse_stays_accurate_on_matching_sensitive_values() -> None:
    """Reusing the analysis for values that break the previous weighted matching must
    still solve accurately, because `update` reanalyzes when the reused factorization
    perturbs its pivots. The zero-diagonal matrix needs weighted matching, so scrambling
    its values makes the first analysis a poor fit for the second."""
    matrix = jnp.asarray(ZERO_DIAGONAL_MATRIX)
    vector = jnp.asarray(ZERO_DIAGONAL_RIGHT_HAND_SIDE)
    indices = BCOO.fromdense(matrix).indices
    data = BCOO.fromdense(matrix).data
    scrambled = data * jnp.asarray(
        np.random.default_rng(2).uniform(-3.0, 3.0, size=data.shape)
    )
    tag = splx.sparsity_pattern_tag(BCOO((data, indices), shape=matrix.shape))
    first = BCOOLinearOperator(
        BCOO((data, indices), shape=matrix.shape, indices_sorted=True), tags=tag
    )
    second_bcoo = BCOO((scrambled, indices), shape=matrix.shape, indices_sorted=True)
    second = BCOOLinearOperator(second_bcoo, tags=tag)

    solver = Pardiso()
    state = solver.init(first, {})
    state = solver.update(state, second)
    solution = lx.linear_solve(second, vector, solver=solver, state=state).value
    solver.release(state)

    residual = float(jnp.abs(second_bcoo.todense() @ solution - vector).max())
    assert residual < 1e-8
