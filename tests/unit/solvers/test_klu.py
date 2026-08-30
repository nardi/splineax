"""KLU-specific tests for factorization reuse and token lifecycle.

These check behaviour unique to `KLU`: `init` analyzes and factorizes so `compute` reuses
the numeric factorization through `solve_with_numeric`, `update` on a matching pattern
reuses the symbolic token and refactors, `transpose` reuses the factorization through
`tsolve_with_numeric`, `conj` reuses the symbolic token, and `release` frees both handles.

The solver-agnostic contract lives in [test_factorization.py](test_factorization.py) and
the basic solve suite in [test_solvers.py](test_solvers.py).
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
from splineax import KLU, BCOOLinearOperator
from splineax.solvers._klu import _KLUState

from .conftest import COMPLEX_MATRIX, RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory

# KLU requires 64-bit mode but no longer enables it as an import side effect, so every
# test here gets it from the shared `enable_x64` fixture (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("enable_x64")


@contextmanager
def _spy(function_name: str) -> Generator[list[bool], None, None]:
    """Record every call to a named function on the `klujax` module.

    `KLU` reaches its klujax functions through `_klujax()` (the module) at call time, so
    replacing a module attribute intercepts both eager and traced paths. A `list[bool]`
    keeps the log truthful under jit, where each trace-time call appends one entry.
    """
    import klujax as klu

    call_log: list[bool] = []
    original = getattr(klu, function_name)

    def spy_function(*args, **kwargs):
        call_log.append(True)
        return original(*args, **kwargs)

    setattr(klu, function_name, spy_function)
    try:
        yield call_log
    finally:
        setattr(klu, function_name, original)


def test_init_computes_with_solve_with_numeric(
    make_operator: OperatorFactory,
) -> None:
    """`init` factorizes eagerly, so `compute` reuses it through `solve_with_numeric`
    rather than the one-shot `solve`."""
    operator = make_operator(SQUARE_MATRIX)
    solver = KLU()
    with _spy("solve_with_numeric") as numeric_calls, _spy("solve") as solve_calls:
        state = solver.init(operator, {})
        solver.compute(state, RIGHT_HAND_SIDE, {})
    assert numeric_calls, "compute did not reuse the numeric factorization"
    assert not solve_calls, "compute used the one-shot klujax.solve"


def test_update_same_pattern_reuses_symbol_and_refactors() -> None:
    """`update` on an operator sharing the sparsity tag reuses the symbolic token and
    the previous numeric factorization, so `analyze` runs once and the pivot-reusing
    `refactor_with_status` is used rather than a fresh analysis."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(SQUARE_MATRIX))
    first = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX), tags=tag)
    second = BCOOLinearOperator(BCOO.fromdense(2.0 * SQUARE_MATRIX), tags=tag)
    solver = KLU()
    with (
        _spy("analyze") as analyze_calls,
        _spy("refactor_with_status") as refactor_calls,
    ):
        state = solver.init(first, {})
        updated = solver.update(state, second)
    assert updated.symbol is state.symbol, "update did not reuse the symbolic token"
    assert len(analyze_calls) == 1, "update re-analyzed a matching pattern"
    assert refactor_calls, "update did not attempt a pivot-reusing refactor"


def _square_jacobian_function(x: jnp.ndarray, args: object) -> jnp.ndarray:
    """A square nonlinear map with an invertible banded Jacobian."""
    del args
    return 3.0 * x + x**2 + 0.5 * jnp.roll(x, 1) * x


def test_update_across_jacobian_points_reuses_analysis() -> None:
    """Operators from one `operator_at` factory carry a pattern tag from their shared
    coloring, so `update` across evaluation points reuses the analysis, and a BCOO
    materialised from such an operator does too, so `analyze` runs once each."""
    point = jnp.linspace(0.5, 1.5, 5)
    factory = splx.SparseJacobianLinearOperatorColoring.detect(
        _square_jacobian_function, point
    )
    first = factory.operator_at(point)
    second = factory.operator_at(point + 0.3)
    solver = KLU()
    with _spy("analyze") as analyze_calls:
        state = solver.init(first, {})
        across_points = solver.update(state, second)
        across_materialised = solver.update(state, lx.materialise(second))
    assert across_points.symbol is state.symbol, "update re-analyzed another point"
    assert across_materialised.symbol is state.symbol, (
        "update re-analyzed a materialised operator"
    )
    assert len(analyze_calls) == 1, (
        "the shared Jacobian pattern was analyzed more than once"
    )


def test_update_falls_back_when_reused_pivots_go_bad() -> None:
    """When new values leave the reused pivots badly scaled, the guarded refactor falls
    back to a fresh factor, so the solve stays accurate. The second matrix zeros out a
    diagonal entry the first factorization pivoted on, which a plain in-place refactor
    would handle poorly."""
    first_dense = SQUARE_MATRIX
    second_dense = SQUARE_MATRIX.at[0, 0].set(1e-9)
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(first_dense))
    first = BCOOLinearOperator(BCOO.fromdense(first_dense), tags=tag)
    second = BCOOLinearOperator(BCOO.fromdense(second_dense), tags=tag)
    solver = KLU()
    state = solver.init(first, {})
    state = solver.update(state, second)
    solution = lx.linear_solve(
        second, RIGHT_HAND_SIDE, solver=solver, state=state
    ).value
    state.release()
    expected = jnp.linalg.solve(np.asarray(second_dense), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-6)


def test_transpose_reuses_factorization_via_tsolve() -> None:
    """`transpose` reuses the same tokens and solves A^T through `tsolve_with_numeric`."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = KLU()
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )
    with _spy("tsolve_with_numeric") as tsolve_calls, _spy("analyze") as analyze_calls:
        state = solver.init(operator, {})
        transposed, _ = solver.transpose(state, {})
        assert transposed.symbol is state.symbol
        assert transposed.numeric is state.numeric
        solution = solver.compute(transposed, RIGHT_HAND_SIDE, {})[0]
    assert tsolve_calls, "transpose did not use tsolve_with_numeric"
    assert len(analyze_calls) == 1, "transpose re-analyzed the pattern"
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_conj_real_is_a_no_op() -> None:
    """For a real matrix, `conj` returns the same state unchanged."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = KLU()
    state = solver.init(operator, {})
    conjugated, _ = solver.conj(state, {})
    assert conjugated is state


def test_conj_complex_reuses_symbol_creates_new_numeric() -> None:
    """For a complex matrix, `conj` reuses the symbolic token (same sparsity) and builds a
    fresh numeric token for conj(A)."""
    operator = BCOOLinearOperator(BCOO.fromdense(COMPLEX_MATRIX))
    solver = KLU()
    state = solver.init(operator, {})
    conjugated, _ = solver.conj(state, {})
    assert isinstance(conjugated, _KLUState)
    assert conjugated.symbol is state.symbol, "conj did not reuse the symbolic token"
    assert conjugated.numeric is not state.numeric, "conj reused the old numeric token"
    assert conjugated.coo is not None and state.coo is not None
    assert jnp.allclose(conjugated.coo[2], state.coo[2].conj())


def test_release_frees_both_handles() -> None:
    """`release` frees the symbolic and the numeric cache slot once each."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = KLU()
    with _spy("free_symbolic") as symbolic_frees, _spy("free_numeric") as numeric_frees:
        state = solver.init(operator, {})
        state.release()
    assert len(symbolic_frees) == 1
    assert len(numeric_frees) == 1


def test_init_symbolic_defers_numeric() -> None:
    """`init_symbolic` analyzes only, so the state carries a symbolic token but no numeric
    one until `update` folds in an operator."""
    solver = KLU()
    state = solver.init_symbolic(BCOO.fromdense(SQUARE_MATRIX))
    assert state.numeric is None
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    updated = solver.update(state, operator)
    assert updated.numeric is not None
