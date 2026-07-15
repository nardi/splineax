"""KLU-specific tests for factorization reuse and handle lifecycle.

This module tests behaviour that is unique to the `KLU` solver's `factorize()`
context manager: that solves inside the block call `solve_with_numeric` /
`tsolve_with_numeric` rather than `klujax.solve`, and that every allocated
handle (symbolic and numeric) is freed exactly once when the block exits.

The generic solve-correctness suite (both solvers, both operator formats) lives
in `test_solvers.py`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from splineax import KLU, BCOOLinearOperator
from splineax.solvers._klu import _KLUNumericState, _KLUSymbolicState

from .conftest import (
    COMPLEX_MATRIX,
    RIGHT_HAND_SIDE,
    SQUARE_MATRIX,
    OperatorFactory,
)

# ---------------------------------------------------------------------------
# Spy helpers
# ---------------------------------------------------------------------------


@contextmanager
def _spy_frees() -> Generator[tuple[list[int], list[int]], None, None]:
    """Context manager that intercepts every `klujax.free_symbolic` and
    `klujax.free_numeric` call made from `KLUHandleAllocationScopeManager`
    and records the Python `id` of the underlying handle.

    Why we intercept at the Python-wrapper level (not at `free_symbolic_p.bind`):
    Under JIT, `klujax.free_symbolic` passes the handle through
    `lax.optimization_barrier`, which produces a *new* Tracer. The id of that
    new Tracer does not match the id of the original `manager.handle`, so
    spying on the primitive's `.bind` would fail the id-equality assertions.
    Intercepting the wrapper functions directly lets us read `manager.handle`
    before the barrier is applied, giving us the same id as
    `numeric_state.factorization.symbolic/numeric`.

    Why we only record when the first argument is a `KLUHandleManager`:
    `KLUHandleAllocationScopeManager.begin_scope()` calls
    `klujax.free_symbolic(manager, deps)` with a `KLUHandleManager`.
    Inside that call, `manager.close()` triggers a *recursive* call to
    `klujax.free_symbolic(raw_handle)` (because `close()` uses
    `self.free_callable` which is the same function captured at manager
    creation time). That recursive call receives a raw Array, not a
    `KLUHandleManager`. The `isinstance` guard below prevents
    double-recording the same handle from this recursive call.

    Under JIT, `manager.close()` short-circuits (the handle is a Tracer and
    cannot be eagerly freed), so `begin_scope()` is the only call site that
    reaches the spy with a `KLUHandleManager`.  In both cases,
    `id(manager.handle) == id(numeric_state.factorization.symbolic/numeric)`.
    """
    import klujax as klu

    freed_symbolic_handle_ids: list[int] = []
    freed_numeric_handle_ids: list[int] = []

    original_free_symbolic = klu.free_symbolic
    original_free_numeric = klu.free_numeric

    def spy_free_symbolic(symbolic_or_handle, dependency=None):
        if isinstance(symbolic_or_handle, klu.KLUHandleManager):
            freed_symbolic_handle_ids.append(id(symbolic_or_handle.handle))
        return original_free_symbolic(symbolic_or_handle, dependency)

    def spy_free_numeric(numeric_or_handle, dependency=None):
        if isinstance(numeric_or_handle, klu.KLUHandleManager):
            freed_numeric_handle_ids.append(id(numeric_or_handle.handle))
        return original_free_numeric(numeric_or_handle, dependency)

    # Replace the module-level attributes so that both `begin_scope()` and any
    # newly-created `KLUHandleManager` (which captures `free_callable` from
    # the module at construction time inside `klujax.analyze` / `klujax.factor`)
    # see the spy.
    klu.free_symbolic = spy_free_symbolic  # type: ignore
    klu.free_numeric = spy_free_numeric  # type: ignore
    try:
        yield freed_symbolic_handle_ids, freed_numeric_handle_ids
    finally:
        klu.free_symbolic = original_free_symbolic
        klu.free_numeric = original_free_numeric


@contextmanager
def _spy_solve(function_name: str) -> Generator[list[bool], None, None]:
    """Context manager that intercepts a named function on the `klujax` module
    and records every invocation.

    `KLU.compute` accesses solve functions via `_klujax()` (which returns the
    `klujax` module object) at call time.  Replacing the module attribute here
    therefore intercepts both non-JIT and JIT-traced paths: JAX tracing
    executes Python code at trace time, so the spy fires during tracing.

    We use a `list[bool]` rather than an integer counter so that the spy is
    mutation-safe and the call log remains truthful under JIT (each trace-time
    call appends one entry).
    """
    import klujax as klu

    call_log: list[bool] = []
    original_function = getattr(klu, function_name)

    def spy_function(*args, **kwargs):
        call_log.append(True)
        return original_function(*args, **kwargs)

    setattr(klu, function_name, spy_function)
    try:
        yield call_log
    finally:
        setattr(klu, function_name, original_function)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_jit", [False, True], ids=["no_jit", "jit"])
def test_factorize_reuses_numeric_solve(
    make_operator: OperatorFactory,
    use_jit: bool,
) -> None:
    """Inside a `factorize()` block, `compute` must call `solve_with_numeric`
    (not `klujax.solve`), and both the symbolic and numeric handles must be
    freed when the block exits.  Verified for both non-JIT and JIT execution.
    """
    operator = make_operator(SQUARE_MATRIX)
    solver = KLU()

    with (
        _spy_frees() as (freed_symbolic_ids, freed_numeric_ids),
        _spy_solve("solve") as solve_calls,
        _spy_solve("solve_with_numeric") as solve_with_numeric_calls,
    ):
        # We capture handle ids via a side-effectful list append rather than
        # returning them from `run`, because `jax.jit` only accepts JAX arrays
        # as return values — Python ints (from `id()`) cannot pass through JIT.
        # The append fires at Python/trace time for both JIT and non-JIT, so
        # `captured_handle_ids` is populated before the assertions below run.
        captured_handle_ids: list[tuple[int, int]] = []

        def run(right_hand_side):
            state_init = solver.init(operator, options={})
            with state_init.factorize() as numeric_state:
                # Under JIT these are Tracer object ids; the spy in _spy_frees
                # captures the same Tracer ids when begin_scope() exits.
                captured_handle_ids.append(
                    (
                        id(numeric_state.factorization.symbolic),
                        id(numeric_state.factorization.numeric),
                    )
                )
                return lx.linear_solve(
                    operator, right_hand_side, solver=solver, state=numeric_state
                ).value

        solve_fn = jax.jit(run) if use_jit else run
        solve_fn(RIGHT_HAND_SIDE)

    symbolic_handle_id, numeric_handle_id = captured_handle_ids[0]

    assert solve_with_numeric_calls, (
        "klujax.solve_with_numeric was not called; factorization was not reused"
    )
    assert not solve_calls, "klujax.solve was called; factorization was not reused"
    assert symbolic_handle_id in freed_symbolic_ids, (
        "symbolic handle was not freed when the factorize block exited"
    )
    assert numeric_handle_id in freed_numeric_ids, (
        "numeric handle was not freed when the factorize block exited"
    )


def test_transpose_in_factorize_reuses_factorization() -> None:
    """Calling `solver.transpose()` on a managed state inside a `factorize()`
    block must reuse the existing symbolic and numeric handles unchanged (KLU's
    `tsolve_with_numeric` solves A^T x = b from the original numeric LU without
    re-factoring).  Both handles must be freed on block exit.
    """
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = KLU()

    with (
        _spy_frees() as (freed_symbolic_ids, freed_numeric_ids),
        _spy_solve("tsolve_with_numeric") as tsolve_calls,
        _spy_solve("solve_with_numeric") as solve_with_numeric_calls,
    ):
        state_init = solver.init(operator, options={})
        with state_init.factorize() as numeric_state:
            symbolic_handle_id = id(numeric_state.factorization.symbolic)
            numeric_handle_id = id(numeric_state.factorization.numeric)

            transposed_state, _ = solver.transpose(numeric_state, options={})

            # `KLU.transpose` on a `_KLUNumericState` reuses the factorization
            # object in-place: no new handles are allocated, and `tsolve_with_numeric`
            # computes the transposed solve using the existing LU decomposition.
            assert (
                isinstance(transposed_state, _KLUNumericState)
                and transposed_state.factorization is numeric_state.factorization
            )

            solver.compute(transposed_state, RIGHT_HAND_SIDE, options={})

    assert tsolve_calls, (
        "klujax.tsolve_with_numeric was not called for a transposed state"
    )
    assert not solve_with_numeric_calls, (
        "klujax.solve_with_numeric was called; should use tsolve_with_numeric for transpose"
    )
    assert symbolic_handle_id in freed_symbolic_ids, (
        "symbolic handle was not freed when the factorize block exited"
    )
    assert numeric_handle_id in freed_numeric_ids, (
        "numeric handle was not freed when the factorize block exited"
    )


def test_conj_real_reuses_both_handles() -> None:
    """For a real matrix, `solver.conj()` is a mathematical no-op and must
    return the original state unchanged.  Exactly one free call must be recorded
    per handle (no extra handles were registered in the scope).
    """
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = KLU()

    with _spy_frees() as (freed_symbolic_ids, freed_numeric_ids):
        state_init = solver.init(operator, options={})
        with state_init.factorize() as numeric_state:
            symbolic_handle_id = id(numeric_state.factorization.symbolic)
            numeric_handle_id = id(numeric_state.factorization.numeric)

            conj_state, _ = solver.conj(numeric_state, options={})

            # For a real matrix, conj(A) = A: `KLU.conj` detects that the
            # values are not complex and returns the same `_ManagedKLUState`
            # object without touching the factorization handles at all.
            assert conj_state is numeric_state

    # Exactly one free per handle: no extra handles were registered in the scope.
    assert freed_symbolic_ids.count(symbolic_handle_id) == 1, (
        "symbolic handle was freed an unexpected number of times"
    )
    assert freed_numeric_ids.count(numeric_handle_id) == 1, (
        "numeric handle was freed an unexpected number of times"
    )


def test_conj_complex_reuses_symbolic_creates_new_numeric() -> None:
    """For a complex matrix, `solver.conj()` must reuse the symbolic analysis
    handle (same sparsity, no re-analyze needed) and allocate a fresh numeric
    handle for conj(A).  All three handles — one symbolic and two numerics —
    must be freed when the `factorize()` block exits.
    """
    operator = BCOOLinearOperator(BCOO.fromdense(COMPLEX_MATRIX))
    solver = KLU()

    with _spy_frees() as (freed_symbolic_ids, freed_numeric_ids):
        state_init = solver.init(operator, options={})
        with state_init.factorize() as numeric_state:
            symbolic_handle_id = id(numeric_state.factorization.symbolic)
            original_numeric_handle_id = id(numeric_state.factorization.numeric)

            conj_state, _ = solver.conj(numeric_state, options={})

            # The symbolic handle is shared because conj(A) has the same
            # sparsity pattern as A.  The numeric handle is new because the
            # numerical values differ.
            assert isinstance(conj_state, _KLUNumericState)
            assert id(conj_state.factorization.symbolic) == symbolic_handle_id, (
                "conj() did not reuse the symbolic handle"
            )
            conj_numeric_handle_id = id(conj_state.factorization.numeric)
            assert conj_numeric_handle_id != original_numeric_handle_id, (
                "conj() on a complex matrix should allocate a new numeric handle"
            )

    # All three registered handles must be freed on scope exit:
    # 1 symbolic + 2 numerics (original and conj).
    assert freed_symbolic_ids.count(symbolic_handle_id) == 1, (
        "symbolic handle was freed an unexpected number of times"
    )
    assert original_numeric_handle_id in freed_numeric_ids, (
        "original numeric handle was not freed"
    )
    assert conj_numeric_handle_id in freed_numeric_ids, (
        "conj numeric handle was not freed"
    )
    assert len(freed_numeric_ids) == 2, (
        f"expected 2 numeric handle frees, got {len(freed_numeric_ids)}"
    )


def test_factorize_symbolic_init_uses_solve_with_symbol(
    make_operator: OperatorFactory,
) -> None:
    """Inside a `factorize_symbolic()` block, calling `.init(operator)` then solving
    must use `klujax.solve_with_symbol` (factors numerically on each call, reusing
    the pre-computed symbolic analysis).  The symbolic handle must be freed when the
    outer block exits and the solution must be numerically correct.
    """
    operator = make_operator(SQUARE_MATRIX)
    solver = KLU()
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with (
        _spy_frees() as (freed_symbolic_ids, freed_numeric_ids),
        _spy_solve("solve_with_symbol") as solve_with_symbol_calls,
        _spy_solve("solve") as solve_calls,
    ):
        with solver.factorize_symbolic(BCOO.fromdense(SQUARE_MATRIX)) as scope:
            symbolic_handle_id = id(scope.symbolic)
            state = scope.init(operator)
            assert isinstance(state, _KLUSymbolicState)
            solution = solver.compute(state, RIGHT_HAND_SIDE, options={})[0]

    assert solve_with_symbol_calls, (
        "klujax.solve_with_symbol was not called; symbolic factorization was not reused"
    )
    assert not solve_calls, "klujax.solve was called; should use solve_with_symbol"
    assert symbolic_handle_id in freed_symbolic_ids, (
        "symbolic handle was not freed when the factorize_symbolic block exited"
    )
    assert not freed_numeric_ids, (
        "a numeric handle was unexpectedly registered in the scope (solve_with_symbol "
        "factors internally without going through the allocation scope manager)"
    )
    assert jnp.allclose(solution, expected, atol=1e-5), (
        "solve_with_symbol produced incorrect solution"
    )


def test_factorize_symbolic_factorize_reuses_symbolic_creates_numeric(
    make_operator: OperatorFactory,
) -> None:
    """Inside a `factorize_symbolic()` block, calling `.factorize(operator)` must
    reuse the pre-computed symbolic handle and create exactly one new numeric handle.
    The symbolic is freed by the outer scope; the numeric by the inner scope.
    The solution must be numerically correct.
    """
    operator = make_operator(SQUARE_MATRIX)
    solver = KLU()
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with (
        _spy_frees() as (freed_symbolic_ids, freed_numeric_ids),
        _spy_solve("solve_with_numeric") as solve_with_numeric_calls,
        _spy_solve("solve_with_symbol") as solve_with_symbol_calls,
    ):
        captured: list[tuple[int, int]] = []

        with solver.factorize_symbolic(BCOO.fromdense(SQUARE_MATRIX)) as scope:
            symbolic_handle_id = id(scope.symbolic)
            with scope.factorize(operator) as numeric_state:
                captured.append(
                    (
                        id(numeric_state.factorization.symbolic),
                        id(numeric_state.factorization.numeric),
                    )
                )
                # Call compute directly rather than through lx.linear_solve to avoid
                # a JIT-cache hit: lx.linear_solve uses eqx.filter_jit internally, and
                # if _KLUNumericState was already compiled by an earlier test, compute
                # won't be re-traced and the spy won't fire.
                solution, _, _ = solver.compute(numeric_state, RIGHT_HAND_SIDE, {})

    captured_symbolic_id, captured_numeric_id = captured[0]

    assert solve_with_numeric_calls, "klujax.solve_with_numeric was not called"
    assert not solve_with_symbol_calls, (
        "klujax.solve_with_symbol was called; numeric factorization should be used"
    )
    assert captured_symbolic_id == symbolic_handle_id, (
        "numeric state did not reuse the symbolic handle from factorize_symbolic"
    )
    assert captured_symbolic_id in freed_symbolic_ids, (
        "symbolic handle was not freed by the outer factorize_symbolic scope"
    )
    assert captured_numeric_id in freed_numeric_ids, (
        "numeric handle was not freed by the inner factorize scope"
    )
    assert len(freed_numeric_ids) == 1, (
        f"expected exactly 1 numeric free, got {len(freed_numeric_ids)}"
    )
    assert jnp.allclose(solution, expected, atol=1e-5), (
        "solve_with_numeric via symbolic reuse produced incorrect solution"
    )


def test_factorize_symbolic_transpose_uses_tsolve_with_symbol(
    make_operator: OperatorFactory,
) -> None:
    """Transposing a `_KLUSymbolicState` and then solving must use
    `klujax.tsolve_with_symbol` and produce the correct A^T solution.
    """
    operator = make_operator(SQUARE_MATRIX)
    solver = KLU()
    expected_transpose = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )

    with (
        _spy_solve("tsolve_with_symbol") as tsolve_calls,
        _spy_solve("solve_with_symbol") as solve_calls,
    ):
        with solver.factorize_symbolic(BCOO.fromdense(SQUARE_MATRIX)) as scope:
            state = scope.init(operator)
            transposed_state, _ = solver.transpose(state, options={})
            solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, options={})[0]

    assert tsolve_calls, (
        "klujax.tsolve_with_symbol was not called for a transposed symbolic state"
    )
    assert not solve_calls, (
        "klujax.solve_with_symbol was called; should use tsolve_with_symbol for transpose"
    )
    assert jnp.allclose(solution, expected_transpose, atol=1e-5), (
        "tsolve_with_symbol produced incorrect transposed solution"
    )


def test_klu_factorize_gives_correct_solution(
    make_operator: OperatorFactory,
) -> None:
    """`KLU().factorize(operator)` is a convenience method equivalent to
    `KLU().init(operator).factorize()`.  It must yield a `_KLUNumericState` that
    produces the same solution as the full lineax solve path.
    """
    operator = make_operator(SQUARE_MATRIX)
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    with KLU().factorize(operator) as state:
        assert isinstance(state, _KLUNumericState)
        solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=KLU(), state=state
        ).value

    assert jnp.allclose(solution, expected, atol=1e-5), (
        "KLU.factorize produced incorrect solution"
    )
