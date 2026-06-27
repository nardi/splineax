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
import lineax as lx
import pytest
from jax.experimental.sparse import BCOO

from splineax import KLU, BCOOLinearOperator
from splineax.solvers._klu import _ManagedKLUState

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
    `managed_state.factorization.symbolic/numeric`.

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
    `id(manager.handle) == id(managed_state.factorization.symbolic/numeric)`.
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
    klu.free_symbolic = spy_free_symbolic
    klu.free_numeric = spy_free_numeric
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
            with state_init.factorize() as managed_state:
                # Under JIT these are Tracer object ids; the spy in _spy_frees
                # captures the same Tracer ids when begin_scope() exits.
                captured_handle_ids.append(
                    (
                        id(managed_state.factorization.symbolic),
                        id(managed_state.factorization.numeric),
                    )
                )
                return lx.linear_solve(
                    operator, right_hand_side, solver=solver, state=managed_state
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
        with state_init.factorize() as managed_state:
            symbolic_handle_id = id(managed_state.factorization.symbolic)
            numeric_handle_id = id(managed_state.factorization.numeric)

            transposed_state, _ = solver.transpose(managed_state, options={})

            # `KLU.transpose` on a `_ManagedKLUState` reuses the factorization
            # object in-place: no new handles are allocated, and `tsolve_with_numeric`
            # computes the transposed solve using the existing LU decomposition.
            assert (
                isinstance(transposed_state, _ManagedKLUState)
                and transposed_state.factorization is managed_state.factorization
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
        with state_init.factorize() as managed_state:
            symbolic_handle_id = id(managed_state.factorization.symbolic)
            numeric_handle_id = id(managed_state.factorization.numeric)

            conj_state, _ = solver.conj(managed_state, options={})

            # For a real matrix, conj(A) = A: `KLU.conj` detects that the
            # values are not complex and returns the same `_ManagedKLUState`
            # object without touching the factorization handles at all.
            assert conj_state is managed_state

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
        with state_init.factorize() as managed_state:
            symbolic_handle_id = id(managed_state.factorization.symbolic)
            original_numeric_handle_id = id(managed_state.factorization.numeric)

            conj_state, _ = solver.conj(managed_state, options={})

            # The symbolic handle is shared because conj(A) has the same
            # sparsity pattern as A.  The numeric handle is new because the
            # numerical values differ.
            assert isinstance(conj_state, _ManagedKLUState)
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
