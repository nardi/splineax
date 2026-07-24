"""Tests for `splineax.solvers._handle`, the stable-id handle-token mechanism shared
by `_klu.py` and `_pardiso.py`.

These test the mechanism directly and in isolation from either solver: that a handle
is only wrapped in a token while being traced, that the wrapping does not force extra
recompilation, that a token's stable id survives being rebuilt with fresh leaf objects
the way `lineax.linear_solve` does to a state pytree, that dependency tracking is
scoped to the trace a dependency was created in, and that registering a dependency in
the wrong trace without going through `splineax.linear_solve` raises rather than
silently dropping it.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from splineax.solvers._handle import (
    HandleDependencies,
    _HandleToken,
    handle_key,
    handle_value,
    mark_via_linear_solve,
    rebind_handle,
    wrap_handle,
)

# int64 handles need 64-bit mode; every test in this module gets it from the shared
# `enable_x64` fixture (tests/conftest.py).
pytestmark = pytest.mark.usefixtures("enable_x64")


def test_wrap_handle_leaves_concrete_values_unwrapped() -> None:
    """A concrete (eager) value stays a plain array: no static aux data is attached,
    since its scope's free loop runs later, eagerly, where plain `id()` tracking
    already works."""
    value = jnp.array(1, dtype=jnp.int64)
    wrapped = wrap_handle(value)
    assert wrapped is value
    assert not isinstance(wrapped, _HandleToken)


def test_wrap_handle_wraps_traced_values() -> None:
    """A value created while tracing is wrapped in a `_HandleToken` carrying a stable
    id, so dependency tracking survives being retraced later."""
    captured = {}

    @jax.jit
    def f(x):
        wrapped = wrap_handle(x)
        captured["is_token"] = isinstance(wrapped, _HandleToken)
        return handle_value(wrapped)

    f(jnp.array(1, dtype=jnp.int64))
    assert captured["is_token"]


def test_rebind_handle_preserves_stable_id() -> None:
    """Advancing a token to a new value keeps its stable id, so dependencies already
    registered against the old value are still found under the new one."""

    @jax.jit
    def f(x, y):
        old = wrap_handle(x)
        new = rebind_handle(old, y)
        assert isinstance(old, _HandleToken)
        assert isinstance(new, _HandleToken)
        assert old.stable_id == new.stable_id
        return handle_value(new)

    f(jnp.array(1, dtype=jnp.int64), jnp.array(2, dtype=jnp.int64))


def test_rebind_handle_on_plain_array_stays_plain() -> None:
    """Rebinding a concrete (untokenized) handle keeps it a plain array."""
    old = jnp.array(1, dtype=jnp.int64)
    new = rebind_handle(old, jnp.array(2, dtype=jnp.int64))
    assert not isinstance(new, _HandleToken)


def test_handle_dependencies_round_trip_through_stable_id() -> None:
    """`HandleDependencies` finds a dependency registered against one token object
    when popped with a *different* token object sharing the same stable id, the
    scenario a handle rewrapped by `lineax.linear_solve` needs to survive."""

    @jax.jit
    def f(x, dependency):
        handle = wrap_handle(x)
        assert isinstance(handle, _HandleToken)
        deps = HandleDependencies()
        deps.register(handle, dependency)

        # A fresh token wrapping the same underlying value, as `lineax.linear_solve`
        # produces by rebuilding the state pytree with new leaf objects: a different
        # Python object, but the same stable id.
        rewrapped = _HandleToken(handle.value, handle.stable_id)
        assert rewrapped is not handle
        assert handle_key(rewrapped) == handle_key(handle)

        found = deps.pop(rewrapped)
        assert found == [dependency]
        return handle_value(handle)

    f(jnp.array(1, dtype=jnp.int64), jnp.array(2.0))


def test_reused_scope_does_not_recompile() -> None:
    """A handle built once, eagerly, and passed as a `jax.jit` argument on repeated
    calls does not force a recompile between calls: `wrap_handle` leaves a concrete
    value as a plain array, so no static aux data changes between calls.

    This is the regression test for the concern that motivated only wrapping traced
    values: an earlier version of this design put a stable id on every handle
    unconditionally, which would fail this test, since a fresh id each time changes
    the pytree treedef and hence `jax.jit`'s cache key.
    """
    trace_count = 0

    @jax.jit
    def f(handle):
        nonlocal trace_count
        trace_count += 1
        return handle_value(handle) + 1

    handle = jnp.array(1, dtype=jnp.int64)  # eager: stays a plain array
    f(handle)
    f(handle)
    assert trace_count == 1, "a concrete handle passed twice should only trace once"


def test_dependencies_from_other_traces_are_dropped() -> None:
    """A dependency registered in a nested trace is dropped when the handle is popped
    from the outer trace, while one registered in the same trace as the pop is kept.

    This is the case `lineax.linear_solve` creates: it runs a solve in a nested trace,
    whose result must not leak into the outer trace's free loop.
    """
    deps = HandleDependencies()

    @jax.jit
    def outer(x, outer_dependency):
        handle = wrap_handle(x)
        assert isinstance(handle, _HandleToken)
        deps.register(handle, outer_dependency)  # registered in the outer trace
        stable_id = handle.stable_id

        def nested(y):
            # Same stable id, but a value and registration belonging to a nested trace,
            # as a `lineax.linear_solve` solve would produce.
            deps.register(_HandleToken(y, stable_id), y)
            return y

        jax.jit(nested)(outer_dependency)

        kept = deps.pop(handle)  # popped in the outer trace
        assert len(kept) == 1
        assert kept[0] is outer_dependency
        return handle_value(handle)

    outer(jnp.array(1, dtype=jnp.int64), jnp.array(2.0))


def test_register_raises_for_unmarked_cross_trace_dependency() -> None:
    """Registering a dependency in a different trace than the one the handle was
    allocated in raises, unless `mark_via_linear_solve` is active.

    This is the check that catches a bare `lineax.linear_solve` call used where
    `splineax.linear_solve` is required: without it, the dependency would simply be
    dropped at free time and the handle could be released while still in use.
    """
    deps = HandleDependencies()

    @jax.jit
    def outer(x, y):
        handle = wrap_handle(x)
        deps.record_allocation(handle)  # allocated in the outer trace

        def nested(z):
            with pytest.raises(RuntimeError, match="splineax.linear_solve"):
                deps.register(handle, z)
            return z

        jax.jit(nested)(y)
        return handle_value(handle)

    outer(jnp.array(1, dtype=jnp.int64), jnp.array(2.0))


def test_register_allows_cross_trace_dependency_when_marked() -> None:
    """The same cross-trace registration does not raise while `mark_via_linear_solve`
    is active, so `splineax.linear_solve` can call `lineax.linear_solve` without
    tripping the check on the nested dependency `compute` registers internally."""
    deps = HandleDependencies()

    @jax.jit
    def outer(x, y):
        handle = wrap_handle(x)
        deps.record_allocation(handle)

        def nested(z):
            with mark_via_linear_solve():
                deps.register(handle, z)  # does not raise
            return z

        jax.jit(nested)(y)
        return handle_value(handle)

    outer(jnp.array(1, dtype=jnp.int64), jnp.array(2.0))


def test_register_same_trace_as_allocation_never_raises() -> None:
    """Registering a dependency in the same trace the handle was allocated in never
    raises, marked or not: there is no cross-trace mismatch to catch."""
    deps = HandleDependencies()

    @jax.jit
    def f(x, y):
        handle = wrap_handle(x)
        deps.record_allocation(handle)
        deps.register(handle, y)  # same trace as record_allocation, does not raise
        return handle_value(handle)

    f(jnp.array(1, dtype=jnp.int64), jnp.array(2.0))


def test_record_allocation_is_a_no_op_for_plain_arrays() -> None:
    """`record_allocation` only tracks tokens: an eagerly allocated handle has no
    allocation trace recorded, so `register` never raises for it, matching the eager
    scope-reuse case where cross-call ordering is already safe (see the module
    docstring in `_handle.py`)."""
    deps = HandleDependencies()
    handle = jnp.array(1, dtype=jnp.int64)  # eager: stays a plain array
    deps.record_allocation(handle)  # no-op, not a token

    @jax.jit
    def f(y):
        deps.register(handle, y)  # does not raise: no allocation trace was recorded
        return y

    f(jnp.array(2.0))


def test_eager_dependencies_round_trip() -> None:
    """Registering and popping a dependency eagerly (no trace) keeps it: there is no
    trace to mismatch, and eager execution order already sequences a free after its
    solves."""
    deps = HandleDependencies()
    handle = jnp.array(1, dtype=jnp.int64)  # eager: a plain array
    dependency = jnp.array(2.0)
    deps.register(handle, dependency)
    kept = deps.pop(handle)
    assert len(kept) == 1
    assert kept[0] is dependency


def test_eqx_module_dynamic_field_can_hold_a_token() -> None:
    """A `_HandleToken` nests correctly inside another `eqx.Module`'s dynamic field,
    the way `_PardisoSymbolicState.handle` / `_KLUFactorization.symbolic` use it: the
    token's own leaf/aux split is respected by the outer module's pytree flattening."""

    class Outer(eqx.Module):
        handle: object

    @jax.jit
    def f(x):
        outer = Outer(wrap_handle(x))
        assert isinstance(outer.handle, _HandleToken)
        return handle_value(outer.handle)

    f(jnp.array(1, dtype=jnp.int64))
