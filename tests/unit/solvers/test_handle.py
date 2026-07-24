"""Tests for `splineax.solvers._handle`, the stable-id handle-token mechanism shared
by `_klu.py` and `_pardiso.py`.

These test the mechanism directly and in isolation from either solver: that a handle
is only wrapped in a token while being traced, that the wrapping does not force extra
recompilation, and that a token's stable id survives being rebuilt with fresh leaf
objects, the way `lineax.linear_solve` does to a state pytree.
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
