"""A stable-id wrapper for factorization handles, shared by `_klu.py` and `_pardiso.py`.

Both solvers free a native handle when the scope that allocated it exits, ordered
after every solve that used it via a dict keyed by the handle's Python object id. That
id tracking breaks whenever `lineax.linear_solve` is involved: it threads state through
`jax.lax.stop_gradient` and a custom primitive bind, both of which mint a fresh Tracer
object for the same logical value under jit. So the object registered when a scope
opens is not the object seen later at the point a solve depends on it, no match is
found, and the handle can be freed before a solve that still needs it finishes.

The fix used here is a JAX pytree fact: a pytree's static aux data survives every
transformation unchanged, even as its dynamic leaves get rewrapped. Wrapping a handle
in a token that carries a stable Python-level integer id as aux data, and keying
dependency tracking by that id instead of the array's object id, survives exactly the
case that breaks today.

This only matters while a scope's whole open-to-close lifecycle is itself being traced,
since only then does its free loop end up traced into the same computation as whatever
`linear_solve` did to the handle in between. An eagerly opened scope's free loop runs
later, as an ordinary eager call, where plain object-id tracking already works. So a
handle is only wrapped in a token when it is created while already being traced;
otherwise it stays a plain array exactly as before. This matters for jit performance,
not just correctness: static aux data is part of a pytree's treedef, which is part of
`jax.jit`'s cache key, so an unconditionally-attached id would force a jitted function
taking a freshly-built scope as an argument to recompile on every call.
"""

import itertools
from typing import Any

import equinox as eqx
import jax
import jax.core
from jaxtyping import Array

_next_stable_id = itertools.count()


class _HandleToken(eqx.Module):
    """Wraps a handle array with a stable id that survives being retraced."""

    value: Array
    stable_id: int = eqx.field(static=True)


def wrap_handle(value: Array) -> Array | _HandleToken:
    """Wrap `value` in a `_HandleToken` if it is being created while tracing.

    Concrete values are returned unchanged: their scope's free loop runs eagerly,
    later, where plain object-id tracking is already reliable, so there is no reason
    to attach static aux data (and its jit-cache-key cost) to them.
    """
    if isinstance(value, jax.core.Tracer):
        return _HandleToken(value, next(_next_stable_id))
    return value


def rebind_handle(old: Array | _HandleToken, new_value: Array) -> Array | _HandleToken:
    """Advance a handle to `new_value`, keeping `old`'s token-or-not kind and, if a
    token, its stable id, so dependencies registered against it are still found."""
    if isinstance(old, _HandleToken):
        return _HandleToken(new_value, old.stable_id)
    return new_value


def handle_value(handle: Array | _HandleToken) -> Array:
    """Unwrap a handle to the raw array a primitive/FFI call expects."""
    if isinstance(handle, _HandleToken):
        return handle.value
    return handle


def handle_key(handle: Array | _HandleToken) -> int:
    """A dependency-dict key for `handle`, stable across retracing when it is a token."""
    if isinstance(handle, _HandleToken):
        return handle.stable_id
    return id(handle)


class HandleDependencies:
    """A `handle_key`-addressed map from a handle to the values that must be computed
    before it is freed, shared by both solvers' scope managers."""

    def __init__(self) -> None:
        self._dependencies: dict[int, list[Any]] = {}

    def register(self, handle: Array | _HandleToken, dependency: Any) -> Any:
        self._dependencies.setdefault(handle_key(handle), []).append(dependency)
        return dependency

    def pop(self, handle: Array | _HandleToken) -> list[Any]:
        return self._dependencies.pop(handle_key(handle), [])
