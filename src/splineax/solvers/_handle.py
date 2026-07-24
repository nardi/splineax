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

The dependency map is also scoped to the trace a dependency was created in. A handle is
allocated and freed inside the same `with` block, so its free only needs to wait for the
solves that ran in the same trace as the free. `lineax.linear_solve` runs a solver's
`compute` in a nested trace, so any solve result registered there is a tracer belonging
to that nested trace, invalid to reference from the outer trace the free runs in. Keying
each dependency by the trace it was registered in, and dropping the ones from other
traces at free time, discards exactly those leaked nested tracers. Eager (untraced)
dependencies fall out of the same rule: there is no trace to match, and eager execution
order already sequences the free after the solves.

Dropping a nested-trace dependency is only safe if some *other*, same-trace dependency
still orders the free correctly. Plain `lineax.linear_solve` cannot supply one when a
handle was allocated while tracing: the only outer-trace value depending on the solve is
`lineax.linear_solve`'s own return value, invisible from inside `compute`. That is what
`splineax.linear_solve` (`_sparse.py`) is for: it calls `lineax.linear_solve` and then
registers the outer-trace result itself. `HandleDependencies.register` checks for exactly
this and raises a clear error, instead of silently dropping the only dependency and
letting a use-after-free surface later as an opaque tracer error or a native crash.
"""

import itertools
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import equinox as eqx
import jax
import jax.core
from jax.extend.core import get_opaque_trace_state
from jaxtyping import Array

_next_stable_id = itertools.count()

# Set by `splineax.linear_solve` for the duration of its call into `lineax.linear_solve`,
# so `HandleDependencies.register` knows not to raise for the nested-trace dependency
# `compute` registers there: `splineax.linear_solve` will register the outer-trace one
# itself once the call returns.
_via_linear_solve: ContextVar[bool] = ContextVar("_via_linear_solve", default=False)


@contextmanager
def mark_via_linear_solve() -> Iterator[None]:
    """Used only by `splineax.linear_solve`, see `_via_linear_solve` above."""
    token = _via_linear_solve.set(True)
    try:
        yield
    finally:
        _via_linear_solve.reset(token)


def current_trace_state() -> Any:
    """An identity for the trace currently being built, or the eager context.

    `OpaqueTraceState` compares equal within one trace and unequal across nested or
    eager traces. It is unhashable, so dependency tracking matches it by equality
    rather than using it as a dict key.
    """
    return get_opaque_trace_state()


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
    before it is freed, shared by both solvers' scope managers.

    Each dependency is tagged with the trace it was registered in, so that `pop` only
    returns the ones from the same trace as the free and drops the rest, see the module
    docstring.
    """

    def __init__(self) -> None:
        self._dependencies: dict[int, list[tuple[Any, Any]]] = {}
        self._allocation_traces: dict[int, Any] = {}

    def record_allocation(self, handle: Array | _HandleToken) -> None:
        """Remember the trace `handle` was allocated in, so `register` can tell whether
        a dependency it is about to add will actually be seen by this handle's free.

        Only meaningful for a token: a handle allocated eagerly has no free to protect
        here, since its free also runs eagerly later, where ordinary execution order is
        enough (see the module docstring).
        """
        if isinstance(handle, _HandleToken):
            self._allocation_traces[handle.stable_id] = current_trace_state()

    def register(self, handle: Array | _HandleToken, dependency: Any) -> Any:
        key = handle_key(handle)
        current = current_trace_state()
        allocated_in = self._allocation_traces.get(key)
        if (
            allocated_in is not None
            and allocated_in != current
            and not _via_linear_solve.get()
        ):
            raise RuntimeError(
                "A solve inside a `factorize_symbolic` scope was run through "
                "`lineax.linear_solve` while the whole scope is traced together with "
                "the solve (opened and closed inside one `jax.jit` call). "
                "`lineax.linear_solve` stages the solve into a nested trace, so this "
                "result cannot be used to order the scope's handle release, which runs "
                "in the outer trace, and freeing it without that ordering is unsafe. "
                "Use `splineax.linear_solve` in place of `lineax.linear_solve` here: "
                "it registers the solve's result so the release orders correctly."
            )
        entry = self._dependencies.setdefault(key, [])
        entry.append((current, dependency))
        return dependency

    def pop(self, handle: Array | _HandleToken) -> list[Any]:
        # Remove both entries so dropped foreign-trace dependencies do not accumulate,
        # and return only the dependencies registered in the trace this free runs in.
        key = handle_key(handle)
        self._allocation_traces.pop(key, None)
        entries = self._dependencies.pop(key, [])
        current = current_trace_state()
        return [dep for trace, dep in entries if trace == current]
