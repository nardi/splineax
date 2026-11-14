"""An opt-in solve profile for debugging the stateful sparse solvers.

`create_solve_profile` creates a `SolveProfile`, which is a context manager that
collects the solver's operations into a log:

```{.python notest}
profile = splineax.create_solve_profile()
with profile:
    solution, state = splineax.linear_solve(operator, vector, solver)
    state.release()
print(profile)
```

Each record is written by a pure `jax.pure_callback` that is entangled with the arrays of
the operation (see `record_operation`). A record therefore fires when the operation runs
and after the operations that it depends on. When XLA removes an operation because no
result depends on it, the record is removed as well. Each time a callback
fires, it appends into whichever `SolveProfile` is currently entered on that thread, or
nowhere if none is. XLA can fire a callback on one of its own threads, which has no
profile entered. Such a callback appends into the most recently entered profile that is
still open on any thread. `SolveProfile.records` is in the order in which the callbacks
fired. When two operations do not depend on each other, XLA can run them in any order.
When no profile is ever active for a given compiled shape, nothing is emitted into the
traced program at all. A function that `jax.jit` compiled while a profile was active keeps
its callbacks. Such a function is slower when it runs with no profile entered.
"""

import contextlib
import dataclasses
import functools
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from types import TracebackType
from typing import Any, NamedTuple, ParamSpec, TypeVar, cast, overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from entangle_jax import entangle
from jax.experimental import io_callback
from jaxtyping import Array, PyTree

# Per-thread state: the stack of open profiles and the current
# `compute` nesting depth (so only the outermost `compute` emits the generic boundary).
_LOCAL = threading.local()


def _stack() -> list["SolveProfile"]:
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _LOCAL.stack = stack
    return stack


def _active() -> "SolveProfile | None":
    """The innermost open profile on this thread, or None when profiling is off."""
    stack = _stack()
    return stack[-1] if stack else None


_OPEN_PROFILES: list["SolveProfile"] = []
"""Every profile that is entered on any thread right now, in the order they were entered."""

_OPEN_PROFILES_LOCK = threading.Lock()
"""Guards `_OPEN_PROFILES`, which every thread that enters or exits a profile changes."""


def _active_for_callback() -> "SolveProfile | None":
    """The profile a firing callback appends into, or None when no profile is open.

    A callback fires on the thread that runs the compiled program. That is the thread that
    entered the profile for a small program, but XLA can run a larger one on a thread of
    its own, where `_stack` is empty. This prefers the profile entered on the current
    thread and otherwise falls back to the most recently entered one that is still open.
    With several threads holding profiles at once, a callback on an XLA thread cannot tell
    which of them it belongs to, so it picks the latest.
    """
    profile = _active()
    if profile is not None:
        return profile
    with _OPEN_PROFILES_LOCK:
        return _OPEN_PROFILES[-1] if _OPEN_PROFILES else None


def profiling_active() -> bool:
    """Whether a `SolveProfile` is currently entered on this thread."""
    return bool(_stack())


# Generic operations that begin a new state-sequence when grouping the log.
_SEQUENCE_STARTS = frozenset({"init", "init_symbolic"})


@dataclasses.dataclass(frozen=True)
class ProfileRecord:
    """One recorded operation in a `SolveProfile`.

    `operation` is the operation name. A **generic** operation (`init`, `init_symbolic`,
    `update`, `compute`, `track`, `release`) has `solver` None. A **solver-specific**
    operation has `solver` set to the solver type (`KLU`, `Pardiso`, `Spsolve`,
    `IterativeRefinement`) and `operation` set to the library function it ran (`analyze`,
    `factor`, `solve_with_numeric`, ...). `inputs` and `outputs` hold the fields rendered as
    `operation[inputs] => (outputs)`.
    """

    operation: str
    solver: str | None = None
    inputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    outputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)


# ANSI colors for the pretty tree. Kept tiny and dependency-free.
_RESET = "\033[0m"
_DIM = "\033[2m"
# Yellow: a factorization built anew.
_CREATED = "\033[33m"
# Green: an existing factorization reused.
_REUSED = "\033[32m"
_BOLD = "\033[1m"

# Solver-specific operations colored as building a factorization anew.
_CREATED_OPS = frozenset({"analyze", "reanalyze", "spsolve", "factorize"})
# Solver-specific operations colored as reusing an existing factorization.
_REUSED_OPS = frozenset(
    {
        "solve_with_numeric",
        "tsolve_with_numeric",
        "solve_stateful",
        "solve",
    }
)
# The floating-point output fields shown in scientific notation.
_SCI_FIELDS = frozenset({"rcond", "residual_norm", "threshold"})

# The rebuild-reason codes, shared by klujax and pardiso_mkl_jax (their RebuildReason
# enums use the same numbering). Rendered by name so a profile line reads `rebuild="none"`
# rather than `rebuild=0`. An unknown code falls back to the raw integer.
_REBUILD_REASON_NAMES = (
    "none",
    "evicted",
    "freed",
    "superseded",
    "dtype",
    "unknown",
    "stale",
)
# Free-text output fields, quoted so a space-joined line stays readable.
_TEXT_FIELDS = frozenset({"reason", "note"})

# A canonical display order for fields, so a line reads the same regardless of dict order
# (the runtime `dynamic` values come back from the callback in JAX's sorted-key order).
_FIELD_ORDER = {
    field: index
    for index, field in enumerate(
        (
            "shape",
            "nse",
            "sparsity_hash",
            "transposed",
            "outcome",
            "reused",
            "rebuild",
            "rcond",
            "perturbed_pivots",
            "zero_pivot",
            "step",
            "residual_norm",
            "threshold",
            "converged",
            "note",
            "reason",
        )
    )
}


def sparsity_hash(tag: object | None) -> str | None:
    """A short hex digest of a sparsity-pattern tag, or None when there is no tag.

    Reuses the tag's own hash, so two operators the solver treats as one pattern (equal tags)
    get the same digest. Consistent within a run, but not stable across runs.
    """
    if tag is None:
        return None
    return f"0x{hash(tag) & 0xFFFFF:05x}"


def _record_color(record: ProfileRecord) -> str:
    """The color for a record, by whether it builds a factorization or reuses one."""
    operation = record.operation
    if record.solver is None:
        if operation == "update":
            outcome = record.outputs.get("outcome")
            if outcome == "reused":
                return _REUSED
            if outcome == "rebuilt":
                return _CREATED
        return _DIM
    if operation in _CREATED_OPS:
        return _CREATED
    match operation:
        case "refactor" | "refactorize":
            return _CREATED if record.outputs.get("reused") is False else _REUSED
        case "factor":
            return _REUSED if record.outputs.get("reused") is True else _CREATED
        case _:
            return _REUSED if operation in _REUSED_OPS else _DIM


def _format_value(field: str, value: Any) -> str:
    if field in _SCI_FIELDS and isinstance(value, (int, float)):
        return f"{field}={value:.3e}"
    if (
        field == "rebuild"
        and isinstance(value, int)
        and 0 <= value < len(_REBUILD_REASON_NAMES)
    ):
        return f'{field}="{_REBUILD_REASON_NAMES[value]}"'
    if field in _TEXT_FIELDS:
        return f'{field}="{value}"'
    return f"{field}={value}"


def _format_fields(fields: Mapping[str, Any]) -> str:
    present = [(key, value) for key, value in fields.items() if value is not None]
    present.sort(key=lambda kv: _FIELD_ORDER.get(kv[0], len(_FIELD_ORDER)))
    return ", ".join(_format_value(key, value) for key, value in present)


def _format_record(record: ProfileRecord) -> str:
    name = (
        f"{record.solver}.{record.operation}"
        if record.solver is not None
        else record.operation
    )
    text = name
    inputs = _format_fields(record.inputs)
    if inputs:
        text += f"[{inputs}]"
    outputs = _format_fields(record.outputs)
    if outputs:
        text += f" => ({outputs})"
    return text


def _want_color(color: bool | None) -> bool:
    if color is not None:
        return color
    if os.environ.get("NO_COLOR") is not None:
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class SolveProfile:
    """A log of the operations taken across one or more state-sequences.

    Create one with `create_solve_profile`. `SolveProfile` is itself a context manager: entering
    it (`with profile:`) makes it the active profile on this thread, so every operation recorded
    while it is entered lands in it, and nothing recorded outside any `with` block, or while a
    different profile is entered, does. A profile can be entered more than once, accumulating
    further records each time.

    `records` is the log in the order in which the operations ran, and `sequences` slices it into one list per
    state-sequence (each generic `init`/`init_symbolic` starts a new one). Printing a
    `SolveProfile` renders the indented tree, with factorizations built anew and
    factorizations reused shown in different colors.
    """

    def __init__(self) -> None:
        self._records: list[ProfileRecord] = []
        """The records in the order in which their callbacks fired."""

    def __enter__(self) -> "SolveProfile":
        """Make this profile the active one on this thread, for every operation inside."""
        _stack().append(self)
        with _OPEN_PROFILES_LOCK:
            _OPEN_PROFILES.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # Flush callbacks dispatched during this block before popping, so a solve issued
        # just before the block closes still lands here rather than after the pop.
        jax.effects_barrier()
        _stack().pop()
        with _OPEN_PROFILES_LOCK:
            _OPEN_PROFILES.remove(self)

    def _append(self, record: ProfileRecord) -> None:
        # Runs on the callback thread. `list.append` is atomic under the GIL.
        self._records.append(record)

    @property
    def records(self) -> list[ProfileRecord]:
        """The recorded operations in the order in which their callbacks fired.

        A record fires after the operations that it depends on. Operations that do not
        depend on each other have no defined order.
        """
        return list(self._records)

    @property
    def sequences(self) -> list[list[ProfileRecord]]:
        """The records grouped into state-sequences, split at each generic init."""
        groups: list[list[ProfileRecord]] = []
        current: list[ProfileRecord] | None = None
        for record in self.records:
            starts = record.solver is None and record.operation in _SEQUENCE_STARTS
            if current is None or starts:
                current = []
                groups.append(current)
            current.append(record)
        return groups

    def render(self, color: bool | None = None) -> str:
        """Render the tree. `color` forces ANSI on/off, and None auto-detects a TTY."""
        use_color = _want_color(color)

        def paint(text: str, code: str) -> str:
            return f"{code}{text}{_RESET}" if use_color else text

        # The color key only means anything in color, so show it only then.
        if use_color:
            header = (
                "solve profile   key: "
                + paint("created", _CREATED)
                + " / "
                + paint("reused", _REUSED)
            )
        else:
            header = "solve profile"
        lines = [header]
        if not self._records:
            lines.append("  (empty)")
            return "\n".join(lines)
        for index, sequence in enumerate(self.sequences):
            lines.append(paint(f"sequence {index}", _BOLD))
            for record in sequence:
                # Solver-specific operations nest one level under their generic operation.
                indent = "    " if record.solver is not None else "  "
                lines.append(
                    indent + paint(_format_record(record), _record_color(record))
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()


def create_solve_profile() -> SolveProfile:
    """Create a new, empty `SolveProfile`, for debugging the sparse solvers.

    Equivalent to `SolveProfile()`. Enter it as a context manager to record the analyze,
    factor, refactor, solve, and iterative-refinement operations run inside the block:

    ```{.python notest}
    profile = splineax.create_solve_profile()
    with profile:
        solution, state = splineax.linear_solve(operator, vector, solver)
        state.release()
    print(profile)
    ```
    """
    return SolveProfile()


_CarrierT = TypeVar("_CarrierT", bound=PyTree)
"""The pytree that `record_operation` returns, in the type that the caller passed."""

_P = ParamSpec("_P")
_ReturnT = TypeVar("_ReturnT")


@overload
def profile_solves(
    fn: Callable[_P, _ReturnT],
    *,
    enabled: bool = True,
) -> Callable[_P, tuple[_ReturnT, SolveProfile | None]]: ...


@overload
def profile_solves(
    fn: None = None,
    *,
    enabled: bool = True,
) -> Callable[
    [Callable[_P, _ReturnT]], Callable[_P, tuple[_ReturnT, SolveProfile | None]]
]: ...


def profile_solves(
    fn: Callable[_P, _ReturnT] | None = None,
    *,
    enabled: bool = True,
) -> (
    Callable[_P, tuple[_ReturnT, SolveProfile | None]]
    | Callable[
        [Callable[_P, _ReturnT]], Callable[_P, tuple[_ReturnT, SolveProfile | None]]
    ]
):
    """Wrap `fn` so every call profiles its solves into a fresh `SolveProfile`.

    The wrapped function returns `(result, profile)` instead of `fn`'s own return value.
    Use it bare as `@profile_solves`, or with arguments as `@profile_solves(enabled=False)`.
    With `enabled=False`, every call runs `fn` directly and returns `(result, None)`. This
    lets you switch profiling off in one place without changing the call sites.

    Apply this directly to a `jax.jit`/`equinox.filter_jit`-decorated function, and always
    call it through the decorated name. Used this way, the very first call for a given
    input shape happens through this wrapper with a profile active, so that shape's
    compiled executable is guaranteed to carry its profiling hooks. The hooks are only
    built in at compile time. Calling the undecorated jit function separately, or through
    a second wrapper with `enabled=False`, will forfeit profiling for any shape it compiles
    first.

    ```{.python notest}
    @splineax.profile_solves
    @equinox.filter_jit
    def solve(operator, vector, solver):
        solution, state = splineax.linear_solve(operator, vector, solver)
        state.release()
        return solution

    solution, profile = solve(operator, vector, solver)
    ```
    """

    def decorate(
        fn: Callable[_P, _ReturnT],
    ) -> Callable[_P, tuple[_ReturnT, SolveProfile | None]]:
        """Wrap one function according to the `enabled` flag of the enclosing call."""

        @functools.wraps(fn)
        def wrapper(
            *args: _P.args, **kwargs: _P.kwargs
        ) -> tuple[_ReturnT, SolveProfile | None]:
            if not enabled:
                return fn(*args, **kwargs), None
            profile = create_solve_profile()
            with profile:
                result = fn(*args, **kwargs)
            return result, profile

        return wrapper

    # A bare `@profile_solves` passes the function in directly. The called form
    # `@profile_solves(enabled=...)` passes nothing and returns the decorator instead.
    if fn is None:
        return decorate
    return decorate(fn)


@contextlib.contextmanager
def compute_scope(carrier: _CarrierT, after: PyTree = ()) -> Iterator[_CarrierT]:
    """Emit the generic `compute` boundary once, at the outermost solver's `compute`.

    A wrapping solver's `compute` (e.g. `HybridDirectIterative`) calls an inner solver's
    `compute`, and this suppresses the inner boundary so one user solve is one generic
    `compute`, with every solver-specific operation nested under it. The block receives
    `carrier` with the boundary record entangled, as `record_operation` returns it. Pass
    the solver state as `after`, so that the boundary fires after the state is ready.
    """
    depth = getattr(_LOCAL, "compute_depth", 0)
    if depth == 0:
        carrier = record_operation(carrier, "compute", after=after)
    _LOCAL.compute_depth = depth + 1
    try:
        yield carrier
    finally:
        _LOCAL.compute_depth = depth


@dataclasses.dataclass(eq=False)
class _RecordCallback:
    """The host callback of one `record_operation`, which appends its records when it fires.

    `eq=False` keeps the identity hash, since the dict fields are unhashable.
    """

    operation: str
    """The operation name the record carries."""

    solver: str | None
    """The solver type for a solver-specific operation, or None for a generic one."""

    input_fields: dict[str, Any]
    """The input fields, known at trace time."""

    static_outputs: dict[str, Any]
    """The output fields known at trace time, used when `outputs_fn` is None."""

    outputs_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
    """Builds the output fields from the converted runtime values, or None."""

    value_ndims: dict[str, int]
    """The number of dimensions of each runtime value, without the dimensions of `jax.vmap`."""

    def __call__(
        self, token: np.ndarray, values: Mapping[str, np.ndarray]
    ) -> np.ndarray:
        """Append one record for each batch element of the values into the active profile.

        Under `jax.vmap` the callback receives the batched arrays and fires one time. The
        leading dimensions of the runtime values are the batch. A record without runtime
        values, or with unbatched ones, is appended one time, because the operation ran one
        time. The callback returns an array of zeros that has the shape of `token`, as
        `jax.pure_callback` requires.
        """
        self.fire(values)
        return np.zeros(np.shape(token), dtype=np.int32)

    def fire(self, values: Mapping[str, np.ndarray]) -> None:
        """Append the records of one firing into the active profile, if there is one."""
        batch_shape = np.broadcast_shapes(
            (),
            *(
                np.shape(value)[: np.ndim(value) - self.value_ndims[key]]
                for key, value in values.items()
            ),
        )
        profile = _active_for_callback()
        if profile is not None:
            size = int(np.prod(batch_shape))
            elements = {
                key: np.broadcast_to(
                    value,
                    batch_shape
                    + np.shape(value)[np.ndim(value) - self.value_ndims[key] :],
                ).reshape(
                    (size,) + np.shape(value)[np.ndim(value) - self.value_ndims[key] :]
                )
                for key, value in values.items()
            }
            for index in range(size):
                self._append(
                    profile, {key: value[index] for key, value in elements.items()}
                )

    def _append(self, profile: SolveProfile, values: Mapping[str, np.ndarray]) -> None:
        """Append the record of one batch element into `profile`.

        A `__cond__` value gates the record and is not a field of the record.
        """
        if "__cond__" in values and not values["__cond__"]:
            return
        merged = {
            key: _to_python(value) for key, value in values.items() if key != "__cond__"
        }
        if self.outputs_fn is not None:
            merged.update(self.outputs_fn(merged))
        else:
            merged.update(self.static_outputs)
        profile._append(
            ProfileRecord(
                operation=self.operation,
                solver=self.solver,
                inputs=self.input_fields,
                outputs=merged,
            )
        )


def _to_python(value: Any) -> Any:
    """Convert a runtime array handed to the callback into a plain Python scalar/list."""
    array = np.asarray(value)
    if array.size == 1:
        return array.reshape(()).item()
    return array.tolist()


def _input_fields(
    inputs: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Return the input fields of a record, calling `inputs` if it is a function."""
    if inputs is None:
        return {}
    # A `Mapping` could itself implement `__call__`, so the type check comes first.
    if isinstance(inputs, Mapping):
        return dict(cast(Mapping[str, Any], inputs))
    return dict(inputs())


class EventRecord(NamedTuple):
    """The fields of one record that `record_event` writes."""

    operation: str
    """The operation name the record carries."""

    solver: str | None = None
    """The solver type for a solver-specific operation, or None for a generic one."""

    inputs: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None
    """The input fields, or a function that returns them when a profile is active."""

    outputs: Mapping[str, Any] | None = None
    """The output fields."""


def record_event(*events: EventRecord) -> None:
    """Record operations as side effects.

    The records are written by one unordered `jax.experimental.io_callback`, in the order
    of `events`. They are written each time the traced program runs, also when no result
    depends on the operation. Their order relative to other records is not defined, and
    under `jax.vmap` each of them is written once. Use this function when an operation
    returns no JAX arrays for `record_operation` to entangle with.
    """
    if _active() is None:
        return
    callbacks = [
        _RecordCallback(
            operation=event.operation,
            solver=event.solver,
            input_fields=_input_fields(event.inputs),
            static_outputs=dict(event.outputs or {}),
            outputs_fn=None,
            value_ndims={},
        )
        for event in events
    ]

    def fire_all() -> None:
        for callback in callbacks:
            callback.fire({})

    io_callback(fire_all, None, ordered=False)


def record_operation(
    carrier: _CarrierT,
    operation: str,
    solver: str | None = None,
    *,
    inputs: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    outputs: Mapping[str, Any]
    | Callable[[Mapping[str, Any]], Mapping[str, Any]]
    | None = None,
    dynamic: Mapping[str, Any] | None = None,
    condition: Array | None = None,
    after: PyTree = (),
) -> _CarrierT:
    """Record one operation, and return `carrier` ordered after the record.

    The record is written by a pure `jax.pure_callback`. Its token is entangled with the
    arrays in `carrier`, so the callback fires after the producers of `carrier`. The
    returned `carrier` is entangled with the callback, so the consumers of the returned
    value run after the callback. Pass the arrays that the operation reads to record the
    start of the operation. Pass the arrays that the operation returns to record its end.
    When XLA removes the operation, the record is removed as well. Only the arrays of
    `carrier` are entangled and the other leaves are returned as they are.

    `solver` is None for a generic operation, or the solver type for a solver-specific one.
    `inputs` and `outputs` are fields known at profile time. `dynamic` holds runtime output
    arrays (rcond, residual norms, step, ...) that are read on the host and merged into the
    outputs. `inputs` may be a callable, evaluated only when a profile is active, so a
    caller can defer work (like reading index arrays) that would otherwise cost something
    when profiling is off. `outputs` may also be a callable taking the converted dynamic
    values, so fields that depend on a runtime branch (like a `reason` chosen by a
    `lax.cond`) can be built on the host, keeping the callback outside the cond.

    The arrays in `after` order the record in the same way as the arrays in `carrier`, but
    the function does not return them. Use it for values that the record must follow and
    that the caller does not pass on.

    `condition` is an optional runtime boolean: the record is appended only when it is true.
    This keeps an operation recordable only when it actually ran (like an iterative-refinement
    step) without placing the callback inside a `lax.cond`.

    Under `jax.vmap`, the callback fires one time and appends one record for each batch
    element. With no profile active this returns `carrier` before emitting anything, so the
    traced program is left untouched and profiling stays free when it is off. Which profile
    a firing appends into is decided when it fires, by looking up the active profile inside
    the callback, so a call to an already-compiled function made outside any `with` block
    records nothing.
    """
    if _active() is None:
        return carrier
    arrays, others = eqx.partition(carrier, eqx.is_array)
    leaves = jtu.tree_leaves(arrays)
    if not leaves:
        return carrier
    witnesses = leaves + jtu.tree_leaves(eqx.filter(after, eqx.is_array))
    input_fields = _input_fields(inputs)
    # Split `outputs` into its two shapes up front, so `_RecordCallback` holds each in a
    # field of its own type. Checked against `Mapping` rather than `callable`, since a
    # `Mapping` could itself implement `__call__` and `callable` alone would not rule
    # that branch out.
    if outputs is None or isinstance(outputs, Mapping):
        outputs_fn = None
        static_outputs = dict(outputs or {})
    else:
        outputs_fn = outputs
        static_outputs = {}
    dynamic_values = {
        key: jnp.asarray(jax.lax.stop_gradient(value))
        for key, value in (dynamic or {}).items()
    }
    if condition is not None:
        dynamic_values["__cond__"] = jnp.asarray(jax.lax.stop_gradient(condition))
    callback = _RecordCallback(
        operation=operation,
        solver=solver,
        input_fields=input_fields,
        static_outputs=static_outputs,
        outputs_fn=outputs_fn,
        value_ndims={key: value.ndim for key, value in dynamic_values.items()},
    )
    # A witness only orders the callback and carries no gradient. The callback is pure,
    # so when XLA removes the operation, the record is removed as well.
    token = entangle(jnp.zeros((), jnp.int32), *map(jax.lax.stop_gradient, witnesses))
    witness = jax.pure_callback(
        callback,
        jax.ShapeDtypeStruct((), jnp.int32),
        token,
        dynamic_values,
        vmap_method="expand_dims",
    )
    return eqx.combine(entangle(arrays, witness), others)
