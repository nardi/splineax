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

The records are appended through `jax.experimental.io_callback`s, which fire during
execution and, each time they fire, append into whichever `SolveProfile` is currently
entered on that thread, or nowhere if none is. Each record carries an order key taken
when it is traced, so the printed tree is in program order even though the callbacks fire
out of order. When no profile is ever active for a given compiled shape, nothing is emitted
into the traced program at all, so profiling costs nothing when it is off.
"""

import contextlib
import dataclasses
import functools
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from types import TracebackType
from typing import Any, ParamSpec, TypeVar

import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from jaxtyping import Array

# Per-thread state: the stack of open profiles, a monotonic order counter stamped on each
# record (so the tree prints in program order despite unordered callbacks), and the current
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


def profiling_active() -> bool:
    """Whether a `SolveProfile` is currently entered on this thread."""
    return bool(_stack())


def _next_order() -> tuple[int, ...]:
    """Return the sort key for the next record, which places it in program order.

    Outside an `order_slot_scope`, the key is the counter alone. Inside one, the key is the
    slot followed by the counter, so the record sorts at the slot's position. The counter
    only grows, so records within one slot keep the order they were traced in.
    """
    counter = getattr(_LOCAL, "order", 0)
    _LOCAL.order = counter + 1
    return getattr(_LOCAL, "order_slot", ()) + (counter,)


def reserve_order_slot() -> tuple[int, ...]:
    """Reserve the current position in program order for records traced later.

    A caller reserves a slot where an operation sits in the program. It then passes that slot
    to `order_slot_scope` around the code that traces the operation, which may run long after
    the code around it has been traced.
    """
    return _next_order()


@contextlib.contextmanager
def order_slot_scope(order_slot: tuple[int, ...]) -> Iterator[None]:
    """Sort every record traced inside this block at `order_slot`.

    `order_slot` comes from `reserve_order_slot`. Scopes nest, and the previous slot is
    restored on exit.
    """
    previous_slot = getattr(_LOCAL, "order_slot", ())
    _LOCAL.order_slot = order_slot
    try:
        yield
    finally:
        _LOCAL.order_slot = previous_slot


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
    `operation[inputs] => (outputs)`. `order` is the sort key for program order, taken
    when the operation is traced.
    """

    operation: str
    solver: str | None = None
    inputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    outputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    order: tuple[int, ...] = ()


# ANSI colors for the pretty tree. Kept tiny and dependency-free.
_RESET = "\033[0m"
_DIM = "\033[2m"
# Yellow: a factorization built anew.
_CREATED = "\033[33m"
# Green: an existing factorization reused.
_REUSED = "\033[32m"
_BOLD = "\033[1m"

# Solver-specific operations colored as building a factorization anew.
_CREATED_OPS = frozenset({"analyze", "reanalyze", "spsolve"})
# Solver-specific operations colored as reusing an existing factorization.
_REUSED_OPS = frozenset({"solve_with_numeric", "tsolve_with_numeric", "solve_stateful"})
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
        case "refactor":
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
    """An ordered log of the operations taken across one or more state-sequences.

    Create one with `create_solve_profile`. `SolveProfile` is itself a context manager: entering
    it (`with profile:`) makes it the active profile on this thread, so every operation recorded
    while it is entered lands in it, and nothing recorded outside any `with` block, or while a
    different profile is entered, does. A profile can be entered more than once, accumulating
    further records each time.

    `records` is the log, and `sequences` slices it into one list per state-sequence (each
    generic `init`/`init_symbolic` starts a new one). Printing a `SolveProfile` renders the
    indented tree, with factorizations built anew and factorizations reused shown in
    different colors.
    """

    def __init__(self) -> None:
        self.records: list[ProfileRecord] = []

    def __enter__(self) -> "SolveProfile":
        """Make this profile the active one on this thread, for every operation inside."""
        _stack().append(self)
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

    def _append(self, record: ProfileRecord) -> None:
        # Runs on the io_callback thread. `list.append` is atomic under the GIL.
        self.records.append(record)

    def _in_order(self) -> list[ProfileRecord]:
        # The callbacks are unordered, so sort by the trace-time order key for a stable,
        # program-order view.
        return sorted(self.records, key=lambda record: record.order)

    @property
    def sequences(self) -> list[list[ProfileRecord]]:
        """The records grouped into state-sequences, split at each generic init."""
        groups: list[list[ProfileRecord]] = []
        current: list[ProfileRecord] | None = None
        for record in self._in_order():
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
        if not self.records:
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


_P = ParamSpec("_P")
_ReturnT = TypeVar("_ReturnT")


def profile_solves(
    fn: Callable[_P, _ReturnT],
) -> Callable[_P, tuple[_ReturnT, SolveProfile | None]]:
    """Wrap `fn` so every call profiles its solves into a fresh `SolveProfile`.

    Returns `(result, profile)` instead of `fn`'s own return value. Pass `enabled=False` on
    a call to skip profiling it, returning `(result, None)` and calling `fn` directly.

    Apply this directly to a `jax.jit`/`equinox.filter_jit`-decorated function, and always
    call it through the decorated name: used this way, the very first call for a given
    input shape necessarily happens through this wrapper, with a profile active, so that
    shape's compiled executable is guaranteed to carry its profiling hooks. Calling the
    undecorated jit function separately, or making the first call for a shape with
    `enabled=False`, permanently forfeits profiling for that shape, since the hooks are
    only ever built in at compile time.

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

    @functools.wraps(fn)
    def wrapper(
        # `enabled` sits between the paramspec halves, which PEP 612 disallows even
        # though it works fine at runtime.
        *args: _P.args,
        enabled: bool = True,  # ty: ignore[invalid-paramspec]
        **kwargs: _P.kwargs,
    ) -> tuple[_ReturnT, SolveProfile | None]:
        if not enabled:
            return fn(*args, **kwargs), None
        profile = create_solve_profile()
        with profile:
            result = fn(*args, **kwargs)
        return result, profile

    return wrapper


@contextlib.contextmanager
def compute_scope() -> Iterator[None]:
    """Emit the generic `compute` boundary once, at the outermost solver's `compute`.

    A wrapping solver's `compute` (e.g. `IterativeRefinement`) calls an inner solver's
    `compute`, and this suppresses the inner boundary so one user solve is one generic
    `compute`, with every solver-specific operation nested under it.
    """
    depth = getattr(_LOCAL, "compute_depth", 0)
    if depth == 0:
        record_operation("compute")
    _LOCAL.compute_depth = depth + 1
    try:
        yield
    finally:
        _LOCAL.compute_depth = depth


def _to_python(value: Any) -> Any:
    """Convert a runtime array handed to the callback into a plain Python scalar/list."""
    array = jax.numpy.asarray(value)
    if array.size == 1:
        return array.reshape(()).item()
    return array.tolist()


def record_operation(
    operation: str,
    solver: str | None = None,
    *,
    inputs: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    outputs: Mapping[str, Any]
    | Callable[[Mapping[str, Any]], Mapping[str, Any]]
    | None = None,
    dynamic: Mapping[str, Any] | None = None,
    condition: Array | None = None,
) -> None:
    """Build a callback that appends one operation into whichever profile is active when it fires.

    `solver` is None for a generic operation, or the solver type for a solver-specific one.
    `inputs` and `outputs` are fields known at profile time. `dynamic` holds runtime output
    arrays (rcond, residual norms, step, ...) read on the host through an unordered
    `io_callback` and merged into the outputs. `inputs` may be a callable, evaluated only when
    a profile is active, so a caller can defer work (like reading index arrays) that would
    otherwise cost something when profiling is off. `outputs` may also be a callable taking the
    converted dynamic values, so fields that depend on a runtime branch (like a `reason`
    chosen by a `lax.cond`) can be built on the host, keeping the callback outside the cond.

    `condition` is an optional runtime boolean: the record is appended only when it is true. This
    keeps an operation recordable only when it actually ran (like an iterative-refinement
    step) without placing the callback inside a `lax.cond`, where an IO effect breaks
    `vmap`-of-cond.

    With no profile active this returns before emitting anything, so the traced program is
    left untouched and profiling stays free when it is off. Which profile a firing appends
    into is decided when it fires, by looking up the active profile inside the callback, so
    a call to an already-compiled function made outside any `with` block records nothing.
    """
    if _active() is None:
        return
    order = _next_order()
    if inputs is None:
        input_fields: dict[str, Any] = {}
    elif isinstance(inputs, Mapping):
        input_fields = dict(inputs)
    else:
        input_fields = dict(inputs())
    # Split `outputs` into its two shapes up front. A nested function does not narrow a
    # union type captured from the enclosing scope, so the branch has to happen here,
    # not inside `_callback`. Checked against `Mapping` rather than `callable`, since a
    # `Mapping` could itself implement `__call__` and `callable` alone would not rule
    # that branch out.
    if outputs is None or isinstance(outputs, Mapping):
        outputs_fn = None
        static_outputs = dict(outputs or {})
    else:
        outputs_fn = outputs
        static_outputs = {}
    dynamic_values = {
        key: jax.lax.stop_gradient(value) for key, value in (dynamic or {}).items()
    }
    conditional = condition is not None
    if conditional:
        dynamic_values["__cond__"] = jax.lax.stop_gradient(jnp.asarray(condition))

    def _callback(values: Mapping[str, Any]) -> None:
        profile = _active()
        if profile is None:
            return
        if conditional and not values["__cond__"]:
            return
        if conditional:
            values = {key: value for key, value in values.items() if key != "__cond__"}
        merged: dict[str, Any] = {}
        for key, value in values.items():
            merged[key] = _to_python(value)
        if outputs_fn is not None:
            merged.update(outputs_fn(merged))
        else:
            merged.update(static_outputs)
        profile._append(
            ProfileRecord(
                operation=operation,
                solver=solver,
                inputs=input_fields,
                outputs=merged,
                order=order,
            )
        )

    io_callback(_callback, (), dynamic_values, ordered=False)
