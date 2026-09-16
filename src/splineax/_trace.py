"""An opt-in solve trace for debugging the stateful sparse solvers.

The sparse solvers in this package do a lot of hidden work: symbolic analyze, numeric
factor, refactor-with-reuse, triangular solve, and iterative-refinement steps. Across a
state-sequence (`init`, `update`, `update`, ..., `release`) the choices they make (reuse a
factorization or rebuild it, refactor or reanalyze, how many refinement steps) are
invisible from the outside.

`solve_trace` turns that on. Inside the context manager every relevant action is recorded
into a `SolveTrace` and printed as an indented tree:

```{.python notest}
with splineax.solve_trace() as trace:
    solution, state = splineax.linear_solve(operator, vector, solver)
    state.release()
print(trace)
```

Each line is one operation. A **generic** operation is one of the stateful-API steps (`init`,
`init_symbolic`, `update`, `compute`, `track`, `release`), written bare. Nested under it are
the **solver-specific** operations it ran, written `SolverType.function` (e.g. `KLU.analyze`).
An operation is rendered `operation[inputs] => (outputs)`.

The records are appended through `jax.experimental.io_callback`s, which fire during execution;
each record carries a trace-time order index, so the printed tree is in program order even
though the callbacks are unordered. When no trace is active nothing is emitted into the traced
program, so tracing costs nothing when it is off.
"""

import contextlib
import dataclasses
import os
import sys
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import jax
from jax.experimental import io_callback

# Per-thread state: the stack of open traces, a monotonic order counter stamped on each
# record (so the tree prints in program order despite unordered callbacks), and the current
# `compute` nesting depth (so only the outermost `compute` emits the generic boundary).
_LOCAL = threading.local()


def _stack() -> list["SolveTrace"]:
    stack = getattr(_LOCAL, "stack", None)
    if stack is None:
        stack = []
        _LOCAL.stack = stack
    return stack


def _active() -> "SolveTrace | None":
    """The innermost open trace on this thread, or None when tracing is off."""
    stack = _stack()
    return stack[-1] if stack else None


def tracing_active() -> bool:
    """Whether a `solve_trace` is currently open on this thread."""
    return bool(_stack())


def _next_order() -> int:
    order = getattr(_LOCAL, "order", 0)
    _LOCAL.order = order + 1
    return order


# Generic operations that begin a new state-sequence when grouping the log.
_SEQUENCE_STARTS = frozenset({"init", "init_symbolic"})


@dataclasses.dataclass(frozen=True)
class TraceRecord:
    """One recorded operation in a `SolveTrace`.

    `operation` is the operation name. A **generic** operation (`init`, `init_symbolic`,
    `update`, `compute`, `track`, `release`) has `solver` None. A **solver-specific**
    operation has `solver` set to the solver type (`KLU`, `Pardiso`, `Spsolve`,
    `IterativeRefinement`) and `operation` set to the library function it ran (`analyze`,
    `factor`, `solve_with_numeric`, ...). `inputs` and `outputs` hold the fields rendered as
    `operation[inputs] => (outputs)`. `order` is the trace-time program order.
    """

    operation: str
    solver: str | None = None
    inputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    outputs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    order: int = 0


# ANSI colours for the pretty tree. Kept tiny and dependency-free.
_RESET = "\033[0m"
_DIM = "\033[2m"
_CREATED = "\033[33m"  # yellow: a factorization built anew
_REUSED = "\033[32m"  # green: an existing factorization reused
_BOLD = "\033[1m"

# Solver-specific operations coloured as building a factorization anew.
_CREATED_OPS = frozenset({"analyze", "reanalyze", "spsolve"})
# Solver-specific operations coloured as reusing an existing factorization.
_REUSED_OPS = frozenset({"solve_with_numeric", "tsolve_with_numeric", "solve_stateful"})
# The floating-point output fields shown in scientific notation.
_SCI_FIELDS = frozenset({"rcond", "residual_norm", "threshold"})
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
    get the same digest. Consistent within a run; not stable across runs.
    """
    if tag is None:
        return None
    return f"0x{hash(tag) & 0xFFFFF:05x}"


def _record_colour(record: TraceRecord) -> str:
    """The colour for a record, by whether it builds a factorization or reuses one."""
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
    if operation == "refactor":
        return _CREATED if record.outputs.get("reused") is False else _REUSED
    if operation == "factor":
        return _REUSED if record.outputs.get("reused") is True else _CREATED
    if operation in _REUSED_OPS:
        return _REUSED
    return _DIM


def _format_value(field: str, value: Any) -> str:
    if field in _SCI_FIELDS and isinstance(value, (int, float)):
        return f"{field}={value:.3e}"
    if field in _TEXT_FIELDS:
        return f'{field}="{value}"'
    return f"{field}={value}"


def _format_fields(fields: Mapping[str, Any]) -> str:
    present = [(key, value) for key, value in fields.items() if value is not None]
    present.sort(key=lambda kv: _FIELD_ORDER.get(kv[0], len(_FIELD_ORDER)))
    return ", ".join(_format_value(key, value) for key, value in present)


def _format_record(record: TraceRecord) -> str:
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


def _want_colour(colour: bool | None) -> bool:
    if colour is not None:
        return colour
    if os.environ.get("NO_COLOR") is not None:
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class SolveTrace:
    """An ordered log of the operations taken across one or more state-sequences.

    Collected by `solve_trace`. `records` is the log; `sequences` slices it into one list per
    state-sequence (each generic `init`/`init_symbolic` starts a new one). Printing a
    `SolveTrace` renders the indented tree, with factorizations built anew and factorizations
    reused shown in different colours.
    """

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def _append(self, record: TraceRecord) -> None:
        # Runs on the io_callback thread; `list.append` is atomic under the GIL.
        self.records.append(record)

    def _in_order(self) -> list[TraceRecord]:
        # The callbacks are unordered, so sort by the trace-time order index for a stable,
        # program-order view.
        return sorted(self.records, key=lambda record: record.order)

    @property
    def sequences(self) -> list[list[TraceRecord]]:
        """The records grouped into state-sequences, split at each generic init."""
        groups: list[list[TraceRecord]] = []
        current: list[TraceRecord] | None = None
        for record in self._in_order():
            starts = record.solver is None and record.operation in _SEQUENCE_STARTS
            if current is None or starts:
                current = []
                groups.append(current)
            current.append(record)
        return groups

    def render(self, colour: bool | None = None) -> str:
        """Render the tree. `colour` forces ANSI on/off; None auto-detects a TTY."""
        use_colour = _want_colour(colour)

        def paint(text: str, code: str) -> str:
            return f"{code}{text}{_RESET}" if use_colour else text

        # The colour key only means anything in colour, so show it only then.
        if use_colour:
            header = (
                "solve trace   key: "
                + paint("created", _CREATED)
                + " / "
                + paint("reused", _REUSED)
            )
        else:
            header = "solve trace"
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
                    indent + paint(_format_record(record), _record_colour(record))
                )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()


@contextlib.contextmanager
def solve_trace() -> Iterator[SolveTrace]:
    """Record every operation the sparse solvers take within this block, for debugging.

    Returns a `SolveTrace` collecting the analyze, factor, refactor, solve, and
    iterative-refinement operations run inside the block. Tracing is scoped to the block and
    to the current thread, and adds nothing to the traced program when no block is open.

    ```{.python notest}
    with splineax.solve_trace() as trace:
        solution, state = splineax.linear_solve(operator, vector, solver)
        state.release()
    print(trace)
    ```

    Because the records are added at trace time, only code traced inside the block is
    recorded: a function compiled outside the block and called within it emits nothing. The
    callbacks do not compose with `jax.vmap` and may conflict with reverse-mode autodiff, so
    trace plain solves rather than solves under `vmap`/`grad`.
    """
    trace = SolveTrace()
    stack = _stack()
    stack.append(trace)
    try:
        yield trace
    finally:
        # The records are appended from `io_callback`s, which run asynchronously. Flush any
        # still in flight so the trace is complete once the block exits.
        jax.effects_barrier()
        stack.pop()


@contextlib.contextmanager
def compute_scope() -> Iterator[None]:
    """Emit the generic `compute` boundary once, at the outermost solver's `compute`.

    A wrapping solver's `compute` (e.g. `IterativeRefinement`) calls an inner solver's
    `compute`; this suppresses the inner boundary so one user solve is one generic `compute`,
    with every solver-specific operation nested under it.
    """
    depth = getattr(_LOCAL, "compute_depth", 0)
    if depth == 0:
        record_event("compute")
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


def record_event(
    operation: str,
    solver: str | None = None,
    *,
    inputs: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    outputs: Mapping[str, Any] | None = None,
    dynamic: Mapping[str, Any] | None = None,
) -> None:
    """Append one operation to the active trace, or do nothing when tracing is off.

    `solver` is None for a generic operation, or the solver type for a solver-specific one.
    `inputs` and `outputs` are fields known at trace time; `dynamic` holds runtime output
    arrays (rcond, residual norms, step, ...) read on the host through an unordered
    `io_callback` and merged into the outputs. `inputs` may be a callable, evaluated only when
    a trace is active, so a caller can defer work (like reading index arrays) that would
    otherwise cost something when tracing is off. When no trace is active this returns before
    emitting any callback, so it leaves the traced program untouched.
    """
    trace = _active()
    if trace is None:
        return
    order = _next_order()
    if inputs is None:
        input_fields: dict[str, Any] = {}
    elif isinstance(inputs, Mapping):
        input_fields = dict(inputs)
    else:
        input_fields = dict(inputs())
    static_outputs = dict(outputs or {})
    dynamic_values = {
        key: jax.lax.stop_gradient(value) for key, value in (dynamic or {}).items()
    }

    def _callback(values: Mapping[str, Any]) -> None:
        merged = dict(static_outputs)
        for key, value in values.items():
            merged[key] = _to_python(value)
        trace._append(
            TraceRecord(
                operation=operation,
                solver=solver,
                inputs=input_fields,
                outputs=merged,
                order=order,
            )
        )

    io_callback(_callback, (), dynamic_values, ordered=False)
