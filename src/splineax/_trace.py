"""An opt-in solve trace for debugging the stateful sparse solvers.

The sparse solvers in this package do a lot of hidden work: symbolic analyze, numeric
factor, refactor-with-reuse, triangular solve, and iterative-refinement steps. Across a
state-sequence (`init`, `update`, `update`, ..., `release`) the choices they make (reuse a
factorization or rebuild it, refactor or reanalyze, how many refinement steps) are
invisible from the outside.

`solve_trace` turns that on. Inside the context manager every relevant action is recorded,
in program order, into a `SolveTrace`:

```{.python notest}
with splineax.solve_trace() as trace:
    solution, state = splineax.linear_solve(operator, vector, solver)
    state.release()
print(trace)
```

The records are appended through ordered `jax.experimental.io_callback`s, so they keep their
true order even under `jax.jit` and inside control flow. When no trace is active nothing is
emitted into the traced program at all, so tracing costs nothing when it is off.
"""

import contextlib
import dataclasses
import os
import sys
import threading
from collections.abc import Iterator, Mapping
from typing import Any

import jax
from jax.experimental import io_callback

# Per-thread stack of the traces currently open, so nested `solve_trace()` blocks work and
# concurrent traces on different threads never share a log.
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


# Actions that begin a new state-sequence when grouping a flat log.
_SEQUENCE_STARTS = frozenset({"init", "init_symbolic"})

# Actions coloured as building a factorization anew, versus reusing an existing one. Every
# other action is neutral. `refactor` and `update` decide their colour from `reused`, so they
# are not listed here.
_CREATED_ACTIONS = frozenset({"analyze", "factor"})
_REUSED_ACTIONS = frozenset({"refactor", "solve", "track"})


@dataclasses.dataclass(frozen=True)
class TraceRecord:
    """One recorded action in a `SolveTrace`.

    `action` is the kind of work (`init`, `init_symbolic`, `update`, `analyze`, `factor`,
    `refactor`, `solve`, `track`, `release`, `ir_start`, `ir_step`, `ir_result`) and
    `backend` is the solver that did it (`klu`, `pardiso`, `spsolve`, `iterative_refinement`).
    The remaining fields are populated only where they apply, so most are None on any given
    record.
    """

    action: str
    backend: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    nnz: int | None = None
    symbolic: bool | None = None
    transposed: bool | None = None
    outcome: str | None = None
    """For `update`: `noop` (same operator), `reused` (analysis reused), or `rebuilt` (a new
    analysis because the sparsity pattern changed)."""
    note: str | None = None
    reused: bool | None = None
    rcond: float | None = None
    residual_norm: float | None = None
    threshold: float | None = None
    step: int | None = None
    perturbed_pivots: int | None = None
    zero_pivot: bool | None = None
    converged: bool | None = None


# ANSI colours for the pretty tree. Kept tiny and dependency-free.
_RESET = "\033[0m"
_DIM = "\033[2m"
_CREATED = "\033[33m"  # yellow: a factorization built anew
_REUSED = "\033[32m"  # green: an existing factorization reused
_BOLD = "\033[1m"

# The fields shown, in order, after `action [backend]` on a record line.
_DETAIL_FIELDS = (
    "outcome",
    "shape",
    "nnz",
    "dtype",
    "symbolic",
    "transposed",
    "reused",
    "rcond",
    "step",
    "residual_norm",
    "threshold",
    "perturbed_pivots",
    "zero_pivot",
    "converged",
    "note",
)


def _record_colour(record: TraceRecord) -> str:
    """The colour for a record, by whether it builds a factorization or reuses one."""
    if record.action in _CREATED_ACTIONS:
        return _CREATED
    if record.action == "refactor":
        return _CREATED if record.reused is False else _REUSED
    if record.action == "update":
        if record.outcome == "rebuilt":
            return _CREATED
        if record.outcome == "reused":
            return _REUSED
        return _DIM
    if record.action in _REUSED_ACTIONS:
        return _REUSED
    return _DIM


def _format_value(field: str, value: Any) -> str:
    if field in ("rcond", "residual_norm", "threshold") and isinstance(
        value, (int, float)
    ):
        return f"{field}={value:.3e}"
    return f"{field}={value}"


def _format_details(record: TraceRecord) -> str:
    parts = [
        _format_value(field, getattr(record, field))
        for field in _DETAIL_FIELDS
        if getattr(record, field) is not None
    ]
    return " ".join(parts)


def _want_colour(colour: bool | None) -> bool:
    if colour is not None:
        return colour
    if os.environ.get("NO_COLOR") is not None:
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class SolveTrace:
    """An ordered log of the actions taken across one or more state-sequences.

    Collected by `solve_trace`. `records` is the flat log in execution order; `sequences`
    slices it into one list per state-sequence (each `init`/`init_symbolic` starts a new
    one). Printing a `SolveTrace` renders an indented, coloured tree, with factorizations
    built anew and factorizations reused shown in different colours.
    """

    def __init__(self) -> None:
        self.records: list[TraceRecord] = []

    def _append(self, record: TraceRecord) -> None:
        # Runs on the io_callback thread; `list.append` is atomic under the GIL.
        self.records.append(record)

    @property
    def sequences(self) -> list[list[TraceRecord]]:
        """The records grouped into state-sequences, split at each `init`/`init_symbolic`."""
        groups: list[list[TraceRecord]] = []
        current: list[TraceRecord] | None = None
        for record in self.records:
            if current is None or record.action in _SEQUENCE_STARTS:
                current = []
                groups.append(current)
            current.append(record)
        return groups

    def render(self, colour: bool | None = None) -> str:
        """Render the tree. `colour` forces ANSI on/off; None auto-detects a TTY."""
        use_colour = _want_colour(colour)

        def paint(text: str, code: str) -> str:
            return f"{code}{text}{_RESET}" if use_colour else text

        lines: list[str] = []
        legend = (
            "solve trace  "
            + paint("created", _CREATED)
            + "  "
            + paint("reused", _REUSED)
        )
        lines.append(legend)
        if not self.records:
            lines.append("  (empty)")
            return "\n".join(lines)

        for index, sequence in enumerate(self.sequences):
            head = sequence[0]
            label = f"sequence {index} [{head.backend}]"
            if head.shape is not None:
                label += f" shape={head.shape}"
            lines.append(paint(label, _BOLD))
            for record in sequence:
                # Refinement bookkeeping nests a level under the solve it belongs to.
                indent = "    " if record.action.startswith("ir_") else "  "
                details = _format_details(record)
                text = f"{record.action} [{record.backend}]"
                if details:
                    text += f"  {details}"
                lines.append(indent + paint(text, _record_colour(record)))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()


@contextlib.contextmanager
def solve_trace() -> Iterator[SolveTrace]:
    """Record every action the sparse solvers take within this block, for debugging.

    Returns a `SolveTrace` collecting the analyze, factor, refactor, solve, and
    iterative-refinement steps run inside the block, in program order. Tracing is scoped to
    the block and to the current thread, and adds nothing to the traced program when no block
    is open.

    ```{.python notest}
    with splineax.solve_trace() as trace:
        solution, state = splineax.linear_solve(operator, vector, solver)
        state.release()
    print(trace)
    ```

    Because the records are added at trace time, only code traced inside the block is
    recorded: a function compiled outside the block and called within it emits nothing.
    Ordered `io_callback`s do not compose with `jax.vmap` and may conflict with reverse-mode
    autodiff, so trace plain solves rather than solves under `vmap`/`grad`.
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


def _to_python(value: Any) -> Any:
    """Convert a runtime array handed to the callback into a plain Python scalar/list."""
    array = jax.numpy.asarray(value)
    if array.size == 1:
        return array.reshape(()).item()
    return array.tolist()


def record_event(
    action: str,
    backend: str,
    *,
    dynamic: Mapping[str, Any] | None = None,
    ordered: bool = True,
    **static: Any,
) -> None:
    """Append one event to the active trace, or do nothing when tracing is off.

    `static` fields are known at trace time (shape, dtype, outcome, ...). `dynamic` holds
    runtime arrays (rcond, residual norms, step, pivot flags, ...) that are read on the host
    through an `io_callback`. When no trace is active this returns before emitting any
    callback, so it leaves the traced program untouched.

    `ordered` uses an ordered `io_callback`, which keeps events in strict program order and is
    the default. Events emitted from inside a solver's `compute` (the solve itself and the
    iterative-refinement steps) must pass `ordered=False`, because they run inside lineax's
    `linear_solve` primitive, which does not carry ordered effects. Those events are still
    logged in order in practice: each sits between the ordered `update` and `track` of its
    solve, and the refinement steps chain through the loop carry.
    """
    trace = _active()
    if trace is None:
        return
    dynamic_values = {
        key: jax.lax.stop_gradient(value) for key, value in (dynamic or {}).items()
    }

    def _callback(values: Mapping[str, Any]) -> None:
        fields = dict(static)
        for key, value in values.items():
            fields[key] = _to_python(value)
        trace._append(TraceRecord(action=action, backend=backend, **fields))

    io_callback(_callback, (), dynamic_values, ordered=ordered)
