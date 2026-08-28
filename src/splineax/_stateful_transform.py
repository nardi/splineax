"""Turn naive `lineax.linear_solve` code into stateful solving.

`stateful_solve_transform` wraps a function that calls `lineax.linear_solve` and threads a
solver state through its solves, so they reuse a factorization, using only the generic
stateful API (`init`, `update`, `release`, and a state's `track`).

A solve inside a `lax.scan`, `lax.while_loop`, or `lax.cond` is not threaded, and raises
rather than silently missing the reuse.
"""

from collections.abc import Callable, Mapping
from typing import Any, Generic, NamedTuple, Protocol, TypeVar, cast, overload

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
from jax import make_jaxpr
from jax._src.interpreters.partial_eval import dce_jaxpr
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Primitive, Var
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver
from lineax._solution import RESULTS
from lineax._solve import linear_solve_p

from splineax.solvers._stateful import StatefulSolver, TrackingState

_OutputT = TypeVar("_OutputT")
"""The return type of the wrapped function."""

_StateT = TypeVar("_StateT")
"""A solver state, whose concrete type depends on the solver."""

_Atom = Var | Literal
"""A jaxpr variable or a literal, the two things an equation reads from."""

_FilterSolver = type | Callable[[AbstractLinearSolver], bool]
"""A rule for choosing which solves to thread.

Either a solver class or protocol matched by `isinstance`, or a boolean predicate on the
solver.
"""

_SolveArguments = tuple[
    AbstractLinearOperator,
    PyTree,
    PyTree[Array],
    Mapping[str, Any],
    AbstractLinearSolver,
    bool,
]
"""The six pytree arguments `lineax.linear_solve` binds.

In order: operator, state, vector, options, solver, and the throw flag. `options` values are
solver-defined, so they stay `Any`.
"""

_SolveResult = tuple[PyTree[Array], RESULTS, dict[str, Any]]
"""What binding `linear_solve_p` returns: the solution, the result code, and the stats.

The stats values are solver-defined, so they stay `Any`.
"""

_INLINE_PRIMITIVES = frozenset({"pjit", "jit", "closed_call", "core_call"})
"""The higher-order primitives whose body we inline by interpreting it.

Each is a pure staging boundary, so running its body in place computes the same values.
`lineax` wraps every solve in one of these, so inlining is required, not an optimisation.
"""


def _solver_matches(solver: AbstractLinearSolver, filter_solver: _FilterSolver) -> bool:
    """Whether the filter selects this solver, as a class or a predicate."""
    if isinstance(filter_solver, type):
        return isinstance(solver, filter_solver)
    return filter_solver(solver)


def _stop_gradient_leaves(tree: PyTree) -> PyTree:
    """Stop gradients on the array leaves of a pytree, leaving the rest untouched.

    `lineax`'s solve primitive asserts its state carries no tangent, so the state and the
    operator it is built from are stopped before the bind. Kept generic here rather than
    relying on a solver stopping its own values.
    """
    dynamic, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.lax.stop_gradient(dynamic), static)


def _is_runtime_value(leaf: object) -> bool:
    """Whether a leaf is a runtime value the interpreter writes to an output variable.

    Tracers are `jax.Array` instances, so this one check covers both concrete arrays and the
    tracers that stand in for them while staging.
    """
    return isinstance(leaf, jax.Array)


def _runtime_value_leaves(tree: PyTree) -> list[Array]:
    """The array leaves of a pytree, matching an equation's runtime output variables."""
    return [leaf for leaf in jax.tree_util.tree_leaves(tree) if _is_runtime_value(leaf)]


def _reconstruct_solve_arguments(eqn: JaxprEqn, operands: list[Any]) -> _SolveArguments:
    """Rebuild the arguments `lineax.linear_solve` bound in this equation.

    `equinox.internal._primitive.filter_primitive_bind` splits the pytree arguments into
    array operands, an equation's invars, and a static half carrying the treedef plus the
    non-array leaves, with a `_missing_dynamic` sentinel where each operand goes. The private
    `_combine` splices the operand values back into that static half, and the treedef
    restores the original pytree. This inverts that encoding, the one private-internals
    dependency, and the round-trip is guarded by a test.
    """
    from equinox.internal._primitive import _combine  # noqa: PLC2701

    static = cast(tuple[Any, ...], eqn.params["static"])
    treedef = cast(jax.tree_util.PyTreeDef, eqn.params["treedef"])
    flat = _combine(list(operands), static)
    return cast(_SolveArguments, jax.tree_util.tree_unflatten(treedef, flat))


def _rebind_solve(arguments: _SolveArguments) -> _SolveResult:
    """Bind `linear_solve_p` with these arguments, returning `(solution, result, stats)`."""
    return cast(
        _SolveResult,
        eqxi.filter_primitive_bind(linear_solve_p, *arguments),
    )


def _nested_jaxprs(eqn: JaxprEqn) -> list[Jaxpr]:
    """Every jaxpr nested in an equation's params."""
    found: list[Jaxpr] = []
    for value in eqn.params.values():
        match value:
            case Jaxpr():
                found.append(value)
            case ClosedJaxpr():
                found.append(value.jaxpr)
            case tuple() | list():
                for item in value:
                    match item:
                        case ClosedJaxpr():
                            found.append(item.jaxpr)
                        case Jaxpr():
                            found.append(item)
    return found


def _jaxpr_has_selected_solve(jaxpr: Jaxpr, filter_solver: _FilterSolver) -> bool:
    """Whether a jaxpr, or any it nests, holds a solve the filter would thread.

    The solver is a static param, so it reads out without running anything, by
    reconstructing with placeholder operands.
    """
    for eqn in jaxpr.eqns:
        if eqn.primitive is linear_solve_p:
            arguments = _reconstruct_solve_arguments(eqn, [None] * len(eqn.invars))
            if _solver_matches(arguments[4], filter_solver):
                return True
        if any(
            _jaxpr_has_selected_solve(inner, filter_solver)
            for inner in _nested_jaxprs(eqn)
        ):
            return True
    return False


class _StateThreadingInterpreter(Generic[_StateT]):
    """Walks a jaxpr like `eval_jaxpr`, threading one solver state through the solves.

    An instance carries the state as it goes and remembers the last solver it threaded, so
    the caller can release the state when it is not handed back.
    """

    filter_solver: _FilterSolver
    """Chooses which solves to thread, as a solver class or a predicate."""

    state: _StateT | None
    """The state threaded so far, or `None` before the first matched solve."""

    solver: StatefulSolver[_StateT] | None
    """The last solver threaded, used by the caller to release the state."""

    def __init__(self, filter_solver: _FilterSolver, state: _StateT | None) -> None:
        self.filter_solver = filter_solver
        self.state = state
        self.solver = None

    def interpret(self, jaxpr: Jaxpr, consts: list[Any], args: list[Any]) -> list[Any]:
        """Evaluate the jaxpr against these argument values, returning its output values."""
        env: dict[Var, Any] = {}

        def read(atom: _Atom) -> Any:
            return atom.val if isinstance(atom, Literal) else env[atom]

        def write(var: Var, value: Any) -> None:
            env[var] = value

        for constvar, constval in zip(jaxpr.constvars, consts):
            write(constvar, constval)
        for invar, arg in zip(jaxpr.invars, args):
            write(invar, arg)

        for eqn in jaxpr.eqns:
            operands = [read(v) for v in eqn.invars]
            outputs = self._process_equation(eqn, operands)
            for outvar, value in zip(eqn.outvars, outputs):
                write(outvar, value)

        return [read(v) for v in jaxpr.outvars]

    def _process_equation(self, eqn: JaxprEqn, operands: list[Any]) -> list[Any]:
        """Produce one equation's output values, threading the state through a matched solve.

        A matched `linear_solve_p` is threaded, an inline primitive has its body interpreted
        in place, a higher-order primitive that nests a matched solve raises, and everything
        else is rebound as `jax.core.eval_jaxpr` would.
        """
        primitive: Primitive = eqn.primitive
        if primitive is linear_solve_p:
            arguments = _reconstruct_solve_arguments(eqn, operands)
            if _solver_matches(arguments[4], self.filter_solver):
                return self._thread_solve(arguments)
            return _runtime_value_leaves(_rebind_solve(arguments))

        if primitive.name in _INLINE_PRIMITIVES:
            inner = cast(ClosedJaxpr, eqn.params["jaxpr"])
            return self.interpret(inner.jaxpr, inner.consts, operands)

        nested = _nested_jaxprs(eqn)
        if any(
            _jaxpr_has_selected_solve(inner, self.filter_solver) for inner in nested
        ):
            raise NotImplementedError(
                "`stateful_solve_transform` cannot thread a solver state through a solve "
                f"inside `{primitive.name}`. Move the solve out of it, or drop the "
                "transform for this function."
            )

        bind_params = primitive.get_bind_params(eqn.params)
        result = primitive.bind(*operands, **bind_params)
        return list(result) if primitive.multiple_results else [result]

    def _thread_solve(self, arguments: _SolveArguments) -> list[Any]:
        """Update the state for this solve's operator, solve, and track the solution.

        The operator and the state are stopped before the bind so the solve's autodiff rule
        sees no state tangent, matching what `lineax.linear_solve` does with the state it
        builds. Returns the solve's output values.
        """
        operator, _old_state, vector, options, solver_any, throw = arguments
        solver = cast(StatefulSolver[_StateT], solver_any)
        stopped_operator = _stop_gradient_leaves(operator)
        if self.state is None:
            self.state = solver.init(stopped_operator, {})
        else:
            self.state = solver.update(self.state, stopped_operator, {})
        self.solver = solver
        solution, result_code, stats = _rebind_solve(
            (
                operator,
                _stop_gradient_leaves(self.state),
                vector,
                options,
                solver_any,
                throw,
            )
        )
        tracking_state = cast(TrackingState, self.state)
        self.state = cast(_StateT, tracking_state.track(solution))
        return _runtime_value_leaves((solution, result_code, stats))


class _StagedComputation(NamedTuple, Generic[_StateT]):
    """The pruned jaxpr and metadata cached for one call signature."""

    jaxpr: Jaxpr
    """The transformed jaxpr, pruned of the dead init `lineax` built."""

    consts: list[Any]
    """The jaxpr's constants, whose leaf types are jaxpr-defined."""

    output_treedef: jax.tree_util.PyTreeDef
    """The structure that rebuilds the function's output from its leaves."""

    state_treedef: jax.tree_util.PyTreeDef
    """The structure that rebuilds the final state from its leaves."""

    num_output_leaves: int
    """How many leading result leaves belong to the output, the rest being the state."""

    solver: StatefulSolver[_StateT] | None
    """The solver whose `release` frees the state when it is not handed back."""


class _WrappedFunction(Protocol[_OutputT]):
    """The function `stateful_solve_transform` returns.

    It takes the wrapped function's own arguments plus a `state` keyword for an initial
    state. It returns the output paired with the final state when the state is kept, and the
    output alone otherwise. The original argument types are typed loosely, since PEP 612
    cannot carry them alongside the added `state` keyword.
    """

    @overload
    def __call__(
        self, *args: Any, state: Any, **kwargs: Any
    ) -> tuple[_OutputT, Any]: ...
    @overload
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, state: Any = ..., **kwargs: Any) -> Any: ...


def stateful_solve_transform(
    fn: Callable[..., _OutputT],
    *,
    filter_solver: _FilterSolver = StatefulSolver,
    return_final_state: bool | None = None,
) -> _WrappedFunction[_OutputT]:
    """Thread a solver state through a function's `lineax.linear_solve` calls.

    The wrapped function takes the original arguments plus a `state` keyword for an initial
    state, defaulting to `None`, in which case `init` runs at the first solve. A function
    whose own signature already has a `state` argument cannot be wrapped, since the keyword
    is taken.

    **Arguments:**

    - `fn`: the function to transform. It calls `lineax.linear_solve` internally.
    - `filter_solver`: which solves to thread, as a solver class matched by `isinstance` or a
        boolean predicate. The default `StatefulSolver` threads only solvers that implement
        the stateful API, so a plain dense `lineax.LU()` passes through.
    - `return_final_state`: when true the wrapped function returns `(output, final_state)`,
        when false it returns the output alone and releases the threaded state. The default is
        true when an initial `state` is passed at call time, false otherwise.

    **Returns:**

    A function with `fn`'s arguments plus a `state` keyword, see `_WrappedFunction`.
    """
    staged_by_signature: dict[Any, _StagedComputation[Any]] = {}

    def stage(
        call: tuple[tuple[Any, ...], dict[str, Any]], state: Any
    ) -> _StagedComputation[Any]:
        """Trace `fn` under this call and state, interpret it, and prune the result.

        Runs the state-threading interpreter once to build the transformed jaxpr, then drops
        the dead init with `dce_jaxpr`. The output and state structures the interpreter
        discovers are captured here so the caller can rebuild both from the evaluated leaves.
        """
        call_leaves, call_treedef = jax.tree_util.tree_flatten(call)
        state_leaves, state_treedef = jax.tree_util.tree_flatten(state)
        num_call_leaves = len(call_leaves)

        output_treedef: jax.tree_util.PyTreeDef | None = None
        state_out_treedef: jax.tree_util.PyTreeDef | None = None
        num_output_leaves = 0
        threaded_solver: StatefulSolver[Any] | None = None

        def call_flat(*flat: Any) -> _OutputT:
            """Call `fn` from a flat list of the call's argument leaves."""
            these_args, these_kwargs = jax.tree_util.tree_unflatten(
                call_treedef, list(flat)
            )
            return fn(*these_args, **these_kwargs)

        def stage_body(*flat: Any) -> list[Any]:
            """Trace and interpret `fn`, returning the output leaves then the state leaves.

            Records the output structure, the final state structure, the output leaf count,
            and the threaded solver in the enclosing scope, so `stage` can read them after.
            """
            nonlocal output_treedef, state_out_treedef
            nonlocal num_output_leaves, threaded_solver
            call_flat_leaves = list(flat[:num_call_leaves])
            state_flat_leaves = list(flat[num_call_leaves:])
            initial_state = jax.tree_util.tree_unflatten(
                state_treedef, state_flat_leaves
            )
            closed, output_shapes = make_jaxpr(call_flat, return_shape=True)(
                *call_flat_leaves
            )
            interpreter: _StateThreadingInterpreter[Any] = _StateThreadingInterpreter(
                filter_solver, initial_state
            )
            outputs = interpreter.interpret(
                closed.jaxpr, closed.consts, call_flat_leaves
            )
            final_leaves, state_out_treedef = jax.tree_util.tree_flatten(
                interpreter.state
            )
            output_treedef = jax.tree_util.tree_structure(output_shapes)
            num_output_leaves = len(outputs)
            threaded_solver = interpreter.solver
            return [*outputs, *final_leaves]

        staged = make_jaxpr(stage_body)(*call_leaves, *state_leaves)
        # Prune the dead init `lineax` built, keeping every input so `eval_jaxpr` can be
        # handed all the arguments. `instantiate=True` keeps the signature and drops only
        # dead internal equations.
        pruned, _ = dce_jaxpr(
            staged.jaxpr, [True] * len(staged.jaxpr.outvars), instantiate=True
        )
        assert output_treedef is not None and state_out_treedef is not None
        return _StagedComputation(
            jaxpr=pruned,
            consts=staged.consts,
            output_treedef=output_treedef,
            state_treedef=state_out_treedef,
            num_output_leaves=num_output_leaves,
            solver=threaded_solver,
        )

    def stateful_function(*args: Any, state: Any = None, **kwargs: Any) -> Any:
        """Run `fn` with its solves threaded, caching the staged jaxpr per signature."""
        keep_state = (
            state is not None if return_final_state is None else return_final_state
        )

        call = (args, kwargs)
        call_leaves, call_treedef = jax.tree_util.tree_flatten(call)
        state_leaves, state_treedef = jax.tree_util.tree_flatten(state)
        signature = (
            call_treedef,
            tuple(jax.typeof(leaf) for leaf in call_leaves),
            state_treedef,
            tuple(jax.typeof(leaf) for leaf in state_leaves),
            keep_state,
        )
        computation = staged_by_signature.get(signature)
        if computation is None:
            computation = stage(call, state)
            staged_by_signature[signature] = computation

        results = jax.core.eval_jaxpr(
            computation.jaxpr, computation.consts, *call_leaves, *state_leaves
        )
        output = jax.tree_util.tree_unflatten(
            computation.output_treedef, results[: computation.num_output_leaves]
        )
        final_state = jax.tree_util.tree_unflatten(
            computation.state_treedef, results[computation.num_output_leaves :]
        )

        if keep_state:
            return output, final_state
        if computation.solver is not None:
            computation.solver.release(final_state)
        return output

    return cast(_WrappedFunction[_OutputT], stateful_function)
