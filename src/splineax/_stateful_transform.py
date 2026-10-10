"""Turn naive `lineax.linear_solve` code into stateful solving.

`stateful_solve_transform` wraps a function that calls `lineax.linear_solve` and threads a
solver state through its solves, so they reuse a factorization. Each threaded solve is
staged through `splineax.linear_solve`, so it behaves exactly like a hand-written call that
passes the state along, and the final state is released with the generic `release`.

A solve inside a `lax.scan` or `lax.while_loop` is threaded by carrying the state through
the loop, and the first iteration is unrolled to create the state when there is none yet.
A solve inside a `lax.cond` is threaded through the branches, and needs a state from an
earlier solve. A solve inside a `jax.checkpoint` (`remat`) is threaded too, keeping the
checkpointing. A solve inside a `custom_jvp` or `custom_vjp` function is threaded by rebuilding
the function as a custom derivative that takes and returns the state.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, NamedTuple, Protocol, TypeVar, cast, overload

import equinox.internal as eqxi
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import make_jaxpr
from jax._src.interpreters.partial_eval import convert_constvars_jaxpr, dce_jaxpr
from jax.custom_derivatives import CustomVJPPrimal, SymbolicZero
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Primitive, Var
from jax.interpreters.ad import Zero
from jaxtyping import Array, Bool, PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver
from lineax._solution import RESULTS
from lineax._solve import linear_solve_p

from splineax._profile import is_record_callback
from splineax.solvers._sparse import linear_solve as splineax_linear_solve
from splineax.solvers._stateful import StatefulSolver

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

_CUSTOM_DIFF_PRIMITIVES = frozenset({"custom_jvp_call", "custom_vjp_call"})
"""The custom-differentiation primitives that the pass-through option leaves unthreaded.

Without the option, `_thread_custom_jvp` and `_thread_custom_vjp` rebuild them with the
state as an extra argument and output. With it, the solves inside them run without reuse.
"""

_INLINE_PRIMITIVES = frozenset({"pjit", "jit", "closed_call", "core_call"})
"""The higher-order primitives whose body we inline by interpreting it.

Each is a pure staging boundary, so running its body in place computes the same values.
`lineax` wraps every solve in one of these, so inlining is required, not an optimisation.
"""


def _instantiate_zero(zero: SymbolicZero) -> Array | np.ndarray:
    """The array of zeros that a symbolic zero tangent stands for."""
    if zero.dtype == jax.dtypes.float0:
        return np.zeros(zero.shape, dtype=jax.dtypes.float0)
    return jnp.zeros(zero.shape, zero.dtype)


def _solver_matches(solver: AbstractLinearSolver, filter_solver: _FilterSolver) -> bool:
    """Whether the filter selects this solver, as a class or a predicate."""
    if isinstance(filter_solver, type):
        return isinstance(solver, filter_solver)
    return filter_solver(solver)


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


def _is_selected_solve(eqn: JaxprEqn, filter_solver: _FilterSolver) -> bool:
    """Whether an equation binds `linear_solve_p` with a solver the filter would thread.

    The solver is a static param, so it reads out without running anything, by
    reconstructing with placeholder operands.
    """
    if eqn.primitive is not linear_solve_p:
        return False
    arguments = _reconstruct_solve_arguments(eqn, [None] * len(eqn.invars))
    return _solver_matches(arguments[4], filter_solver)


def _is_profile_record(eqn: JaxprEqn) -> bool:
    """Whether an equation is the `io_callback` that appends a solve-profile record."""
    if eqn.primitive.name != "io_callback":
        return False
    callback = getattr(eqn.params.get("callback"), "callback_func", None)
    return is_record_callback(callback)


def _state_substitutions(eqn: JaxprEqn, new_state: PyTree) -> dict[Var, Any]:
    """Map the invars carrying a solve's own state to the matching leaves of `new_state`.

    `lineax.linear_solve` returns the state it bound on the `Solution`, so its outputs read
    these invars. Pointing them at the threaded state leaves the state `lineax` built dead,
    which lets DCE drop it. Each operand is swapped for a unique marker and the arguments
    are rebuilt, so the state's leaves say which invars they came from.

    Returns no substitutions when the two states differ in structure or leaf type, which
    keeps the state `lineax` built. A var the solve also reads outside its state is skipped,
    so its other readers keep their value.
    """
    markers = [object() for _ in eqn.invars]
    index_by_marker = {id(marker): index for index, marker in enumerate(markers)}
    marked_state = _reconstruct_solve_arguments(eqn, markers)[1]
    marked_leaves, marked_treedef = jax.tree_util.tree_flatten(marked_state)
    new_leaves, new_treedef = jax.tree_util.tree_flatten(new_state)
    if marked_treedef != new_treedef:
        return {}

    substitutions: dict[Var, Any] = {}
    for marked, new in zip(marked_leaves, new_leaves):
        index = index_by_marker.get(id(marked))
        if index is None:
            # A static leaf, which is no operand.
            continue
        var = eqn.invars[index]
        if not _is_runtime_value(new):
            return {}
        # A solve operand is always an array, so its aval is a `ShapedArray`.
        aval = cast(jax.core.ShapedArray, var.aval)
        if new.shape != aval.shape or new.dtype != aval.dtype:
            return {}
        if isinstance(var, Literal) or eqn.invars.count(var) > 1:
            continue
        substitutions[var] = new
    return substitutions


def _jaxpr_has_selected_solve(jaxpr: Jaxpr, filter_solver: _FilterSolver) -> bool:
    """Whether a jaxpr, or any it nests, holds a solve the filter would thread."""
    for eqn in jaxpr.eqns:
        if _is_selected_solve(eqn, filter_solver):
            return True
        if any(
            _jaxpr_has_selected_solve(inner, filter_solver)
            for inner in _nested_jaxprs(eqn)
        ):
            return True
    return False


class _StateThreadingInterpreter(Generic[_StateT]):
    """Walks a jaxpr like `eval_jaxpr`, threading one solver state through the solves."""

    filter_solver: _FilterSolver
    """Chooses which solves to thread, as a solver class or a predicate."""

    state: _StateT | None
    """The state threaded so far, or `None` before the first matched solve."""

    pass_through_custom_diff: bool
    """Whether a `custom_jvp` or `custom_vjp` call runs as it is, instead of being threaded."""

    def __init__(
        self,
        filter_solver: _FilterSolver,
        state: _StateT | None,
        pass_through_custom_diff: bool = False,
    ) -> None:
        self.filter_solver = filter_solver
        self.state = state
        self.pass_through_custom_diff = pass_through_custom_diff

    def interpret(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        args: list[Any],
        drop_profile_records: bool = False,
        tangent_args: Sequence[bool] | None = None,
    ) -> list[Any]:
        """Evaluate the jaxpr against these argument values, returning its output values.

        `tangent_args` flags the arguments that carry tangents. A solve that depends on
        one of them runs with the threaded state but does not store the state it used, so
        the state stays a function of the primal values alone.

        Profile records that come before a threaded solve in the same jaxpr are dropped,
        and so are all records when `drop_profile_records` is true. A jaxpr that binds a
        solve itself is the body of `lineax.linear_solve`, where everything before the bind
        is `lineax`'s own `init`. The threaded state replaces that init, so its records
        describe work that no longer runs.
        """
        env: dict[Var, Any] = {}

        def read(atom: _Atom) -> Any:
            return atom.val if isinstance(atom, Literal) else env[atom]

        def write(var: Var, value: Any) -> None:
            env[var] = value

        for constvar, constval in zip(jaxpr.constvars, consts):
            write(constvar, constval)
        for invar, arg in zip(jaxpr.invars, args):
            write(invar, arg)
        tangent_vars: set[Var] = {
            invar
            for invar, is_tangent in zip(jaxpr.invars, tangent_args or ())
            if is_tangent
        }

        last_solve_index = max(
            (
                index
                for index, eqn in enumerate(jaxpr.eqns)
                if _is_selected_solve(eqn, self.filter_solver)
            ),
            default=-1,
        )
        for index, eqn in enumerate(jaxpr.eqns):
            drops_records = drop_profile_records or index < last_solve_index
            if drops_records and _is_profile_record(eqn):
                continue
            operands = [read(v) for v in eqn.invars]
            operand_is_tangent = [
                not isinstance(v, Literal) and v in tangent_vars for v in eqn.invars
            ]
            has_tangent_operand = any(operand_is_tangent)
            state_before = self.state
            outputs = self._process_equation(
                eqn, operands, env, drops_records, operand_is_tangent
            )
            if has_tangent_operand and eqn.primitive is linear_solve_p:
                self.state = state_before
            if has_tangent_operand:
                tangent_vars.update(eqn.outvars)
            for outvar, value in zip(eqn.outvars, outputs):
                write(outvar, value)

        return [read(v) for v in jaxpr.outvars]

    def _process_equation(
        self,
        eqn: JaxprEqn,
        operands: list[Any],
        env: dict[Var, Any],
        drop_profile_records: bool,
        operand_is_tangent: Sequence[bool] = (),
    ) -> list[Any]:
        """Produce one equation's output values, threading the state through a matched solve.

        A matched `linear_solve_p` is threaded, an inline primitive has its body interpreted
        in place, a higher-order primitive that nests a matched solve raises, and everything
        else is rebound as `jax.core.eval_jaxpr` would. A threaded solve may rewrite `env`,
        see `_thread_solve`, and an inlined body drops its profile records when
        `drop_profile_records` is true.
        """
        primitive: Primitive = eqn.primitive
        if primitive is linear_solve_p:
            arguments = _reconstruct_solve_arguments(eqn, operands)
            if _solver_matches(arguments[4], self.filter_solver):
                return self._thread_solve(eqn, arguments, env, any(operand_is_tangent))
            return _runtime_value_leaves(_rebind_solve(arguments))

        if primitive.name in _INLINE_PRIMITIVES:
            inner = cast(ClosedJaxpr, eqn.params["jaxpr"])
            return self.interpret(
                inner.jaxpr,
                inner.consts,
                operands,
                drop_profile_records,
                operand_is_tangent,
            )

        nested = _nested_jaxprs(eqn)
        if any(
            _jaxpr_has_selected_solve(inner, self.filter_solver) for inner in nested
        ):
            if primitive.name == "cond":
                return self._thread_cond(eqn, operands)
            if primitive.name == "scan":
                return self._thread_scan(eqn, operands)
            if primitive.name == "while":
                return self._thread_while(eqn, operands)
            if primitive.name == "remat2":
                return self._thread_remat(eqn, operands)
            if (
                primitive.name == "custom_jvp_call"
                and not self.pass_through_custom_diff
            ):
                return self._thread_custom_jvp(eqn, operands)
            if (
                primitive.name == "custom_vjp_call"
                and not self.pass_through_custom_diff
            ):
                return self._thread_custom_vjp(eqn, operands)
            passes_through = (
                self.pass_through_custom_diff
                and primitive.name in _CUSTOM_DIFF_PRIMITIVES
            )
            if not passes_through:
                raise NotImplementedError(
                    "`stateful_solve_transform` cannot thread a solver state through a "
                    f"solve inside `{primitive.name}`. Move the solve out of it, or drop "
                    "the transform for this function."
                )
            # The option asks for the primitive to run unchanged, so its solve runs without
            # reusing a factorization.

        bind_params = primitive.get_bind_params(eqn.params)
        result = primitive.bind(*operands, **bind_params)
        return list(result) if primitive.multiple_results else [result]

    def _thread_solve(
        self,
        eqn: JaxprEqn,
        arguments: _SolveArguments,
        env: dict[Var, Any],
        depends_on_tangent: bool = False,
    ) -> list[Any]:
        """Solve through `splineax.linear_solve`, threading the state in and out.

        The solve is staged exactly as a hand-written `splineax.linear_solve` call would
        stage it, with the same `init` or `update`, `track`, differentiation rule, and
        profile order. The vars that held the old state then point at the state this solve
        used, see `_state_substitutions`. Returns the solve's output values.
        """
        operator, _old_state, vector, options, solver, throw = arguments
        solution, self.state = splineax_linear_solve(
            operator,
            vector,
            solver,
            options=dict(options),
            state=self.state,
            # A solve of a tangent has no error check, which could not be transposed.
            # The primal solve it follows has been checked already.
            throw=throw and not depends_on_tangent,
        )
        env.update(_state_substitutions(eqn, solution.state))
        return _runtime_value_leaves((solution.value, solution.result, solution.stats))

    def _require_prior_state(self, region: str) -> None:
        """Raise if no state exists yet to seed a control-flow region's carry."""
        if self.state is None:
            raise NotImplementedError(
                "`stateful_solve_transform` cannot thread a solver state into a "
                f"`{region}` before a solve has created one. Solve once before it, or pass "
                "an initial state."
            )

    def _thread_nested_body(
        self,
        jaxpr: Jaxpr,
        consts: list[Any],
        operands: list[Any],
        state_leaves: list[Any],
        state_treedef: jax.tree_util.PyTreeDef,
        tangent_args: Sequence[bool] | None = None,
    ) -> tuple[list[Any], list[Any], jax.tree_util.PyTreeDef]:
        """Interpret a nested body seeded with the carried state, returning its outputs.

        Runs a fresh interpreter over the body, threading the state rebuilt from
        `state_leaves`, and returns the body's own outputs, the threaded state's leaves, and
        that state's structure. The outgoing structure may differ from the incoming one when a
        first solve creates the state, which is why it is returned rather than assumed.
        """
        incoming = jax.tree_util.tree_unflatten(state_treedef, state_leaves)
        inner: _StateThreadingInterpreter[_StateT] = _StateThreadingInterpreter(
            self.filter_solver, incoming, self.pass_through_custom_diff
        )
        outputs = inner.interpret(jaxpr, consts, operands, tangent_args=tangent_args)
        if inner.state is not None:
            self.state = inner.state
        out_leaves, out_treedef = jax.tree_util.tree_flatten(self.state)
        return outputs, out_leaves, out_treedef

    @staticmethod
    def _prune_dead(traced: ClosedJaxpr) -> ClosedJaxpr:
        """Drop equations left dead by state substitution, keeping the signature.

        Rewriting a body rebinds `lineax`'s own init, whose result the substituted state then
        replaces, leaving it dead. Outer DCE reaches a dead init inside a `scan` but not one
        inside a `while`, so each rewritten body is pruned here for a uniform result.
        """
        pruned, _ = dce_jaxpr(
            traced.jaxpr, [True] * len(traced.jaxpr.outvars), instantiate=True
        )
        return ClosedJaxpr(pruned, traced.consts)

    @staticmethod
    def _hoist_consts(traced: ClosedJaxpr) -> tuple[ClosedJaxpr, list[Any]]:
        """Move a traced jaxpr's constants to leading invars, returning it and the consts.

        `scan` and `while` reject a body that closes over constants, so the constants become
        extra const operands the loop passes in.
        """
        return ClosedJaxpr(convert_constvars_jaxpr(traced.jaxpr), ()), list(
            traced.consts
        )

    def _thread_cond(self, eqn: JaxprEqn, operands: list[Any]) -> list[Any]:
        """Thread the state through a `cond` whose branches hold a matched solve.

        Each branch is rewritten to take the state's leaves as extra operands and return the
        threaded state's leaves as extra outputs, so every branch has the same signature no
        matter how many times it solves. The `cond` is rebound with the state leaves added to
        its operands, and the trailing outputs become the new state. Requires a state to
        already exist, since a branch cannot build one the untaken branch would not match.
        """
        self._require_prior_state("cond")
        branches = cast(tuple[ClosedJaxpr, ...], eqn.params["branches"])
        index, branch_operands = operands[0], operands[1:]
        num_operands = len(branch_operands)
        state_leaves, state_treedef = jax.tree_util.tree_flatten(self.state)

        def rewrite_branch(branch: ClosedJaxpr) -> ClosedJaxpr:
            """Trace one branch into a jaxpr that also threads the state."""

            def threaded(*args: Any) -> list[Any]:
                outputs, out_leaves, _ = self._thread_nested_body(
                    branch.jaxpr,
                    branch.consts,
                    list(args[:num_operands]),
                    list(args[num_operands:]),
                    state_treedef,
                )
                return [*outputs, *out_leaves]

            return self._prune_dead(
                make_jaxpr(threaded)(*branch_operands, *state_leaves)
            )

        new_branches = tuple(rewrite_branch(branch) for branch in branches)
        bind_params = dict(eqn.primitive.get_bind_params(eqn.params))
        bind_params["branches"] = new_branches
        results = eqn.primitive.bind(
            index, *branch_operands, *state_leaves, **bind_params
        )
        num_outputs = len(eqn.outvars)
        self.state = jax.tree_util.tree_unflatten(
            state_treedef, list(results[num_outputs:])
        )
        return list(results[:num_outputs])

    def _rebind_unchanged(self, eqn: JaxprEqn, operands: list[Any]) -> list[Any]:
        """Rebind a primitive with its operands unchanged, threading no state."""
        bind_params = dict(eqn.primitive.get_bind_params(eqn.params))
        result = eqn.primitive.bind(*operands, **bind_params)
        return list(result) if eqn.primitive.multiple_results else [result]

    def _check_loop_state_structure(
        self,
        region: str,
        seed_treedef: jax.tree_util.PyTreeDef,
        body_out_treedef: jax.tree_util.PyTreeDef,
    ) -> None:
        """Raise if the body returns a state whose structure differs from the carried one.

        A loop carry has one fixed structure, so a solver whose `update` changes the state's
        pytree cannot be carried. This turns the raw carry-mismatch into a clear message.
        """
        if seed_treedef != body_out_treedef:
            raise ValueError(
                f"A solve threaded through `{region}` produced a state whose structure "
                "changed between iterations. A solver's states must share one pytree "
                "structure across `init`, `update`, and `track` to be carried through a "
                "loop. A state from `init_symbolic` differs, so `update` it before the loop."
            )

    def _augmented_scan(
        self,
        eqn: JaxprEqn,
        num_consts: int,
        num_carry: int,
        length: int,
        consts_values: list[Any],
        carry_values: list[Any],
        stacked_xs: list[Any],
    ) -> tuple[list[Any], list[Any]]:
        """Rebind a `scan` that also carries `self.state`, seeded from it.

        Requires `self.state` to already hold a loop-carry-shaped state. Returns the final
        carry leaves and the stacked per-iteration output leaves, and updates `self.state`.
        """
        body = cast(ClosedJaxpr, eqn.params["jaxpr"])
        per_iteration_xs = [leaf[0] for leaf in stacked_xs]
        state_leaves, state_treedef = jax.tree_util.tree_flatten(self.state)
        num_state = len(state_leaves)
        body_out_treedef = state_treedef

        def new_body(*args: Any) -> list[Any]:
            """Run one scan step, threading the state carried alongside the loop carry."""
            nonlocal body_out_treedef
            offset = num_consts
            consts = list(args[:offset])
            carry = list(args[offset : offset + num_carry])
            offset += num_carry
            carried_state = list(args[offset : offset + num_state])
            offset += num_state
            per_iteration = list(args[offset:])
            outputs, out_leaves, body_out_treedef = self._thread_nested_body(
                body.jaxpr,
                body.consts,
                [*consts, *carry, *per_iteration],
                carried_state,
                state_treedef,
            )
            return [*outputs[:num_carry], *out_leaves, *outputs[num_carry:]]

        traced = self._prune_dead(
            make_jaxpr(new_body)(
                *consts_values, *carry_values, *state_leaves, *per_iteration_xs
            )
        )
        self._check_loop_state_structure("scan", state_treedef, body_out_treedef)
        hoisted_body, hoisted_consts = self._hoist_consts(traced)
        bind_params = dict(eqn.primitive.get_bind_params(eqn.params))
        bind_params["jaxpr"] = hoisted_body
        bind_params["num_consts"] = num_consts + len(hoisted_consts)
        bind_params["num_carry"] = num_carry + num_state
        bind_params["length"] = length
        results = eqn.primitive.bind(
            *hoisted_consts,
            *consts_values,
            *carry_values,
            *state_leaves,
            *stacked_xs,
            **bind_params,
        )
        self.state = jax.tree_util.tree_unflatten(
            state_treedef, list(results[num_carry : num_carry + num_state])
        )
        carry_final = list(results[:num_carry])
        stacked_ys = list(results[num_carry + num_state :])
        return carry_final, stacked_ys

    def _thread_scan(self, eqn: JaxprEqn, operands: list[Any]) -> list[Any]:
        """Thread the state through a `scan` whose body holds a matched solve.

        The state's leaves become extra carries, placed after the existing carries, so each
        iteration reuses the factorization from the last and the final carry holds the state
        after the loop. With no prior state the first iteration is unrolled to
        create it, since `None` cannot be threaded as a carry.
        """
        num_consts = eqn.params["num_consts"]
        num_carry = eqn.params["num_carry"]
        length = eqn.params["length"]
        consts_values = operands[:num_consts]
        carry_values = operands[num_consts : num_consts + num_carry]
        stacked_xs = operands[num_consts + num_carry :]

        if length == 0:
            # The body never runs, so no solve happens and there is nothing to thread.
            return self._rebind_unchanged(eqn, operands)

        if self.state is not None:
            carry_final, stacked_ys = self._augmented_scan(
                eqn,
                num_consts,
                num_carry,
                length,
                consts_values,
                carry_values,
                stacked_xs,
            )
            return [*carry_final, *stacked_ys]

        # Unroll iteration zero to create the state, then scan the remaining iterations.
        body = cast(ClosedJaxpr, eqn.params["jaxpr"])
        first_xs = [leaf[0] for leaf in stacked_xs]
        _, none_treedef = jax.tree_util.tree_flatten(None)
        outputs, first_state_leaves, first_state_treedef = self._thread_nested_body(
            body.jaxpr,
            body.consts,
            [*consts_values, *carry_values, *first_xs],
            [],
            none_treedef,
        )
        first_carry = list(outputs[:num_carry])
        first_ys = list(outputs[num_carry:])
        self.state = jax.tree_util.tree_unflatten(
            first_state_treedef, first_state_leaves
        )
        if length == 1:
            return [*first_carry, *[leaf[None] for leaf in first_ys]]
        tail_xs = [leaf[1:] for leaf in stacked_xs]
        carry_final, tail_ys = self._augmented_scan(
            eqn, num_consts, num_carry, length - 1, consts_values, first_carry, tail_xs
        )
        stacked_ys = [
            jnp.concatenate([first[None], rest], axis=0)
            for first, rest in zip(first_ys, tail_ys)
        ]
        return [*carry_final, *stacked_ys]

    def _augmented_while(
        self,
        eqn: JaxprEqn,
        cond_consts: list[Any],
        body_consts: list[Any],
        carry_values: list[Any],
        seed_leaves: list[Any],
        seed_treedef: jax.tree_util.PyTreeDef,
        start_condition: Bool[Array, ""] | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """Rebind a `while` that also carries a state, seeded from `seed_leaves`.

        The condition takes the state leaves and ignores them, the body threads them. When
        `start_condition` is given, the loop also carries it unchanged and only runs while
        it is true. Returns the final carry leaves and the final state leaves.
        """
        cond_jaxpr = cast(ClosedJaxpr, eqn.params["cond_jaxpr"])
        body_jaxpr = cast(ClosedJaxpr, eqn.params["body_jaxpr"])
        cond_nconsts = eqn.params["cond_nconsts"]
        body_nconsts = eqn.params["body_nconsts"]
        num_carry = len(carry_values)
        num_state = len(seed_leaves)
        body_out_treedef = seed_treedef
        start_condition_carry = [] if start_condition is None else [start_condition]

        def new_cond(*args: Any) -> list[Any]:
            """Evaluate the loop condition, ignoring the extra state carry."""
            consts = list(args[:cond_nconsts])
            carry = list(args[cond_nconsts : cond_nconsts + num_carry])
            (condition,) = jax.core.eval_jaxpr(
                cond_jaxpr.jaxpr, cond_jaxpr.consts, *consts, *carry
            )
            if start_condition is not None:
                condition = jnp.logical_and(condition, args[-1])
            return [condition]

        def new_body(*args: Any) -> list[Any]:
            """Run one loop step, threading the state carried alongside the loop carry."""
            nonlocal body_out_treedef
            offset = body_nconsts
            consts = list(args[:offset])
            carry = list(args[offset : offset + num_carry])
            offset += num_carry
            carried_state = list(args[offset : offset + num_state])
            outputs, out_leaves, body_out_treedef = self._thread_nested_body(
                body_jaxpr.jaxpr,
                body_jaxpr.consts,
                [*consts, *carry],
                carried_state,
                seed_treedef,
            )
            return [*outputs, *out_leaves, *start_condition_carry]

        traced_cond = self._prune_dead(
            make_jaxpr(new_cond)(
                *cond_consts, *carry_values, *seed_leaves, *start_condition_carry
            )
        )
        traced_body = self._prune_dead(
            make_jaxpr(new_body)(
                *body_consts, *carry_values, *seed_leaves, *start_condition_carry
            )
        )
        self._check_loop_state_structure("while_loop", seed_treedef, body_out_treedef)
        cond_closed, cond_hoisted = self._hoist_consts(traced_cond)
        body_closed, body_hoisted = self._hoist_consts(traced_body)
        bind_params = dict(eqn.primitive.get_bind_params(eqn.params))
        bind_params["cond_jaxpr"] = cond_closed
        bind_params["body_jaxpr"] = body_closed
        bind_params["cond_nconsts"] = cond_nconsts + len(cond_hoisted)
        bind_params["body_nconsts"] = body_nconsts + len(body_hoisted)
        results = eqn.primitive.bind(
            *cond_hoisted,
            *cond_consts,
            *body_hoisted,
            *body_consts,
            *carry_values,
            *seed_leaves,
            *start_condition_carry,
            **bind_params,
        )
        carry_final = list(results[:num_carry])
        state_final = list(results[num_carry : num_carry + num_state])
        return carry_final, state_final

    def _thread_while(self, eqn: JaxprEqn, operands: list[Any]) -> list[Any]:
        """Thread the state through a `while_loop` whose body holds a matched solve.

        The state's leaves become extra carries. With a prior state the loop is rebound to
        carry it. With no prior state the first iteration is unrolled and run once to create
        the state, guarded by the loop condition so a loop that would run zero times keeps its
        original carry.
        """
        cond_nconsts = eqn.params["cond_nconsts"]
        body_nconsts = eqn.params["body_nconsts"]
        cond_consts = operands[:cond_nconsts]
        body_consts = operands[cond_nconsts : cond_nconsts + body_nconsts]
        carry_values = operands[cond_nconsts + body_nconsts :]

        if self.state is not None:
            seed_leaves, seed_treedef = jax.tree_util.tree_flatten(self.state)
            carry_final, state_final = self._augmented_while(
                eqn, cond_consts, body_consts, carry_values, seed_leaves, seed_treedef
            )
            self.state = jax.tree_util.tree_unflatten(seed_treedef, state_final)
            return carry_final

        cond_jaxpr = cast(ClosedJaxpr, eqn.params["cond_jaxpr"])
        body_jaxpr = cast(ClosedJaxpr, eqn.params["body_jaxpr"])
        _, none_treedef = jax.tree_util.tree_flatten(None)
        first_carry, first_state_leaves, first_state_treedef = self._thread_nested_body(
            body_jaxpr.jaxpr,
            body_jaxpr.consts,
            [*body_consts, *carry_values],
            [],
            none_treedef,
        )
        predicate = jax.core.eval_jaxpr(
            cond_jaxpr.jaxpr, cond_jaxpr.consts, *cond_consts, *carry_values
        )[0]

        # The loop after the first iteration only runs when the condition held at the start.
        # A zero-trip loop keeps its original carry, chosen by a select, since a `cond`
        # could not hold the solves' callbacks when it is batched.
        carry_final, state_final = self._augmented_while(
            eqn,
            cond_consts,
            body_consts,
            list(first_carry),
            first_state_leaves,
            first_state_treedef,
            start_condition=predicate,
        )
        self.state = jax.tree_util.tree_unflatten(first_state_treedef, state_final)
        return [
            jax.lax.select(predicate, final_leaf, original_leaf)
            for final_leaf, original_leaf in zip(carry_final, carry_values)
        ]

    def _thread_remat(self, eqn: JaxprEqn, operands: list[Any]) -> list[Any]:
        """Thread the state through a `remat` whose body holds a matched solve.

        `remat` is not inlined, since that would drop the rematerialisation it exists for.
        Its body instead takes the state's leaves as extra operands and returns the threaded
        state's leaves as extra outputs, and the wrapper is rebound so the checkpointing is
        kept. Unlike a loop, a `remat` runs once and imposes no fixed carry, so a first solve
        inside it may create the state, and the incoming and outgoing states may differ in
        structure.
        """
        body = cast(Jaxpr, eqn.params["jaxpr"])
        state_leaves, state_treedef = jax.tree_util.tree_flatten(self.state)
        num_operands = len(operands)
        output_state_treedef = state_treedef

        def new_body(*args: Any) -> list[Any]:
            """Run the checkpointed body, threading the state alongside its operands."""
            nonlocal output_state_treedef
            incoming = jax.tree_util.tree_unflatten(
                state_treedef, list(args[num_operands:])
            )
            inner: _StateThreadingInterpreter[_StateT] = _StateThreadingInterpreter(
                self.filter_solver, incoming, self.pass_through_custom_diff
            )
            outputs = inner.interpret(body, [], list(args[:num_operands]))
            if inner.state is not None:
                self.state = inner.state
            out_leaves, output_state_treedef = jax.tree_util.tree_flatten(self.state)
            return [*outputs, *out_leaves]

        traced = self._prune_dead(make_jaxpr(new_body)(*operands, *state_leaves))
        bind_params = dict(eqn.primitive.get_bind_params(eqn.params))
        bind_params["jaxpr"] = convert_constvars_jaxpr(traced.jaxpr)
        results = eqn.primitive.bind(
            *traced.consts, *operands, *state_leaves, **bind_params
        )
        num_outputs = len(eqn.outvars)
        self.state = jax.tree_util.tree_unflatten(
            output_state_treedef, list(results[num_outputs:])
        )
        return list(results[:num_outputs])

    def _thread_custom_jvp(self, eqn: JaxprEqn, operands: list[Array]) -> list[Array]:
        """Thread the state through a `custom_jvp` call whose function holds a matched solve.

        The function is rebuilt as a `custom_jvp` that takes the state's leaves as extra
        arguments and returns the threaded state's leaves as extra outputs. Its primal runs
        the original function under the interpreter, so its solves thread the state. Its
        differentiation rule runs the jaxpr of the original rule under the interpreter in the
        same way, so a solve in the rule threads the state as well. The state leaves get
        zero tangents, since a state is not a differentiable quantity.

        Like `remat` and unlike a loop, a custom-derivative call runs once and imposes no
        fixed carry, so a first solve inside it may create the state, and the incoming and
        outgoing states may differ in structure.
        """
        call_jaxpr = cast(ClosedJaxpr, eqn.params["call_jaxpr"])
        num_consts = cast(int, eqn.params["num_consts"])
        jvp_jaxpr_fun = eqn.params["jvp_jaxpr_fun"]
        original_symbolic_zeros = cast(bool, eqn.params["symbolic_zeros"])
        num_operands = len(operands)
        state_leaves, state_treedef = jax.tree_util.tree_flatten(self.state)
        output_state_treedef = state_treedef

        def threaded_primal(*arguments: Array) -> list[Array]:
            """Interpret the original function, threading the state alongside its arguments."""
            nonlocal output_state_treedef
            outputs, out_leaves, output_state_treedef = self._thread_nested_body(
                call_jaxpr.jaxpr,
                call_jaxpr.consts,
                list(arguments[:num_operands]),
                list(arguments[num_operands:]),
                state_treedef,
            )
            return [*outputs, *out_leaves]

        threaded_function = jax.custom_jvp(threaded_primal)

        def threaded_jvp_rule(
            primals: tuple[Array, ...], tangents: tuple[Array | SymbolicZero, ...]
        ) -> tuple[list[Array], list[Array | SymbolicZero]]:
            """Interpret the jaxpr of the original rule with the state threaded through it."""
            operand_primals = list(primals[:num_operands])
            operand_tangents = list(tangents[:num_operands])[num_consts:]
            if not original_symbolic_zeros:
                # A rule that did not ask for symbolic zeros gets arrays, as JAX would give.
                operand_tangents = [
                    _instantiate_zero(tangent)
                    if type(tangent) is SymbolicZero
                    else tangent
                    for tangent in operand_tangents
                ]
            # The rule takes the primals after the constants, and only the nonzero tangents.
            tangent_is_zero = [
                type(tangent) is SymbolicZero for tangent in operand_tangents
            ]
            rule_jaxpr, rule_consts, output_tangent_is_zero = (
                jvp_jaxpr_fun.call_wrapped(*tangent_is_zero)
            )
            nonzero_tangents = [
                tangent
                for tangent in operand_tangents
                if type(tangent) is not SymbolicZero
            ]
            num_primals = len(operand_primals) - num_consts
            num_rule_arguments = num_primals + len(nonzero_tangents)

            def interpret_rule(*flat_arguments: Array) -> list[Array]:
                """Interpret the rule's jaxpr, returning its outputs then the state leaves."""
                outputs, out_leaves, _ = self._thread_nested_body(
                    rule_jaxpr,
                    rule_consts,
                    list(flat_arguments[:num_rule_arguments]),
                    list(flat_arguments[num_rule_arguments:]),
                    state_treedef,
                    tangent_args=[False] * num_primals + [True] * len(nonzero_tangents),
                )
                return [*outputs, *out_leaves]

            # Staging the rule keeps its primal values traced. Run eagerly, a concrete scalar
            # would reach a rebuilt operator as a Python scalar, which equinox treats as
            # static, so a closure-converted function would reject it.
            flat_results = jax.jit(interpret_rule)(
                *operand_primals[num_consts:],
                *nonzero_tangents,
                *primals[num_operands:],
            )
            rule_outputs = flat_results[: len(rule_jaxpr.outvars)]
            out_leaves = flat_results[len(rule_jaxpr.outvars) :]
            num_outputs = len(output_tangent_is_zero)
            out_primals = rule_outputs[:num_outputs]
            nonzero_out_tangents = iter(rule_outputs[num_outputs:])
            out_tangents = [
                SymbolicZero(jax.typeof(primal_out).to_tangent_aval())
                if is_zero
                else next(nonzero_out_tangents)
                for primal_out, is_zero in zip(out_primals, output_tangent_is_zero)
            ]
            state_tangents = [
                SymbolicZero(jax.typeof(leaf).to_tangent_aval()) for leaf in out_leaves
            ]
            return [*out_primals, *out_leaves], [*out_tangents, *state_tangents]

        # The type stubs of JAX leave symbolic zeros out of the tangent type of a rule.
        threaded_function.defjvp(
            cast(Callable[..., tuple[list[Array], list[Array]]], threaded_jvp_rule),
            symbolic_zeros=True,
        )
        threaded_results = threaded_function(*operands, *state_leaves)
        num_outputs = len(eqn.outvars)
        self.state = jax.tree_util.tree_unflatten(
            output_state_treedef, list(threaded_results[num_outputs:])
        )
        return list(threaded_results[:num_outputs])

    def _thread_custom_vjp(self, eqn: JaxprEqn, operands: list[Array]) -> list[Array]:
        """Thread the state through a `custom_vjp` call whose function holds a matched solve.

        The function is rebuilt as a `custom_vjp` that takes the state's leaves as extra
        arguments and returns the threaded state's leaves as extra outputs. Its primal and
        its forward function run the original ones under the interpreter, so their solves
        thread the state. The forward function also saves the state as a residual. Its
        backward function is the original one, traced to a jaxpr and interpreted with that
        state, so its solves reuse the forward factorization. The state leaves get no
        cotangent, since a state is not a differentiable quantity.
        """
        call_jaxpr = cast(ClosedJaxpr, eqn.params["call_jaxpr"])
        num_consts = cast(int, eqn.params["num_consts"])
        fwd_jaxpr_thunk = eqn.params["fwd_jaxpr_thunk"]
        symbolic_zeros = cast(bool, eqn.params["symbolic_zeros"])
        _, _, original_backward = eqn.primitive.get_bind_params(eqn.params)["subfuns"]
        num_operands = len(operands)
        num_outputs = len(eqn.outvars)
        state_leaves, state_treedef = jax.tree_util.tree_flatten(self.state)
        output_state_treedef = state_treedef
        num_original_residuals = 0

        def threaded_primal(*arguments: Array) -> list[Array]:
            """Interpret the original function, threading the state alongside its arguments."""
            nonlocal output_state_treedef
            outputs, out_leaves, output_state_treedef = self._thread_nested_body(
                call_jaxpr.jaxpr,
                call_jaxpr.consts,
                list(arguments[:num_operands]),
                list(arguments[num_operands:]),
                state_treedef,
            )
            return [*outputs, *out_leaves]

        threaded_function = jax.custom_vjp(threaded_primal)

        def threaded_forward(
            *arguments: Array | CustomVJPPrimal,
        ) -> tuple[list[Array], list[Array]]:
            """Interpret the original forward function, which returns the outputs and residuals.

            The state it returns is a residual too, for the backward function to use.
            """
            nonlocal output_state_treedef, num_original_residuals
            if symbolic_zeros:
                # The rule sees each argument as a value with a flag for a nonzero tangent.
                operand_values = [cast(CustomVJPPrimal, a).value for a in arguments]
                nonzero_flags = [cast(CustomVJPPrimal, a).perturbed for a in arguments]
            else:
                operand_values = [cast(Array, a) for a in arguments]
                nonzero_flags = [True] * len(arguments)
            forward_jaxpr, forward_consts = fwd_jaxpr_thunk.call_wrapped(
                *nonzero_flags[num_consts:num_operands]
            )
            results, out_leaves, output_state_treedef = self._thread_nested_body(
                forward_jaxpr,
                forward_consts,
                operand_values[num_consts:num_operands],
                operand_values[num_operands:],
                state_treedef,
            )
            # The forward jaxpr returns the residuals first and leaves out the ones that are
            # just an operand, which the original backward function expects in place. The
            # indices count the constants, too.
            num_residuals = len(results) - num_outputs
            pruned_residuals = iter(results[:num_residuals])
            _, _, input_forwards = eqn.params["out_trees"]()
            residuals = [
                next(pruned_residuals) if index is None else operand_values[index]
                for index in input_forwards
            ]
            num_original_residuals = len(residuals)
            return [*results[num_residuals:], *out_leaves], [*residuals, *out_leaves]

        def threaded_backward(
            residuals: list[Array], cotangents: list[Array | SymbolicZero]
        ) -> tuple[Array | None, ...]:
            """Interpret the original backward function, threading the state it was given.

            The backward function is traced to a jaxpr, so its solves are found and run under
            an interpreter seeded with the state that the forward function returned. A
            solve of a cotangent reads that state without replacing it.
            """
            original_residuals = residuals[:num_original_residuals]
            state_residuals = residuals[num_original_residuals:]
            output_cotangents = cotangents[:num_outputs]
            live_cotangents = [
                cotangent
                for cotangent in output_cotangents
                if type(cotangent) is not SymbolicZero
            ]
            result_is_zero: list[bool] = []
            # A residual that is a concrete value stays one while the function is traced,
            # since a backward function may branch on it in Python.
            residual_is_traced = [
                isinstance(residual, jax.core.Tracer) for residual in original_residuals
            ]

            def call_backward(*flat_arguments: Array) -> list[Array]:
                """Call the original backward function, leaving out its zero cotangents."""
                dynamic_values = iter(flat_arguments)
                full_residuals = [
                    next(dynamic_values) if is_traced else residual
                    for residual, is_traced in zip(
                        original_residuals, residual_is_traced
                    )
                ]
                restored_cotangents = [
                    cotangent
                    if type(cotangent) is SymbolicZero
                    else next(dynamic_values)
                    for cotangent in output_cotangents
                ]
                backward_results = original_backward.call_wrapped(
                    *full_residuals, *restored_cotangents
                )
                result_is_zero[:] = [
                    type(result) is Zero for result in backward_results
                ]
                return [
                    result for result in backward_results if type(result) is not Zero
                ]

            dynamic_residuals = [
                residual
                for residual, is_traced in zip(original_residuals, residual_is_traced)
                if is_traced
            ]
            backward_arguments = [*dynamic_residuals, *live_cotangents]
            traced_backward = make_jaxpr(call_backward)(*backward_arguments)
            if _jaxpr_has_selected_solve(traced_backward.jaxpr, self.filter_solver):
                num_arguments = len(backward_arguments)

                def interpret_backward(*flat_arguments: Array) -> list[Array]:
                    """Interpret the traced backward function with the state seeded."""
                    interpreter: _StateThreadingInterpreter[_StateT] = (
                        _StateThreadingInterpreter(
                            self.filter_solver,
                            jax.tree_util.tree_unflatten(
                                output_state_treedef,
                                list(flat_arguments[num_arguments:]),
                            ),
                            self.pass_through_custom_diff,
                        )
                    )
                    return interpreter.interpret(
                        traced_backward.jaxpr,
                        traced_backward.consts,
                        list(flat_arguments[:num_arguments]),
                        tangent_args=[False] * len(dynamic_residuals)
                        + [True] * len(live_cotangents),
                    )

                # Staging the interpreted function lets the `init` that `lineax` built for
                # each solve, which the threaded state replaces, be pruned as dead.
                staged_backward = self._prune_dead(
                    make_jaxpr(interpret_backward)(
                        *backward_arguments, *state_residuals
                    )
                )
                live_results = jax.core.eval_jaxpr(
                    staged_backward.jaxpr,
                    staged_backward.consts,
                    *backward_arguments,
                    *state_residuals,
                )
            else:
                live_results = jax.core.eval_jaxpr(
                    traced_backward.jaxpr, traced_backward.consts, *backward_arguments
                )
            live_result_iterator = iter(live_results)
            operand_cotangents = [
                None if is_zero else next(live_result_iterator)
                for is_zero in result_is_zero
            ]
            return (*operand_cotangents, *[None] * len(state_leaves))

        threaded_function.defvjp(
            threaded_forward, threaded_backward, symbolic_zeros=symbolic_zeros
        )
        threaded_results = threaded_function(*operands, *state_leaves)
        # The state is not differentiable, so later equations see it as having no tangent.
        self.state = jax.tree_util.tree_unflatten(
            output_state_treedef,
            [jax.lax.stop_gradient(leaf) for leaf in threaded_results[num_outputs:]],
        )
        return list(threaded_results[:num_outputs])


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
    pass_through_custom_diff: bool = False,
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
    - `pass_through_custom_diff`: by default a matched solve inside a `custom_jvp` or
        `custom_vjp` function is threaded, through its primal and its rule. Set this true to run
        such a function as it is, so its solves work but do not reuse a factorization.

    **Returns:**

    A function taking `fn`'s arguments plus a `state` keyword for an initial state. It returns
    `fn`'s output, or the output paired with the final state when a state is kept.
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

        def call_flat(*flat: Any) -> _OutputT:
            """Call `fn` from a flat list of the call's argument leaves."""
            these_args, these_kwargs = jax.tree_util.tree_unflatten(
                call_treedef, list(flat)
            )
            return fn(*these_args, **these_kwargs)

        def stage_body(*flat: Any) -> list[Any]:
            """Trace and interpret `fn`, returning the output leaves then the state leaves.

            Records the output structure, the final state structure, and the output leaf
            count in the enclosing scope, so `stage` can read them after.
            """
            nonlocal output_treedef, state_out_treedef
            nonlocal num_output_leaves
            call_flat_leaves = list(flat[:num_call_leaves])
            state_flat_leaves = list(flat[num_call_leaves:])
            initial_state = jax.tree_util.tree_unflatten(
                state_treedef, state_flat_leaves
            )
            closed, output_shapes = make_jaxpr(call_flat, return_shape=True)(
                *call_flat_leaves
            )
            interpreter: _StateThreadingInterpreter[Any] = _StateThreadingInterpreter(
                filter_solver, initial_state, pass_through_custom_diff
            )
            outputs = interpreter.interpret(
                closed.jaxpr, closed.consts, call_flat_leaves
            )
            final_leaves, state_out_treedef = jax.tree_util.tree_flatten(
                interpreter.state
            )
            output_treedef = jax.tree_util.tree_structure(output_shapes)
            num_output_leaves = len(outputs)
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
        if final_state is not None:
            final_state.release()
        return output

    return cast(_WrappedFunction[_OutputT], stateful_function)
