"""Turn naive `lineax.linear_solve` code into stateful solving.

`stateful_solve_transform` wraps a function that calls `lineax.linear_solve` and threads a
solver state through its solves, so they reuse a factorization, using only the generic
stateful API (the solver's `init` and `update`, and a state's `track` and `release`).

A solve inside a `lax.scan` or `lax.while_loop` is threaded by carrying the state through
the loop, and the first iteration is unrolled to create the state when there is none yet.
A solve inside a `lax.cond` is threaded through the branches, and needs a state from an
earlier solve. A solve inside a `jax.checkpoint` (`remat`) is threaded too, keeping the
checkpointing.
"""

from collections.abc import Callable, Mapping
from typing import Any, Generic, NamedTuple, Protocol, TypeVar, cast, overload

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.core
import jax.numpy as jnp
from jax import make_jaxpr
from jax._src.interpreters.partial_eval import convert_constvars_jaxpr, dce_jaxpr
from jax.extend.core import ClosedJaxpr, Jaxpr, JaxprEqn, Literal, Primitive, Var
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver
from lineax._solution import RESULTS
from lineax._solve import linear_solve_p

from splineax._solve import (
    _restore_operator,
    _splineax_linear_solve_p,
)
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
"""The custom-differentiation primitives whose solves can pass through unthreaded.

Their differentiation rule lives in an opaque callable, so the primal cannot be rewritten
to thread the state without desyncing that rule. With the pass-through option the solve runs
without reuse rather than raising.
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
    """Walks a jaxpr like `eval_jaxpr`, threading one solver state through the solves."""

    filter_solver: _FilterSolver
    """Chooses which solves to thread, as a solver class or a predicate."""

    state: _StateT | None
    """The state threaded so far, or `None` before the first matched solve."""

    pass_through_custom_diff: bool
    """Whether a solve inside a `custom_jvp` or `custom_vjp` passes through instead of raising."""

    def __init__(
        self,
        filter_solver: _FilterSolver,
        state: _StateT | None,
        pass_through_custom_diff: bool = False,
    ) -> None:
        self.filter_solver = filter_solver
        self.state = state
        self.pass_through_custom_diff = pass_through_custom_diff

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
            if primitive.name == "cond":
                return self._thread_cond(eqn, operands)
            if primitive.name == "scan":
                return self._thread_scan(eqn, operands)
            if primitive.name == "while":
                return self._thread_while(eqn, operands)
            if primitive.name == "remat2":
                return self._thread_remat(eqn, operands)
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
            # Pass the custom-diff primitive through unchanged. Its solve runs but does not
            # reuse a factorization, since the state cannot cross its opaque rule.

        bind_params = primitive.get_bind_params(eqn.params)
        result = primitive.bind(*operands, **bind_params)
        return list(result) if primitive.multiple_results else [result]

    def _thread_solve(self, arguments: _SolveArguments) -> list[Any]:
        """Update the state for this solve's operator, then solve through splineax's own
        primitive and thread the state it returns.

        The operator's values are stopped before `init` and `update`, so the factorization is
        an autodiff constant, but the operator handed to the solve keeps its tangent so the
        matrix gradient flows through its `matvec`. Binding `_splineax_linear_solve_p` rather
        than `lineax`'s primitive is what makes a reused factorization differentiate
        correctly: its transpose and forward rules factor an independent slot per
        differentiated solve, so an earlier solve's adjoint does not read a slot a later
        refactor overwrote. The returned state carries the ordering token, so no separate
        `track` is needed.
        """
        operator, _old_state, vector, options, solver_any, throw = arguments
        solver = cast(StatefulSolver[_StateT], solver_any)
        stopped_operator = _stop_gradient_leaves(operator)
        if self.state is None:
            self.state = solver.init(stopped_operator, {})
        else:
            self.state = solver.update(self.state, stopped_operator, {})
        prior_operator = getattr(self.state, "operator", None)
        solution, result_code, stats, new_state = cast(
            tuple[PyTree[Array], RESULTS, dict[str, Any], _StateT],
            eqxi.filter_primitive_bind(
                _splineax_linear_solve_p,
                operator,
                self.state,
                vector,
                options,
                solver_any,
                throw,
            ),
        )
        # The primitive strips the operator from the threaded state, so restore it with the
        # updated state's identity, which keeps the next `update`'s no-op check working.
        self.state = _restore_operator(new_state, prior_operator)
        return _runtime_value_leaves((solution, result_code, stats))

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
        outputs = inner.interpret(jaxpr, consts, operands)
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
        bind_params = eqn.primitive.get_bind_params(eqn.params)
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
        bind_params = eqn.primitive.get_bind_params(eqn.params)
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
        bind_params = eqn.primitive.get_bind_params(eqn.params)
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
    ) -> tuple[list[Any], list[Any]]:
        """Rebind a `while` that also carries a state, seeded from `seed_leaves`.

        The condition takes the state leaves and ignores them, the body threads them. Returns
        the final carry leaves and the final state leaves.
        """
        cond_jaxpr = cast(ClosedJaxpr, eqn.params["cond_jaxpr"])
        body_jaxpr = cast(ClosedJaxpr, eqn.params["body_jaxpr"])
        cond_nconsts = eqn.params["cond_nconsts"]
        body_nconsts = eqn.params["body_nconsts"]
        num_carry = len(carry_values)
        num_state = len(seed_leaves)
        body_out_treedef = seed_treedef

        def new_cond(*args: Any) -> list[Any]:
            """Evaluate the loop condition, ignoring the extra state carry."""
            consts = list(args[:cond_nconsts])
            carry = list(args[cond_nconsts : cond_nconsts + num_carry])
            return jax.core.eval_jaxpr(
                cond_jaxpr.jaxpr, cond_jaxpr.consts, *consts, *carry
            )

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
            return [*outputs, *out_leaves]

        traced_cond = self._prune_dead(
            make_jaxpr(new_cond)(*cond_consts, *carry_values, *seed_leaves)
        )
        traced_body = self._prune_dead(
            make_jaxpr(new_body)(*body_consts, *carry_values, *seed_leaves)
        )
        self._check_loop_state_structure("while_loop", seed_treedef, body_out_treedef)
        cond_closed, cond_hoisted = self._hoist_consts(traced_cond)
        body_closed, body_hoisted = self._hoist_consts(traced_body)
        bind_params = eqn.primitive.get_bind_params(eqn.params)
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
        num_carry = len(carry_values)
        _, none_treedef = jax.tree_util.tree_flatten(None)
        first_carry, first_state_leaves, first_state_treedef = self._thread_nested_body(
            body_jaxpr.jaxpr,
            body_jaxpr.consts,
            [*body_consts, *carry_values],
            [],
            none_treedef,
        )
        num_state = len(first_state_leaves)
        predicate = jax.core.eval_jaxpr(
            cond_jaxpr.jaxpr, cond_jaxpr.consts, *cond_consts, *carry_values
        )[0]

        def run_rest() -> list[Any]:
            """Continue the loop from the unrolled carry and state."""
            carry_final, state_final = self._augmented_while(
                eqn,
                cond_consts,
                body_consts,
                list(first_carry),
                first_state_leaves,
                first_state_treedef,
            )
            return [*carry_final, *state_final]

        def skip() -> list[Any]:
            """Keep the original carry when the loop would run zero times."""
            return [*carry_values, *first_state_leaves]

        combined = jax.lax.cond(predicate, run_rest, skip)
        self.state = jax.tree_util.tree_unflatten(
            first_state_treedef, list(combined[num_carry : num_carry + num_state])
        )
        return list(combined[:num_carry])

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
        bind_params = eqn.primitive.get_bind_params(eqn.params)
        bind_params["jaxpr"] = convert_constvars_jaxpr(traced.jaxpr)
        results = eqn.primitive.bind(
            *traced.consts, *operands, *state_leaves, **bind_params
        )
        num_outputs = len(eqn.outvars)
        self.state = jax.tree_util.tree_unflatten(
            output_state_treedef, list(results[num_outputs:])
        )
        return list(results[:num_outputs])


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
        `custom_vjp` raises, since the state cannot cross the custom rule. Set this true to let
        such a solve run without threading, so it works but does not reuse a factorization.

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
