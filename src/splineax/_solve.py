"""The explicit stateful linear-solve API.

`linear_solve` folds an operator into a solver state, solves, and returns the solution
paired with the updated state. It owns its own forward-mode and reverse-mode rules through
a primitive, `_splineax_linear_solve_p`, rather than reusing `lineax.linear_solve`. The
reason is the state. A stateful sparse solve threads a factorization token through the
solve so a later refactor of the same cache slot waits on it, and that token has to be
carried out of the primitive alongside the solution. `lineax.linear_solve` returns only the
solution, so it cannot thread the token.

The differentiation rules mirror `lineax`'s: the forward rule solves the tangent system and
the transpose rule solves the transposed system, so the operator-entry gradient flows
through the operator's own matvec. The sparse stateful solvers are full-rank and square, so
the least-squares terms `lineax` carries drop out.
"""

import functools as ft
from typing import Any, TypeVar, overload

import equinox as eqx
import equinox.internal as eqxi
import jax.core
import jax.interpreters.ad as ad
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import Array, PyTree
from lineax import (
    AbstractLinearOperator,
    TangentLinearOperator,
    linearise,
)
from lineax import (
    linear_solve as _lx_linear_solve,
)
from lineax._misc import inexact_asarray
from lineax._solution import RESULTS, Solution
from lineax._solve import sentinel

from splineax.solvers._stateful import StatefulSolver, TrackingState

StateT = TypeVar("StateT", bound=TrackingState)


def _is_none(x: Any) -> bool:
    return x is None


def _sum(*args: Any) -> Any:
    return sum(args[1:], args[0])


def _witness_dtype(state: Any) -> Any:
    return getattr(state, "order_witness", jnp.zeros(())).dtype


def _local_witness(rhs: PyTree[Array], dtype: Any) -> Array:
    """A witness value that depends on `rhs`, for a solve with no preceding solve in its
    chain to receive one from.

    Multiplying by `0.0` keeps the value zero-valued but, since `0.0` times a `NaN` or an
    infinity is `NaN` under IEEE 754, XLA cannot fold it away as a disconnected constant.
    Reading one element makes the result depend on the whole tangent seed.
    """
    leaf = jtu.tree_leaves(rhs)[0]
    return (0.0 * jnp.real(jnp.ravel(leaf)[0])).astype(dtype)


def _read_witness(result: PyTree[Array], dtype: Any) -> Array:
    """A witness value that depends on `result`, a solve's own output.

    Used the same way as `_local_witness`, but derived from what a solve actually
    produced, so the witness a transpose call returns for its own `state` input carries a
    genuine dependency on the read it just performed.
    """
    leaf = jtu.tree_leaves(result)[0]
    return (0.0 * jnp.real(jnp.ravel(leaf)[0])).astype(dtype)


def _to_struct(x: Any) -> Any:
    if isinstance(x, jax.core.ShapedArray):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)
    if isinstance(x, jax.core.AbstractValue):
        raise NotImplementedError(
            "`splineax.linear_solve` only supports working with JAX arrays; not other "
            f"abstract values. Got abstract value {x}."
        )
    return x


def _to_shapedarray(x: Any) -> Any:
    if isinstance(x, jax.ShapeDtypeStruct):
        return jax.core.ShapedArray(x.shape, x.dtype)
    return x


def _strip_operator(state: Any) -> Any:
    """Replace the state's operator with None so it is not carried through the primitive.

    The operator is a differentiated input. Passing it out as an output too would duplicate
    its arrays in the graph, so it is dropped here and grafted back on with its original
    identity in `linear_solve`.
    """
    if not hasattr(state, "operator"):
        return state
    return eqx.tree_at(lambda s: s.operator, state, None, is_leaf=_is_none)


def _restore_operator(state: Any, operator: Any) -> Any:
    """Graft `operator` back onto a state whose operator was stripped for the primitive."""
    if not hasattr(state, "operator"):
        return state
    return eqx.tree_at(lambda s: s.operator, state, operator, is_leaf=_is_none)


def _linear_solve_impl(
    operator, state, vector, options, solver, throw, *, check_closure
):
    out = solver.compute_stateful(state, vector, options)
    if check_closure:
        out = eqxi.nontraceable(
            out, name="splineax.linear_solve with respect to a closed-over value"
        )
    solution, result, new_state, stats = out
    # A stripped-operator copy of the state leaves the primitive. The token it carries is
    # what a later refactor waits on.
    new_state = _strip_operator(new_state)
    has_nonfinite = jnp.any(
        jnp.stack(
            [jnp.any(jnp.invert(jnp.isfinite(x))) for x in jtu.tree_leaves(solution)]
        )
    )
    result = RESULTS.where(
        (result == RESULTS.successful) & has_nonfinite,
        RESULTS.singular,
        result,
    )
    if throw:
        solution, result, stats, new_state = result.error_if(
            (solution, result, stats, new_state),
            result != RESULTS.successful,
        )
    return solution, result, stats, new_state


@eqxi.filter_primitive_def
def _linear_solve_abstract_eval(operator, state, vector, options, solver, throw):
    state, vector, options, solver = jtu.tree_map(
        _to_struct, (state, vector, options, solver)
    )
    out = eqx.filter_eval_shape(
        _linear_solve_impl,
        operator,
        state,
        vector,
        options,
        solver,
        throw,
        check_closure=False,
    )
    out = jtu.tree_map(_to_shapedarray, out)
    return out


@eqxi.filter_primitive_jvp
def _linear_solve_jvp(primals, tangents):
    operator, state, vector, options, solver, throw = primals
    t_operator, t_state, t_vector, t_options, t_solver, t_throw = tangents
    del t_options, t_solver, t_throw

    solution, result, stats, new_state = eqxi.filter_primitive_bind(
        _splineax_linear_solve_p, operator, state, vector, options, solver, throw
    )

    # Full-rank square solve, so x = A^-1 b and x' = A^-1 (b' - A' x). The tangent solve
    # applies A^-1 after the primal, so it factors a fresh, independent slot rather than the
    # state's shared one, which a later refactor of the same cache slot would overwrite.
    vecs = []
    if any(t is not None for t in jtu.tree_leaves(t_vector, is_leaf=_is_none)):
        vecs.append(
            jtu.tree_map(eqxi.materialise_zeros, vector, t_vector, is_leaf=_is_none)
        )
    if any(t is not None for t in jtu.tree_leaves(t_operator, is_leaf=_is_none)):
        t_operator = linearise(TangentLinearOperator(operator, t_operator))
        vecs.append((-(t_operator.mv(solution) ** ω)).ω)

    # A state that keeps a shared, mutable cache slot carries `order_witness`, a leaf with
    # no factorization meaning of its own. Threading a real (not symbolic-zero) value
    # through it here is what lets the transpose rule below chain a later solve's adjoint
    # read into an earlier one's adjoint refactor, so reusing the slot stays correct
    # without giving each adjoint an independent factorization. See `_linear_solve_transpose`.
    incoming_witness = (
        None if t_state is None else getattr(t_state, "order_witness", None)
    )
    chains_reuse = hasattr(new_state, "order_witness")

    if len(vecs) == 0:
        t_solution = jtu.tree_map(jnp.zeros_like, solution)
        t_new_state = jtu.tree_map(lambda _: None, new_state)
        if chains_reuse and incoming_witness is not None:
            # Nothing here differentiates this solve, but a chain passing through it must
            # still carry the witness onward, or a solve before it would wrongly conclude
            # nothing downstream reuses its slot.
            t_new_state = eqx.tree_at(
                lambda s: s.order_witness,
                t_new_state,
                incoming_witness,
                is_leaf=_is_none,
            )
    else:
        rhs = jtu.tree_map(_sum, *vecs) if len(vecs) > 1 else vecs[0]
        state_isolated, options_isolated = solver.isolate(state, options)
        if chains_reuse:
            witness_in = incoming_witness
            if witness_in is None:
                witness_in = _local_witness(rhs, _witness_dtype(state_isolated))
            state_isolated = eqx.tree_at(
                lambda s: s.order_witness, state_isolated, witness_in
            )
        t_solution, _, _, t_new_state_inner = eqxi.filter_primitive_bind(
            _splineax_linear_solve_p,
            operator,
            state_isolated,
            rhs,
            options_isolated,
            solver,
            True,
        )
        t_new_state = jtu.tree_map(lambda _: None, new_state)
        if chains_reuse:
            t_new_state = eqx.tree_at(
                lambda s: s.order_witness,
                t_new_state,
                t_new_state_inner.order_witness,
                is_leaf=_is_none,
            )

    out = solution, result, stats, new_state
    t_out = (
        t_solution,
        jtu.tree_map(lambda _: None, result),
        jtu.tree_map(lambda _: None, stats),
        t_new_state,
    )
    return out, t_out


@eqxi.filter_primitive_transpose
def _linear_solve_transpose(inputs, cts_out):
    cts_solution, _, _, cts_new_state = cts_out
    operator, state, vector, options, solver, _ = inputs
    cts_solution = jtu.tree_map(
        ft.partial(eqxi.materialise_zeros, allow_struct=True),
        operator.in_structure(),
        cts_solution,
    )
    operator_transpose = operator.transpose()
    # `cts_new_state` is the witness a later solve's own transpose call produced for its
    # `state` input, or `None` when nothing downstream reuses this state's cache slot (the
    # last solve in a reused chain). Threading it into `transpose` is what lets a solver
    # reuse its shared slot for this adjoint's refactor, ordered after that later read,
    # rather than always factoring an independent one. See `StatefulSolver.transpose`.
    order_after = (
        None if cts_new_state is None else getattr(cts_new_state, "order_witness", None)
    )
    state_transpose, options_transpose = solver.transpose(
        state, options, order_after=order_after
    )
    cts_vector, _, _, _ = eqxi.filter_primitive_bind(
        _splineax_linear_solve_p,
        operator_transpose,
        state_transpose,
        cts_solution,
        options_transpose,
        solver,
        True,
    )
    cts_vector = jtu.tree_map(
        lambda v, ct: ct if isinstance(v, ad.UndefinedPrimal) else None,
        vector,
        cts_vector,
        is_leaf=lambda x: isinstance(x, ad.UndefinedPrimal),
    )
    operator_none = jtu.tree_map(lambda _: None, operator)
    state_cts = jtu.tree_map(lambda _: None, state)
    witness_in = getattr(state, "order_witness", None)
    if isinstance(witness_in, ad.UndefinedPrimal):
        # A later solve's transpose call wants a cotangent for our own `state` input,
        # chaining this read into it. Derive it from `cts_vector`, this solve's own
        # adjoint result, so the value genuinely depends on this read having happened.
        outgoing_witness = _read_witness(cts_vector, witness_in.aval.dtype)
        state_cts = eqx.tree_at(
            lambda s: s.order_witness, state_cts, outgoing_witness, is_leaf=_is_none
        )
    options_none = jtu.tree_map(lambda _: None, options)
    solver_none = jtu.tree_map(lambda _: None, solver)
    return operator_none, state_cts, cts_vector, options_none, solver_none, None


_splineax_linear_solve_p = eqxi.create_vprim(
    "splineax_linear_solve",
    eqxi.filter_primitive_def(ft.partial(_linear_solve_impl, check_closure=False)),
    _linear_solve_abstract_eval,
    _linear_solve_jvp,
    _linear_solve_transpose,
)
_splineax_linear_solve_p.def_impl(
    eqxi.filter_primitive_def(ft.partial(_linear_solve_impl, check_closure=True))
)
eqxi.register_impl_finalisation(_splineax_linear_solve_p)


@overload
def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Array],
    solver: StatefulSolver[StateT],
    *,
    options: dict[str, Any] | None = ...,
    state: StateT = ...,
    throw: bool = ...,
) -> tuple[Solution, StateT]: ...


@overload
def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Array],
    solver: Any = ...,
    *,
    options: dict[str, Any] | None = ...,
    state: PyTree[Any] = ...,
    throw: bool = ...,
) -> tuple[Solution, Any]: ...


def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Array],
    solver: Any = None,
    *,
    options: dict[str, Any] | None = None,
    state: PyTree[Any] = sentinel,
    throw: bool = True,
) -> tuple[Solution, Any]:
    """Solve `operator @ x = vector`, returning the solution and an updated state.

    This is the explicit stateful API. It runs the solver's `init` or `update` to fold the
    operator into a state, solves, and threads the factorization token into the returned
    state so a later `update` on the same state is ordered after this solve. It returns a
    `(solution, state)` tuple:

    ```python
    solution, state = splineax.linear_solve(operator, vector, solver, state=state)
    ```

    With no `state`, a fresh one is built with `init`. The default solver is
    `AutoSparseLinearSolver`, which picks a backend for the platform and precision.

    Unlike `lineax.linear_solve`, this owns its own differentiation, so `jax.grad` and
    `jax.jvp` stay correct across a reused factorization. A non-stateful solver, for example
    a dense `lineax.LU`, is solved through `lineax.linear_solve` instead, and the state
    carried on its `Solution` is returned so the tuple shape holds.
    """
    if solver is None:
        # Imported here to avoid a cycle: `_auto` reaches this module through `_sparse`.
        from splineax.solvers._auto import AutoSparseLinearSolver

        solver = AutoSparseLinearSolver()
    if not isinstance(solver, StatefulSolver):
        # A non-stateful solver keeps lineax's own state and differentiation.
        solution = _lx_linear_solve(
            operator, vector, solver, options=options, state=state, throw=throw
        )
        return solution, solution.state
    # Match lineax and cast an integer or weakly-typed right-hand side to inexact, so it
    # lines up with the operator's structure the solvers ravel against.
    vector = jtu.tree_map(inexact_asarray, vector)
    opts = {} if options is None else options
    # `init`/`update` build the factorization. The operator is passed through as-is, so
    # `update` can compare it by identity, and the solvers stop gradients on the values
    # before handing them to the native analyze and factor.
    if state is sentinel:
        state = solver.init(operator, opts)
    else:
        state = solver.update(state, operator, opts)
    solution, result, stats, new_state = eqxi.filter_primitive_bind(
        _splineax_linear_solve_p, operator, state, vector, opts, solver, throw
    )
    # The primitive strips the operator from the threaded state, so restore it here with
    # its original identity, which keeps `update`'s no-op check working for a repeated
    # operator.
    new_state = _restore_operator(new_state, state.operator)
    return Solution(
        value=solution, result=result, stats=stats, state=new_state
    ), new_state
