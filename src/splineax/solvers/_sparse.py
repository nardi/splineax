import warnings
from typing import (
    Any,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import numpy as np
from asdex import ColoredPattern
from equinox.internal import ω
from jax._src.ad_util import SymbolicZero  # noqa: PLC2701
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Inexact, Integer, PyTree
from lineax import (
    AbstractLinearOperator,
    AbstractLinearSolver,
    TangentLinearOperator,
    linearise,
)
from lineax import linear_solve as _lx_linear_solve
from lineax._solution import RESULTS, Solution

from splineax._profile import (
    order_slot_scope,
    profiling_active,
    reserve_order_slot,
    sparsity_hash,
)
from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._tagged import materialise_as_bcoo
from splineax.operators._tags import (
    TaggedOperator,
    _ContentPatternTag,
    coloring_index_array,
    find_pattern_tag,
    pattern_indices,
)
from splineax.operators._tags import operator_pattern_tag as operator_pattern_tag
from splineax.operators._tags import sparse_indices_sorted as sparse_indices_sorted
from splineax.operators._tags import sparsity_pattern_tag as sparsity_pattern_tag
from splineax.solvers._stateful import StatefulSolver

# Everything `init_symbolic` accepts as a sparsity pattern.
_Sparsity = (
    BCOO
    | BCSR
    | BCOOLinearOperator
    | BCSRLinearOperator
    | ColoredPattern
    | _ContentPatternTag
    | TaggedOperator
)


class PatternCoordinates(eqx.Module):
    """The coordinates of a sparsity pattern, with values when the source has them."""

    rows: Integer[Array, " nse"]
    """The int32 row index of each entry."""

    columns: Integer[Array, " nse"]
    """The int32 column index of each entry."""

    shape: tuple[int, ...] = eqx.field(static=True)
    """The shape of the matrix."""

    values: Inexact[Array, " nse"] | None
    """The values of a `BCOO` or `BCSR` source, or None for a source without values."""


def sparse_operator(
    operator: AbstractLinearOperator, solver_name: str
) -> BCOOLinearOperator | BCSRLinearOperator:
    """Return `operator` as one of the sparse operators the solvers read.

    A `BCOOLinearOperator` or `BCSRLinearOperator` is returned as it is. A tagged
    `lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator`, also inside a
    `lineax.TaggedLinearOperator`, is materialised with `materialise_as_bcoo`.
    """
    match operator:
        case BCOOLinearOperator() | BCSRLinearOperator():
            return operator
        case (
            lx.JacobianLinearOperator()
            | lx.FunctionLinearOperator()
            | lx.TaggedLinearOperator()
        ):
            return materialise_as_bcoo(operator)
        case _:
            raise TypeError(
                f"`{solver_name}` requires a sparse operator backed by a `BCOO` or "
                "`BCSR` matrix (e.g. `splineax.BCOOLinearOperator` or "
                "`splineax.BCSRLinearOperator`), or a `lineax.JacobianLinearOperator` or "
                "`lineax.FunctionLinearOperator` carrying a sparsity-pattern tag; got "
                f"{type(operator).__name__}."
            )


def _coordinates_from_indices(
    indices: Array | np.ndarray,
    shape: tuple[int, ...],
    values: Inexact[Array, " nse"] | None = None,
) -> PatternCoordinates:
    """Split an `(nse, 2)` index array into int32 rows and columns."""
    indices = jnp.asarray(indices)
    return PatternCoordinates(
        indices[:, 0].astype(jnp.int32),
        indices[:, 1].astype(jnp.int32),
        tuple(shape),
        values,
    )


def pattern_coordinates(sparsity: _Sparsity, solver_name: str) -> PatternCoordinates:
    """Read the coordinates of a sparsity pattern for `init_symbolic`.

    A tag, a tagged lineax operator and an `asdex.ColoredPattern` carry the pattern
    host-side, so their indices are read without materialising anything. Their entry order is the
    order in which a later solve will pair values with the indices.
    """
    match sparsity:
        case ColoredPattern():
            return _coordinates_from_indices(*coloring_index_array(sparsity))
        case _ContentPatternTag(indices=indices, shape=shape):
            return _coordinates_from_indices(indices, shape)
        case BCOO():
            return _coordinates_from_indices(
                sparsity.indices, sparsity.shape, sparsity.data
            )
        case BCSR():
            return pattern_coordinates(sparsity.to_bcoo(), solver_name)
        case BCOOLinearOperator() | BCSRLinearOperator():
            return pattern_coordinates(sparsity.matrix, solver_name)
        case (
            lx.JacobianLinearOperator()
            | lx.FunctionLinearOperator()
            | lx.TaggedLinearOperator()
        ):
            tag = find_pattern_tag(sparsity.tags)
            if not isinstance(tag, _ContentPatternTag):
                raise TypeError(
                    f"`{solver_name}.init_symbolic` needs a sparsity-pattern tag with "
                    f"concrete indices on a `lineax.{type(sparsity).__name__}`."
                )
            return pattern_coordinates(tag, solver_name)
        case _:
            raise TypeError(
                f"`{solver_name}.init_symbolic` requires a `BCOO`, `BCSR`, "
                "`BCOOLinearOperator`, `BCSRLinearOperator`, a sparsity-pattern tag, a "
                "tagged `lineax.JacobianLinearOperator` or "
                "`lineax.FunctionLinearOperator`, or an `asdex.ColoredPattern`; got "
                f"{type(sparsity).__name__}."
            )


class PerformanceWarning(UserWarning):
    """Raised when a sparse solver has to do work that a differently prepared input
    would have avoided.

    Currently only used by `Spsolve` and `Pardiso`, when their `init` sorts an
    unsorted `BCOO` or `BCSR` operator before solving. Both need row-major sorted
    indices and will silently sort them for you, but doing so on every `init` is
    wasted work if the same operator is solved more than once. Passing an
    already-sorted matrix (for a `BCOO`, call `.sort_indices()` once yourself)
    avoids the warning and the repeated cost.
    """


def warn_if_unsorted(matrix: BCOO | BCSR, solver_name: str) -> None:
    """Raises a `PerformanceWarning` if `matrix`'s indices are not sorted.

    Shared by `Spsolve.init` and `Pardiso.init`, both of which sort an unsorted
    `BCOO` or `BCSR` operator (via a `BCSR.from_bcoo` round-trip) before solving.
    """
    if not matrix.indices_sorted:
        warnings.warn(
            f"`{solver_name}` received a `{type(matrix).__name__}` matrix with "
            "unsorted indices, and must sort them before solving. Passing an "
            "already-sorted matrix avoids this overhead.",
            PerformanceWarning,
            stacklevel=2,
        )


def profile_inputs(
    pattern: Any,
    tag: object | None,
    shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Build the `shape`/`nse`/`sparsity_hash` inputs a solve profile records for an operation.

    Reads the pattern's concrete indices (None under `jit`) for `nse`, and hashes `tag` for
    `sparsity_hash`. Meant to be called lazily (only when a profile is active), since reading
    the index array is not free.
    """
    indices, index_shape = pattern_indices(pattern)
    nse = None if indices is None else int(indices.shape[0])
    return {
        "shape": shape if shape is not None else index_shape,
        "nse": nse,
        "sparsity_hash": sparsity_hash(tag),
    }


def sparsity_reuse_block(
    state_tag: object | None, operator_tag: object | None
) -> str | None:
    """Why a state's analysis cannot be reused for an operator, or None if it can.

    `update` reuses a symbolic analysis only when both the state and the new operator carry
    the same sparsity-pattern tag. This returns a short reason a solve profile can record when
    that fails, so a rebuilt-from-scratch analysis is motivated rather than mysterious.
    """
    if state_tag is None:
        return "State has no sparsity tag"
    if operator_tag is None:
        return "Operator has no sparsity tag"
    if state_tag != operator_tag:
        return "Different sparsity tag"
    return None


_StateT = TypeVar("_StateT")


def _tangent_zeros(primal: Any) -> Any:
    """The all-zero tangent pytree matching `primal`, one `SymbolicZero` per leaf.

    A custom JVP rule must return tangent outputs with the same container structure as
    the primal outputs, and a solve returns a rich pytree: the solution value, the
    RESULTS code, the stats, and the solver state. `SymbolicZero` leaves keep both
    structures equal (unlike `Zero`, which is itself a pytree node and would change the
    structure) while telling JAX no tangent flows there. Only the solution value carries
    a real tangent.

    A leaf that is not an array, such as the function inside a stored
    `lineax.JacobianLinearOperator`, gets `None`. `_StaticOutputs` drops those leaves
    from the primal outputs too, so both structures still match.
    """
    return jtu.tree_map(
        lambda x: (
            SymbolicZero(jax.typeof(x).to_tangent_aval()) if eqx.is_array(x) else None
        ),
        primal,
    )


def _has_tangent(tangent: Any) -> bool:
    """Whether a tangent pytree holds any leaf that is not a symbolic zero."""
    return any(type(t) is not SymbolicZero for t in jtu.tree_leaves(tangent))


def _assert_zero_tangent(tangent: Any, name: str) -> None:
    """Raise if a tangent pytree holds any leaf that is not a symbolic zero.

    The solver, the options, and the state are constants of the differentiation: the
    factorization does not depend differentiably on anything, so a tangent on them
    means the caller differentiated something the solve cannot thread through.
    """
    if _has_tangent(tangent):
        raise ValueError(
            f"`splineax.linear_solve` received a tangent for `{name}`, which is a "
            "constant of the solve and cannot be differentiated."
        )


def _instantiate_tangent(primal: Any, tangent: Any) -> Any:
    """Replace symbolic-zero leaves with concrete zero arrays, keeping real tangents."""
    return jtu.tree_map(
        lambda _, t: (
            jnp.zeros(t.aval.shape, t.aval.dtype) if type(t) is SymbolicZero else t
        ),
        primal,
        tangent,
    )


def _zero_leaves_as_none(tangent: Any) -> Any:
    """Replace symbolic-zero leaves with `None`, as equinox's filter JVP expects."""
    return jtu.tree_map(lambda t: None if type(t) is SymbolicZero else t, tangent)


def _prepare_state(
    operator: AbstractLinearOperator,
    state: Any,
    solver: Any,
    opts: dict[str, Any],
) -> Any:
    """Fold the operator into the state: `init` with no state, `update` with one."""
    if state is None:
        return solver.init(operator, opts)
    return solver.update(state, operator, opts)


class _OrderedSolveState(eqx.Module):
    """A solver state paired with the profile order slot of the solve that reads it."""

    state: PyTree[Any]
    """The wrapped solver's own state."""

    order_slot: tuple[int, ...] = eqx.field(static=True)
    """The position in program order reserved for the solve's profile records."""


class _ProfileOrderedSolver(AbstractLinearSolver):
    """A solver wrapper that sorts a solve's profile records at the solve's call site.

    lineax's `linear_solve` primitive traces `compute` late, during abstract evaluation,
    lowering, and batching. By then the code after the solve has already been traced, so
    the solve's records would sort after it. This wrapper's `compute` records inside the
    order slot its `_OrderedSolveState` carries, which `_solve_factorization` reserves at
    the call site.
    """

    solver: AbstractLinearSolver
    """The solver that does the work."""

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _OrderedSolveState:
        """Initialize the wrapped solver and reserve a slot at this point."""
        return _OrderedSolveState(
            self.solver.init(operator, options), reserve_order_slot()
        )

    def compute(
        self,
        state: _OrderedSolveState,
        vector: PyTree[Any],
        options: dict[str, Any],
    ) -> tuple[PyTree[Any], RESULTS, dict[str, Any]]:
        """Run the wrapped solver's `compute` with its records sorted at the state's slot."""
        with order_slot_scope(state.order_slot):
            return self.solver.compute(state.state, vector, options)

    def transpose(
        self, state: _OrderedSolveState, options: dict[str, Any]
    ) -> tuple[_OrderedSolveState, dict[str, Any]]:
        """Transpose the wrapped state and reserve a new slot for the transposed solve.

        lineax calls this where it stages the transposed solve. Under reverse mode, that is
        during transposition, after the whole forward pass, which is also when the
        backward solve runs. So the transposed solve takes a new slot at this point
        instead of the forward solve's slot.
        """
        transposed_state, transposed_options = self.solver.transpose(
            state.state, options
        )
        return (
            _OrderedSolveState(transposed_state, reserve_order_slot()),
            transposed_options,
        )

    def conj(
        self, state: _OrderedSolveState, options: dict[str, Any]
    ) -> tuple[_OrderedSolveState, dict[str, Any]]:
        """Conjugate the wrapped state and reserve a new slot for the conjugated solve.

        lineax calls this where it stages the conjugated solve, so the new slot sits at
        that point in the program. See `transpose`.
        """
        conjugated_state, conjugated_options = self.solver.conj(state.state, options)
        return (
            _OrderedSolveState(conjugated_state, reserve_order_slot()),
            conjugated_options,
        )

    def assume_full_rank(self) -> bool:
        """Whether the wrapped solver assumes a full-rank operator."""
        return self.solver.assume_full_rank()


def _solve_factorization(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    solver: Any,
    opts: dict[str, Any],
    state: Any,
    throw: bool,
) -> Solution:
    """Solve against a prepared `state`, without tracking the solution onto it.

    `linear_solve` and its JVP rule share this body. The rule passes the same operator
    object it used for the primal solve, so `update` is an identity no-op and the tangent
    solve reuses the primal's factorization.

    While a `SolveProfile` is active, the solver and state are wrapped in
    `_ProfileOrderedSolver` and `_OrderedSolveState`. The solve's records then sort at
    this call site. With no profile active, the solve goes to lineax unchanged.
    """
    if not profiling_active():
        return _lx_linear_solve(
            operator, vector, solver, options=opts, state=state, throw=throw
        )
    # Reserve the slot now, while the code around this solve is being traced.
    ordered_state = _OrderedSolveState(state, reserve_order_slot())
    solution = _lx_linear_solve(
        operator,
        vector,
        _ProfileOrderedSolver(solver),
        options=opts,
        state=ordered_state,
        throw=throw,
    )
    # Return the solver's own state on the solution, as the unprofiled path does.
    return Solution(
        value=solution.value,
        result=solution.result,
        state=state,
        stats=solution.stats,
    )


def update_then_compute(
    solver: Any,
    state: Any,
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    options: dict[str, Any],
) -> tuple[PyTree[Any], RESULTS, dict[str, Any], Any]:
    """Update `state` for `operator`, then solve through `lineax.linear_solve`.

    This is the `update_and_compute` of a solver that has nothing to decide after the
    solve. It never raises on a failed result. The caller checks the result code.
    """
    updated_state = solver.update(state, operator, options)
    solution = _solve_factorization(
        operator, vector, solver, options, updated_state, throw=False
    )
    return solution.value, solution.result, solution.stats, updated_state


def _stateful_solve_impl(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    state: Any,
    *,
    solver: Any,
    options: dict[str, Any] | None,
    throw: bool,
) -> tuple[Solution, Any]:
    """The solve body shared by `linear_solve` and its custom JVP rule.

    Runs `init` when there is no state, then `update_and_compute`, and tracks the
    solution, exactly as `linear_solve` does for a stateful solver.
    """
    opts = {} if options is None else options
    if state is None:
        # A state fresh from `init` matches the operator, so there is nothing to update.
        state = solver.init(operator, opts)
        initial = _solve_factorization(
            operator, vector, solver, opts, state, throw=False
        )
        value, result, stats = initial.value, initial.result, initial.stats
    else:
        value, result, stats, state = solver.update_and_compute(
            state, operator, vector, opts
        )
    if throw:
        value = result.error_if(value, result != RESULTS.successful)
    solution = Solution(value=value, result=result, state=state, stats=stats)
    # Order any later `release` after this solve. A no-op for solvers whose state owns
    # nothing, such as `Spsolve`.
    if hasattr(state, "track"):
        state = state.track(solution)
    return solution, state


def _stateful_solve_jvp(
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    *,
    solver: Any,
    options: dict[str, Any] | None,
    throw: bool,
) -> tuple[tuple[Solution, Any], tuple[Solution, Any]]:
    """Tie the primal and tangent solves into one factorization, sharing the state.

    The tangent of `x = A^(-1) b` for a square nonsingular `A` is
    `x' = A^(-1) (b' - A'x)`, so both systems share the matrix and its factorization.
    Rather than letting lineax's own JVP issue the tangent solve against a stopped
    state, this rule runs the primal solve and then issues the tangent solve through
    the same body with the same operator, so the tangent solve's `update` is an identity
    no-op and it reuses the primal's factorization.

    Both solves go through the `lineax.linear_solve` primitive. This rule's own staging may
    be differentiated again, and the primitive's JVP handles that. If `compute` were staged
    directly, a higher derivative would hit the native library solve, which may not support
    the tangent behavior that we require.

    The track happens last, after both solves. The tracked state's tokens are
    entangled with the primal solution, which orders a later (re)factor after it. A
    tangent solve scheduled after the track could be overtaken by the next solve's
    refactor, re-keying the handle it reads and forcing a rebuild (the benign
    SUPERSEDED). Solving first and tracking once after both keeps the tangent solve
    inside the primal's factorization window, and its solution is a witness of the
    same track, so a later refactor waits for both solves.

    Higher derivatives need no separate rule: taking a derivative of this rule's tangent
    solve hits lineax's own JVP on the staged solves (which shares the state), and
    reverse mode transposes the staged equations through JAX's built-in `custom_jvp`
    transpose, so no custom transpose rule is required either.
    """
    operator, vector, state = primals
    t_operator, t_vector, t_state = tangents
    _assert_zero_tangent(t_state, "state")
    if not solver.assume_full_rank():
        raise NotImplementedError(
            "`splineax.linear_solve` cannot differentiate a solve whose solver does "
            "not assume a full-rank (square, nonsingular) operator."
        )
    opts = {} if options is None else options
    prepared = _prepare_state(operator, state, solver, opts)
    solution = _solve_factorization(operator, vector, solver, opts, prepared, throw)

    # Build the tangent right-hand side `b' - A'x`. With no tangent on the vector or
    # the operator, the tangent solve is skipped entirely: the tangent outputs are
    # symbolic zeros, so nothing is staged beyond the primal.
    has_vector_tangent = _has_tangent(t_vector)
    has_operator_tangent = _has_tangent(t_operator)
    if not (has_vector_tangent or has_operator_tangent):
        out_state = prepared
        if hasattr(out_state, "track"):
            out_state = out_state.track(solution)
        return (solution, out_state), (
            _tangent_zeros(solution),
            _tangent_zeros(out_state),
        )

    vecs = []
    if has_vector_tangent:
        vecs.append(_instantiate_tangent(vector, t_vector))
    if has_operator_tangent:
        t_operator_op = linearise(
            TangentLinearOperator(operator, _zero_leaves_as_none(t_operator))
        )
        # The `-A'x` term, conjugated the way lineax's own JVP does.
        vecs.append((-(t_operator_op.mv(solution.value) ** ω)).ω)
    t_rhs = vecs[0]
    for vec in vecs[1:]:
        t_rhs = jtu.tree_map(lambda a, b: a + b, t_rhs, vec)

    # The tangent solve: the same operator object and the same prepared state, so
    # `update` is a no-op and the factorization is shared with the primal solve.
    t_solution = _solve_factorization(operator, t_rhs, solver, opts, prepared, throw)

    # Track the primal solution onto the state once, after both solves. Only the
    # primal solution can be a witness: JAX requires a custom-JVP rule's primal
    # outputs to depend only on the primal inputs, and under `jacfwd`'s vmap the
    # tangent solution is batched, so tracking it would batch the returned state.
    # The tangent solve still reads the prepared state's tokens, which the track
    # chains from, keeping it inside the primal's factorization window.
    out_state = prepared
    if hasattr(out_state, "track"):
        out_state = out_state.track(solution)

    # Only the solution value carries a tangent. The result code, the stats, and the
    # states are constants of the differentiation, so their tangents are symbolic zeros
    # with the primal's structure.
    tangent_solution = Solution(
        value=t_solution.value,
        result=_tangent_zeros(solution.result),
        stats=_tangent_zeros(solution.stats),
        state=_tangent_zeros(solution.state),
    )
    return (solution, out_state), (tangent_solution, _tangent_zeros(out_state))


class _StaticOutputs:
    """Holds the parts of the custom-JVP solve's outputs that are not arrays.

    A `jax.custom_jvp` may only return arrays. A solver state stores its operator,
    and a `lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator` holds a
    Python function as a pytree leaf. The solve therefore returns only the array part,
    keeps the rest here while it is traced, and `linear_solve` joins the two again.
    """

    def __init__(self) -> None:
        self.static: Any = None
        """The non-array part of the outputs, set each time the solve is traced."""

    def array_part(self, outputs: Any) -> Any:
        """Keep the non-array part of `outputs` here and return the array part."""
        array_outputs, self.static = eqx.partition(outputs, eqx.is_array)
        return array_outputs


def _array_outputs_impl(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    state: Any,
    *,
    solver: Any,
    options: dict[str, Any] | None,
    throw: bool,
    static_outputs: _StaticOutputs,
) -> Any:
    """Run `_stateful_solve_impl` and return the array part of its outputs."""
    outputs = _stateful_solve_impl(
        operator, vector, state, solver=solver, options=options, throw=throw
    )
    return static_outputs.array_part(outputs)


def _array_outputs_jvp(
    primals: tuple[Any, ...],
    tangents: tuple[Any, ...],
    *,
    solver: Any,
    options: dict[str, Any] | None,
    throw: bool,
    static_outputs: _StaticOutputs,
) -> tuple[Any, Any]:
    """Run `_stateful_solve_jvp` and return the array part of its primal outputs.

    The tangent outputs already hold `None` where the primal outputs are not arrays.
    """
    outputs, tangent_outputs = _stateful_solve_jvp(
        primals, tangents, solver=solver, options=options, throw=throw
    )
    return static_outputs.array_part(outputs), tangent_outputs


_stateful_solve = eqx.filter_custom_jvp(_array_outputs_impl)
_stateful_solve.def_jvp(_array_outputs_jvp)


@runtime_checkable
class SparseLinearSolver(StatefulSolver[_StateT], Protocol[_StateT]):
    """Structural type for the sparse stateful solvers in this package.

    Extends the solver-agnostic `StatefulSolver` (init, update, compute, transpose, conj,
    assume_full_rank) with `init_symbolic`, which analyzes a known
    sparsity pattern into a reusable state before any values are available. `KLU`,
    `Pardiso`, `Spsolve`, and `AutoSparseLinearSolver` all satisfy it structurally.
    """

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> _StateT:
        """Analyze a sparsity pattern into a state, reused by a later `update`."""
        ...


@overload
def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    solver: Any = ...,
    *,
    options: dict[str, Any] | None = ...,
    throw: bool = ...,
) -> tuple[Solution, Any]: ...


@overload
def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    solver: Any = ...,
    *,
    options: dict[str, Any] | None = ...,
    state: _StateT,
    throw: bool = ...,
) -> tuple[Solution, _StateT]: ...


def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    solver: Any = None,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
    throw: bool = True,
) -> tuple[Solution, Any]:
    """Solve `operator @ x = vector`, returning the solution and an updated state.

    A wrapper over `lineax.linear_solve` for the stateful sparse API. It runs the
    solver's `init` or `update` to fold the operator into a state, solves, then tracks the
    solution against the state so a later release is ordered after it.
    Unlike `lineax.linear_solve`, it returns a `(solution, state)` tuple:

    ```python
    solution, state = splineax.linear_solve(operator, vector, solver, state=state)
    ```

    With no `state`, a fresh one is built with `solver.init`. The default solver is
    `AutoSparseLinearSolver`, which picks a backend for the platform and precision. When a
    `state` is passed, the returned state has the same type, so it can be threaded straight
    back into the next call.

    A solver that does not implement the stateful API (the `StatefulSolver` protocol),
    such as a plain dense `lineax.LU()`, has no reusable state to thread. For such a
    solver this behaves like `lineax.linear_solve`, returning the incoming `state`
    unchanged alongside the solution so the `(solution, state)` return shape stays stable.
    """
    if solver is None:
        # Imported here to avoid a cycle: `_auto` imports this module.
        from splineax.solvers._auto import AutoSparseLinearSolver

        solver = AutoSparseLinearSolver()
    opts = {} if options is None else options
    if not isinstance(solver, StatefulSolver):
        # No reusable state, so defer to `lineax.linear_solve` and hand back the
        # incoming `state` unchanged, keeping the return shape stable.
        solution = _lx_linear_solve(
            operator, vector, solver, options=opts, state=state, throw=throw
        )
        return solution, state
    # The state is a physical thread, not a differentiable quantity: a threaded state
    # carries the previous operator's values in its token arrays, but the next `update`
    # refactors from the new operator's values, so no solution genuinely depends on
    # the old ones through it. Stop its gradient here so those token tangents (and the
    # tangents of the entanglement edges) are dropped before crossing the boundary.
    # The primal pass is unchanged, so the ordering the entanglement provides holds.
    # Only array leaves can take a gradient, so leave the others, such as the function
    # of a stored `lineax.JacobianLinearOperator`, as they are.
    array_state, static_state = eqx.partition(state, eqx.is_array)
    state = eqx.combine(jax.lax.stop_gradient(array_state), static_state)
    # Route through the custom-JVP solve, whose rule ties the primal and tangent
    # solves into one factorization and threads the state between them. The solver, the
    # options, and the `throw` flag cross as static keyword arguments, so they stay
    # Python objects across the boundary rather than traced pytrees.
    static_outputs = _StaticOutputs()
    array_outputs = _stateful_solve(
        operator,
        vector,
        state,
        solver=solver,
        options=opts,
        throw=throw,
        static_outputs=static_outputs,
    )
    solution, out_state = eqx.combine(array_outputs, static_outputs.static)
    return solution, out_state
