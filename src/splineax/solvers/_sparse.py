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
import numpy as np
from asdex import ColoredPattern
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver
from lineax import linear_solve as _lx_linear_solve
from lineax._solution import RESULTS, Solution
from lineax._solve import sentinel

from splineax._profile import (
    order_slot_scope,
    profiling_active,
    reserve_order_slot,
    sparsity_hash,
)
from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
)
from splineax.operators._tags import (
    _ContentPatternTag,
    _IdentityPatternTag,
    coloring_index_array,
)
from splineax.operators._tags import sparse_indices_sorted as sparse_indices_sorted
from splineax.solvers._stateful import StatefulSolver

# Everything `init_symbolic` accepts as a sparsity pattern.
_Sparsity = (
    BCOO
    | BCSR
    | BCOOLinearOperator
    | BCSRLinearOperator
    | SparseJacobianLinearOperator
    | SparseJacobianLinearOperatorColoring
    | JacobianColoring
    | ColoredPattern
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


def _pattern_indices(
    pattern: "_Sparsity",
) -> tuple[np.ndarray | None, tuple[int, ...] | None]:
    """Read a pattern's COO index array and shape as concrete numpy data.

    Returns `(None, None)` when the indices are traced, which sends
    `sparsity_pattern_tag` to its random-id fallback. The Jacobian and coloring forms read
    their pattern from the precomputed asdex coloring, whose indices are always concrete.
    """
    match pattern:
        case BCOO():
            indices, shape = pattern.indices, pattern.shape
        case BCSR():
            bcoo = pattern.to_bcoo()
            indices, shape = bcoo.indices, bcoo.shape
        case BCOOLinearOperator():
            indices, shape = pattern.matrix.indices, pattern.matrix.shape
        case BCSRLinearOperator():
            bcoo = pattern.matrix.to_bcoo()
            indices, shape = bcoo.indices, bcoo.shape
        case SparseJacobianLinearOperator() | JacobianColoring():
            return coloring_index_array(pattern.coloring)
        case SparseJacobianLinearOperatorColoring():
            return coloring_index_array(pattern.coloring.coloring)
        case ColoredPattern():
            return coloring_index_array(pattern)
        case _:
            return None, None
    if isinstance(indices, jax.core.Tracer):
        return None, None
    return np.asarray(indices), tuple(shape)


def sparsity_pattern_tag(pattern: "_Sparsity | None" = None) -> object:
    """Create a tag marking an operator's structural sparsity pattern.

    Attach the tag to operators through their `tags` argument. Two operators carrying
    equal tags are asserted to have exactly the same index arrays, in the same order, so
    a solver may reuse one operator's factorization for the other.

    Given a concrete `pattern`, the tag is content-hashed, so independently tagged
    operators with the same indices get equal tags. With no argument, or a pattern whose
    indices are traced under jit, the tag instead carries a random id. Thread that one
    tag object onto every operator sharing the pattern to mark them as equal.
    """
    if pattern is None:
        return _IdentityPatternTag()
    indices, shape = _pattern_indices(pattern)
    if indices is None or shape is None:
        return _IdentityPatternTag()
    return _ContentPatternTag(indices, shape)


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
    indices, index_shape = _pattern_indices(pattern)
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


def operator_pattern_tag(operator: AbstractLinearOperator) -> object | None:
    """Return the operator's sparsity-pattern tag, or None if it carries none.

    Solvers read this in `update` to decide whether an operator shares a state's pattern. A
    `SparseJacobianLinearOperator` carries a tag derived from its coloring, so operators built
    by one `operator_at` factory, and any BCOO materialised from them, reuse a factorization
    without the caller tagging them.
    """
    for tag in getattr(operator, "tags", ()):
        if isinstance(tag, (_ContentPatternTag, _IdentityPatternTag)):
            return tag
    return None


_StateT = TypeVar("_StateT")


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
    order slot its `_OrderedSolveState` carries, which `linear_solve` reserves at the
    call site.
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
    state: Any = sentinel,
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
        # A non-stateful solver keeps no reusable state, so there is nothing to `init`,
        # `update`, or `track`. Defer the solve to `lineax.linear_solve` and hand back the
        # incoming `state` untouched, keeping the `(solution, state)` return shape stable.
        solution = _lx_linear_solve(
            operator, vector, solver, options=opts, state=state, throw=throw
        )
        return solution, state
    # `init`/`update` build the factorization. The operator is passed through as-is, so
    # `update` can compare it by identity, and the solvers stop gradients on the values
    # themselves before handing them to the native analyze and factor.
    if state is sentinel:
        state = solver.init(operator, opts)
    else:
        state = solver.update(state, operator, opts)
    if profiling_active():
        # While a `SolveProfile` is active, wrap the solver and state so the solve's
        # records sort at this call site (see `_ProfileOrderedSolver`). The slot is
        # reserved now, while the code around this solve is being traced.
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
        solution = Solution(
            value=solution.value,
            result=solution.result,
            state=state,
            stats=solution.stats,
        )
    else:
        solution = _lx_linear_solve(
            operator, vector, solver, options=options, state=state, throw=throw
        )
    # Order any later `release` after this solve. A no-op for solvers whose state owns
    # nothing, such as `Spsolve`.
    if hasattr(state, "track"):
        state = state.track(solution)
    return solution, state
