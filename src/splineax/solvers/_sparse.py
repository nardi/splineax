import warnings
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import jax
import jax.core
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from asdex import ColoredPattern
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import PyTree
from lineax import AbstractLinearOperator
from lineax import linear_solve as _lx_linear_solve
from lineax._solution import RESULTS, Solution
from lineax._solve import sentinel

from splineax._trace import tracing_active
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


def sparsity_reuse_block(
    state_tag: object | None, operator_tag: object | None
) -> str | None:
    """Why a state's analysis cannot be reused for an operator, or None if it can.

    `update` reuses a symbolic analysis only when both the state and the new operator carry
    the same sparsity-pattern tag. This returns a short reason a solve trace can record when
    that fails, so a rebuilt-from-scratch analysis is motivated rather than mysterious.
    """
    if state_tag is None:
        return "state carries no sparsity tag to reuse"
    if operator_tag is None:
        return "operator carries no sparsity tag"
    if state_tag != operator_tag:
        return "operator's sparsity tag differs from the state's"
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


def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    solver: Any = None,
    *,
    options: dict[str, Any] | None = None,
    state: PyTree[Any] = sentinel,
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
    `AutoSparseLinearSolver`, which picks a backend for the platform and precision.

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
    if tracing_active():
        # While a `solve_trace` is open, run `compute` directly instead of through lineax's
        # `linear_solve` primitive, which does not propagate the trace's in-`compute`
        # `io_callback`s (the solve and iterative-refinement steps). Off the trace path this
        # is never taken, so ordinary solves keep lineax's full behaviour.
        solution = _traced_compute(operator, vector, solver, opts, state, throw)
    else:
        solution = _lx_linear_solve(
            operator, vector, solver, options=options, state=state, throw=throw
        )
    # Order any later `release` after this solve. A no-op for solvers whose state owns
    # nothing, such as `Spsolve`.
    if hasattr(state, "track"):
        state = state.track(solution)
    return solution, state


def _any_nonfinite(tree: PyTree[Any]) -> Any:
    leaves = jtu.tree_leaves(tree)
    if not leaves:
        return jnp.bool_(False)
    return jnp.any(
        jnp.stack([jnp.any(jnp.invert(jnp.isfinite(leaf))) for leaf in leaves])
    )


def _traced_compute(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    solver: Any,
    options: dict[str, Any],
    state: PyTree[Any],
    throw: bool,
) -> Solution:
    """Solve by calling `solver.compute` directly, for use while a solve trace is open.

    Mirrors `lineax._solve._linear_solve_impl` (the non-finite result adjustment and the
    `throw` check) so the returned `Solution` matches the normal path, but keeps `compute`
    outside lineax's `linear_solve` primitive so the trace's in-`compute` callbacks run.
    """
    solution, result, stats = solver.compute(state, vector, options)
    result = RESULTS.where(
        (result == RESULTS.successful) & _any_nonfinite(solution),
        RESULTS.singular,
        result,
    )
    result = RESULTS.where(
        (result == RESULTS.singular) & _any_nonfinite(vector),
        RESULTS.nonfinite_input,
        result,
    )
    if throw:
        solution, result, stats = result.error_if(
            (solution, result, stats), result != RESULTS.successful
        )
    return Solution(value=solution, result=result, state=state, stats=stats)
