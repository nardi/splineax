import secrets
import warnings
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import jax
import jax.core
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import PyTree
from lineax import AbstractLinearOperator
from lineax import linear_solve as _lx_linear_solve
from lineax._solution import Solution
from lineax._solve import sentinel

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
)
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


class _HasRepr:
    """A tag object whose only content is its repr, matching lineax's own tags."""

    def __init__(self, string: str) -> None:
        self.string = string

    def __repr__(self) -> str:
        return self.string


sparse_indices_sorted = _HasRepr("sparse_indices_sorted")
"""One global assertion that an operator's indices are already row-major sorted, so
`Pardiso` and `Spsolve` may skip the sort they would otherwise do in `init`."""


class _ContentPatternTag:
    """A sparsity-pattern tag identified by the content of its index arrays.

    Two tags are equal when their indices match exactly, so operators built with the same
    pattern reuse each other's factorization even when tagged separately. This follows
    asdex's `_HashableEntries`, which carries concrete index arrays as hashable static aux
    data.
    """

    __slots__ = ("_hash", "indices", "shape")

    def __init__(self, indices: np.ndarray, shape: tuple[int, ...]) -> None:
        self.indices = indices
        self.shape = shape
        self._hash: int | None = None

    def __hash__(self) -> int:
        # Content hashing is O(nnz), so compute it lazily and cache it. A frozenset
        # hashes the tag every time it is built.
        if self._hash is None:
            self._hash = hash(
                (self.indices.tobytes(), str(self.indices.dtype), self.shape)
            )
        return self._hash

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, _ContentPatternTag):
            return NotImplemented
        # The dtype is compared alongside the contents to stay consistent with the hash.
        return (
            self.shape == other.shape
            and self.indices.dtype == other.indices.dtype
            and np.array_equal(self.indices, other.indices)
        )


class _IdentityPatternTag:
    """A sparsity-pattern tag identified by a random id, for use under jit.

    A traced index array cannot be hashed, so this stands in for the content tag. Two
    instances differ, and the same instance threaded onto several operators marks them as
    sharing a pattern. The random id keeps the tag hashable and stable across pytree
    flatten and unflatten, where a bare `object()` identity would not survive.
    """

    __slots__ = ("_id",)

    def __init__(self) -> None:
        self._id = secrets.randbits(128)

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _IdentityPatternTag):
            return NotImplemented
        return self._id == other._id


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
        case SparseJacobianLinearOperator() | SparseJacobianLinearOperatorColoring():
            asdex_pattern = pattern.coloring.sparsity
            indices = np.stack(
                [np.asarray(asdex_pattern.rows), np.asarray(asdex_pattern.cols)], axis=1
            )
            return indices, tuple(asdex_pattern.shape)
        case JacobianColoring():
            asdex_pattern = pattern.sparsity
            indices = np.stack(
                [np.asarray(asdex_pattern.rows), np.asarray(asdex_pattern.cols)], axis=1
            )
            return indices, tuple(asdex_pattern.shape)
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


def operator_pattern_tag(operator: AbstractLinearOperator) -> object | None:
    """Return the operator's sparsity-pattern tag, or None if it carries none.

    Solvers read this in `update` to decide whether an operator shares a state's pattern.
    A user-attached tag wins. Otherwise a `SparseJacobianLinearOperator` derives one from
    its coloring, so operators built by one `operator_at` factory reuse a factorization
    across evaluation points without the caller tagging them. The tag lives only on the
    solver state, never on the operator, so it does not change the operator's pytree.
    """
    for tag in getattr(operator, "tags", ()):
        if isinstance(tag, (_ContentPatternTag, _IdentityPatternTag)):
            return tag
    if isinstance(operator, SparseJacobianLinearOperator) and not operator.transposed:
        return sparsity_pattern_tag(operator)
    return None


_StateT = TypeVar("_StateT")


@runtime_checkable
class SparseLinearSolver(StatefulSolver[_StateT], Protocol[_StateT]):
    """Structural type for the sparse stateful solvers in this package.

    Extends the solver-agnostic `StatefulSolver` (init, update, release, compute,
    transpose, conj, assume_full_rank) with `init_symbolic`, which analyzes a known
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
    solution against the state so a later `solver.release(state)` is ordered after it.
    Unlike `lineax.linear_solve`, it returns a `(solution, state)` tuple:

    ```python
    solution, state = splineax.linear_solve(operator, vector, solver, state=state)
    ```

    With no `state`, a fresh one is built with `solver.init`. The default solver is
    `AutoSparseLinearSolver`, which picks a backend for the platform and precision.
    """
    if solver is None:
        # Imported here to avoid a cycle: `_auto` imports this module.
        from splineax.solvers._auto import AutoSparseLinearSolver

        solver = AutoSparseLinearSolver()
    opts = {} if options is None else options
    # `init`/`update` build the factorization. The operator is passed through as-is, so
    # `update` can compare it by identity, and the solvers stop gradients on the values
    # themselves before handing them to the native analyze and factor.
    if state is sentinel:
        state = solver.init(operator, opts)
    else:
        state = solver.update(state, operator, opts)
    solution = _lx_linear_solve(
        operator, vector, solver, options=options, state=state, throw=throw
    )
    # Order any later `release` after this solve. A no-op for solvers whose state owns
    # nothing, such as `Spsolve`.
    if hasattr(state, "track"):
        state = state.track(solution)
    return solution, state
