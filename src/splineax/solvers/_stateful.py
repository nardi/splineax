"""Solver-agnostic protocols for the stateful linear-solve API.

These describe the surface a solver exposes so a caller can give it information about
the operators it will solve, as soon as that information is known, and thread a solver
state through solves. The protocols are detached from the sparse-solving domain on
purpose, so the same shape could describe any solver that keeps reusable state.

The sparse-specific extension (`init_symbolic`, the sparsity tags) lives in `_sparse.py`.
"""

from typing import Any, Protocol, Self, TypeVar, runtime_checkable

from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS

_StateT = TypeVar("_StateT")


@runtime_checkable
class TrackingState(Protocol):
    """A solver state that records solves depending on it and frees its own memory.

    A state may own memory that must outlive every solve made with it. `track` marks a
    solution as a dependency, so a later `release` is ordered after that solve, and
    `release` frees that memory once the state is done. Both live on the state, so a caller
    releases without a reference to the solver. A state that owns nothing implements `track`
    as a no-op returning `self`, and `release` as a no-op.
    """

    def track(self, solution: PyTree[Array]) -> Self:
        """Return a new state whose eventual `release` is ordered after `solution`."""
        ...

    def release(self) -> None:
        """Free any memory this state owns, ordered after its tracked solves."""
        ...


@runtime_checkable
class StatefulSolver(Protocol[_StateT]):
    """A solver that creates and updates a reusable state.

    This is the part of the lineax `AbstractLinearSolver` interface we rely on, plus
    `update`. A solver satisfies it structurally, so no base class is needed. `update` folds
    new information about the operator into an existing state.

    The states a solver produces from `init`, `update`, and a state's `track` should share
    one pytree structure, so a state can be carried through a `scan` or `while_loop`, whose
    carry has a fixed structure. The sparse `init_symbolic` state may differ, since it holds
    only a symbolic analysis. Such a state must be `update`d before it is carried through a
    loop, for example by unrolling the first iteration.
    """

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _StateT: ...

    def update(
        self,
        state: _StateT,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _StateT:
        """Fold a new operator into `state`, reusing prior work where possible."""
        ...

    def compute(
        self, state: _StateT, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]: ...

    def compute_stateful(
        self, state: _StateT, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, _StateT, dict[str, Any]]:
        """Solve and return the solution with a state ordered after this solve.

        The returned state carries a factorization token that waits on this solve, so a
        later `update` refactoring the same cache slot is ordered after it rather than
        racing it under `jit`. `compute` is the same solve with the state dropped, kept for
        lineax's own differentiation.
        """
        ...

    def transpose(
        self,
        state: _StateT,
        options: dict[str, Any],
        *,
        order_after: Any = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Return the transposed state, reusing a shared cache slot when it is safe to.

        `order_after` is `None` when nothing else reuses this state's cache slot after this
        adjoint runs, so the slot already holds the values this call needs and it can solve
        directly. Otherwise `order_after` is a value a later adjoint's own read produced;
        threading it into this call's own refactor, ordered after that value, is what keeps
        the refactor from racing that read under `jit`. A solver with no shared cache slot
        ignores `order_after`.
        """
        ...

    def isolate(
        self, state: _StateT, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        """Return a state backed by a fresh, independent factorization of the same operator.

        A differentiated solve applies the factorization after the primal solve, so reusing
        a shared cache slot a later `update` overwrites would solve the wrong matrix. An
        isolated state owns a slot no other solve writes, so a tangent or adjoint solve
        against it stays correct with no ordering between the solves. A solver that keeps no
        shared handle implements this as a no-op returning `state`.
        """
        ...

    def conj(
        self, state: _StateT, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]: ...

    def assume_full_rank(self) -> bool: ...
