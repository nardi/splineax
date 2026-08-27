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
    """A solver state that can record a solve depending on it.

    A state may own memory that must outlive every solve made with it. `track` marks a
    solution as a dependency, so a later `release` is ordered after that solve. It is
    optional to use, and a solver whose state owns nothing implements it as a no-op that
    returns `self`.
    """

    def track(self, solution: PyTree[Array]) -> Self:
        """Return a new state whose eventual `release` is ordered after `solution`."""
        ...


@runtime_checkable
class StatefulSolver(Protocol[_StateT]):
    """A solver that creates, updates, and releases a reusable state.

    This is the part of the lineax `AbstractLinearSolver` interface we rely on, plus
    `update` and `release`. A solver satisfies it structurally, so no base class is
    needed. `update` folds new information about the operator into an existing state, and
    `release` says the state is done and its memory may go.
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

    def release(self, state: _StateT) -> None:
        """Signal that `state` is done, so any memory it owns may be freed."""
        ...

    def compute(
        self, state: _StateT, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]: ...

    def transpose(
        self, state: _StateT, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]: ...

    def conj(
        self, state: _StateT, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]: ...

    def assume_full_rank(self) -> bool: ...
