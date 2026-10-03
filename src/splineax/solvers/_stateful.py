"""Solver-agnostic protocols for the stateful linear-solve API.

These describe the surface a solver exposes so a caller can give it information about
the operators it will solve, as soon as that information is known, and thread a solver
state through solves. The protocols are detached from the sparse-solving domain on
purpose, so the same shape could describe any solver that keeps reusable state.

The sparse-specific extension (`init_symbolic`, the sparsity tags) lives in `_sparse.py`.
"""

from collections.abc import Callable
from typing import Any, Protocol, Self, TypeVar, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jaxtyping import Array, Bool, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS

from splineax._profile import record_operation, suppress_records

_StateT = TypeVar("_StateT")
_TreeT = TypeVar("_TreeT")


@runtime_checkable
class TrackingSolverState(Protocol):
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
    new information about the operator into an existing state, and `update_and_compute`
    does that and solves in one call, so a solver can choose how much of the state to
    rebuild after seeing the result of the solve.

    The states a solver produces from `init`, `update`/`update_and_compute`, and a state's `track` should share
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

    def update_and_compute(
        self,
        state: _StateT,
        operator: AbstractLinearOperator,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any], _StateT]:
        """Fold `operator` into `state` and solve against `vector`.

        Returns the solution, the result code, the solver statistics and the updated
        state. A solver may defer or skip the work in `update` when the solve shows that
        the existing state was good enough.
        """
        ...

    def compute(
        self, state: _StateT, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]: ...

    def transpose(
        self, state: _StateT, options: dict[str, Any]
    ) -> tuple[_StateT, dict[str, Any]]: ...

    def conj(
        self, state: _StateT, options: dict[str, Any]
    ) -> tuple[_StateT, dict[str, Any]]: ...

    def assume_full_rank(self) -> bool: ...


def conditional_update(
    update: Callable[
        [_StateT, AbstractLinearOperator, dict[str, Any]],
        _StateT,
    ],
    state: _StateT,
    operator: AbstractLinearOperator,
    options: dict[str, Any],
    refactor: Bool[Array, ""],
) -> _StateT:
    """Run `update` only where the traced `refactor` is true, and keep `state` elsewhere.

    The cond covers the whole `update`, so the factorization work is skipped at runtime
    when `refactor` is false. Both branches must return one pytree structure, which holds
    when `operator` shares the sparsity pattern of the operator in `state`. Records inside
    the branch are dropped (see `suppress_records`), and one record outside carries the
    outcome.
    """
    # Static fields, such as the shape, are not arrays. They come from the refreshed
    # state and must equal the ones in `state`, which the matching pattern guarantees.
    static_parts: list[_StateT] = []

    def refresh() -> _StateT:
        with suppress_records():
            refreshed = update(state, operator, options)
        dynamic, static = eqx.partition(refreshed, eqx.is_array)
        static_parts.append(static)
        return dynamic

    def keep() -> _StateT:
        return eqx.filter(state, eqx.is_array)

    try:
        dynamic = jax.lax.cond(refactor, refresh, keep)
    except TypeError as error:
        raise ValueError(
            "A traced `refactor` requires an operator with the same sparsity pattern as "
            "the operator in the state."
        ) from error
    record_operation(
        "update",
        dynamic={"refactored": refactor},
        outputs=lambda values: {
            "outcome": "reused" if values["refactored"] else "noop",
            "reason": "Refactor requested"
            if values["refactored"]
            else "Existing factorization kept",
        },
    )
    return eqx.combine(dynamic, static_parts[0])


def select_pytree(
    predicate: Bool[Array, ""], on_true: _TreeT, on_false: _TreeT
) -> _TreeT:
    """Choose `on_true` or `on_false` leaf by leaf, where `predicate` is a traced boolean.

    Both pytrees must share one structure. Leaves that are not arrays are taken from
    `on_true`, since a static value cannot depend on a traced predicate.
    """

    def select_leaf(true_leaf: object, false_leaf: object) -> object:
        match true_leaf, false_leaf:
            case ((jax.Array() | np.ndarray()), (jax.Array() | np.ndarray())):
                return jnp.where(predicate, true_leaf, false_leaf)
            case _:
                return true_leaf

    return jtu.tree_map(select_leaf, on_true, on_false)
