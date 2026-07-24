import abc
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Generic, Iterator, Protocol, TypeVar, runtime_checkable

from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import PyTree
from lineax import AbstractLinearOperator, AutoLinearSolver
from lineax import linear_solve as _lx_linear_solve
from lineax._solution import Solution
from lineax._solve import AbstractLinearSolver, sentinel

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.solvers._handle import mark_via_linear_solve

_Sparsity = BCOO | BCSR | BCOOLinearOperator | BCSRLinearOperator


class SparseNumericState(Protocol):
    """A fully factorized sparse solver state, ready to pass to `lineax.linear_solve`.

    Marker protocol: a terminal state with no further factorization step.
    """


@runtime_checkable
class SparseBasicState(Protocol):
    """The state returned by `SparseLinearSolver.init`.

    Can be turned into a numeric factorization for reuse across solves.
    """

    def factorize(self) -> AbstractContextManager[SparseNumericState]:
        """Pre-compute a numeric factorization, yielding a reusable state."""
        ...


@runtime_checkable
class SparseSymbolicState(Protocol):
    """A state that reuses a pre-computed symbolic factorization.

    Returned by `SparseSymbolicScope.init`. Directly solvable, and can additionally
    be turned into a numeric factorization.
    """

    def factorize(self) -> AbstractContextManager[SparseNumericState]:
        """Pre-compute a numeric factorization, reusing the symbolic one."""
        ...


@runtime_checkable
class SparseSymbolicScope(Protocol):
    """A pre-analyzed symbolic-factorization scope yielded by
    `SparseLinearSolver.factorize_symbolic`."""

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> SparseSymbolicState:
        """Build a directly-solvable state reusing the scope's symbolic factorization."""
        ...

    def factorize(
        self, operator: AbstractLinearOperator
    ) -> AbstractContextManager[SparseNumericState]:
        """Also pre-compute the numeric factorization, reusing the symbolic one."""
        ...


@runtime_checkable
class SparseLinearSolver(Protocol):
    """Structural type for sparse direct solvers that expose factorization reuse on
    top of the lineax `AbstractLinearSolver` interface."""

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> SparseBasicState: ...

    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        """Pre-compute a full factorization for reuse across multiple solves."""
        ...

    def factorize_symbolic(
        self, sparsity: _Sparsity
    ) -> AbstractContextManager[SparseSymbolicScope]:
        """Pre-compute a symbolic factorization from a known sparsity pattern."""
        ...


_SolverState = TypeVar("_SolverState")


class AbstractSparseLinearSolver(
    AbstractLinearSolver[_SolverState], Generic[_SolverState]
):
    """Abstract base for sparse direct solvers that support factorization reuse.

    Extends the lineax `AbstractLinearSolver` interface with `factorize` and
    `factorize_symbolic`. Concrete subclasses (`KLU`, `Spsolve`,
    `AutoSparseLinearSolver`) are therefore usable both with `lineax.linear_solve`
    (which requires an `AbstractLinearSolver`) and the factorization-reuse API. They
    also structurally satisfy the `SparseLinearSolver` protocol.
    """

    @abc.abstractmethod
    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        """Pre-compute a full factorization for reuse across multiple solves."""

    @abc.abstractmethod
    def factorize_symbolic(
        self, sparsity: _Sparsity
    ) -> AbstractContextManager[SparseSymbolicScope]:
        """Pre-compute a symbolic factorization from a known sparsity pattern."""


@contextmanager
def factorize_through_init(
    solver: SparseLinearSolver,
    operator: AbstractLinearOperator,
    options: dict[str, Any],
) -> Iterator[SparseNumericState]:
    """Shared `factorize` behaviour: run `init`, then numeric-factorize its state.

    Reused by both `KLU.factorize` and `Spsolve.factorize` (behaviour reuse via a
    function instead of inheritance).
    """
    with solver.init(operator, options).factorize() as numeric_state:
        yield numeric_state


@runtime_checkable
class _HandleOwningState(Protocol):
    """A state that owns a native handle and must register a solve's result against
    it, implemented by `KLU`'s and `Pardiso`'s symbolic/numeric state classes."""

    def _register_solve_dependency(self, value: Any) -> None: ...


def linear_solve(
    operator: AbstractLinearOperator,
    vector: PyTree[Any],
    solver: AbstractLinearSolver = AutoLinearSolver(well_posed=True),
    *,
    options: dict[str, Any] | None = None,
    state: PyTree[Any] = sentinel,
    throw: bool = True,
) -> Solution:
    """Drop-in replacement for `lineax.linear_solve`, needed for a state derived from
    `KLU`'s or `Pardiso`'s `factorize_symbolic` scope when the scope is opened and
    closed entirely inside one `jax.jit` call.

    `lineax.linear_solve` stages the solve into a trace nested inside whichever trace
    calls it. When the whole scope is traced together with the solve, that leaves the
    scope's handle-freeing free loop, which runs in the *outer* trace, unable to see
    the solve's result and unable to order the free after it. This function also
    registers the result against `state`'s handle(s) itself, from the outer trace,
    right here, once `lineax.linear_solve` returns.

    Solving without `state` set, or with a state that owns no handle (`Spsolve`, or a
    solver's `.init()` state before any factorization), behaves exactly like
    `lineax.linear_solve`: there is nothing to register.
    """
    with mark_via_linear_solve():
        solution = _lx_linear_solve(
            operator, vector, solver, options=options, state=state, throw=throw
        )
    if isinstance(state, _HandleOwningState):
        state._register_solve_dependency(solution.value)
    return solution
