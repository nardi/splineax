from contextlib import AbstractContextManager, contextmanager
from typing import Any, Iterator, Protocol, runtime_checkable

from jax.experimental.sparse import BCOO, BCSR
from lineax import AbstractLinearOperator

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator

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
