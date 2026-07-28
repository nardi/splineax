import abc
import warnings
from contextlib import AbstractContextManager, contextmanager
from typing import (
    Any,
    Generic,
    Iterator,
    Literal,
    Protocol,
    TypeVar,
    overload,
    runtime_checkable,
)

import equinox as eqx
import jax
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, AutoLinearSolver
from lineax import linear_solve as _lx_linear_solve
from lineax._solution import RESULTS, Solution
from lineax._solve import AbstractLinearSolver, sentinel

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
)
from splineax.solvers._handle import mark_via_linear_solve

# Everything `factorize_symbolic` accepts as a sparsity pattern.
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

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[False] = False
    ) -> AbstractContextManager[SparseSymbolicScope]: ...

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[True]
    ) -> AbstractContextManager["SymbolicScopedSparseLinearSolver"]: ...

    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: bool = False
    ) -> AbstractContextManager[
        "SparseSymbolicScope | SymbolicScopedSparseLinearSolver"
    ]:
        """Pre-compute a symbolic factorization from a known sparsity pattern.

        Yields a `SparseSymbolicScope`, or a `SymbolicScopedSparseLinearSolver`
        bundling that scope with this solver when `as_solver=True`.
        """
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

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[False] = False
    ) -> AbstractContextManager[SparseSymbolicScope]: ...

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[True]
    ) -> AbstractContextManager["SymbolicScopedSparseLinearSolver"]: ...

    @abc.abstractmethod
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: bool = False
    ) -> AbstractContextManager[
        "SparseSymbolicScope | SymbolicScopedSparseLinearSolver"
    ]:
        """Pre-compute a symbolic factorization from a known sparsity pattern.

        Yields a `SparseSymbolicScope`, or a `SymbolicScopedSparseLinearSolver`
        bundling that scope with this solver when `as_solver=True`.
        """


class SymbolicScopedSparseLinearSolver(
    AbstractLinearSolver["SparseSymbolicState | SparseNumericState"]
):
    """A solver bound to one open symbolic-factorization scope.

    Returned by `solver.factorize_symbolic(sparsity, as_solver=True)`, for the common
    case where a solver and a scope derived from it are used together and would
    otherwise have to be passed around as a pair:

    ```python
    with solver.factorize_symbolic(sparsity, as_solver=True) as scoped_solver:
        x = lx.linear_solve(operator, b, solver=scoped_solver).value
    ```

    This is an ordinary `lineax.AbstractLinearSolver`, so it can be passed as
    `solver=` anywhere the solver it came from can. The difference is its `init`,
    which is the scope's `init`: every solve made through it reuses the scope's
    symbolic factorization, and it is therefore only valid for operators sharing the
    sparsity pattern the scope was opened with. Solving happens through the original
    solver, so the numerical result is exactly that of passing `solver=solver,
    state=scope.init(operator)` by hand.

    Like the scope it wraps, it is only usable inside the `with` block that yielded
    it: once that block exits, the underlying factorization is freed.
    """

    solver: AbstractSparseLinearSolver[Any]
    """The solver the scope was opened from, which performs every solve."""
    scope: SparseSymbolicScope
    """The open symbolic-factorization scope every state is derived from."""

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> SparseSymbolicState:
        """Build a state reusing the scope's symbolic factorization.

        Identical to the scope's own `init`, which is what restricts this solver to
        operators with the scope's sparsity pattern.
        """
        return self.scope.init(operator, options)

    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        """Also pre-compute the numeric factorization, reusing the symbolic one.

        Identical to the scope's own `factorize`. `options` is accepted for parity
        with `AbstractSparseLinearSolver.factorize` and unused, since a scope's
        `factorize` takes none.
        """
        del options
        return self.scope.factorize(operator)

    def compute(
        self,
        state: SparseSymbolicState | SparseNumericState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        return self.solver.compute(state, vector, options)

    def transpose(
        self, state: SparseSymbolicState | SparseNumericState, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        return self.solver.transpose(state, options)

    def conj(
        self, state: SparseSymbolicState | SparseNumericState, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        return self.solver.conj(state, options)

    def assume_full_rank(self) -> bool:
        return self.solver.assume_full_rank()


SymbolicScopedSparseLinearSolver.__init__.__doc__ = """**Arguments:**

- `solver`: the sparse solver performing the solves.
- `scope`: an open symbolic-factorization scope, as opened by that solver's
    `factorize_symbolic`.

Usually not constructed directly: call `solver.factorize_symbolic(sparsity,
as_solver=True)` instead, which opens the scope and pairs it with the solver.
"""


@contextmanager
def as_scoped_solver(
    solver: AbstractSparseLinearSolver[Any],
    scope_manager: AbstractContextManager[SparseSymbolicScope],
) -> Iterator[SymbolicScopedSparseLinearSolver]:
    """Wrap an unopened `factorize_symbolic` scope into a scoped solver.

    Shared implementation of `factorize_symbolic(..., as_solver=True)` for every
    solver: opening and closing the scoped solver opens and closes the scope itself,
    so the factorization lives exactly as long as it would have.
    """
    with scope_manager as scope:
        yield SymbolicScopedSparseLinearSolver(solver, scope)


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

    A `SymbolicScopedSparseLinearSolver` carries its scope instead of being handed a
    state, so there `lineax.linear_solve` would build the (handle-owning) state itself
    and never hand it back in time to be registered. This function runs that `init`
    first, exactly as `lineax.linear_solve` would have, and then registers against the
    state it gets, so a scoped solver is safe under one traced scope too.
    """
    if state is sentinel and isinstance(solver, SymbolicScopedSparseLinearSolver):
        # Same stop-gradient treatment `lineax.linear_solve` gives the operator before
        # its own `init` call, so the state carries no tangents either way.
        dynamic_operator, static_operator = eqx.partition(operator, eqx.is_array)
        stopped_operator = eqx.combine(
            jax.lax.stop_gradient(dynamic_operator), static_operator
        )
        state = solver.init(stopped_operator, {} if options is None else options)
    with mark_via_linear_solve():
        solution = _lx_linear_solve(
            operator, vector, solver, options=options, state=state, throw=throw
        )
    if isinstance(state, _HandleOwningState):
        state._register_solve_dependency(solution.value)
    return solution
