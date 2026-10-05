"""Hybrid direct and iterative solving for sparse systems.

A direct solver factors a matrix once and then solves cheaply, but a new matrix needs a
new factorization. When the matrix changes slowly, the factorization of an earlier matrix
is still a good approximation of the inverse. `HybridDirectIterative` uses that
factorization as the preconditioner of an iterative solver, and only factors again when
the iterative solve cannot reach the tolerance within a small number of steps.

The iterative solver is one of the options in `IterativeOptions`. `RichardsonOptions`
repeats the direct solve on the residual, which is iterative refinement. The Krylov
options (`GMRESOptions`) use the same factorization as their preconditioner.

Every result is checked against the true residual `||b - A x||`. A solution that passes
the check is returned as it is. One that fails, after a fresh factorization of the current
matrix, is returned as NaN with a failed result. So the hybrid reaches the tolerance
whenever a direct solve followed by refinement would.
"""

import dataclasses
from collections.abc import Callable
from functools import cached_property
from typing import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import Array, Bool, Float, Int, PyTree
from lineax import (
    CG,
    GMRES,
    AbstractLinearOperator,
    BiCGStab,
    conj,
    has_unit_diagonal,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_tridiagonal,
    is_upper_triangular,
    linearise,
)
from lineax._misc import inexact_asarray, strip_weak_dtype
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver
from lineax._solver.misc import preconditioner_and_y0

from splineax._profile import compute_scope, record_operation, suppress_records
from splineax.solvers._sparse import (
    SparseLinearSolver,
    _Sparsity,
    operator_pattern_tag,
    sparsity_reuse_block,
    state_operator,
)
from splineax.solvers._stateful import (
    TrackingSolverState,
    conditional_update,
    select_pytree,
)

_StateT = TypeVar("_StateT")

_CONVERGENCE_FLOOR_ULPS = 100.0
"""How close to machine precision the residual is allowed to demand.

A solve in a fixed working precision cannot push the relative residual below a small
multiple of that precision's machine epsilon, since the residual itself is formed with
rounding error of that size. The convergence threshold is floored at this many ulps times
the working epsilon, so a healthy solve at a tolerance tighter than the precision can
reach still reports success. An ill-conditioned solve that cannot reach the floored
tolerance still fails."""


def _tree_norm(tree: PyTree[Array]) -> Float[Array, ""]:
    """Euclidean norm over all leaves of a pytree, treating them as one flat vector."""
    squared = sum(jnp.sum(jnp.abs(leaf) ** 2) for leaf in jtu.tree_leaves(tree))
    return jnp.sqrt(squared)


def _tree_add(left: PyTree[Array], right: PyTree[Array]) -> PyTree[Array]:
    return jtu.tree_map(lambda first, second: first + second, left, right)


def _tree_sub(left: PyTree[Array], right: PyTree[Array]) -> PyTree[Array]:
    return jtu.tree_map(lambda first, second: first - second, left, right)


def _residual_norm(
    operator: AbstractLinearOperator, solution: PyTree[Array], vector: PyTree[Array]
) -> Float[Array, ""]:
    """The norm of `vector - operator @ solution`."""
    return _tree_norm(_tree_sub(vector, operator.mv(solution)))


def _convergence_threshold(
    operator: AbstractLinearOperator, vector: PyTree[Array], tolerance: float
) -> Float[Array, ""]:
    """The residual norm a solution must reach, `tolerance * ||vector||` with a floor.

    The floor is `_CONVERGENCE_FLOOR_ULPS` times the machine epsilon of the precision
    that the residual is computed in.
    """
    dtypes = [leaf.dtype for leaf in jtu.tree_leaves(operator.out_structure())]
    dtypes += [leaf.dtype for leaf in jtu.tree_leaves(vector)]
    floor = _CONVERGENCE_FLOOR_ULPS * jnp.finfo(jnp.result_type(*dtypes)).eps
    return jnp.maximum(tolerance, floor) * _tree_norm(vector)


class _RichardsonState(eqx.Module):
    """The state of a `Richardson` solve, which is the operator being solved."""

    operator: AbstractLinearOperator
    """The operator that residuals are formed with."""


class Richardson(AbstractLinearSolver[_RichardsonState]):
    """Preconditioned Richardson iteration, which is iterative refinement.

    Starting from `y0`, each step adds the preconditioner applied to the residual,
    `y = y + M (b - A y)`. With the inverse of a factorization as the preconditioner,
    every step is one back-substitution and one matrix-vector product.

    It takes the same `options` as the Krylov solvers in `lineax`. A `preconditioner`
    operator defaults to the identity, and `y0` defaults to zeros. The iteration stops
    when `||b - A y|| <= rtol * ||b|| + atol`, with the relative part floored at
    machine precision (see `_CONVERGENCE_FLOOR_ULPS`), or after `max_steps` steps.
    """

    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(default=0.0, static=True)
    max_steps: int = eqx.field(default=10, static=True)

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _RichardsonState:
        del options
        return _RichardsonState(state_operator(operator))

    def compute(
        self,
        state: _RichardsonState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        operator = state.operator
        preconditioner, initial_guess = preconditioner_and_y0(operator, vector, options)
        initial_residual = _tree_sub(vector, operator.mv(initial_guess))
        threshold = _convergence_threshold(operator, vector, self.rtol) + self.atol

        # A start that already meets the tolerance runs no step, and records nothing so
        # that the profile looks as if no refinement was used. The gate is a dynamic
        # value instead of a branch, since a record inside a cond breaks `vmap`.
        started = (_tree_norm(initial_residual) > threshold) & (self.max_steps > 0)
        record_operation(
            "refine_start",
            "Richardson",
            dynamic={
                "residual_norm": _tree_norm(initial_residual),
                "threshold": threshold,
            },
            condition=started,
        )

        def keep_going(
            carry: tuple[PyTree[Array], PyTree[Array], Int[Array, ""]],
        ) -> Bool[Array, ""]:
            _, residual, step = carry
            return (step < self.max_steps) & (_tree_norm(residual) > threshold)

        def refine(
            carry: tuple[PyTree[Array], PyTree[Array], Int[Array, ""]],
        ) -> tuple[PyTree[Array], PyTree[Array], Int[Array, ""]]:
            solution, residual, step = carry
            solution = _tree_add(solution, preconditioner.mv(residual))
            residual = _tree_sub(vector, operator.mv(solution))
            record_operation(
                "refine_step",
                "Richardson",
                dynamic={"step": step + 1, "residual_norm": _tree_norm(residual)},
            )
            return solution, residual, step + 1

        solution, residual, steps = jax.lax.while_loop(
            keep_going, refine, (initial_guess, initial_residual, jnp.array(0))
        )
        converged = _tree_norm(residual) <= threshold
        record_operation(
            "refine_result",
            "Richardson",
            dynamic={
                "step": steps,
                "residual_norm": _tree_norm(residual),
                "converged": converged,
            },
            condition=started,
        )
        result = RESULTS.where(converged, RESULTS.successful, RESULTS.max_steps_reached)
        stats = {"num_steps": steps, "max_steps": self.max_steps}
        return solution, result, stats

    def transpose(
        self, state: _RichardsonState, options: dict[str, Any]
    ) -> tuple[_RichardsonState, dict[str, Any]]:
        transposed_options = {}
        if "preconditioner" in options:
            transposed_options["preconditioner"] = options["preconditioner"].transpose()
        return _RichardsonState(state.operator.transpose()), transposed_options

    def conj(
        self, state: _RichardsonState, options: dict[str, Any]
    ) -> tuple[_RichardsonState, dict[str, Any]]:
        conjugated_options = {}
        if "preconditioner" in options:
            conjugated_options["preconditioner"] = conj(options["preconditioner"])
        return _RichardsonState(conj(state.operator)), conjugated_options

    def assume_full_rank(self) -> bool:
        return True


Richardson.__init__.__doc__ = """**Arguments:**

- `rtol`: the target relative residual, `||b - A y|| <= rtol * ||b||`.
- `atol`: an absolute residual added to the target. Defaults to `0`.
- `max_steps`: the maximum number of steps. Defaults to `10`.
"""


@runtime_checkable
class IterativeOptions(Protocol):
    """The fields and builder that every iterative solver option type has."""

    max_steps: int
    """The step cap when the factorization was made for the current operator."""
    max_steps_stale: int
    """The step cap when the factorization was made for an earlier operator."""
    max_restarts: int
    """How many more passes to run from the current solution when a pass ends above the
    tolerance of the true residual. Zero means one pass."""

    def build(
        self, relative_tolerance: float, max_steps: int
    ) -> AbstractLinearSolver[Any]:
        """Build the solver, with a relative tolerance and a step cap."""
        ...


class RichardsonOptions(eqx.Module):
    """Options for iterative refinement with the factorization as the preconditioner.

    Each step costs one back-substitution and one matrix-vector product. It converges
    when the preconditioned operator is close enough to the identity, so a stale
    factorization needs few steps only when the operator has changed a little.
    """

    max_steps: int = eqx.field(default=10, static=True)
    """The step cap when the factorization was made for the current operator."""
    max_steps_stale: int = eqx.field(default=3, static=True)
    """The step cap when the factorization was made for an earlier operator."""
    max_restarts: int = eqx.field(default=0, static=True)
    """Passes after the first. Richardson checks the true residual itself, so zero."""

    def build(
        self, relative_tolerance: float, max_steps: int
    ) -> AbstractLinearSolver[Any]:
        return Richardson(rtol=relative_tolerance, max_steps=max_steps)


class GMRESOptions(eqx.Module):
    """Options for GMRES with the factorization as the preconditioner.

    GMRES finds the best correction in a Krylov space, so it can converge when the
    factorization is too stale for Richardson iteration.
    """

    restart: int = eqx.field(default=20, static=True)
    """The size of the Krylov space built before a restart."""
    stagnation_iters: int = eqx.field(default=20, static=True)
    """How many restarts without a decrease of the residual end the solve."""
    max_steps: int = eqx.field(default=100, static=True)
    """The step cap when the factorization was made for the current operator."""
    max_steps_stale: int = eqx.field(default=10, static=True)
    """The step cap when the factorization was made for an earlier operator."""
    max_restarts: int = eqx.field(default=2, static=True)
    """Passes after the first. GMRES stops on a preconditioned residual, so the true
    residual can end above the target and a pass from the current solution fixes it."""

    def build(
        self, relative_tolerance: float, max_steps: int
    ) -> AbstractLinearSolver[Any]:
        return GMRES(
            rtol=relative_tolerance,
            atol=0.0,
            max_steps=max_steps,
            restart=self.restart,
            stagnation_iters=self.stagnation_iters,
        )


class CGOptions(eqx.Module):
    """Options for conjugate gradients with the factorization as the preconditioner.

    The operator must be symmetric and positive definite, and carry
    `lineax.positive_semidefinite_tag`. The factorization of an earlier matrix of that
    kind is a positive definite preconditioner. Each step needs one matrix-vector product
    and one solve, with short recurrences and no growing Krylov space.
    """

    max_steps: int = eqx.field(default=100, static=True)
    """The step cap when the factorization was made for the current operator."""
    max_steps_stale: int = eqx.field(default=10, static=True)
    """The step cap when the factorization was made for an earlier operator."""
    max_restarts: int = eqx.field(default=2, static=True)
    """Passes after the first. CG stops on a preconditioned residual, so the true
    residual can end above the target and a pass from the current solution fixes it."""

    def build(
        self, relative_tolerance: float, max_steps: int
    ) -> AbstractLinearSolver[Any]:
        return CG(rtol=relative_tolerance, atol=0.0, max_steps=max_steps)


class BiCGStabOptions(eqx.Module):
    """Options for BiCGStab with the factorization as the preconditioner.

    BiCGStab handles a nonsymmetric operator with short recurrences. Each step needs two
    matrix-vector products and two solves.
    """

    max_steps: int = eqx.field(default=100, static=True)
    """The step cap when the factorization was made for the current operator."""
    max_steps_stale: int = eqx.field(default=10, static=True)
    """The step cap when the factorization was made for an earlier operator."""
    max_restarts: int = eqx.field(default=2, static=True)
    """Passes after the first. BiCGStab stops on a preconditioned residual, so the true
    residual can end above the target and a pass from the current solution fixes it."""

    def build(
        self, relative_tolerance: float, max_steps: int
    ) -> AbstractLinearSolver[Any]:
        return BiCGStab(rtol=relative_tolerance, atol=0.0, max_steps=max_steps)


SupportedIterativeOptions = (
    RichardsonOptions | GMRESOptions | CGOptions | BiCGStabOptions
)
"""The iterative solver options that `HybridDirectIterative` accepts."""


class ReuseOptions(eqx.Module):
    """Options for when `HybridDirectIterative` keeps a factorization for a new operator."""

    max_reuses: int | None = eqx.field(default=None, static=True)
    """How many solves in a row may use the same factorization before it is made again.
    `None` sets no limit."""
    slow_fraction: float | None = eqx.field(default=0.5, static=True)
    """The fraction of `max_steps_stale` that a solve with an old factorization may use
    before the next solve makes a new one. `None` never makes one for this reason."""


class _FactorizationInverse(AbstractLinearOperator):
    """The inverse of the matrix in a direct solver state, as a linear operator.

    Applying it is a solve against the stored factorization. It is the preconditioner
    of the iterative solver.
    """

    solver: SparseLinearSolver[Any] = eqx.field(static=True)
    """The direct solver. Its state type depends on the solver, so it is not narrowed."""
    state: PyTree[Array]
    """The state of the direct solver."""
    options: dict[str, Any] = eqx.field(static=True)
    """The options that the direct solver receives on each solve."""
    structure: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    """The structure of the vectors, which is the same on both sides for a square matrix."""
    positive_semidefinite: bool = eqx.field(default=False, static=True)
    """Whether the inverse is symmetric and positive semidefinite, which holds when the
    matrix is. A conjugate gradient solver needs this of its preconditioner."""

    def mv(self, vector: PyTree[Array]) -> PyTree[Array]:
        solution, _, _ = self.solver.compute(self.state, vector, self.options)
        # A direct solver may work in a higher precision than the vector.
        return jtu.tree_map(
            lambda leaf, reference: leaf.astype(reference.dtype),
            solution,
            self.structure,
        )

    def as_matrix(self) -> Array:
        raise NotImplementedError(
            "The inverse of a factorization is only available as a matrix-vector product."
        )

    def transpose(self) -> "_FactorizationInverse":
        state, options = self.solver.transpose(self.state, self.options)
        return _FactorizationInverse(
            self.solver, state, options, self.structure, self.positive_semidefinite
        )

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        return self.structure


@linearise.register(_FactorizationInverse)
def _linearise_factorization_inverse(
    operator: _FactorizationInverse,
) -> _FactorizationInverse:
    """The operator is applied through a solve, which is already linear and not traced."""
    return operator


@is_symmetric.register(_FactorizationInverse)
@is_positive_semidefinite.register(_FactorizationInverse)
def _is_symmetric_factorization_inverse(operator: _FactorizationInverse) -> bool:
    return operator.positive_semidefinite


@is_negative_semidefinite.register(_FactorizationInverse)
@is_diagonal.register(_FactorizationInverse)
@is_tridiagonal.register(_FactorizationInverse)
@is_lower_triangular.register(_FactorizationInverse)
@is_upper_triangular.register(_FactorizationInverse)
@has_unit_diagonal.register(_FactorizationInverse)
def _has_no_tag_factorization_inverse(operator: _FactorizationInverse) -> bool:
    return False


class HybridState(eqx.Module, Generic[_StateT]):
    """The state of a `HybridDirectIterative` solve.

    It wraps the state of the direct solver, together with the operator being solved and
    the bookkeeping for the reuse of a factorization. A state straight from
    `init_symbolic` has no operator and is not solvable until `update` gives it one.
    """

    inner_state: _StateT
    """The state of the direct solver. Its factorization may be stale."""
    operator: AbstractLinearOperator | None
    """The operator that this state represents. `None` for a symbolic-only state."""
    stale: Bool[Array, ""]
    """Whether the factorization was made for an earlier operator."""
    reuses: Int[Array, ""]
    """How many solves in a row have used the current factorization while it was stale."""
    refactor_next: Bool[Array, ""]
    """Whether the next solve makes a new factorization before solving."""
    transposed: bool = eqx.field(default=False, static=True)
    """Whether this state solves the transposed system. A transposed state always has a
    current factorization, since `transpose` makes a new one where it was stale."""

    def track(self, solution: PyTree[Array]) -> Self:
        """Order a later `release` after `solution`, through the inner state.

        This does nothing for an inner state that owns no memory.
        """
        inner = self.inner_state
        tracked = (
            inner.track(solution) if isinstance(inner, TrackingSolverState) else inner
        )
        return dataclasses.replace(self, inner_state=tracked)

    def release(self) -> None:
        """Release the inner state, which owns any memory that this state holds."""
        inner = self.inner_state
        if isinstance(inner, TrackingSolverState):
            inner.release()


def _initial_state(
    inner_state: _StateT, operator: AbstractLinearOperator | None
) -> HybridState[_StateT]:
    """A state whose factorization matches `operator`."""
    return HybridState(
        inner_state,
        None if operator is None else state_operator(operator),
        jnp.array(False),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(False),
    )


def _check_vector_structure(
    operator: AbstractLinearOperator, vector: PyTree[Array]
) -> None:
    """Raise a `ValueError` unless `vector` has the structure that `operator` returns.

    This is the check that `lineax.linear_solve` makes before it calls a solver.
    """
    vector_structure = strip_weak_dtype(jax.eval_shape(lambda: vector))
    operator_structure = strip_weak_dtype(operator.out_structure())
    # The comparison is `is True` to handle the possibility of a tracer.
    if eqx.tree_equal(vector_structure, operator_structure) is not True:
        raise ValueError(
            "Vector and operator structures do not match. Got a vector with structure "
            f"{vector_structure} and an operator with out-structure {operator_structure}."
        )


def _nan_unless_accepted(
    solution: PyTree[Array], accepted: Bool[Array, ""], vector: PyTree[Array]
) -> tuple[PyTree[Array], RESULTS]:
    """Return the solution and its result, with NaN in place of a solution that failed.

    The caller sees the failure in the result code and in the values, so a solution that
    never met the tolerance cannot be mistaken for a good one. A vector with a
    non-finite entry gets the result `nonfinite_input`, as in `lineax.linear_solve`.
    """
    finite_input = jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in jtu.tree_leaves(vector)])
    )
    failure = RESULTS.where(
        finite_input, RESULTS.max_steps_reached, RESULTS.nonfinite_input
    )
    result = RESULTS.where(accepted, RESULTS.successful, failure)
    checked = jtu.tree_map(lambda leaf: jnp.where(accepted, leaf, jnp.nan), solution)
    return checked, result


def _skip_unless(
    predicate: Bool[Array, ""],
    run: Callable[[], tuple[PyTree[Array], Int[Array, ""]]],
) -> tuple[PyTree[Array], Int[Array, ""]]:
    """Run `run` only where `predicate` is true, and return zeros elsewhere.

    The branch that does not run returns zeros of the same structure. Records inside
    `run` are dropped, since a record in a cond breaks `vmap`.
    """

    def run_silently() -> tuple[PyTree[Array], Int[Array, ""]]:
        with suppress_records():
            return run()

    shapes = jax.eval_shape(run_silently)
    zeros = jtu.tree_map(lambda shape: jnp.zeros(shape.shape, shape.dtype), shapes)
    return jax.lax.cond(predicate, run_silently, lambda: zeros)


class HybridDirectIterative(AbstractLinearSolver[HybridState[Any]]):
    """A direct solver and an iterative solver that share one factorization.

    The direct solver factors the matrix, and the iterative solver uses that
    factorization as its preconditioner. When a solve gets a new operator and `reuse_direct`
    is on, the factorization of the previous operator is tried first. The iterative solve
    runs with a low step cap (`max_steps_stale`). If its true residual reaches the
    tolerance, the old factorization stays. Otherwise the factorization is made again for
    the new operator, and the iterative solve runs with the high step cap (`max_steps`).

    The factorization is only reused when the operators share a sparsity pattern tag (see
    `operator_pattern_tag`), and only through `update_and_compute`, which `linear_solve`
    uses. Plain `update` always makes a new factorization. Reuse covers solves that are
    not differentiated. A derivative solve updates the factorization first.

    A solution that never reaches the tolerance is returned as NaN with the result
    `max_steps_reached`. Reuse never causes this, because the check after a new
    factorization is the same as without reuse.

    The wrapped solver must be square and nonsingular.
    """

    direct: SparseLinearSolver[Any]
    """The direct solver. Its state type depends on the solver, so it is not narrowed."""
    iterative: SupportedIterativeOptions = GMRESOptions()
    """The iterative solver, with the options that are specific to it."""
    reuse_direct: bool | ReuseOptions = True
    """Whether to try the factorization of the previous operator first. `False` makes a
    new factorization for each new operator."""
    tol: float = eqx.field(default=1e-10, static=True)
    """The target relative residual, `||b - A x|| <= tol * ||b||`."""

    @cached_property
    def _stale_solver(self) -> AbstractLinearSolver[Any]:
        """The iterative solver for a factorization of an earlier operator."""
        return self.iterative.build(self.tol, self.iterative.max_steps_stale)

    @cached_property
    def _fresh_solver(self) -> AbstractLinearSolver[Any]:
        """The iterative solver for a factorization of the current operator."""
        return self.iterative.build(self.tol, self.iterative.max_steps)

    @cached_property
    def _reuse_options(self) -> ReuseOptions | None:
        """The options for reuse, or `None` when every new operator is factored."""
        match self.reuse_direct:
            case True:
                return ReuseOptions()
            case False:
                return None
            case ReuseOptions() as reuse_options:
                return reuse_options
            case _:
                raise TypeError("`reuse_direct` must be a boolean or `ReuseOptions`.")

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> HybridState[Any]:
        return _initial_state(self.direct.init(operator, options), operator)

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> HybridState[Any]:
        """Analyze a sparsity pattern through the direct solver.

        The state has no operator, so `update` must give it one before a solve. This raises
        `AttributeError` if the direct solver has no symbolic phase.
        """
        return _initial_state(self.direct.init_symbolic(sparsity, options), None)

    def update(
        self,
        state: HybridState[Any],
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> HybridState[Any]:
        """Fold a new operator into `state` through the direct solver.

        This makes a new factorization, so the returned state is not stale. It returns the
        same state object when the direct solver did, so an unchanged operator is a no-op.
        """
        inner_state = self.direct.update(state.inner_state, operator, options)
        if inner_state is state.inner_state:
            return state
        return _initial_state(inner_state, operator)

    def update_and_compute(
        self,
        state: HybridState[Any],
        operator: AbstractLinearOperator,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any], HybridState[Any]]:
        """Fold `operator` into `state` and solve, keeping the factorization if it suffices.

        When reuse applies, the factorization stays and the iterative solve decides whether
        it is good enough. Otherwise this is `update` followed by `compute`.
        """
        previous = state.operator
        # `update` makes a new factorization before the solve, which `_solve` does not see.
        factored_by_update = False
        if operator is previous:
            updated_state = state
        elif (
            self._reuse_options is None
            or previous is None
            or sparsity_reuse_block(
                operator_pattern_tag(previous), operator_pattern_tag(operator)
            )
            is not None
        ):
            updated_state = self.update(state, operator, options)
            factored_by_update = True
        else:
            # The factorization stays as it is until the solve shows that it is too far
            # off, see `_solve`.
            updated_state = dataclasses.replace(
                state, operator=state_operator(operator), stale=jnp.array(True)
            )
        with compute_scope():
            solution, result, stats, new_state = self._solve(
                updated_state, vector, options
            )
        if factored_by_update:
            stats = {**stats, "refactored": jnp.array(True)}
        return solution, result, stats, new_state

    def compute(
        self,
        state: HybridState[Any],
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        """Solve against `state`.

        A stale factorization that fails to reach the tolerance is made again for this
        solve only. Use `update_and_compute` to keep the new factorization.
        """
        with compute_scope():
            solution, result, stats, _ = self._solve(state, vector, options)
        return solution, result, stats

    def _iterate(
        self,
        solver: AbstractLinearSolver[Any],
        operator: AbstractLinearOperator,
        vector: PyTree[Array],
        inner_state: PyTree[Array],
        options: dict[str, Any],
        threshold: Float[Array, ""],
        max_restarts: int,
    ) -> tuple[PyTree[Array], Int[Array, ""]]:
        """Solve with the iterative solver, preconditioned by the factorization.

        The start is one direct solve. Each pass of the iterative solver then starts from
        the current solution, and runs while the true residual is above `threshold`. The
        loop allows one pass and up to `max_restarts` more. A start that is already
        within the threshold runs no pass, which also keeps a Krylov method away from a
        residual of exactly zero. Returns the solution and the number of steps taken.
        """
        inverse = _FactorizationInverse(
            self.direct,
            inner_state,
            options,
            operator.in_structure(),
            isinstance(self.iterative, CGOptions),
        )

        def above_target(solution: PyTree[Array]) -> Bool[Array, ""]:
            return ~(_residual_norm(operator, solution, vector) <= threshold)

        def needs_pass(
            carry: tuple[PyTree[Array], Int[Array, ""], Int[Array, ""]],
        ) -> Bool[Array, ""]:
            solution, _, passes = carry
            return (passes < 1 + max_restarts) & above_target(solution)

        def run_pass(
            carry: tuple[PyTree[Array], Int[Array, ""], Int[Array, ""]],
        ) -> tuple[PyTree[Array], Int[Array, ""], Int[Array, ""]]:
            solution, steps, passes = carry
            iterative_state = solver.init(operator, {"preconditioner": inverse})
            new_solution, _, stats = solver.compute(
                iterative_state,
                vector,
                {"preconditioner": inverse, "y0": solution},
            )
            new_steps = jnp.asarray(stats["num_steps"], dtype=jnp.int32)
            return new_solution, steps + new_steps, passes + 1

        solution, steps, _ = jax.lax.while_loop(
            needs_pass,
            run_pass,
            (inverse.mv(vector), jnp.array(0, dtype=jnp.int32), jnp.array(0)),
        )
        return solution, steps

    def _solve(
        self,
        state: HybridState[Any],
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any], HybridState[Any]]:
        """Solve against `state` and return the solution, result, statistics and new state.

        With reuse off, this is one iterative solve against the current factorization.
        With reuse on, it first tries the stale factorization with the low step cap, and
        then makes a new factorization wherever that attempt failed.
        """
        operator = state.operator
        if operator is None:
            raise ValueError(
                "`HybridDirectIterative` cannot solve with a symbolic-only state. Call "
                "`update` with an operator first."
            )
        # The solvers in `lineax` expect JAX arrays, as `lineax.linear_solve` converts them.
        vector = jtu.tree_map(inexact_asarray, vector)
        _check_vector_structure(operator, vector)
        threshold = _convergence_threshold(operator, vector, self.tol)
        reuse_options = self._reuse_options

        if reuse_options is None or state.transposed:
            solution, steps = self._iterate(
                self._fresh_solver,
                operator,
                vector,
                state.inner_state,
                options,
                threshold,
                self.iterative.max_restarts,
            )
            accepted = _residual_norm(operator, solution, vector) <= threshold
            stats = {
                "num_steps": steps,
                "refactored": jnp.array(False),
                "reused": jnp.array(False),
            }
            checked, result = _nan_unless_accepted(solution, accepted, vector)
            return checked, result, stats, state

        # The attempt is skipped when a new factorization is already due.
        attempt = state.stale & ~state.refactor_next
        if reuse_options.max_reuses is not None:
            attempt = attempt & (state.reuses < reuse_options.max_reuses)

        solution_stale, steps_stale = _skip_unless(
            attempt,
            lambda: self._iterate(
                self._stale_solver,
                operator,
                vector,
                state.inner_state,
                options,
                threshold,
                0,
            ),
        )
        accepted_stale = attempt & (
            _residual_norm(operator, solution_stale, vector) <= threshold
        )
        record_operation(
            "reuse_attempt",
            "HybridDirectIterative",
            dynamic={
                "steps": steps_stale,
                "residual_norm": _residual_norm(operator, solution_stale, vector),
                "accepted": accepted_stale,
            },
            condition=attempt,
        )

        refactor = state.stale & ~accepted_stale
        inner_state = conditional_update(
            self.direct.update, state.inner_state, operator, options, refactor
        )
        solution_fresh, steps_fresh = _skip_unless(
            ~accepted_stale,
            lambda: self._iterate(
                self._fresh_solver,
                operator,
                vector,
                inner_state,
                options,
                threshold,
                self.iterative.max_restarts,
            ),
        )
        accepted_fresh = ~accepted_stale & (
            _residual_norm(operator, solution_fresh, vector) <= threshold
        )
        record_operation(
            "iterative_result",
            "HybridDirectIterative",
            dynamic={
                "steps": steps_fresh,
                "residual_norm": _residual_norm(operator, solution_fresh, vector),
                "accepted": accepted_fresh,
            },
            condition=~accepted_stale,
        )

        slow_steps = (
            jnp.inf
            if reuse_options.slow_fraction is None
            else reuse_options.slow_fraction * self.iterative.max_steps_stale
        )
        new_state = dataclasses.replace(
            state,
            inner_state=inner_state,
            stale=state.stale & ~refactor,
            reuses=jnp.where(
                refactor, 0, jnp.where(state.stale, state.reuses + 1, state.reuses)
            ),
            refactor_next=accepted_stale & (steps_stale > slow_steps),
        )
        solution = select_pytree(accepted_stale, solution_stale, solution_fresh)
        stats = {
            "num_steps": steps_stale + steps_fresh,
            "refactored": refactor,
            "reused": accepted_stale,
        }
        checked, result = _nan_unless_accepted(
            solution, accepted_stale | accepted_fresh, vector
        )
        return checked, result, stats, new_state

    def transpose(
        self, state: HybridState[Any], options: dict[str, Any]
    ) -> tuple[HybridState[Any], dict[str, Any]]:
        inner_state = state.inner_state
        operator = state.operator
        if operator is not None:
            # Transposing a state fixes its factorization to one orientation, so a stale
            # factorization is made again first, where it is stale.
            inner_state = conditional_update(
                self.direct.update, inner_state, operator, options, state.stale
            )
            operator = operator.transpose()
        inner_state, transposed_options = self.direct.transpose(inner_state, options)
        return (
            HybridState(
                inner_state,
                operator,
                jnp.array(False),
                jnp.array(0, dtype=jnp.int32),
                jnp.array(False),
                not state.transposed,
            ),
            transposed_options,
        )

    def conj(
        self, state: HybridState[Any], options: dict[str, Any]
    ) -> tuple[HybridState[Any], dict[str, Any]]:
        inner_state, conjugated_options = self.direct.conj(state.inner_state, options)
        operator = None if state.operator is None else conj(state.operator)
        return (
            dataclasses.replace(state, inner_state=inner_state, operator=operator),
            conjugated_options,
        )

    def assume_full_rank(self) -> bool:
        return self.direct.assume_full_rank()


HybridDirectIterative.__init__.__doc__ = """**Arguments:**

- `direct`: the stateful direct solver. It supplies the factorization and must be square
    and nonsingular.
- `iterative`: the iterative solver and its options (see `IterativeOptions`). Defaults to
    `GMRESOptions()`.
- `reuse_direct`: whether to try the factorization of the previous operator before making
    a new one. `True` uses the default `ReuseOptions`, `False` always makes a new
    factorization, and a `ReuseOptions` sets the limits. Defaults to `True`.
- `tol`: the target relative residual, `||b - A x|| <= tol * ||b||`. The threshold is
    floored at machine precision, so a tolerance tighter than the working precision can
    reach still succeeds. Defaults to `1e-10`.
"""


class IterativeRefinementSettings(eqx.Module):
    """The `tol` and `max_steps` of an iterative refinement, without a solver bound yet.

    A solver that offers refinement as an option, such as `AutoSparseLinearSolver`, takes
    one of these instead of separate tolerance and step-cap arguments. See
    `IterativeRefinement` for what each does.
    """

    tol: float = eqx.field(default=1e-10, static=True)
    """Target relative residual, `||b - A x|| <= tol * ||b||`."""
    max_steps: int = eqx.field(default=10, static=True)
    """Maximum correction steps before returning NaN."""


class HybridSettings(eqx.Module):
    """The options of a `HybridDirectIterative`, without a direct solver bound yet."""

    iterative: SupportedIterativeOptions = GMRESOptions()
    """The iterative solver and its options."""
    reuse_direct: bool | ReuseOptions = True
    """Whether to try the factorization of the previous operator first."""
    tol: float = eqx.field(default=1e-10, static=True)
    """Target relative residual, `||b - A x|| <= tol * ||b||`."""


def IterativeRefinement(
    solver: SparseLinearSolver[Any], tol: float = 1e-10, max_steps: int = 10
) -> HybridDirectIterative:
    """Wrap a stateful solver so that each solve is refined with iterative refinement.

    The wrapped `solver` supplies the factorization. Each solve starts from one direct
    solve and then corrects it with the residual until `||b - A x|| <= tol * ||b||`, or
    returns NaN after `max_steps` corrections. Every new operator is factored, so this is
    a `HybridDirectIterative` with `RichardsonOptions` and `reuse_direct=False`.

    **Arguments:**

    - `solver`: the stateful solver to wrap. It must be square and nonsingular.
    - `tol`: the target relative residual. Defaults to `1e-10`.
    - `max_steps`: the maximum number of correction steps. Defaults to `10`.
    """
    return HybridDirectIterative(
        solver,
        RichardsonOptions(max_steps=max_steps),
        reuse_direct=False,
        tol=tol,
    )


def _hybrid_from_settings(
    solver: SparseLinearSolver[Any],
    settings: HybridSettings | IterativeRefinementSettings,
) -> HybridDirectIterative:
    """Bind a direct solver to settings."""
    match settings:
        case HybridSettings(iterative, reuse_direct, tol):
            return HybridDirectIterative(solver, iterative, reuse_direct, tol)
        case IterativeRefinementSettings(tol, max_steps):
            return IterativeRefinement(solver, tol, max_steps)
