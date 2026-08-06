from contextlib import AbstractContextManager, contextmanager
from typing import Any, Iterator, Literal, NamedTuple, overload

import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from lineax import CG, GMRES, AbstractLinearOperator, conj, materialise
from lineax import linear_solve as _lx_linear_solve
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver
from lineax._solver.misc import (
    PackedStructures,
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)
from lineax._tags import transpose_tags

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import SparseJacobianLinearOperator
from splineax.operators._pattern import MatrixSparsity, as_matrix_sparsity
from splineax.solvers._sparse import (
    AbstractSparseLinearSolver,
    SparseNumericState,
    SymbolicScopedSparseLinearSolver,
    _Sparsity,
    analyze_numeric_through_init,
    as_scoped_solver,
)
from splineax.transforms._block_jacobi import BlockJacobi
from splineax.transforms._clustering import AggregationClustering
from splineax.transforms._compose import compose_transforms
from splineax.transforms._equilibration import RuizEquilibration
from splineax.transforms._protocols import (
    AnalyzedPreconditioner,
    AnalyzedTransform,
    AppliedTransform,
    Preconditioner,
    SystemTransform,
)


def _as_bcoo_operator(
    operator: AbstractLinearOperator, *, context: str
) -> BCOOLinearOperator:
    match operator:
        case SparseJacobianLinearOperator():
            # Materialise the Jacobian into a `BCOOLinearOperator` and reuse the BCOO
            # path below.
            return _as_bcoo_operator(materialise(operator), context=context)
        case BCSRLinearOperator(matrix):
            return BCOOLinearOperator(matrix.to_bcoo(), operator.tags)
        case BCOOLinearOperator():
            return operator
        case _:
            raise TypeError(
                f"`{context}` requires a sparse operator backed by a `BCOO` or `BCSR` "
                "matrix (e.g. `splineax.BCOOLinearOperator` or "
                "`splineax.BCSRLinearOperator`), or a "
                f"`splineax.SparseJacobianLinearOperator`; "
                f"got {type(operator).__name__}."
            )


def _pattern_of(matrix: BCOO) -> MatrixSparsity:
    return MatrixSparsity(
        matrix.indices[:, 0].astype(jnp.int32),
        matrix.indices[:, 1].astype(jnp.int32),
        (matrix.shape[0], matrix.shape[1]),
    )


class _IterativeAnalyzedState(NamedTuple):
    """Bound to one matrix's pattern (via `transform_plan`/`preconditioner_plan`), not
    yet numerically analyzed. Satisfies both `SparseBasicState` and
    `SparseSymbolicState`: the two protocols share the same shape (just
    `.analyze_numeric()`), so one type plays both roles, built either straight from
    `init` or from a `PreconditionedIterativeSolver.analyze_symbolic` scope.
    """

    matrix: BCOO
    tags: frozenset[object]
    packed_structures: PackedStructures
    transform_plan: AnalyzedTransform
    preconditioner_plan: AnalyzedPreconditioner

    @contextmanager
    def analyze_numeric(self) -> Iterator["_IterativeNumericState"]:
        transformed_matrix, applied = self.transform_plan.analyze_numeric(self.matrix)
        # Tags only survive a congruence: anything else can break symmetry or
        # definiteness, and a wrong tag is worse than none (see
        # `AnalyzedTransform.is_congruence`).
        transformed_tags = (
            self.tags if self.transform_plan.is_congruence else frozenset()
        )
        with self.preconditioner_plan.analyze_numeric(
            transformed_matrix, transformed_tags
        ) as preconditioner:
            operator = BCOOLinearOperator(transformed_matrix, transformed_tags)
            yield _IterativeNumericState(
                operator, applied, preconditioner, self.packed_structures
            )


class _IterativeNumericState(NamedTuple):
    operator: BCOOLinearOperator
    applied: AppliedTransform
    preconditioner: AbstractLinearOperator
    packed_structures: PackedStructures


_IterativeState = _IterativeAnalyzedState | _IterativeNumericState


class _IterativeScope(NamedTuple):
    transform_plan: AnalyzedTransform
    preconditioner_plan: AnalyzedPreconditioner

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _IterativeAnalyzedState:
        del options
        bcoo_operator = _as_bcoo_operator(
            operator, context="PreconditionedIterativeSolver.analyze_symbolic"
        )
        return _IterativeAnalyzedState(
            bcoo_operator.matrix,
            bcoo_operator.tags,
            pack_structures(operator),
            self.transform_plan,
            self.preconditioner_plan,
        )

    @contextmanager
    def analyze_numeric(
        self, operator: AbstractLinearOperator
    ) -> Iterator[_IterativeNumericState]:
        with self.init(operator).analyze_numeric() as state:
            yield state


class PreconditionedIterativeSolver(AbstractSparseLinearSolver[_IterativeState]):
    """Preconditioned Krylov solve, staged the same way the direct solvers are.

    `A x = b` is rewritten by `transform` into `A' y = b'` (`A' = L A R`, `b' = L b`,
    `x = R y`), and `preconditioner` builds an operator approximating `A'^-1` that
    `solver` uses to condition its iteration. See the
    [preconditioning guide](../guide/preconditioning.md) for the full picture and
    [`block_jacobi_solver`][splineax.block_jacobi_solver] for a ready-made instance.

    `transform` and `preconditioner` are analyzed from the sparsity pattern alone in
    `analyze_symbolic`, and reused across every matrix sharing that pattern; each
    matrix's own `analyze_numeric` reruns only the numeric step (e.g. Ruiz iterations,
    block inversion).
    """

    transform: SystemTransform
    preconditioner: Preconditioner
    solver: AbstractLinearSolver

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _IterativeAnalyzedState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`PreconditionedIterativeSolver` may only be used for linear solves "
                "with square matrices"
            )
        bcoo_operator = _as_bcoo_operator(
            operator, context="PreconditionedIterativeSolver.init"
        )
        matrix = bcoo_operator.matrix
        transform_plan = self.transform.analyze_symbolic(_pattern_of(matrix))
        preconditioner_plan = self.preconditioner.analyze_symbolic(
            transform_plan.pattern
        )
        return _IterativeAnalyzedState(
            matrix,
            bcoo_operator.tags,
            pack_structures(operator),
            transform_plan,
            preconditioner_plan,
        )

    def analyze_numeric(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        """Pre-compute the transform and preconditioner for reuse across solves.

        Equivalent to `self.init(operator, options).analyze_numeric()`.
        """
        return analyze_numeric_through_init(self, operator, options)

    @overload
    def analyze_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[False] = False
    ) -> AbstractContextManager[_IterativeScope]: ...

    @overload
    def analyze_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[True]
    ) -> AbstractContextManager[SymbolicScopedSparseLinearSolver]: ...

    def analyze_symbolic(
        self, sparsity: _Sparsity, *, as_solver: bool = False
    ) -> AbstractContextManager[_IterativeScope | SymbolicScopedSparseLinearSolver]:
        """Pre-analyze `transform` and `preconditioner` from a known sparsity pattern.

        Yields an `_IterativeScope`. Inside the block, `.init(operator)` builds a
        state reusing this pattern-only analysis for any matrix sharing it; that
        state's own `.analyze_numeric()` (or the scope's `.analyze_numeric(operator)`
        shortcut) runs the numeric step.

        Args:
            sparsity: sparse matrix whose sparsity pattern to pre-analyze. Accepts the
                same types as `KLU.analyze_symbolic`.
            as_solver: yield a `SymbolicScopedSparseLinearSolver` pairing the scope
                       with this solver, instead of the bare scope.
        """
        scope = self._analyze_symbolic(sparsity)
        return as_scoped_solver(self, scope) if as_solver else scope

    @contextmanager
    def _analyze_symbolic(self, sparsity: _Sparsity) -> Iterator[_IterativeScope]:
        pattern = as_matrix_sparsity(
            sparsity, context="PreconditionedIterativeSolver.analyze_symbolic"
        )
        if pattern.shape[0] != pattern.shape[1]:
            raise ValueError(
                "`PreconditionedIterativeSolver.analyze_symbolic` requires a square "
                f"matrix; got shape {pattern.shape}."
            )
        transform_plan = self.transform.analyze_symbolic(pattern)
        preconditioner_plan = self.preconditioner.analyze_symbolic(
            transform_plan.pattern
        )
        yield _IterativeScope(transform_plan, preconditioner_plan)

    def compute(
        self, state: _IterativeState, vector: Any, options: dict[str, Any]
    ) -> tuple[Any, RESULTS, dict[str, Any]]:
        del options
        match state:
            case _IterativeNumericState():
                numeric_state = state
            case _IterativeAnalyzedState():
                # No cheaper solve exists without a preconditioner having been built
                # at all, so this tier refactors on every call, exactly as `KLU`'s
                # symbolic tier does.
                with state.analyze_numeric() as numeric_state:
                    return self.compute(numeric_state, vector, {})

        b = ravel_vector(vector, numeric_state.packed_structures)
        transformed_b = numeric_state.applied.transform_vector(b)
        # `throw=False`: a Krylov solve can genuinely fail to converge, unlike the
        # direct solvers. Returning `solution.result` below, rather than letting this
        # inner solve raise, is what lets the *outer* `lineax.linear_solve` apply its
        # own `throw` setting to that failure, exactly as it would for any other
        # solver's non-`successful` result.
        solution = _lx_linear_solve(
            numeric_state.operator,
            transformed_b,
            solver=self.solver,
            options={"preconditioner": numeric_state.preconditioner},
            throw=False,
        )
        x = numeric_state.applied.recover_solution(solution.value)
        # Forward the inner solve's stats (e.g. `num_steps`) rather than discarding
        # them: they describe the Krylov iteration this solver runs, which is exactly
        # what a caller inspecting `Solution.stats` wants to see.
        return (
            unravel_solution(x, numeric_state.packed_structures),
            solution.result,
            solution.stats,
        )

    def transpose(
        self, state: _IterativeState, options: dict[str, Any]
    ) -> tuple[_IterativeState, dict[str, Any]]:
        del options
        match state:
            case _IterativeNumericState(
                operator=operator,
                applied=applied,
                preconditioner=preconditioner,
                packed_structures=packed_structures,
            ):
                return _IterativeNumericState(
                    operator.transpose(),
                    applied.transpose(),
                    preconditioner.transpose(),
                    transpose_packed_structures(packed_structures),
                ), {}
            case _IterativeAnalyzedState(
                matrix=matrix, tags=tags, packed_structures=packed_structures
            ):
                # Cheap to redo from scratch: `analyze_symbolic` is pure XLA, and
                # there is no native handle whose reuse would be worth preserving.
                transposed_matrix: BCOO = matrix.T
                transposed_tags = transpose_tags(tags)
                transform_plan = self.transform.analyze_symbolic(
                    _pattern_of(transposed_matrix)
                )
                preconditioner_plan = self.preconditioner.analyze_symbolic(
                    transform_plan.pattern
                )
                return _IterativeAnalyzedState(
                    transposed_matrix,
                    transposed_tags,
                    transpose_packed_structures(packed_structures),
                    transform_plan,
                    preconditioner_plan,
                ), {}

    def conj(
        self, state: _IterativeState, options: dict[str, Any]
    ) -> tuple[_IterativeState, dict[str, Any]]:
        del options
        match state:
            case _IterativeNumericState(
                operator=operator,
                applied=applied,
                preconditioner=preconditioner,
                packed_structures=packed_structures,
            ):
                conj_matrix = BCOO(
                    (operator.matrix.data.conj(), operator.matrix.indices),
                    shape=operator.matrix.shape,
                )
                conj_operator = BCOOLinearOperator(conj_matrix, operator.tags)
                return _IterativeNumericState(
                    conj_operator,
                    applied.conj(),
                    conj(preconditioner),
                    packed_structures,
                ), {}
            case _IterativeAnalyzedState(
                matrix=matrix,
                tags=tags,
                packed_structures=packed_structures,
                transform_plan=transform_plan,
                preconditioner_plan=preconditioner_plan,
            ):
                conj_matrix = BCOO(
                    (matrix.data.conj(), matrix.indices), shape=matrix.shape
                )
                return _IterativeAnalyzedState(
                    conj_matrix,
                    tags,
                    packed_structures,
                    transform_plan,
                    preconditioner_plan,
                ), {}

    def assume_full_rank(self) -> bool:
        return True


PreconditionedIterativeSolver.__init__.__doc__ = """**Arguments:**

- `transform`: the system transformation (symbolic and numeric stages composed
    together, e.g. with `compose_transforms`). Rewrites `A x = b` before the solve.
- `preconditioner`: builds the operator `solver` uses to condition its iteration.
- `solver`: the lineax iterative solver performing the actual Krylov iteration (e.g.
    `lineax.GMRES()`).
"""


def block_jacobi_solver(
    block_size: int | None = None,
    solver: AbstractLinearSolver = GMRES(rtol=1e-6, atol=1e-6),
) -> PreconditionedIterativeSolver:
    """Builds a `PreconditionedIterativeSolver` from `AggregationClustering`,
    `RuizEquilibration` and `BlockJacobi` at one consistent block size.

    **Arguments:**

    - `block_size`: shared block size for the clustering and the preconditioner. If
        `None`, both derive one from the pattern independently, but by the same rule,
        so they still agree.
    - `solver`: the lineax iterative solver to condition. Defaults to
        `lineax.GMRES()`. When it is `lineax.CG`, the equilibration is set symmetric,
        since `CG` requires a positive semidefinite preconditioner and only a
        congruence is guaranteed to preserve that tag.
    """
    equilibration = RuizEquilibration(symmetric=isinstance(solver, CG))
    transform = compose_transforms(
        AggregationClustering(block_size=block_size), equilibration
    )
    preconditioner = BlockJacobi(block_size=block_size)
    return PreconditionedIterativeSolver(transform, preconditioner, solver)
