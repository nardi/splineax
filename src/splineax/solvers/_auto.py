from contextlib import AbstractContextManager
from functools import cached_property
from typing import Any

import jax
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator

from ._klu import KLU
from ._sparse import (
    AbstractSparseLinearSolver,
    SparseBasicState,
    SparseNumericState,
    SparseSymbolicScope,
    SparseSymbolicState,
)
from ._spsolve import Spsolve


class AutoSparseLinearSolver(
    AbstractSparseLinearSolver[
        SparseBasicState | SparseSymbolicState | SparseNumericState
    ]
):
    """Selects a sparse direct solver based on the JAX platform.

    Dispatches to `KLU` on CPU (SuiteSparse direct solve with factorization reuse) and
    to `Spsolve` on any other backend. Exposes the same factorization API as `KLU`
    (`factorize`, `factorize_symbolic`), so it can be substituted for `KLU` verbatim; on
    non-CPU backends these degrade to no-ops via `Spsolve`.
    """

    platform: str | None = None
    """Platform to select for. If None, `jax.default_backend()` is used. Set to e.g.
    "cpu", "gpu", or "tpu" to override the choice explicitly."""

    @cached_property
    def _chosen_solver(self) -> KLU | Spsolve:
        platform = self.platform if self.platform is not None else jax.default_backend()
        match platform:
            case "cpu":
                return KLU()
            case _:
                return Spsolve()

    def select_solver(self, operator: AbstractLinearOperator) -> AbstractLinearSolver:
        """Check which solver `AutoSparseLinearSolver` will dispatch to.

        Mirrors `lineax.AutoLinearSolver.select_solver`. The operator is accepted for
        signature parity but selection depends only on the platform.
        """
        del operator
        return self._chosen_solver

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> SparseBasicState:
        return self._chosen_solver.init(operator, options)

    def compute(
        self, state: Any, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        return self._chosen_solver.compute(state, vector, options)

    def transpose(
        self, state: Any, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        return self._chosen_solver.transpose(state, options)

    def conj(self, state: Any, options: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return self._chosen_solver.conj(state, options)

    def assume_full_rank(self) -> bool:
        return self._chosen_solver.assume_full_rank()

    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        return self._chosen_solver.factorize(operator, options)

    def factorize_symbolic(
        self, sparsity: BCOO | BCSR | BCOOLinearOperator | BCSRLinearOperator
    ) -> AbstractContextManager[SparseSymbolicScope]:
        return self._chosen_solver.factorize_symbolic(sparsity)


AutoSparseLinearSolver.__init__.__doc__ = """**Arguments:**

- `platform`: optional platform string ("cpu", "gpu", "tpu") overriding the
    automatically detected `jax.default_backend()`.
"""
