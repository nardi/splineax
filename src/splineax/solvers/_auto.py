from functools import cached_property
from typing import Any

import jax
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver

from ._iterative import IterativeRefinement, IterativeRefinementSettings
from ._klu import KLU, _KLUState
from ._pardiso import Pardiso, _pardiso_available, _PardisoState
from ._sparse import SparseLinearSolver, _Sparsity
from ._spsolve import Spsolve, _SpsolveState

_State = _KLUState | _PardisoState | _SpsolveState


class _AutoDispatch(AbstractLinearSolver[_State]):
    """Selects a sparse direct solver based on the JAX platform, precision, and what is
    installed, and forwards the whole stateful API to it.

    This is the platform dispatch behind `AutoSparseLinearSolver`, split out so the
    refinement layer can wrap it. It carries no iterative refinement of its own.
    """

    platform: str | None = None
    """Platform to select for. If None, `jax.default_backend()` is used."""

    @cached_property
    def _chosen_solver(self) -> Pardiso | KLU | Spsolve:
        platform = self.platform if self.platform is not None else jax.default_backend()
        x64_enabled = jax.config.read("jax_enable_x64")
        # Pardiso and KLU are both double precision only, so either is only a valid choice
        # on CPU with x64 enabled. Pardiso is preferred when its optional dependency is
        # installed. KLU (a hard dependency) is always available as a fallback. Everything
        # else falls back to Spsolve, which works in single or double precision and on any
        # backend.
        if platform == "cpu" and x64_enabled:
            return Pardiso() if _pardiso_available() else KLU()
        return Spsolve()

    def select_solver(self, operator: AbstractLinearOperator) -> AbstractLinearSolver:
        """Check which solver this dispatch will use. Selection depends only on the
        platform, so the operator is accepted for signature parity only."""
        del operator
        return self._chosen_solver

    def _solver_for_state(self, state: Any) -> Pardiso | KLU | Spsolve:
        """The concrete solver that must handle `state`.

        Usually `self._chosen_solver`, except when it is `Pardiso` but `state` is not a
        Pardiso state: that means `init` fell back to `KLU` for a complex operator (see
        `AutoSparseLinearSolver`), and later calls on that same state must keep using
        `KLU`.
        """
        chosen = self._chosen_solver
        if isinstance(chosen, Pardiso) and not isinstance(state, _PardisoState):
            return KLU()
        return chosen

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _State:
        chosen = self._chosen_solver
        if isinstance(chosen, Pardiso):
            try:
                return chosen.init(operator, options)
            except TypeError:
                # `pardiso_mkl_jax` does not support complex matrices. Fall back to `KLU`,
                # which does, rather than surfacing Pardiso's error for a case Auto can
                # handle. `KLU` is a hard dependency, always available here.
                return KLU().init(operator, options)
        return chosen.init(operator, options)

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> _State:
        # No complex fallback is possible here, since a bare pattern carries no values.
        # Pardiso's `init_symbolic` is a no-op that only records the pattern, so this is
        # safe to delegate directly.
        return self._chosen_solver.init_symbolic(sparsity, options)

    def update(
        self,
        state: Any,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _State:
        return self._solver_for_state(state).update(state, operator, options)

    def release(self, state: Any) -> None:
        self._solver_for_state(state).release(state)

    def compute(
        self, state: Any, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        return self._solver_for_state(state).compute(state, vector, options)

    def transpose(
        self, state: Any, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        return self._solver_for_state(state).transpose(state, options)

    def conj(self, state: Any, options: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return self._solver_for_state(state).conj(state, options)

    def assume_full_rank(self) -> bool:
        return self._chosen_solver.assume_full_rank()


class AutoSparseLinearSolver(AbstractLinearSolver[Any]):
    """Selects a sparse direct solver based on the JAX platform, precision, and what is
    installed, and by default refines its solution with iterative refinement.

    On CPU with x64 enabled, dispatches to `Pardiso` (Intel oneMKL Pardiso, factorization
    reuse) if the optional `pardiso-mkl-jax` dependency is installed, otherwise `KLU`
    (SuiteSparse, factorization reuse). Both are double precision only, hence the x64
    requirement. On any other backend, or on CPU when x64 is disabled, it dispatches to
    `Spsolve`, which works in single or double precision and on any backend. It exposes
    the same stateful API as `Pardiso` and `KLU` (`update`, `release`, `init_symbolic`),
    so it can be substituted for either. When it dispatches to `Spsolve`, the reuse calls
    degrade to no-ops.

    `pardiso_mkl_jax` does not support complex matrices (see `Pardiso`'s docstring), so
    `init` falls back to `KLU` for a complex operator even when `Pardiso` was otherwise
    selected, keeping `Auto` able to solve anything `KLU` can. `init_symbolic` cannot make
    the same check, since a bare sparsity pattern carries no values to inspect, so it stays
    on `Pardiso`. Construct `KLU()` directly for symbolic-pattern reuse on a complex
    operator.

    By default the chosen solver is wrapped in `IterativeRefinement`, which improves each
    solution until its relative residual is within tolerance or a step cap is spent. Pass
    an `IterativeRefinementSettings` to tune those, or `iterative_refinement=False` to
    solve with the chosen direct solver alone.
    """

    platform: str | None = None
    """Platform to select for. If None, `jax.default_backend()` is used. Set to e.g.
    "cpu", "gpu", or "tpu" to override the choice explicitly. `Pardiso`/`KLU` are chosen
    only when this resolves to "cpu" and x64 is enabled, otherwise `Spsolve` is
    chosen."""
    iterative_refinement: bool | IterativeRefinementSettings = True
    """Whether to refine the direct solve, and with what settings. `True` refines with the
    `IterativeRefinementSettings` defaults, `False` disables it, and an explicit
    `IterativeRefinementSettings` sets the tolerance and step cap."""

    @cached_property
    def _solver(self) -> SparseLinearSolver[Any]:
        """The exact solver `AutoSparseLinearSolver` runs.

        The platform dispatch, wrapped in `IterativeRefinement` unless refinement is
        disabled. Every stateful-API call and `select_solver` forward here, so with
        refinement on the state is an `_IterativeRefinementState` and with it off the state
        is the chosen direct solver's own.
        """
        dispatch = _AutoDispatch(self.platform)
        settings = self.iterative_refinement
        if settings is False:
            return dispatch
        if settings is True:
            settings = IterativeRefinementSettings()
        return IterativeRefinement(dispatch, settings.tol, settings.max_steps)

    def select_solver(
        self, operator: AbstractLinearOperator
    ) -> SparseLinearSolver[Any]:
        """The exact solver `AutoSparseLinearSolver` will run, including any refinement.

        Mirrors `lineax.AutoLinearSolver.select_solver`. With refinement on this is an
        `IterativeRefinement` wrapping the chosen direct solver. The operator is accepted
        for signature parity but selection depends only on the platform.
        """
        del operator
        return self._solver

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> Any:
        return self._solver.init(operator, options)

    def init_symbolic(self, sparsity: _Sparsity, options: dict[str, Any] = {}) -> Any:
        return self._solver.init_symbolic(sparsity, options)

    def update(
        self,
        state: Any,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> Any:
        return self._solver.update(state, operator, options)

    def release(self, state: Any) -> None:
        self._solver.release(state)

    def compute(
        self, state: Any, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        return self._solver.compute(state, vector, options)

    def transpose(
        self, state: Any, options: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        return self._solver.transpose(state, options)

    def conj(self, state: Any, options: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        return self._solver.conj(state, options)

    def assume_full_rank(self) -> bool:
        return self._solver.assume_full_rank()


AutoSparseLinearSolver.__init__.__doc__ = """**Arguments:**

- `platform`: optional platform string ("cpu", "gpu", "tpu") overriding the
    automatically detected `jax.default_backend()`. `Pardiso` (if installed) or `KLU`
    are chosen only when this resolves to "cpu" and x64 is enabled, otherwise
    `Spsolve` is chosen.
- `iterative_refinement`: whether to refine the direct solve with iterative refinement.
    `True` (the default) refines with the `IterativeRefinementSettings` defaults, `False`
    disables it, and an explicit `IterativeRefinementSettings` sets the tolerance and
    step cap.
"""
