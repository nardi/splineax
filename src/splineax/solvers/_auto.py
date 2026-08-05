from contextlib import AbstractContextManager
from functools import cached_property
from typing import Any, Literal, overload

import jax
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver

from ._klu import KLU
from ._pardiso import (
    Pardiso,
    _pardiso_available,
    _PardisoBasicState,
    _PardisoNumericState,
    _PardisoSymbolicState,
)
from ._sparse import (
    AbstractSparseLinearSolver,
    SparseBasicState,
    SparseNumericState,
    SparseSymbolicScope,
    SparseSymbolicState,
    SymbolicScopedSparseLinearSolver,
    _Sparsity,
    as_scoped_solver,
)
from ._spsolve import Spsolve

_PARDISO_STATE_TYPES = (_PardisoBasicState, _PardisoNumericState, _PardisoSymbolicState)


class AutoSparseLinearSolver(
    AbstractSparseLinearSolver[
        SparseBasicState | SparseSymbolicState | SparseNumericState
    ]
):
    """Selects a sparse direct solver based on the JAX platform, precision, and what's
    installed.

    On CPU with x64 enabled, dispatches to `Pardiso` (Intel oneMKL Pardiso, factorization
    reuse) if the optional `pardiso-mkl-jax` dependency is installed, otherwise `KLU`
    (SuiteSparse, factorization reuse). Both are double precision only, hence the x64
    requirement. On any other backend, or on CPU when x64 is disabled, it dispatches to
    `Spsolve`, which works in single or double precision and on any backend. Exposes the
    same factorization API as `Pardiso`/`KLU` (`analyze_numeric`, `analyze_symbolic`), so
    it can be substituted for either verbatim. When it dispatches to `Spsolve`, these
    factorization calls degrade to no-ops.

    `pardiso_mkl_jax` does not support complex matrices (see `Pardiso`'s docstring), so
    `init`/`analyze_numeric` fall back to `KLU` for a complex operator even when
    `Pardiso` was otherwise selected, keeping `Auto` able to solve anything `KLU` can.
    `analyze_symbolic` cannot make the same check, since a bare sparsity pattern carries
    no values to inspect, so it stays on `Pardiso`. Construct `KLU()` directly for
    symbolic-factorization reuse on a complex operator.
    """

    platform: str | None = None
    """Platform to select for. If None, `jax.default_backend()` is used. Set to e.g.
    "cpu", "gpu", or "tpu" to override the choice explicitly. `Pardiso`/`KLU` are chosen
    only when this resolves to "cpu" and x64 is enabled, otherwise `Spsolve` is
    chosen."""

    @cached_property
    def _chosen_solver(self) -> Pardiso | KLU | Spsolve:
        platform = self.platform if self.platform is not None else jax.default_backend()
        x64_enabled = jax.config.read("jax_enable_x64")
        # Pardiso and KLU are both double precision only, so either is only a valid
        # choice on CPU when x64 is enabled. Pardiso is preferred when its optional
        # dependency is installed. KLU (a hard dependency) is always available as a
        # fallback. Everything else falls back to Spsolve, which works in single or
        # double precision and on any backend.
        if platform == "cpu" and x64_enabled:
            return Pardiso() if _pardiso_available() else KLU()
        return Spsolve()

    def select_solver(self, operator: AbstractLinearOperator) -> AbstractLinearSolver:
        """Check which solver `AutoSparseLinearSolver` will dispatch to.

        Mirrors `lineax.AutoLinearSolver.select_solver`. The operator is accepted for
        signature parity but selection depends only on the platform.
        """
        del operator
        return self._chosen_solver

    def _solver_for_state(self, state: Any) -> Pardiso | KLU | Spsolve:
        """The concrete solver that must handle `state`.

        Usually `self._chosen_solver`, except when it's `Pardiso` but `state` isn't
        one of Pardiso's own state types: that means `init`/`analyze_numeric` fell back
        to `KLU` for a complex operator (see the class docstring), and later calls on
        that same state need to keep using `KLU` too.
        """
        chosen = self._chosen_solver
        if isinstance(chosen, Pardiso) and not isinstance(state, _PARDISO_STATE_TYPES):
            return KLU()
        return chosen

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> SparseBasicState:
        chosen = self._chosen_solver
        if isinstance(chosen, Pardiso):
            try:
                return chosen.init(operator, options)
            except TypeError:
                # `pardiso_mkl_jax` doesn't support complex matrices. Fall back to
                # `KLU`, which does, rather than surfacing Pardiso's error for a case
                # Auto can transparently handle. `KLU` is a hard dependency, always
                # available here.
                return KLU().init(operator, options)
        return chosen.init(operator, options)

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

    def analyze_numeric(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        chosen = self._chosen_solver
        if isinstance(chosen, Pardiso):
            # `Pardiso.analyze_numeric`/`KLU.analyze_numeric` are both just
            # `self.init(...).analyze_numeric()` (see `analyze_numeric_through_init`),
            # and unlike `init` itself, `analyze_numeric()` doesn't raise eagerly: it
            # returns a context manager that only runs `init` once entered, by which
            # point it's too late to switch solvers. Calling `init` here instead, and
            # reusing its state, lets the same try/except fallback as `init` above
            # work without that trap.
            try:
                init_state = chosen.init(operator, options)
            except TypeError:
                # See `init`'s matching fallback.
                return KLU().analyze_numeric(operator, options)
            return init_state.analyze_numeric()
        return chosen.analyze_numeric(operator, options)

    @overload
    def analyze_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[False] = False
    ) -> AbstractContextManager[SparseSymbolicScope]: ...

    @overload
    def analyze_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[True]
    ) -> AbstractContextManager[SymbolicScopedSparseLinearSolver]: ...

    def analyze_symbolic(
        self, sparsity: _Sparsity, *, as_solver: bool = False
    ) -> AbstractContextManager[SparseSymbolicScope | SymbolicScopedSparseLinearSolver]:
        scope = self._chosen_solver.analyze_symbolic(sparsity)
        # Pair the scope with `self`, not with the chosen solver: solving through
        # `AutoSparseLinearSolver` dispatches to the same solver anyway (see
        # `_solver_for_state`), and this keeps the scoped solver as substitutable for
        # `AutoSparseLinearSolver` as the rest of its API is.
        return as_scoped_solver(self, scope) if as_solver else scope


AutoSparseLinearSolver.__init__.__doc__ = """**Arguments:**

- `platform`: optional platform string ("cpu", "gpu", "tpu") overriding the
    automatically detected `jax.default_backend()`. `Pardiso` (if installed) or `KLU`
    are chosen only when this resolves to "cpu" and x64 is enabled, otherwise
    `Spsolve` is chosen.
"""
