"""A lineax iterative solver, with a preconditioner and optional system transforms."""

from contextlib import AbstractContextManager, contextmanager
from typing import Any, Iterator, Literal, NamedTuple, TypeAlias, overload

import equinox as eqx
import lineax as lx
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS

from splineax._pattern import CooPattern, _Sparsity, as_coo_pattern
from splineax.preconditioners._preconditioner import (
    NumericPreconditioner,
    Preconditioner,
    Side,
    SymbolicPreconditioner,
    operator_for,
)
from splineax.preconditioners._transform import (
    IdentityTransform,
    NumericTransform,
    SymbolicTransform,
    SystemTransform,
    compose,
    conj_transform,
    transform_operator,
    transform_pattern,
    transform_solution,
    transform_vector,
    transpose_transform,
    untransform_solution,
)
from splineax.solvers._sparse import (
    AbstractSparseLinearSolver,
    SparseNumericState,
    SymbolicScopedSparseLinearSolver,
    as_scoped_solver,
    factorize_through_init,
)

IterativeLinearSolver: TypeAlias = lx.CG | lx.BiCGStab | lx.GMRES

# Which side each lineax Krylov method applies `options["preconditioner"]` on. Not a
# choice this package makes: it is how each solver is written. `GMRES` preconditions
# the residual (`M(b - Ay)`), `BiCGStab`'s own source calls its preconditioner "K2^-1
# (i.e. right preconditioning)", and `CG` splits it across both sides.
_REQUIRED_SIDES: dict[type, frozenset[Side]] = {
    lx.GMRES: frozenset({Side.LEFT}),
    lx.BiCGStab: frozenset({Side.RIGHT}),
    lx.CG: frozenset({Side.LEFT, Side.RIGHT}),
}


def _required_sides(solver: IterativeLinearSolver) -> frozenset[Side]:
    for cls, sides in _REQUIRED_SIDES.items():
        if isinstance(solver, cls):
            return sides
    raise TypeError(
        "`PreconditionedIterativeLinearSolver` supports `lineax.CG`, "
        "`lineax.BiCGStab` and `lineax.GMRES`; got "
        f"{type(solver).__name__}. A direct solver has nothing to precondition -- use "
        "it on its own instead."
    )


def _pattern_of(sparsity: "_Sparsity | CooPattern", context: str) -> CooPattern:
    return (
        sparsity
        if isinstance(sparsity, CooPattern)
        else as_coo_pattern(sparsity, context)
    )


class _Analysis(eqx.Module):
    """Everything derivable from the sparsity pattern alone, computed once."""

    transform: SystemTransform
    symbolic: SymbolicPreconditioner


class _PreconditionedState(NamedTuple):
    """Everything a preconditioned solve needs, built once by `init`."""

    inner_state: Any
    """The wrapped lineax solver's own state, built from the transformed operator."""
    preconditioner: AbstractLinearOperator
    """The operator handed to lineax as `options["preconditioner"]`."""
    transform: SystemTransform
    """Applied to the right-hand side, and undone on the solution."""

    @contextmanager
    def factorize(self) -> Iterator["_PreconditionedState"]:
        # Building the preconditioner *is* this solver's numeric factorization, and
        # `init` has already done it, so there is nothing further to pre-compute.
        yield self


class _PreconditionedSymbolicScope(NamedTuple):
    """A scope holding the pattern analysis, reusable across sets of values."""

    solver: "PreconditionedIterativeLinearSolver"
    transform: SystemTransform
    """The composed symbolic transforms, already derived from the pattern."""
    symbolic: SymbolicPreconditioner
    """The preconditioner with its parameters resolved against the transformed pattern."""

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _PreconditionedState:
        """Build a solvable state, redoing only the value-dependent work."""
        return self.solver._init_from_symbolic(
            operator, options, self.transform, self.symbolic
        )

    @contextmanager
    def factorize(
        self, operator: AbstractLinearOperator
    ) -> Iterator[_PreconditionedState]:
        with self.init(operator).factorize() as state:
            yield state


class PreconditionedIterativeLinearSolver(
    AbstractSparseLinearSolver[_PreconditionedState]
):
    """Wraps a lineax Krylov method with a preconditioner built from the sparsity
    pattern.

    Lineax provides `CG`, `BiCGStab` and `GMRES`, and each accepts a preconditioner
    through `options["preconditioner"]` -- but leaves building one to the caller. That
    is the gap this fills, and it is a sparse problem: a good preconditioner is built in
    two phases, analysing the pattern once and rebuilding cheaply from each new set of
    values, which is the same split `factorize_symbolic` already exposes for the direct
    solvers here.

    ```python
    import jax.numpy as jnp
    import lineax as lx
    from jax.experimental.sparse import BCOO
    import splineax as splx

    matrix = jnp.eye(6) * 4.0 + jnp.diag(jnp.ones(5), 1) + jnp.diag(jnp.ones(5), -1)
    operator = splx.BCOOLinearOperator(BCOO.fromdense(matrix))

    solver = splx.PreconditionedIterativeLinearSolver(
        lx.GMRES(rtol=1e-10, atol=1e-10),
        splx.BlockJacobi(blocks=3),
        sparsity=operator,
    )
    solution = lx.linear_solve(operator, jnp.arange(6.0), solver=solver)
    assert jnp.allclose(solution.value, jnp.linalg.solve(matrix, jnp.arange(6.0)))
    ```

    This is an ordinary `lineax.AbstractLinearSolver`, so it goes anywhere a solver
    does, and it additionally supports this package's `factorize` /
    `factorize_symbolic` reuse API.

    !!! note "Why `sparsity` is given up front"

        Analysing a pattern -- detecting blocks, computing an ordering -- happens in
        Python, on concrete index values. `lineax.linear_solve` is itself jit-wrapped,
        so by the time it calls `init` the operator's indices are *tracers* and no
        host-side analysis is possible, even outside any `jax.jit` of your own. Passing
        `sparsity` does the analysis once, when the solver is built, which is also
        where you want it: it is reused by every solve.

        Two other routes do the analysis outside the trace and so need no `sparsity`:
        calling `solver.init(operator)` yourself and passing the result as
        `state=`, or opening a `factorize_symbolic` scope. Omitting all three raises,
        rather than silently falling back to something slower.

    !!! note "Transforms are global, and belong here rather than in the preconditioner"

        A reordering applies to the whole system or not at all, so it is not a
        parameter of any one preconditioner. `transforms` runs before the
        preconditioner's parameters are resolved, and this solver is what permutes the
        right-hand side and un-permutes the solution. The preconditioner only ever sees
        the pattern the transforms produced, and needs to know nothing about them.
    """

    solver: IterativeLinearSolver
    """The lineax Krylov method that does the solving."""
    preconditioner: Preconditioner
    """What to build as `M`. Its parameters may be values or injected providers."""
    transforms: tuple[SymbolicTransform | NumericTransform, ...] = ()
    """Applied to the whole system in order, before the preconditioner is built."""
    analysis: "_Analysis | None" = None
    """The pre-computed pattern analysis, when `sparsity` was given. Never set by hand:
    it is derived in `__init__`."""

    def __init__(
        self,
        solver: IterativeLinearSolver,
        preconditioner: Preconditioner,
        transforms: tuple[SymbolicTransform | NumericTransform, ...] = (),
        sparsity: _Sparsity | CooPattern | None = None,
    ):
        self.solver = solver
        self.preconditioner = preconditioner
        self.transforms = tuple(transforms)
        self.analysis = (
            None
            if sparsity is None
            else _Analysis(
                *self._analyze(
                    _pattern_of(
                        sparsity, "`PreconditionedIterativeLinearSolver(sparsity=...)`"
                    )
                )
            )
        )

    def __check_init__(self):
        for position, stage in enumerate(self.transforms):
            if not isinstance(stage, (SymbolicTransform, NumericTransform)):
                raise TypeError(
                    f"`transforms[{position}]` is a {type(stage).__name__}, which is "
                    "neither a `SymbolicTransform` (a `symbolic(pattern)` method) nor a "
                    "`NumericTransform` (a `numeric(operator)` method), so it would "
                    "never be applied."
                )
        required = _required_sides(self.solver)
        available = self.preconditioner.sides
        if not required <= available:
            missing = ", ".join(sorted(side.name for side in required - available))
            raise ValueError(
                f"`{type(self.solver).__name__}` applies its preconditioner on the "
                f"{', '.join(sorted(s.name for s in required))} side, but "
                f"{type(self.preconditioner).__name__} does not support: {missing}."
            )

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _PreconditionedState:
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`PreconditionedIterativeLinearSolver` may only be used for linear "
                "solves with square matrices."
            )
        if self.analysis is not None:
            analysis = self.analysis
        else:
            pattern = as_coo_pattern(operator, "`PreconditionedIterativeLinearSolver`")
            try:
                analysis = _Analysis(*self._analyze(pattern))
            except ValueError as error:
                if "concrete values rather than tracers" not in str(error):
                    raise
                raise ValueError(
                    "`PreconditionedIterativeLinearSolver` must analyze the sparsity "
                    "pattern host-side, but this operator's indices are traced -- "
                    "which is always the case inside `lineax.linear_solve`, since it "
                    "is jit-wrapped. Build the solver with `sparsity=` so the analysis "
                    "happens once up front, call `solver.init(operator)` yourself and "
                    "pass it as `state=`, or open a `factorize_symbolic` scope."
                ) from error
        return self._init_from_symbolic(
            operator, options, analysis.transform, analysis.symbolic
        )

    def _analyze(
        self, pattern: CooPattern
    ) -> tuple[SystemTransform, SymbolicPreconditioner]:
        """Run the symbolic transforms, then resolve the preconditioner's parameters.

        Each stage re-patterns before the next, so a transform analyses its
        predecessor's output rather than the original -- which is what lets a matching
        and a reordering compose without either knowing about the other.
        """
        transform: SystemTransform = IdentityTransform()
        for stage in self.transforms:
            if isinstance(stage, SymbolicTransform):
                step = stage.symbolic(pattern)
                pattern = transform_pattern(step, pattern)
                transform = compose(transform, step)
        return transform, self.preconditioner.symbolic(pattern)

    def _init_from_symbolic(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any],
        transform: SystemTransform,
        symbolic: SymbolicPreconditioner,
    ) -> _PreconditionedState:
        if "preconditioner" in options:
            raise ValueError(
                "`PreconditionedIterativeLinearSolver` builds the preconditioner "
                "itself, so `options['preconditioner']` cannot also be given. Pass the "
                "preconditioner to the solver instead, or use the bare lineax solver."
            )
        # Numeric transforms need the values, so they run here rather than in
        # `_analyze`, on the operator the symbolic transforms already produced. A stage
        # offering both is taken as symbolic, so that it is analysed once per pattern
        # rather than once per set of values.
        transformed = transform_operator(transform, operator)
        for stage in self.transforms:
            if not isinstance(stage, SymbolicTransform) and isinstance(
                stage, NumericTransform
            ):
                step = stage.numeric(transformed)
                transformed = transform_operator(step, transformed)
                transform = compose(transform, step)
        numeric: NumericPreconditioner = symbolic.numeric(transformed)
        # Lineax takes a single `options["preconditioner"]`, so even `CG` -- which
        # applies it on both sides -- needs just one operator; requiring both sides is
        # about what the preconditioner must *support*, not about handing over two.
        required = _required_sides(self.solver)
        side = Side.LEFT if Side.LEFT in required else Side.RIGHT
        preconditioner = operator_for(numeric, side)
        # `CG` checks definiteness here, on the *transformed* operator -- which is why
        # tags have to survive the transform rather than being dropped wholesale.
        inner_state = self.solver.init(transformed, options)
        return _PreconditionedState(inner_state, preconditioner, transform)

    def compute(
        self,
        state: _PreconditionedState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        inner = {**options, "preconditioner": state.preconditioner}
        if "y0" in inner:
            # `y0` is given in the untransformed solution space, like the answer.
            inner["y0"] = transform_solution(state.transform, inner["y0"])
        value, result, stats = self.solver.compute(
            state.inner_state, transform_vector(state.transform, vector), inner
        )
        return untransform_solution(state.transform, value), result, stats

    def transpose(
        self, state: _PreconditionedState, options: dict[str, Any]
    ) -> tuple[_PreconditionedState, dict[str, Any]]:
        inner_options = {**options, "preconditioner": state.preconditioner}
        inner_state, transposed_options = self.solver.transpose(
            state.inner_state, inner_options
        )
        # Every lineax Krylov solver transposes the preconditioner in `options` for us.
        preconditioner = transposed_options.pop("preconditioner", state.preconditioner)
        return (
            _PreconditionedState(
                inner_state, preconditioner, transpose_transform(state.transform)
            ),
            transposed_options,
        )

    def conj(
        self, state: _PreconditionedState, options: dict[str, Any]
    ) -> tuple[_PreconditionedState, dict[str, Any]]:
        inner_options = {**options, "preconditioner": state.preconditioner}
        inner_state, conj_options = self.solver.conj(state.inner_state, inner_options)
        preconditioner = conj_options.pop("preconditioner", state.preconditioner)
        return (
            _PreconditionedState(
                inner_state, preconditioner, conj_transform(state.transform)
            ),
            conj_options,
        )

    def assume_full_rank(self) -> bool:
        return self.solver.assume_full_rank()

    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        """Pre-compute the preconditioner for reuse across solves.

        Building it is the numeric factorization, and `init` already does that, so this
        yields the state `init` returns. There are no native handles to free.
        """
        return factorize_through_init(self, operator, options)

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[False] = False
    ) -> AbstractContextManager[_PreconditionedSymbolicScope]: ...

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[True]
    ) -> AbstractContextManager[SymbolicScopedSparseLinearSolver]: ...

    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: bool = False
    ) -> AbstractContextManager[
        _PreconditionedSymbolicScope | SymbolicScopedSparseLinearSolver
    ]:
        """Pre-analyze a sparsity pattern, for reuse across many sets of values.

        This is where the two-phase split pays: the reorderings, the block detection and
        the scatter indices are computed once per *pattern*, and each `scope.init` redoes
        only the scatter-add and the batched inverse.

        Unlike `KLU`, the analysis runs host-side, so the pattern's indices must be
        concrete values rather than tracers.

        Args:
            sparsity: the pattern to pre-analyze.
            as_solver: yield a `SymbolicScopedSparseLinearSolver` pairing the scope with
                       this solver, instead of the bare scope.
        """
        scope = self._factorize_symbolic(sparsity)
        return as_scoped_solver(self, scope) if as_solver else scope

    @contextmanager
    def _factorize_symbolic(
        self, sparsity: _Sparsity
    ) -> Iterator[_PreconditionedSymbolicScope]:
        # Kept separate from `factorize_symbolic` so that the public method can be
        # overloaded on `as_solver` (`@contextmanager` and `@overload` do not compose).
        pattern = as_coo_pattern(
            sparsity, "`PreconditionedIterativeLinearSolver.factorize_symbolic`"
        )
        if pattern.shape[0] != pattern.shape[1]:
            raise ValueError(
                "`PreconditionedIterativeLinearSolver.factorize_symbolic` requires a "
                f"square matrix; got shape {pattern.shape}."
            )
        transform, symbolic = self._analyze(pattern)
        yield _PreconditionedSymbolicScope(self, transform, symbolic)


PreconditionedIterativeLinearSolver.__init__.__doc__ = """**Arguments:**

- `solver`: the lineax Krylov method to use: `lineax.CG`, `lineax.BiCGStab` or
    `lineax.GMRES`. Its tolerances and iteration limits are its own.
- `preconditioner`: what to build as `M`, such as
    [`splineax.BlockJacobi`][]. Each of its parameters may be given as a value or as an
    injected provider that derives one from the sparsity pattern.
- `transforms`: transformations applied to the whole system before the preconditioner
    is built, in order, such as [`splineax.ReverseCuthillMcKee`][]. The solver permutes
    the right-hand side and un-permutes the solution, so the transformation is invisible
    from the outside.
- `sparsity`: the sparsity pattern to analyze up front, as any of the types
    `factorize_symbolic` accepts (an operator will do). Needed to use this solver
    directly with `lineax.linear_solve`, which is jit-wrapped and so can only hand
    `init` traced indices. Omit it when passing `state=solver.init(operator)` yourself
    or going through `factorize_symbolic`, both of which analyze outside the trace.

Raises `ValueError` if the preconditioner cannot be applied on the side the Krylov
method needs -- `GMRES` preconditions on the left, `BiCGStab` on the right, and `CG`
requires both.
"""
