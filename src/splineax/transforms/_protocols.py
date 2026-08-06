"""Protocols for system transformations and preconditioners.

A [preconditioned iterative solve](../guide/preconditioning.md) rewrites `A x = b` into
`(L A R) y = L b`, then recovers `x = R y`, and hands a preconditioner to a Krylov
solver. Every transform (a permutation, a scaling, ...) is one `(L, R)` pair, and
transforms compose by chaining. Three degrees of binding lead up to that pair:

- `SystemTransform`: the rule, bound to nothing yet. What you configure and pass to a
  solver, e.g. `AggregationClustering(block_size=32)`.
- `AnalyzedTransform`: the same rule bound to one sparsity pattern, built by
  `analyze_symbolic`. Knows the pattern it produces, so a later stage in a chain can
  plan against it before any matrix exists.
- `AppliedTransform`: bound to one matrix, built by `analyze_numeric`. The concrete
  `(L, R)` pair: `transform_vector` applies `L`, `recover_solution` applies `R`.

`Preconditioner` mirrors the first two tiers. There is no third: its bound form is
already a `lineax.AbstractLinearOperator`, which is what a Krylov solver wants directly.

None of these are base classes. A `NamedTuple` or `eqx.Module` satisfies a protocol
just by having the right fields and methods, so a new transform is written by
implementing `SystemTransform` and `AnalyzedTransform`, nothing more.
"""

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from jax.experimental.sparse import BCOO
from jaxtyping import Array
from lineax import AbstractLinearOperator

from splineax.operators._pattern import MatrixSparsity


@runtime_checkable
class AppliedTransform(Protocol):
    """One concrete `(L, R)` pair, bound to one matrix.

    Built by `AnalyzedTransform.analyze_numeric`. `transform_vector` applies `L` to a
    right-hand side, `recover_solution` applies `R` to a solution of the transformed
    system to recover the solution of the original one.
    """

    def transform_vector(self, b: Array) -> Array:
        """Applies `L`: `A x = b` becomes `(L A R) y = L b`."""
        ...

    def recover_solution(self, y: Array) -> Array:
        """Applies `R`: recovers `x = R y` from a solution `y` of the transformed
        system."""
        ...

    def transpose(self) -> "AppliedTransform":
        """The pair for the transposed system, `(R^T, L^T)`."""
        ...

    def conj(self) -> "AppliedTransform":
        """The pair with every array conjugated."""
        ...


@runtime_checkable
class AnalyzedTransform(Protocol):
    """A `SystemTransform` bound to one sparsity pattern.

    Built by `SystemTransform.analyze_symbolic`.
    """

    @property
    def pattern(self) -> MatrixSparsity:
        """The sparsity pattern of `L A R`, given the pattern of `A` this was
        analyzed against. What a later transform in a chain plans against."""
        ...

    @property
    def is_congruence(self) -> bool:
        """Whether `L == R^T`, so `L A R = R^T A R`.

        A congruence preserves symmetry and positive/negative semidefiniteness.
        `PreconditionedIterativeSolver` reads this to decide whether an operator's
        tags survive onto the transformed operator.
        """
        ...

    def analyze_numeric(self, matrix: BCOO) -> tuple[BCOO, AppliedTransform]:
        """Rewrites `matrix` to `L matrix R`, returning it with the applied `(L, R)`
        pair."""
        ...


@runtime_checkable
class SystemTransform(Protocol):
    """A rule for rewriting `A x = b`, bound to nothing yet.

    What you configure and pass to `PreconditionedIterativeSolver`, or chain with
    `compose_transforms`.
    """

    def analyze_symbolic(self, pattern: MatrixSparsity) -> AnalyzedTransform:
        """Analyzes `pattern`, the sparsity of `A`, producing a plan for the numeric
        step."""
        ...


@runtime_checkable
class AnalyzedPreconditioner(Protocol):
    """A `Preconditioner` bound to one sparsity pattern.

    Built by `Preconditioner.analyze_symbolic`.
    """

    def analyze_numeric(
        self, matrix: BCOO, tags: frozenset[object]
    ) -> AbstractContextManager[AbstractLinearOperator]:
        """Builds the preconditioning operator for `matrix`, as a context manager.

        A context manager because some preconditioners own a resource (a native
        factorization, say) that must be released once the caller is done solving
        with it. `tags` are the (already transformed) operator's lineax tags, passed
        through so the built operator can carry the ones it still honours, e.g.
        `positive_semidefinite_tag` for a preconditioner `CG` can use.
        """
        ...


@runtime_checkable
class Preconditioner(Protocol):
    """A rule for approximating `A^-1`, bound to nothing yet.

    What you configure and pass to `PreconditionedIterativeSolver`.
    """

    def analyze_symbolic(self, pattern: MatrixSparsity) -> AnalyzedPreconditioner:
        """Analyzes `pattern`, the sparsity of the (already transformed) matrix,
        producing a plan for the numeric step."""
        ...
