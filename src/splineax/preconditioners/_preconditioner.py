"""The interface a preconditioner implements, in three tiers.

A preconditioner is built the same way the direct solvers in this package are: analyse
the sparsity pattern once, then rebuild cheaply from each new set of values. That gives
three objects rather than one, mirroring the `solver` -> `scope` -> `state` progression
of `factorize_symbolic`:

```text
BlockJacobi(blocks=...)          the spec:      static settings, holds no values
  .symbolic(pattern)     ->      the symbolic:  parameters resolved against a pattern
  .numeric(operator)     ->      the numeric:   an operator ready to hand to lineax
```

Each tier is a `Protocol` rather than a base class, matching how `SparseLinearSolver`
and friends are typed in `splineax.solvers`. Nothing needs to subclass anything to be
usable as a preconditioner.
"""

from enum import Enum, auto
from typing import Protocol, runtime_checkable

from lineax import AbstractLinearOperator

from splineax._pattern import CooPattern, _Sparsity


class Side(Enum):
    """Which side of the system a preconditioner is applied to.

    Not a free choice: it is fixed by the Krylov method, since each of lineax's
    iterative solvers applies `options["preconditioner"]` in exactly one way.

    - `LEFT` solves `M A x = M b`, as `lineax.GMRES` does.
    - `RIGHT` solves `A M y = b` and recovers `x = M y`, as `lineax.BiCGStab` does.

    `lineax.CG` splits the preconditioner across both sides, so it requires a
    preconditioner offering both.
    """

    LEFT = auto()
    RIGHT = auto()


@runtime_checkable
class Preconditioner(Protocol):
    """A preconditioner specification: what to build, with no values yet.

    This is the tier the user constructs and passes to
    `PreconditionedIterativeLinearSolver`. It is a description, so it is cheap, static,
    and reusable across any number of patterns and matrices.
    """

    @property
    def sides(self) -> frozenset[Side]:
        """The sides this preconditioner can be applied to.

        Declared on the spec tier so that pairing a preconditioner with a Krylov method
        that needs a side it cannot supply fails when the solver is *constructed*,
        rather than at the first solve.
        """
        ...

    def symbolic(self, sparsity: _Sparsity | CooPattern) -> "SymbolicPreconditioner":
        """Resolve this specification's parameters against a sparsity pattern."""
        ...


@runtime_checkable
class SymbolicPreconditioner(Protocol):
    """A preconditioner whose parameters are resolved, but which holds no values.

    Everything derived from the pattern alone -- block boundaries, orderings, scatter
    indices -- has been computed by this point and is reused by every `numeric` call.
    """

    @property
    def sides(self) -> frozenset[Side]:
        """The sides this preconditioner can be applied to."""
        ...

    def numeric(self, operator: AbstractLinearOperator) -> "NumericPreconditioner":
        """Build the preconditioner for an operator with this symbolic's pattern."""
        ...


@runtime_checkable
class NumericPreconditioner(Protocol):
    """A fully built preconditioner.

    Marker protocol: what it can actually do comes from `LeftPreconditioner` and
    `RightPreconditioner` below, at least one of which every numeric preconditioner
    implements.
    """


@runtime_checkable
class LeftPreconditioner(Protocol):
    """A numeric preconditioner that can be applied on the left, solving `M A x = M b`."""

    def left_operator(self) -> AbstractLinearOperator:
        """The operator `M` to apply on the left."""
        ...


@runtime_checkable
class RightPreconditioner(Protocol):
    """A numeric preconditioner applied on the right, solving `A M y = b`, `x = M y`."""

    def right_operator(self) -> AbstractLinearOperator:
        """The operator `M` to apply on the right.

        A separate method rather than a flag on one operator because the two sides are
        genuinely different for some preconditioners: a split incomplete factorisation
        applies `L^-1` on the left and `U^-1` on the right. For a symmetric
        preconditioner such as `BlockJacobi` this returns the same operator as
        `left_operator`.
        """
        ...


def operator_for(
    preconditioner: NumericPreconditioner, side: Side
) -> AbstractLinearOperator:
    """The operator a numeric preconditioner supplies for `side`.

    Raises `TypeError` if it does not support that side -- which the solver's
    construction-time check exists to prevent reaching.
    """
    match side:
        case Side.LEFT:
            if not isinstance(preconditioner, LeftPreconditioner):
                raise TypeError(
                    f"{type(preconditioner).__name__} cannot be applied on the left."
                )
            return preconditioner.left_operator()
        case Side.RIGHT:
            if not isinstance(preconditioner, RightPreconditioner):
                raise TypeError(
                    f"{type(preconditioner).__name__} cannot be applied on the right."
                )
            return preconditioner.right_operator()
