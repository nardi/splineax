"""The sparsity pattern of a matrix, in coordinate form.

Every part of this package that pre-analyzes a matrix before it has any values --
`KLU.factorize_symbolic`, `Pardiso.factorize_symbolic`, and the preconditioners --
starts by reducing whatever the caller passed to the same thing: row indices, column
indices, and a shape. `as_coo_pattern` is that reduction, and `_Sparsity` is the union
of everything it accepts.

The awkward part is that the accepted types disagree about where the indices live.
`BCOO`/`BCSR` carry them as (possibly traced) JAX arrays, while the Jacobian operators
carry a host-side asdex pattern, and a transposed `SparseJacobianLinearOperator`
describes its *forward* Jacobian, so its rows and columns must be swapped to stay
aligned with the values they are later paired with. Collecting that in one place keeps
the three call sites from drifting apart.
"""

from typing import NamedTuple

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Integer
from lineax import AbstractLinearOperator

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
)

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


class ConcreteCooPattern(NamedTuple):
    """A `CooPattern` whose indices are known host-side, as NumPy arrays.

    Produced by `CooPattern.concrete`. Analyses that run in Python rather than as
    staged-out JAX operations -- block detection, orderings -- need this rather than the
    traced form.
    """

    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]

    @property
    def nse(self) -> int:
        """The number of stored entries, including any duplicates."""
        return int(self.rows.size)


class CooPattern(NamedTuple):
    """Row indices, column indices and shape of a matrix's stored entries.

    Entry `k` of the matrix is stored at `(rows[k], cols[k])`. Duplicate coordinates are
    permitted (an uncoalesced `BCOO` is a valid pattern) and the entries are in no
    particular order; what matters is that the order matches the order of the *values*
    of any operator later paired with this pattern, which is the same contract
    `factorize_symbolic` already places on its caller.

    The indices may be traced, since `KLU` can pre-analyze a pattern from inside
    `jax.jit`. Call `concrete` to obtain host-side NumPy indices, which is what the
    Python-level analyses require.
    """

    rows: Integer[Array, " nse"]
    cols: Integer[Array, " nse"]
    shape: tuple[int, int]

    @property
    def nse(self) -> int:
        """The number of stored entries, including any duplicates."""
        return int(self.rows.shape[0])

    def concrete(self, context: str) -> ConcreteCooPattern:
        """Read the indices host-side, raising if they are traced.

        **Arguments:**

        - `context`: what needed the concrete pattern, named in the error message.
        """
        if isinstance(self.rows, jax.core.Tracer) or isinstance(
            self.cols, jax.core.Tracer
        ):
            raise ValueError(
                f"{context} analyzes the sparsity pattern host-side, so its indices "
                "must be concrete values rather than tracers. This happens when the "
                "pattern is derived from a matrix built inside `jax.jit` (or another "
                "transform). Pass a concrete pattern instead -- the indices of a "
                "sparse matrix are usually known outside the trace even when its "
                "values are not."
            )
        return ConcreteCooPattern(
            np.asarray(self.rows), np.asarray(self.cols), self.shape
        )


def as_coo_pattern(
    sparsity: _Sparsity | AbstractLinearOperator, context: str
) -> CooPattern:
    """Reduce any accepted sparsity specification to row/column indices and a shape.

    **Arguments:**

    - `sparsity`: the pattern, as any member of `_Sparsity`.
    - `context`: the calling API, named in the error message raised for an
        unsupported type.
    """
    match sparsity:
        case SparseJacobianLinearOperator(transposed=True):
            # The stored pattern describes the forward Jacobian. asdex emits `BCOO`
            # values in the pattern's index order and `BCOO.T` swaps the index columns
            # without reordering entries, so swapping rows and columns here keeps the
            # indices aligned with the values they are later paired with.
            pattern = sparsity.coloring.sparsity
            return CooPattern(
                jnp.asarray(pattern.cols, dtype=jnp.int32),
                jnp.asarray(pattern.rows, dtype=jnp.int32),
                pattern.shape[::-1],
            )
        case SparseJacobianLinearOperator() | SparseJacobianLinearOperatorColoring():
            # Both hold the coloring one level in: the operator stores an
            # `asdex.ColoredPattern` whose `.sparsity` is the pattern, and the operator
            # coloring stores a `JacobianColoring` whose `.sparsity` property returns
            # the same pattern.
            pattern = sparsity.coloring.sparsity
            return CooPattern(
                jnp.asarray(pattern.rows, dtype=jnp.int32),
                jnp.asarray(pattern.cols, dtype=jnp.int32),
                pattern.shape,
            )
        case JacobianColoring():
            # A bare coloring exposes the pattern directly through its `.sparsity`.
            pattern = sparsity.sparsity
            return CooPattern(
                jnp.asarray(pattern.rows, dtype=jnp.int32),
                jnp.asarray(pattern.cols, dtype=jnp.int32),
                pattern.shape,
            )
        case BCSRLinearOperator():
            return _from_bcoo(sparsity.matrix.to_bcoo())
        case BCOOLinearOperator():
            return _from_bcoo(sparsity.matrix)
        case BCSR():
            return _from_bcoo(sparsity.to_bcoo())
        case BCOO():
            return _from_bcoo(sparsity)
        case _:
            raise TypeError(
                f"{context} requires a `BCOO`, `BCSR`, `BCOOLinearOperator`, "
                "`BCSRLinearOperator`, `SparseJacobianLinearOperator`, "
                "`SparseJacobianLinearOperatorColoring`, or `JacobianColoring`; "
                f"got {type(sparsity).__name__}."
            )


def sparsity_values(sparsity: _Sparsity) -> Array | None:
    """The stored values of a sparsity specification, when it carries any.

    A pattern given as a matrix (`BCOO`, `BCSR`, or an operator wrapping one) also
    carries values, which `Pardiso` inspects to reject complex dtypes up front rather
    than at the first solve. A pattern given as a coloring carries none, hence `None`.
    """
    match sparsity:
        case BCSRLinearOperator():
            return sparsity.matrix.to_bcoo().data
        case BCOOLinearOperator():
            return sparsity.matrix.data
        case BCSR():
            return sparsity.data
        case BCOO():
            return sparsity.data
        case _:
            return None


def _from_bcoo(matrix: BCOO) -> CooPattern:
    if matrix.ndim != 2:
        raise ValueError(
            f"A sparsity pattern must be 2-dimensional; got {matrix.ndim} dimensions."
        )
    rows, cols = matrix.indices[:, 0], matrix.indices[:, 1]
    return CooPattern(
        rows.astype(jnp.int32),
        cols.astype(jnp.int32),
        (int(matrix.shape[0]), int(matrix.shape[1])),
    )
