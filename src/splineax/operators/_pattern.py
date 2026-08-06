"""Reads a bare sparsity pattern out of anything a solver's `analyze_symbolic` accepts.

`MatrixSparsity` and `as_matrix_sparsity` are shared by every symbolic-analysis entry
point in the package: `KLU.analyze_symbolic` and `PreconditionedIterativeSolver`'s
transforms both start from the same row/column indices, read the same way, regardless
of whether the caller handed over a bare `BCOO`, an operator, or an asdex coloring.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Integer

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
)

# Everything `analyze_symbolic` accepts as a sparsity pattern.
_Sparsity = (
    BCOO
    | BCSR
    | BCOOLinearOperator
    | BCSRLinearOperator
    | SparseJacobianLinearOperator
    | SparseJacobianLinearOperatorColoring
    | JacobianColoring
)


class MatrixSparsity(NamedTuple):
    """Row and column indices of a matrix's nonzero pattern, with its shape.

    Values are dropped: only where the nonzeros are, not what they hold. This is what
    every symbolic-analysis stage in the package (a solver's or a transform's) starts
    from.
    """

    rows: Integer[Array, " nse"]
    cols: Integer[Array, " nse"]
    shape: tuple[int, int]


def as_matrix_sparsity(sparsity: _Sparsity, *, context: str) -> MatrixSparsity:
    """Reads the row/column indices and shape out of any `_Sparsity` source.

    Host-side for the coloring cases: `SparsityPattern.rows`/`.cols` are plain numpy
    arrays read from the precomputed asdex pattern, so `sparsity` must be concrete
    there, not a traced value inside a jitted function. The `BCOO`/`BCSR` cases read
    ordinary JAX arrays and work fine under a trace.

    **Arguments:**

    - `sparsity`: a `BCOO`, `BCSR`, `BCOOLinearOperator`, `BCSRLinearOperator`,
        `SparseJacobianLinearOperator`, `SparseJacobianLinearOperatorColoring`, or
        `JacobianColoring`.
    - `context`: names the caller in the raised `TypeError`, e.g.
        `"KLU.analyze_symbolic"`.
    """
    match sparsity:
        case SparseJacobianLinearOperator(transposed=True):
            # The stored pattern describes the forward Jacobian. asdex emits `BCOO`
            # values in the pattern's index order and `BCOO.T` swaps the index columns
            # without reordering entries, so swapping rows and columns here keeps the
            # indices aligned with the values a later numeric step pairs them with.
            pattern = sparsity.coloring.sparsity
            rows = jnp.asarray(pattern.cols, dtype=jnp.int32)
            cols = jnp.asarray(pattern.rows, dtype=jnp.int32)
            shape = pattern.shape[::-1]
        case SparseJacobianLinearOperator() | SparseJacobianLinearOperatorColoring():
            # Both hold the coloring one level in: the operator stores an
            # `asdex.ColoredPattern` whose `.sparsity` is the pattern, and the operator
            # coloring stores a `JacobianColoring` whose `.sparsity` property returns
            # the same pattern.
            pattern = sparsity.coloring.sparsity
            rows = jnp.asarray(pattern.rows, dtype=jnp.int32)
            cols = jnp.asarray(pattern.cols, dtype=jnp.int32)
            shape = pattern.shape
        case JacobianColoring():
            # A bare coloring exposes the pattern directly through its `.sparsity`
            # property.
            pattern = sparsity.sparsity
            rows = jnp.asarray(pattern.rows, dtype=jnp.int32)
            cols = jnp.asarray(pattern.cols, dtype=jnp.int32)
            shape = pattern.shape
        case BCSRLinearOperator():
            bcoo = sparsity.matrix.to_bcoo()
            rows = bcoo.indices[:, 0].astype(jnp.int32)
            cols = bcoo.indices[:, 1].astype(jnp.int32)
            shape = bcoo.shape
        case BCOOLinearOperator():
            bcoo = sparsity.matrix
            rows = bcoo.indices[:, 0].astype(jnp.int32)
            cols = bcoo.indices[:, 1].astype(jnp.int32)
            shape = bcoo.shape
        case BCSR():
            bcoo = sparsity.to_bcoo()
            rows = bcoo.indices[:, 0].astype(jnp.int32)
            cols = bcoo.indices[:, 1].astype(jnp.int32)
            shape = bcoo.shape
        case BCOO():
            rows = sparsity.indices[:, 0].astype(jnp.int32)
            cols = sparsity.indices[:, 1].astype(jnp.int32)
            shape = sparsity.shape
        case _:
            raise TypeError(
                f"`{context}` requires a `BCOO`, `BCSR`, `BCOOLinearOperator`, "
                "`BCSRLinearOperator`, `SparseJacobianLinearOperator`, "
                "`SparseJacobianLinearOperatorColoring`, or `JacobianColoring`; "
                f"got {type(sparsity).__name__}."
            )

    return MatrixSparsity(rows, cols, (shape[0], shape[1]))
