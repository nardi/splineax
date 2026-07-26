"""Sparse matrix conversion helpers shared by `_klu.py`, `_pardiso.py`, and `_cudss.py`.

These solvers each accept the same handful of operator/sparsity types and need the
same COO/CSR array plumbing to feed their native libraries. Kept separate from
`_sparse.py`, which is the protocol and public `linear_solve` module and shouldn't
grow array plumbing.
"""

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from jax.typing import DTypeLike
from jaxtyping import Array, Inexact, Integer
from lineax import AbstractLinearOperator, materialise

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
)
from splineax.solvers._sparse import _Sparsity


def operator_to_sparse_matrix(
    operator: AbstractLinearOperator, *, error_prefix: str
) -> BCOO | BCSR:
    """Unwrap a sparse operator into its underlying `BCOO` or `BCSR` matrix.

    Materialises `SparseJacobianLinearOperator` first, then unwraps
    `BCOOLinearOperator`/`BCSRLinearOperator` as-is, keeping whichever native form
    (COO or CSR) the operator already stored. Callers convert to their own preferred
    form afterwards, see `KLU.init`/`Pardiso.init`.

    Args:
        operator: The operator to unwrap.
        error_prefix: Solver name to quote in the `TypeError` raised for an
                      unsupported operator, e.g. "`KLU`".
    """
    match operator:
        case SparseJacobianLinearOperator():
            return operator_to_sparse_matrix(
                materialise(operator), error_prefix=error_prefix
            )
        case BCSRLinearOperator(matrix):
            return matrix
        case BCOOLinearOperator(matrix):
            return matrix
        case _:
            raise TypeError(
                f"{error_prefix} requires a sparse operator backed by a `BCOO` or "
                "`BCSR` matrix (e.g. `splineax.BCOOLinearOperator` or "
                "`splineax.BCSRLinearOperator`), or a "
                f"`splineax.SparseJacobianLinearOperator`; got {type(operator).__name__}."
            )


def sparsity_to_coo_pattern(
    sparsity: _Sparsity, *, error_prefix: str
) -> tuple[
    Integer[Array, " nse"],
    Integer[Array, " nse"],
    tuple[int, ...],
    Inexact[Array, " nse"] | None,
]:
    """Read a sparsity pattern's row/column indices, shape, and values (if any).

    Values come back `None` for a bare coloring, since `JacobianColoring` and its
    relatives only carry index information, not a matrix. Every other input carries
    values through unchanged, for callers (like `Pardiso`) that use them as a
    representative sample for symbolic analysis.

    Args:
        sparsity: One of the types `factorize_symbolic` accepts: `BCOO`, `BCSR`,
                  `BCOOLinearOperator`, `BCSRLinearOperator`,
                  `SparseJacobianLinearOperator`,
                  `SparseJacobianLinearOperatorColoring`, or `JacobianColoring`.
        error_prefix: Method name to quote in the `TypeError` raised for an
                      unsupported input, e.g. "`KLU.factorize_symbolic`".
    """
    values: Inexact[Array, " nse"] | None = None
    match sparsity:
        case SparseJacobianLinearOperator(transposed=True):
            # The stored pattern describes the forward Jacobian. asdex emits `BCOO`
            # values in the pattern's index order and `BCOO.T` swaps the index
            # columns without reordering entries, so swapping rows and columns here
            # keeps the indices aligned with the values a caller later pairs them
            # with.
            pattern = sparsity.coloring.sparsity
            rows = jnp.asarray(pattern.cols, dtype=jnp.int32)
            cols = jnp.asarray(pattern.rows, dtype=jnp.int32)
            shape = pattern.shape[::-1]
        case SparseJacobianLinearOperator() | SparseJacobianLinearOperatorColoring():
            # Both hold the coloring one level in: the operator stores an
            # `asdex.ColoredPattern` whose `.sparsity` is the pattern, and the
            # operator coloring stores a `JacobianColoring` whose `.sparsity`
            # property returns the same pattern.
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
            values = bcoo.data
        case BCOOLinearOperator():
            bcoo = sparsity.matrix
            rows = bcoo.indices[:, 0].astype(jnp.int32)
            cols = bcoo.indices[:, 1].astype(jnp.int32)
            shape = bcoo.shape
            values = bcoo.data
        case BCSR():
            bcoo = sparsity.to_bcoo()
            rows = bcoo.indices[:, 0].astype(jnp.int32)
            cols = bcoo.indices[:, 1].astype(jnp.int32)
            shape = bcoo.shape
            values = bcoo.data
        case BCOO():
            rows = sparsity.indices[:, 0].astype(jnp.int32)
            cols = sparsity.indices[:, 1].astype(jnp.int32)
            shape = sparsity.shape
            values = sparsity.data
        case _:
            raise TypeError(
                f"{error_prefix} requires a `BCOO`, `BCSR`, `BCOOLinearOperator`, "
                "`BCSRLinearOperator`, `SparseJacobianLinearOperator`, "
                "`SparseJacobianLinearOperatorColoring`, or `JacobianColoring`; got "
                f"{type(sparsity).__name__}."
            )

    return rows, cols, tuple(shape), values


def csr_from_coo_pattern(
    rows: Integer[Array, " nse"],
    cols: Integer[Array, " nse"],
    shape: tuple[int, ...],
    values: Inexact[Array, " nse"] | None,
    *,
    dtype: DTypeLike,
) -> tuple[Integer[Array, " n+1"], Integer[Array, " nse"], Inexact[Array, " nse"]]:
    """Convert a COO `(row, col)` sparsity pattern to sorted CSR `(indptr, indices, values)`.

    `values` is optional because some `factorize_symbolic` inputs (a bare sparsity
    pattern, with no associated matrix) carry no numeric data. When omitted, a dummy
    `1.0` is used instead: the symbolic analysis this feeds only needs *some*
    representative values to run, not necessarily meaningful ones, and every later
    solve refactors with the real values from the operator being solved.

    `dtype` is the numeric dtype to cast (or fill) `values` to. `Pardiso` passes
    `jnp.float64` (its solver is double-precision only); `CuDSS` passes the
    pattern's own dtype, since it supports f32/f64/complex directly.
    """
    if values is None:
        values = jnp.ones(rows.shape[0], dtype=dtype)
    else:
        values = values.astype(dtype)
    bcsr = BCSR.from_bcoo(BCOO((values, jnp.stack([rows, cols], axis=1)), shape=shape))
    return bcsr.indptr.astype(jnp.int32), bcsr.indices.astype(jnp.int32), bcsr.data
