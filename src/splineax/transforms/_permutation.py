"""`AppliedTransform` for a row/column permutation."""

from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Integer


class AppliedPermutation(NamedTuple):
    """`A' = A[row_perm][:, col_perm]`, `b' = b[row_perm]`, `x = y[argsort(col_perm)]`.

    `row_perm[k]` is the old row index that ends up at new row `k`, and likewise for
    `col_perm`. Built by any transform that reorders rows and columns, whether
    together (`row_perm is col_perm`, as `AggregationClustering` does) or
    independently, e.g. a bipartite matching that only permutes rows.

    When `row_perm == col_perm`, `A' = P^T A P` for the permutation matrix `P`, a
    congruence: it cannot change the matrix's conditioning, only where its entries
    sit.
    """

    row_perm: Integer[Array, " n"]
    col_perm: Integer[Array, " m"]

    def transform_vector(self, b: Array) -> Array:
        return b[self.row_perm]

    def recover_solution(self, y: Array) -> Array:
        return y[jnp.argsort(self.col_perm)]

    def transpose(self) -> "AppliedPermutation":
        return AppliedPermutation(self.col_perm, self.row_perm)

    def conj(self) -> "AppliedPermutation":
        # Permutation matrices are real, so conjugation is a no-op.
        return self
