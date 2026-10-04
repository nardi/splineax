import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Inexact
from lineax import AbstractLinearOperator, is_symmetric
from lineax._operator import _frozenset
from lineax._tags import transpose_tags

from ._operations import (
    register_sparse_operator,
    sparse_as_matrix,
    sparse_in_structure,
    sparse_mv,
    sparse_out_structure,
)
from ._tags import _ContentPatternTag, find_pattern_tag, sparse_indices_sorted


class BCSRLinearOperator(AbstractLinearOperator):
    """Wraps a `jax.experimental.sparse.BCSR` array into a linear operator.

    If the matrix has shape `(a, b)` then matrix-vector multiplication (`self.mv`) is
    defined in the usual way: as accepting a vector of shape `(b,)` and returning a
    vector of shape `(a,)`.
    """

    matrix: Inexact[BCSR, "a b"]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self, matrix: Inexact[BCSR, "a b"], tags: object | frozenset[object] = ()
    ):
        """**Arguments:**

        - `matrix`: a two-dimensional `BCSR` array. For an array with shape `(a, b)`
            then this operator can perform matrix-vector products on a vector of shape
            `(b,)` to return a vector of shape `(a,)`.
        - `tags`: any tags indicating whether this matrix has any particular properties,
            like symmetry or positive-definite-ness. Note that these properties are
            unchecked and you may get incorrect values elsewhere if these tags are
            wrong. A matrix carrying `indices_sorted` adds the `sparse_indices_sorted`
            tag automatically, so a solver can skip its sort.
        """
        if matrix.ndim != 2:
            raise ValueError(
                "`BCSRLinearOperator(matrix=...)` should be 2-dimensional."
            )
        if not jnp.issubdtype(matrix.dtype, jnp.inexact):
            matrix = BCSR(
                (matrix.data.astype(jnp.float32), matrix.indices, matrix.indptr),
                shape=matrix.shape,
                indices_sorted=matrix.indices_sorted,
            )
        self.matrix = matrix
        tags = _frozenset(tags)
        # A sorted matrix lets a solver skip its sort, so record that through the tag.
        if matrix.indices_sorted:
            tags = tags | {sparse_indices_sorted}
        self.tags = tags

    def mv(self, vector: Inexact[Array, " b"]) -> Inexact[Array, " a"]:
        return sparse_mv(self.matrix, vector)

    def as_matrix(self) -> Inexact[Array, "a b"]:
        return sparse_as_matrix(self.matrix)

    def transpose(self) -> "BCSRLinearOperator":
        if is_symmetric(self):
            return self
        # `BCSR.transpose` is not implemented in JAX; round-trip through `BCOO`.
        matrix_T_bcoo: BCOO = self.matrix.to_bcoo().T
        matrix_T = BCSR.from_bcoo(matrix_T_bcoo)
        # `BCSR.from_bcoo` always sorts, but (unlike `BCOO.from_bcoo`) never sets
        # `indices_sorted` on the result, so it comes back looking unsorted. Correcting
        # it here is what lets a solver skip re-sorting an already-transposed operator.
        matrix_T = BCSR(
            (matrix_T.data, matrix_T.indices, matrix_T.indptr),
            shape=matrix_T.shape,
            indices_sorted=True,
        )
        return BCSRLinearOperator(matrix_T, _row_major_pattern_tags(self.tags))

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return sparse_in_structure(self)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return sparse_out_structure(self)

    def _conj(self) -> "BCSRLinearOperator":
        matrix = BCSR(
            (self.matrix.data.conj(), self.matrix.indices, self.matrix.indptr),
            shape=self.matrix.shape,
        )
        return BCSRLinearOperator(matrix, self.tags)


def _row_major_pattern_tags(tags: frozenset[object]) -> frozenset[object]:
    """Transpose `tags` for a `BCSR` matrix, whose entries are always row-major sorted.

    The transposed pattern tag keeps the entry order of the original, which is
    column-major for the transposed matrix. A content tag is sorted to match the
    transposed `BCSR`. An identity tag carries no indices, so it needs no change.
    """
    transposed_tags = transpose_tags(tags)
    pattern_tag = find_pattern_tag(transposed_tags)
    if not isinstance(pattern_tag, _ContentPatternTag):
        return transposed_tags
    return (transposed_tags - {pattern_tag}) | {pattern_tag.row_major_sorted()}


register_sparse_operator(BCSRLinearOperator)
