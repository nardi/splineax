import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Array, Inexact
from lineax import AbstractLinearOperator, is_symmetric
from lineax._tags import transpose_tags

from ._operations import (
    register_sparse_operator,
    sparse_as_matrix,
    sparse_in_structure,
    sparse_mv,
    sparse_out_structure,
)
from ._tags import sparse_indices_sorted


class BCOOLinearOperator(AbstractLinearOperator):
    """Wraps a `jax.experimental.sparse.BCOO` array into a linear operator.

    If the matrix has shape `(a, b)` then matrix-vector multiplication (`self.mv`) is
    defined in the usual way: as accepting a vector of shape `(b,)` and returning a
    vector of shape `(a,)`.
    """

    matrix: Inexact[BCOO, "a b"]
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self, matrix: Inexact[BCOO, "a b"], tags: object | frozenset[object] = ()
    ):
        """**Arguments:**

        - `matrix`: a two-dimensional `BCOO` array. For an array with shape `(a, b)`
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
                "`BCOOLinearOperator(matrix=...)` should be 2-dimensional."
            )
        if not jnp.issubdtype(matrix.dtype, jnp.inexact):
            matrix = BCOO(
                (matrix.data.astype(float), matrix.indices),
                shape=matrix.shape,
                indices_sorted=matrix.indices_sorted,
            )
        self.matrix = matrix
        tags = tags if isinstance(tags, frozenset) else frozenset([tags])
        # A sorted matrix lets a solver skip its sort, so record that through the tag.
        if matrix.indices_sorted:
            tags = tags | {sparse_indices_sorted}
        self.tags = tags

    def mv(self, vector: Inexact[Array, " b"]) -> Inexact[Array, " a"]:
        return sparse_mv(self.matrix, vector)

    def as_matrix(self) -> Inexact[Array, "a b"]:
        return sparse_as_matrix(self.matrix)

    def transpose(self) -> "BCOOLinearOperator":
        if is_symmetric(self):
            return self
        matrix_T: BCOO = self.matrix.T
        return BCOOLinearOperator(matrix_T, transpose_tags(self.tags))

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return sparse_in_structure(self)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return sparse_out_structure(self)

    def _conj(self) -> "BCOOLinearOperator":
        matrix = BCOO(
            (self.matrix.data.conj(), self.matrix.indices), shape=self.matrix.shape
        )
        return BCOOLinearOperator(matrix, self.tags)


register_sparse_operator(BCOOLinearOperator)
