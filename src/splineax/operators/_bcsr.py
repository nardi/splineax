import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
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
            wrong.
        """
        if matrix.ndim != 2:
            raise ValueError(
                "`BCSRLinearOperator(matrix=...)` should be 2-dimensional."
            )
        if not jnp.issubdtype(matrix.dtype, jnp.inexact):
            matrix = BCSR(
                (matrix.data.astype(jnp.float32), matrix.indices, matrix.indptr),
                shape=matrix.shape,
            )
        self.matrix = matrix
        self.tags = tags if isinstance(tags, frozenset) else frozenset([tags])

    def mv(self, vector: Inexact[Array, " b"]) -> Inexact[Array, " a"]:
        return sparse_mv(self.matrix, vector)

    def as_matrix(self) -> Inexact[Array, "a b"]:
        return sparse_as_matrix(self.matrix)

    def transpose(self) -> "BCSRLinearOperator":
        if is_symmetric(self):
            return self
        # `BCSR.transpose` is not implemented in JAX; round-trip through `BCOO`.
        matrix_T_bcoo: BCOO = self.matrix.to_bcoo().T  # type: ignore
        matrix_T = BCSR.from_bcoo(matrix_T_bcoo)
        return BCSRLinearOperator(matrix_T, transpose_tags(self.tags))

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


register_sparse_operator(BCSRLinearOperator)
