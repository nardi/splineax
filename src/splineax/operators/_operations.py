"""Shared, format-agnostic behaviour for the sparse linear operators.

`BCOOLinearOperator` and `BCSRLinearOperator` do not share a base class (beyond
`lineax.AbstractLinearOperator`); instead they *compose* their behaviour from the
helpers here. The identical methods (`mv`, `as_matrix`, `in_structure`,
`out_structure`) delegate to the `sparse_*` functions, and all of the Lineax
`singledispatch` registrations are installed by `register_sparse_operator`, which each
operator calls on its own class. The few format-specific operations (`transpose`,
`_as_bcoo`, `_conj`) remain methods on the concrete operators.

Those registrations come in two layers. `register_operator_tags` installs everything
answerable from an operator's tags and shape, and is shared with
`BlockDiagonalLinearOperator`, which has no `matrix` to speak of;
`register_sparse_operator` adds the band extraction (`diagonal`, `tridiagonal`) that
reads one.
"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Inexact
from lineax import (
    AbstractLinearOperator,
    conj,
    diagonal,
    has_unit_diagonal,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_tridiagonal,
    is_upper_triangular,
    linearise,
    materialise,
    tridiagonal,
)
from lineax._tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
)

from splineax.operators._sparse import SparseLinearOperator, TaggedLinearOperator


def _bcoo_band(matrix: BCOO, offset: int) -> Inexact[Array, " size"]:
    """Extracts the diagonal band at `offset` from a 2-dimensional `BCOO`, without
    materialising the dense matrix.

    Entries on the band satisfy `col - row == offset`. They are scattered (and summed,
    in case of duplicate indices) into a vector of length `min(rows, cols) - |offset|`,
    indexed by the row for the main/lower bands and by the column for the upper bands.
    Off-band entries are dropped by assigning them an out-of-range segment id.
    """
    rows = matrix.indices[:, 0]
    cols = matrix.indices[:, 1]
    nrows, ncols = matrix.shape
    size = min(nrows, ncols) - abs(offset)
    # Index along the band by the smaller of the two coordinates.
    segment = rows if offset >= 0 else cols
    on_band = (cols - rows) == offset
    data = jnp.where(on_band, matrix.data, 0)
    segment = jnp.where(on_band, segment, size)
    return jax.ops.segment_sum(data, segment, num_segments=size)


# Shared operator methods (delegated to from each operator).


def sparse_mv(
    matrix: Inexact[BCOO | BCSR, "a b"], vector: Inexact[Array, " b"]
) -> Inexact[Array, " a"]:
    return matrix @ vector


def sparse_as_matrix(matrix: Inexact[BCOO | BCSR, "a b"]) -> Inexact[Array, "a b"]:
    return matrix.todense()


def sparse_in_structure(
    operator: SparseLinearOperator[BCOO | BCSR],
) -> jax.ShapeDtypeStruct:
    matrix = operator.matrix
    _, in_size = matrix.shape
    return jax.ShapeDtypeStruct(shape=(in_size,), dtype=matrix.dtype)


def sparse_out_structure(
    operator: SparseLinearOperator[BCOO | BCSR],
) -> jax.ShapeDtypeStruct:
    matrix = operator.matrix
    out_size, _ = matrix.shape
    return jax.ShapeDtypeStruct(shape=(out_size,), dtype=matrix.dtype)


# Shared `singledispatch` implementations. These mirror `lineax.MatrixLinearOperator`.


def _has_real_dtype(operator: TaggedLinearOperator) -> bool:
    # Reads the structures through the operator's own methods rather than
    # `sparse_in_structure`, so that this works for storage-free operators too.
    leaves = jtu.tree_leaves((operator.in_structure(), operator.out_structure()))
    dtype = jnp.result_type(*leaves)
    return not jnp.issubdtype(dtype, jnp.complexfloating)


def _is_symmetric(operator: TaggedLinearOperator) -> bool:
    if symmetric_tag in operator.tags or diagonal_tag in operator.tags:
        return True
    if (
        positive_semidefinite_tag in operator.tags
        or negative_semidefinite_tag in operator.tags
    ):
        return _has_real_dtype(operator)
    return False


def _is_diagonal(operator: TaggedLinearOperator) -> bool:
    return diagonal_tag in operator.tags or (
        operator.in_size() == 1 and operator.out_size() == 1
    )


def _is_tridiagonal(operator: TaggedLinearOperator) -> bool:
    return tridiagonal_tag in operator.tags or diagonal_tag in operator.tags


def _has_unit_diagonal(operator: TaggedLinearOperator) -> bool:
    return unit_diagonal_tag in operator.tags


def _is_lower_triangular(operator: TaggedLinearOperator) -> bool:
    return lower_triangular_tag in operator.tags


def _is_upper_triangular(operator: TaggedLinearOperator) -> bool:
    return upper_triangular_tag in operator.tags


def _is_positive_semidefinite(operator: TaggedLinearOperator) -> bool:
    return positive_semidefinite_tag in operator.tags


def _is_negative_semidefinite(operator: TaggedLinearOperator) -> bool:
    return negative_semidefinite_tag in operator.tags


# Used for both `linearise` and `materialise`: keep the operator sparse.
def _identity(obj: AbstractLinearOperator) -> AbstractLinearOperator:
    return obj


def _as_bcoo(matrix: BCOO | BCSR) -> BCOO:
    if isinstance(matrix, BCSR):
        return matrix.to_bcoo()

    return matrix


def _diagonal(operator: SparseLinearOperator[BCOO | BCSR]) -> Inexact[Array, " size"]:
    matrix = _as_bcoo(operator.matrix)
    return _bcoo_band(matrix, 0)


def _tridiagonal(
    operator: SparseLinearOperator[BCOO | BCSR],
) -> tuple[
    Inexact[Array, " size"], Inexact[Array, " size-1"], Inexact[Array, " size-1"]
]:
    matrix = _as_bcoo(operator.matrix)
    return _bcoo_band(matrix, 0), _bcoo_band(matrix, -1), _bcoo_band(matrix, 1)


def _conj(operator: TaggedLinearOperator) -> AbstractLinearOperator:
    return operator._conj()


def register_operator_tags(cls: type[TaggedLinearOperator]) -> None:
    """Registers the Lineax `singledispatch` operations that every operator here shares.

    These are exactly the ones answerable from `tags` and the operator's shape, plus the
    two that must stay away from a dense fallback. Split out from
    `register_sparse_operator` because `BlockDiagonalLinearOperator` stores a stack of
    dense blocks rather than a `matrix`, so it can take all of this but none of the
    band extraction below.

    Registering `linearise` and `materialise` as the identity is load-bearing rather
    than an optimisation. Lineax's default rules would densify the operator into an
    `n x n` array -- and `preconditioner_and_y0` calls `linearise` on whatever is passed
    as `options["preconditioner"]`, so without this every preconditioned solve would
    quietly allocate a dense matrix of the very size the sparse operator exists to
    avoid.
    """
    is_symmetric.register(cls, _is_symmetric)
    is_diagonal.register(cls, _is_diagonal)
    is_tridiagonal.register(cls, _is_tridiagonal)
    has_unit_diagonal.register(cls, _has_unit_diagonal)
    is_lower_triangular.register(cls, _is_lower_triangular)
    is_upper_triangular.register(cls, _is_upper_triangular)
    is_positive_semidefinite.register(cls, _is_positive_semidefinite)
    is_negative_semidefinite.register(cls, _is_negative_semidefinite)
    linearise.register(cls, _identity)
    materialise.register(cls, _identity)
    conj.register(cls, _conj)


def register_sparse_operator(cls: type[SparseLinearOperator[BCOO | BCSR]]) -> None:
    """Registers all of Lineax's `singledispatch` operations for a sparse operator
    `cls`, so that it works with Lineax's solvers. Shared by both operators in lieu of
    a common base class.
    """
    register_operator_tags(cls)
    # Band extraction, which needs the operator's `matrix`.
    diagonal.register(cls, _diagonal)
    tridiagonal.register(cls, _tridiagonal)
