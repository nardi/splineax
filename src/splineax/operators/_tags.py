"""Tag objects for sparse operators.

These mark a property a solver may rely on, matching lineax's own repr-identified tags.
"""

import secrets

import numpy as np
from asdex import ColoredPattern


class _HasRepr:
    """A tag object whose only content is its repr, matching lineax's own tags."""

    def __init__(self, string: str) -> None:
        self.string = string

    def __repr__(self) -> str:
        return self.string


sparse_indices_sorted = _HasRepr("sparse_indices_sorted")
"""One global assertion that an operator's indices are already row-major sorted, so
`Pardiso` and `Spsolve` may skip the sort they would otherwise do in `init`.

`BCOOLinearOperator` and `BCSRLinearOperator` add this automatically when the matrix they
wrap already carries `indices_sorted`.
"""


class _ContentPatternTag:
    """A sparsity-pattern tag identified by the content of its index arrays.

    Two tags are equal when their indices match exactly, so operators built with the same
    pattern reuse each other's factorization even when tagged separately. This follows
    asdex's `_HashableEntries`, which carries concrete index arrays as hashable static aux
    data.
    """

    __slots__ = ("_hash", "indices", "shape")

    def __init__(self, indices: np.ndarray, shape: tuple[int, ...]) -> None:
        self.indices = indices
        self.shape = shape
        self._hash: int | None = None

    def __hash__(self) -> int:
        # Content hashing is O(nnz), so compute it lazily and cache it. A frozenset
        # hashes the tag every time it is built.
        if self._hash is None:
            self._hash = hash(
                (self.indices.tobytes(), str(self.indices.dtype), self.shape)
            )
        return self._hash

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, _ContentPatternTag):
            return NotImplemented
        # The dtype is compared alongside the contents to stay consistent with the hash.
        return (
            self.shape == other.shape
            and self.indices.dtype == other.indices.dtype
            and np.array_equal(self.indices, other.indices)
        )


class _IdentityPatternTag:
    """A sparsity-pattern tag identified by a random id, for use under jit.

    A traced index array cannot be hashed, so this stands in for the content tag. Two
    instances differ, and the same instance threaded onto several operators marks them as
    sharing a pattern. The random id keeps the tag hashable and stable across pytree
    flatten and unflatten, where a bare `object()` identity would not survive.
    """

    __slots__ = ("_id",)

    def __init__(self) -> None:
        self._id = secrets.randbits(128)

    def __hash__(self) -> int:
        return hash(self._id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _IdentityPatternTag):
            return NotImplemented
        return self._id == other._id


def coloring_index_array(
    coloring: ColoredPattern,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Read a coloring's COO index array and shape as concrete numpy data.

    The pattern comes from the precomputed asdex coloring, whose row and column indices are
    always concrete, so this never sees a traced array.
    """
    sparsity = coloring.sparsity
    indices = np.stack([np.asarray(sparsity.rows), np.asarray(sparsity.cols)], axis=1)
    return indices, tuple(sparsity.shape)


def sparsity_tag_from_coloring(coloring: ColoredPattern) -> _ContentPatternTag:
    """Build a content pattern tag from a coloring's sparsity.

    A `SparseJacobianLinearOperator` uses this to carry a tag for the fixed pattern its
    coloring describes, so operators built for that pattern reuse each other's factorization.
    """
    indices, shape = coloring_index_array(coloring)
    return _ContentPatternTag(indices, shape)
