"""Tag objects for sparse operators.

These mark a property a solver may rely on, matching lineax's own repr-identified tags.
"""


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
