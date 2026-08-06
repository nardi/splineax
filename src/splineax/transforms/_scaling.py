"""`AppliedTransform` for a row/column diagonal scaling."""

from typing import NamedTuple

from jaxtyping import Array, Inexact


class AppliedScaling(NamedTuple):
    """`A' = diag(r) A diag(c)`, `b' = r * b`, `x = c * y`.

    Built by any transform that rescales rows and columns, whether together
    (`r is c`, a congruence) or independently, as `RuizEquilibration` does by
    default.
    """

    r: Inexact[Array, " n"]
    c: Inexact[Array, " m"]

    def transform_vector(self, b: Array) -> Array:
        return self.r * b

    def recover_solution(self, y: Array) -> Array:
        return self.c * y

    def transpose(self) -> "AppliedScaling":
        # diag(r) and diag(c) are their own transpose, so (R^T, L^T) is just the pair
        # swapped, same as `AppliedPermutation.transpose`.
        return AppliedScaling(self.c, self.r)

    def conj(self) -> "AppliedScaling":
        return AppliedScaling(self.r.conj(), self.c.conj())
