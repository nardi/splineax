"""Ruiz equilibration, a numeric `SystemTransform`.

Repeatedly rescales rows and columns so every row and column maximum approaches 1 in
magnitude. Two `segment_max` reductions and two gathers per iteration, all GPU
friendly, and the pattern never changes, only the values, so this is a numeric
transform: `analyze_symbolic` just remembers the pattern for the reductions to use.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Array, Integer

from splineax.operators._pattern import MatrixSparsity
from splineax.transforms._protocols import AppliedTransform
from splineax.transforms._scaling import AppliedScaling


class _EquilibrationPlan(NamedTuple):
    rows: Integer[Array, " nse"]
    cols: Integer[Array, " nse"]
    shape: tuple[int, int]
    iterations: int
    symmetric: bool
    is_congruence: bool

    @property
    def pattern(self) -> MatrixSparsity:
        # Only the values change; the pattern this stage hands to the next one is
        # exactly the pattern it was analyzed against.
        return MatrixSparsity(self.rows, self.cols, self.shape)

    def analyze_numeric(self, matrix: BCOO) -> tuple[BCOO, AppliedTransform]:
        n, m = self.shape
        magnitude = jnp.abs(matrix.data)
        r = jnp.ones(n, dtype=magnitude.dtype)
        c = jnp.ones(m, dtype=magnitude.dtype)

        for _ in range(self.iterations):
            row_max = jax.ops.segment_max(magnitude, self.rows, num_segments=n)
            col_max = jax.ops.segment_max(magnitude, self.cols, num_segments=m)
            if self.symmetric:
                row_max = col_max = jnp.maximum(row_max, col_max)
            row_scale = jnp.where(row_max > 0, jax.lax.rsqrt(row_max), 1.0)
            col_scale = jnp.where(col_max > 0, jax.lax.rsqrt(col_max), 1.0)
            r = r * row_scale
            c = c * col_scale
            magnitude = magnitude * row_scale[self.rows] * col_scale[self.cols]

        new_data = matrix.data * r[self.rows] * c[self.cols]
        new_matrix = BCOO(
            (new_data, jnp.stack([self.rows, self.cols], axis=1)), shape=self.shape
        )
        return new_matrix, AppliedScaling(r, c)


class RuizEquilibration(eqx.Module):
    """Numeric transform that rescales rows and columns towards unit maximum magnitude.

    Leaves the sparsity pattern untouched: only the values change. Run after a
    symbolic transform (e.g. `AggregationClustering`) so the values it sees are
    already in their final positions.
    """

    iterations: int = eqx.field(static=True, default=5)
    symmetric: bool = eqx.field(static=True, default=False)

    def analyze_symbolic(self, pattern: MatrixSparsity) -> _EquilibrationPlan:
        is_congruence = self.symmetric and pattern.shape[0] == pattern.shape[1]
        return _EquilibrationPlan(
            pattern.rows,
            pattern.cols,
            pattern.shape,
            self.iterations,
            self.symmetric,
            is_congruence,
        )


RuizEquilibration.__init__.__doc__ = """**Arguments:**

- `iterations`: number of scaling rounds. 5 is enough for row and column maxima to
    settle close to 1 for most matrices.
- `symmetric`: scale rows and columns by the same factor (`r is c`), so the transform
    is a congruence and preserves symmetry and definiteness. Needed to keep a
    `positive_semidefinite_tag` operator usable with `CG`. Off by default, which
    scales rows and columns independently for a better-conditioned result on a general
    (non-symmetric) matrix.
"""
