"""Chains several transforms into one, applied in order.

Each stage's `(L, R)` pair composes into `L = L_n ... L_1` and `R = R_1 ... R_n` for
stages `1..n` applied in that order, which is why `transform_vector` folds forward
(applies stage 1 first) and `recover_solution` folds backward (applies stage n first).
`transpose` on the composed pair works out to transposing each stage in place, same
order: see `_ComposedAppliedTransform.transpose` for the derivation.
"""

from typing import NamedTuple

from jax.experimental.sparse import BCOO
from jaxtyping import Array

from splineax.operators._pattern import MatrixSparsity
from splineax.transforms._protocols import (
    AnalyzedTransform,
    AppliedTransform,
    SystemTransform,
)


class _ComposedAppliedTransform(NamedTuple):
    stages: tuple[AppliedTransform, ...]

    def transform_vector(self, b: Array) -> Array:
        for stage in self.stages:
            b = stage.transform_vector(b)
        return b

    def recover_solution(self, y: Array) -> Array:
        for stage in reversed(self.stages):
            y = stage.recover_solution(y)
        return y

    def transpose(self) -> "_ComposedAppliedTransform":
        # Write L = L_n...L_1, R = R_1...R_n. The transposed pair is (R^T, L^T), and
        # R^T = R_n^T...R_1^T, L^T = L_1^T...L_n^T. Each stage's own `.transpose()`
        # already gives (R_i^T, L_i^T) as its local pair, so folding those forward and
        # backward the same way `transform_vector`/`recover_solution` do reproduces
        # exactly R^T and L^T: no reordering needed, only a per-stage transpose.
        return _ComposedAppliedTransform(
            tuple(stage.transpose() for stage in self.stages)
        )

    def conj(self) -> "_ComposedAppliedTransform":
        # Conjugation distributes over a matrix product without reordering it, unlike
        # transpose, so this is the same elementwise map with no other change.
        return _ComposedAppliedTransform(tuple(stage.conj() for stage in self.stages))


class _ComposedAnalyzedTransform(NamedTuple):
    stages: tuple[AnalyzedTransform, ...]
    pattern: MatrixSparsity
    is_congruence: bool

    def analyze_numeric(self, matrix: BCOO) -> tuple[BCOO, AppliedTransform]:
        applied = []
        for stage in self.stages:
            matrix, one_applied = stage.analyze_numeric(matrix)
            applied.append(one_applied)
        return matrix, _ComposedAppliedTransform(tuple(applied))


class _ComposedTransform(NamedTuple):
    stages: tuple[SystemTransform, ...]

    def analyze_symbolic(self, pattern: MatrixSparsity) -> _ComposedAnalyzedTransform:
        plans = []
        for stage in self.stages:
            plan = stage.analyze_symbolic(pattern)
            plans.append(plan)
            pattern = plan.pattern
        is_congruence = all(plan.is_congruence for plan in plans)
        return _ComposedAnalyzedTransform(tuple(plans), pattern, is_congruence)


def compose_transforms(*transforms: SystemTransform) -> SystemTransform:
    """Chains `transforms`, applied in the order given.

    `A x = b` becomes `(L_n ... L_1 A R_1 ... R_n) y = (L_n ... L_1) b`, one transform
    at a time: each stage analyzes the sparsity pattern the previous one produced, and
    at the numeric step rewrites the matrix the previous one produced. With no
    arguments, the result is the identity transform.
    """
    return _ComposedTransform(transforms)
