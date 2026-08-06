# Transforms

The building blocks [`PreconditionedIterativeSolver`][splineax.PreconditionedIterativeSolver]
composes. See the [preconditioning guide](../guide/preconditioning.md) for how they fit
together.

::: splineax.transforms.AggregationClustering

---

::: splineax.transforms.RuizEquilibration

---

::: splineax.transforms.BlockJacobi

---

::: splineax.transforms.compose_transforms

## Applied transforms

The concrete `(L, R)` pairs the transforms above build, reusable by any transform of
the same shape (a permutation or a scaling).

::: splineax.transforms.AppliedPermutation

---

::: splineax.transforms.AppliedScaling

## Protocols

Structural types, not base classes: implement these to write a new transform or
preconditioner.

::: splineax.transforms.SystemTransform

---

::: splineax.transforms.AnalyzedTransform

---

::: splineax.transforms.AppliedTransform

---

::: splineax.transforms.Preconditioner

---

::: splineax.transforms.AnalyzedPreconditioner
