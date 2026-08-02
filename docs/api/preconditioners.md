# Preconditioners

A preconditioner is *parametrized*, and each of its parameters can be fulfilled two
ways: with a fixed value, or with an object that derives one from the sparsity pattern
and is injected as a dependency. [`splineax.BlockJacobi`][] needs a block partition, so
`blocks=4` gives it one directly and
`blocks=MaximalCaptureBlockPartitioner()` injects something that works one out.

Providers are typed by minimal protocols, never by base classes: anything with a
`partition` method can be injected as a `BlockPartitioner`. See the
[iterative solvers guide](../guide/iterative.md) for the whole picture.

::: splineax.BlockJacobi

## Block partitions

::: splineax.MaximalCaptureBlockPartitioner

---

::: splineax.BlockPartition

---

::: splineax.BlockPartitioner

---

::: splineax.coverage

## System transforms

Transforms apply to the whole system rather than to one preconditioner, so they live on
[`splineax.PreconditionedIterativeLinearSolver`][]'s `transforms` argument.

::: splineax.ReverseCuthillMcKee

---

::: splineax.SymbolicTransform

---

::: splineax.NumericTransform

## Protocols

The three tiers a preconditioner passes through, mirroring the `solver` -> `scope` ->
`state` progression of the direct solvers' `factorize_symbolic`.

::: splineax.Preconditioner

---

::: splineax.SymbolicPreconditioner

---

::: splineax.NumericPreconditioner

---

::: splineax.Side

---

::: splineax.LeftPreconditioner

---

::: splineax.RightPreconditioner
