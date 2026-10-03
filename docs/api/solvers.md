# Solvers

::: splineax.linear_solve

---

::: splineax.AutoSparseLinearSolver

---

::: splineax.KLU

---

::: splineax.Pardiso

---

::: splineax.CuDSS

---

::: splineax.Spsolve

---

::: splineax.HybridDirectIterative

---

::: splineax.RichardsonOptions

---

::: splineax.GMRESOptions

---

::: splineax.BiCGStabOptions

---

::: splineax.CGOptions

---

::: splineax.ReuseOptions

---

::: splineax.HybridSettings

---

::: splineax.HybridState

---

::: splineax.Richardson

---

::: splineax.IterativeRefinement

---

::: splineax.IterativeRefinementSettings

---

::: splineax.solvers.ReorderingScheme

---

::: splineax.solvers.CuDSSReordering

---

::: splineax.solvers.CuDSSMemory

## Stateful solve transform

Threads a solver state through a function's `lineax.linear_solve` calls, so its solves reuse
a factorization. See [Transforming existing Lineax code](../guide/transform.md).

::: splineax.stateful_solve_transform

## Sparsity tags

::: splineax.sparsity_pattern_tag

---

::: splineax.sparse_indices_sorted

## Protocols

Solvers structurally satisfy the `SparseLinearSolver` protocol, which extends the
solver-agnostic `StatefulSolver`. The stateful reuse API is described in
[Stateful solves](../guide/stateful.md).

::: splineax.SparseLinearSolver

---

::: splineax.StatefulSolver

---

::: splineax.TrackingSolverState

---

::: splineax.PerformanceWarning
