# Debugging solves with a solve trace

A sparse direct solver does a lot of work you never see: it analyzes a sparsity pattern,
factorizes it, reuses or rebuilds that factorization as the values change, solves against
it, and (with [iterative refinement](solvers.md)) corrects the solution step by step. When a
solve is slower than expected, refactors when you thought it would reuse, or refines for more
steps than it should, the [stateful API](stateful.md) hides exactly the information you need.

`splineax.solve_trace` is a context manager that records those operations, in order, for
debugging. Enable it around your solves and read the log afterwards:

```python
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

import splineax as splx

jax.config.update("jax_enable_x64", True)

dense = jnp.array(
    [
        [10.0, 2.0, 0.0, 0.0],
        [3.0, 14.0, 5.0, 0.0],
        [0.0, 6.0, 18.0, 9.0],
        [0.0, 0.0, 1.0, 12.0],
    ]
)
b1 = jnp.array([1.0, 2.0, 3.0, 4.0])
b2 = b1[::-1]

# Two matrices that share a sparsity pattern, tagged so `update` can reuse the analysis.
tag = splx.sparsity_pattern_tag(BCOO.fromdense(dense))
first = splx.BCOOLinearOperator(BCOO.fromdense(dense), tags=tag)
second = splx.BCOOLinearOperator(BCOO.fromdense(2.0 * dense), tags=tag)
solver = splx.KLU()

with splx.solve_trace() as trace:
    solution, state = splx.linear_solve(first, b1, solver)
    solution, state = splx.linear_solve(second, b2, solver, state=state)
    state.release()

ordered = sorted(trace.records, key=lambda record: record.order)
assert [record.operation for record in ordered] == [
    "init", "analyze", "factor", "compute", "solve_with_numeric", "track",
    "update", "refactor", "compute", "solve_with_numeric", "track",
    "release", "free_numeric", "free_symbolic",
]
```

Each line is one **operation**. A *generic* operation is one of the stateful-API steps
(`init`, `init_symbolic`, `update`, `compute`, `track`, `release`), written bare. Nested under
it are the *solver-specific* operations it ran, written `SolverType.function`. The first solve
analyzes and factorizes; the second `update` shares the pattern, so it **reuses** the analysis
and only refactors. Printing the trace shows this as an indented, coloured tree, with
factorizations built anew and factorizations reused in different colours:

```{.python continuation}
print(trace)
```

```text
solve trace  created  reused
sequence 0
  init[shape=(4, 4), nse=10, sparsity_hash=0x60347]
    KLU.analyze
    KLU.factor
  compute
    KLU.solve_with_numeric
  track
  update[shape=(4, 4), nse=10, sparsity_hash=0x60347] => (outcome=reused, reason="Identical sparsity tag")
    KLU.refactor => (reused=True, rcond=8.329e-01, reason="Pivots stable: no error and rcond > 1e-08")
  compute
    KLU.solve_with_numeric
  track
  release
    KLU.free_numeric
    KLU.free_symbolic
```

Each operation is `operation[inputs] => (outputs)`. The `sparsity_hash` identifies the
pattern: two operators the solver treats as one pattern share it, so the equal hashes above
confirm the reuse (its value is arbitrary and changes between runs). Each `init` (or
`init_symbolic`) starts a new **state-sequence**; `trace.records` is the flat log and
`trace.sequences` slices it into one list per sequence:

```{.python continuation}
assert len(trace.sequences) == 1
```

## When reuse is lost

If you `update` with an operator whose sparsity pattern differs from the state's, the analysis
cannot be reused and the solver rebuilds it from scratch. The trace makes that explicit and
motivated: the `update` records `outcome="rebuilt"` and a `reason` saying why, and a fresh
`KLU.analyze` follows. Every choice carries a `reason` like this — why an analysis was rebuilt
rather than reused, or why a fresh factorization was taken instead of a refactor.

```{.python continuation}
other = BCOO.fromdense(dense[::-1])  # a different sparsity pattern
changed = splx.BCOOLinearOperator(other, tags=splx.sparsity_pattern_tag(other))
with splx.solve_trace() as trace:
    solution, state = splx.linear_solve(first, b1, solver)
    solution, state = splx.linear_solve(changed, b1, solver, state=state)
    state.release()

update = next(record for record in trace.records if record.operation == "update")
assert update.outputs["outcome"] == "rebuilt"
assert update.outputs["reason"] == "Different sparsity tag"
```

## What each backend records

- `KLU` records `KLU.analyze`, `KLU.factor`, `KLU.refactor` (with the reciprocal condition
  estimate and whether the factorization was reused), `KLU.solve_with_numeric` (or
  `KLU.tsolve_with_numeric` when transposed), and `KLU.free_numeric` / `KLU.free_symbolic`
  under `release`. `Pardiso` is analogous, with `Pardiso.reanalyze` and pivot-stability flags.
- `Spsolve` factors and solves in one fused call, so under `compute` it records a single
  `Spsolve.spsolve`; its reuse API is a set of no-ops.
- `IterativeRefinement` records `IterativeRefinement.refine_start`, a `refine_step` per
  correction (with the step index and residual norm), and a `refine_result` (with the final
  residual and whether it converged), nested under the `compute` alongside the inner solver's
  per-step solves. `AutoSparseLinearSolver` forwards to whichever backend it picked.

## Limitations

- **Tracing is applied at trace time.** Only code traced while a `solve_trace` block is open
  is recorded. A function compiled by `jax.jit` outside the block emits nothing when later
  called inside it — trace it fresh, or call it eagerly. Eager (non-jit) code always
  re-traces, so it just works.
- **Full detail needs `splineax.linear_solve`.** While a trace is open, `splineax.linear_solve`
  runs the solver's `compute` directly so the `compute` operations (the triangular solve and
  the iterative-refinement steps) are captured. A solve issued through `lineax.linear_solve`
  (or a `stateful_solve_transform`) still records the generic and structural operations
  (`init`, `update`, `analyze`, `factor`, `refactor`, ...), but not the ones inside `compute`.
- **Not for `vmap`/`grad`.** The log is appended through `io_callback`s, which do not compose
  with `jax.vmap` and may conflict with reverse-mode autodiff. Trace plain solves rather than
  solves under `vmap` or `grad`. The callbacks are unordered, but each record carries a
  trace-time order index, so the printed tree stays in program order (sort `trace.records` by
  `record.order`).
- **Zero cost when off.** With no trace open nothing is emitted into the traced program, so
  leaving the instrumentation in place costs nothing.
