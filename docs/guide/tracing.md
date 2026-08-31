# Debugging solves with a solve trace

A sparse direct solver does a lot of work you never see: it analyzes a sparsity pattern,
factorizes it, reuses or rebuilds that factorization as the values change, solves against
it, and (with [iterative refinement](solvers.md)) corrects the solution step by step. When a
solve is slower than expected, refactors when you thought it would reuse, or refines for more
steps than it should, the [stateful API](stateful.md) hides exactly the information you need.

`splineax.solve_trace` is a context manager that records those actions, in order, for
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

actions = [record.action for record in trace.records]
assert actions == [
    "init", "analyze", "factor", "solve", "track",
    "update", "refactor", "solve", "track", "release",
]
```

The first solve analyzes and factorizes; the second `update` shares the pattern, so it
**reuses** the analysis and only refactors. Printing the trace shows this as an indented,
coloured tree, with factorizations built anew and factorizations reused in different colours:

```{.python continuation}
print(trace)
```

```text
solve trace  created  reused
sequence 0 [klu] shape=(4, 4)
  init [klu]  shape=(4, 4)
  analyze [klu]  shape=(4, 4) nnz=10
  factor [klu]  shape=(4, 4) nnz=10
  solve [klu]  shape=(4, 4) transposed=False
  track [klu]  shape=(4, 4)
  update [klu]  outcome=reused shape=(4, 4)
  refactor [klu]  nnz=10 reused=True rcond=8.329e-01
  solve [klu]  shape=(4, 4) transposed=False
  track [klu]  shape=(4, 4)
  release [klu]  shape=(4, 4)
```

Each `init` (or `init_symbolic`) starts a new **state-sequence**. `trace.records` is the flat
log, and `trace.sequences` slices it into one list per sequence:

```{.python continuation}
assert len(trace.sequences) == 1
```

## When reuse is lost

If you `update` with an operator whose sparsity pattern differs from the state's, the analysis
cannot be reused and the solver rebuilds it from scratch. The trace makes that explicit: the
`update` is recorded with `outcome="rebuilt"` and a `note`, and a fresh `analyze` follows.

```{.python continuation}
changed = splx.BCOOLinearOperator(BCOO.fromdense(dense[::-1]))  # a different pattern
with splx.solve_trace() as trace:
    solution, state = splx.linear_solve(first, b1, solver)
    solution, state = splx.linear_solve(changed, b1, solver, state=state)
    state.release()

update = next(record for record in trace.records if record.action == "update")
assert update.outcome == "rebuilt"
assert update.note == "sparsity pattern changed"
```

## What each backend records

- `KLU` and `Pardiso` record `analyze`, `factor`, `refactor` (with the reciprocal condition
  estimate or pivot-stability flags and whether the factorization was reused), `solve`,
  `track`, and `release`.
- `Spsolve` factors and solves in one fused call, so it records an `init` boundary and a
  fused `solve`; its reuse API is a set of no-ops.
- `IterativeRefinement` records `ir_start`, an `ir_step` per correction (with the step index
  and residual norm), and an `ir_result` (with the final residual and whether it converged),
  nested under the solve they belong to. `AutoSparseLinearSolver` forwards to whichever
  backend it picked, so the log carries that backend's events.

## Limitations

- **Tracing is applied at trace time.** Only code traced while a `solve_trace` block is open
  is recorded. A function compiled by `jax.jit` outside the block emits nothing when later
  called inside it — trace it fresh, or call it eagerly. Eager (non-jit) code always
  re-traces, so it just works.
- **Full detail needs `splineax.linear_solve`.** While a trace is open, `splineax.linear_solve`
  runs the solver's `compute` directly so the `solve` and iterative-refinement steps are
  captured. A solve issued through `lineax.linear_solve` (or a `stateful_solve_transform`) still
  records the structural events (`analyze`, `factor`, `refactor`, `update`, ...), but not the
  in-`compute` steps.
- **Not for `vmap`/`grad`.** The log is appended through `io_callback`s, which do not compose
  with `jax.vmap` and may conflict with reverse-mode autodiff. Trace plain solves rather than
  solves under `vmap` or `grad`. Under `jax.jit` the structural events keep their order and the
  in-`compute` events keep theirs, but the two groups may interleave; run eagerly for an exact
  linear order.
- **Zero cost when off.** With no trace open nothing is emitted into the traced program, so
  leaving the instrumentation in place costs nothing.
