# The solve profile

When using a sparse direct solver, every linear solve is part of a larger chain of
operations. The solver analyzes a matrix's sparsity pattern, factorizes it, reuses or
rebuilds that factorization as the values change, solves against it, and (with
[iterative refinement](solvers.md)) corrects the solution step by step. When a solve is
slower than expected, it might be the case that some work is unexpectedly being repeated.
For example, the solver might decide a new factorization needs to be rebuilt, while
actually it could reuse the factorization it already had. The [stateful API](stateful.md)
hides exactly these decisions.

The solve profiler exists to make these decisions visible. The main interface is
`splineax.SolveProfile`, a context manager that records all solver operations in an
ordered log. Create one with `create_solve_profile`, activate it using a `with` block,
and inspect the log afterwards:

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

profile = splx.create_solve_profile()
with profile:
    solution, state = splx.linear_solve(first, b1, solver)
    solution, state = splx.linear_solve(second, b2, solver, state=state)
    state.release()

print(profile)
```

```text
solve profile
sequence 0
  init[shape=(4, 4), nse=10, sparsity_hash=0x60347]
    KLU.analyze
    KLU.factor
  compute
    KLU.solve_with_numeric => (rebuild="none")
  track
  update[shape=(4, 4), nse=10, sparsity_hash=0x60347] => (outcome=reused, reason="Identical sparsity tag")
    KLU.refactor => (reused=True, rebuild="none", rcond=8.329e-01, reason="Pivots stable: no error and rcond > 1e-08")
  compute
    KLU.solve_with_numeric => (rebuild="none")
  track
  release
    KLU.free_numeric
    KLU.free_symbolic
```

Each operation is formatted as `operation[inputs] => (outputs)`. Each `init` (or
`init_symbolic`) starts a new **state-sequence**, a number of operations that build on the
same base factorization. `sparsity_hash` identifies a unique sparsity pattern: two
operators with the same hash should reuse a single factorization (note that the value of
the hash is arbitrary and may change between runs). `profile.records` is a flat log of
operations, and `profile.sequences` slices it into one list per sequence:

```{.python continuation}
assert len(profile.sequences) == 1
```

A *generic* operation is one of the stateful-API steps (`init`, `init_symbolic`, `update`,
`compute`, `track`, `release`), written bare. Nested under it are the *solver-specific*
operations it ran, written `SolverType.function`. The first solve analyzes and
factorizes. The second `update` shares the pattern, so it **reuses** the analysis and only
refactors. When printed to a terminal the tree is colored, with factorizations built anew
and factorizations reused in different colors.

## Profiling and JIT

When profiling a compiled function, it is important that the initial tracing of the
function happens under a solve profile as well. This is because the log callbacks are
triggered by Python control flow and so are only seen by the JAX tracers when a profile
is active.

In this case, it is recommended to use the `splineax.profile_solves` decorator. If you
apply it after `jax.jit`/`equinox.filter_jit`, the function in question will always be
executed under a profile, and the profile object will be returned as a second return
value:

```{.python notest}
import equinox as eqx

@splx.profile_solves
@eqx.filter_jit
def solve(operator, vector, solver):
    solution, state = splx.linear_solve(operator, vector, solver)
    state.release()
    return solution

solution, profile = solve(operator, vector, solver)
```

One edge remains, and it comes from JAX compiling once per shape rather than from
anything `profile_solves` gets wrong. If the first call for a shape happens with
`enabled=False`, or you call the undecorated jitted function directly instead of going
through `profile_solves`, that shape's compiled executable never gets profiling hooks, and
no later call for it will ever be profiled, even with `enabled=True`.

## Reuse mechanics

If you `update` a state with an operator that has a different sparsity pattern, the
symbolic analysis cannot be reused and the solver rebuilds it from scratch. The profile
will show this happened and why: the `update` operation records `outcome="rebuilt"` and a
`reason`, followed by a fresh `KLU.analyze`. Every choice carries a `reason` like this,
saying why an analysis was rebuilt rather than reused, or why a fresh factorization was
taken instead of a refactor.

```{.python continuation}
other = BCOO.fromdense(dense[::-1])  # a different sparsity pattern
changed = splx.BCOOLinearOperator(other, tags=splx.sparsity_pattern_tag(other))
profile = splx.create_solve_profile()
with profile:
    solution, state = splx.linear_solve(first, b1, solver)
    solution, state = splx.linear_solve(changed, b1, solver, state=state)
    state.release()

update = next(record for record in profile.records if record.operation == "update")
assert update.outputs["outcome"] == "rebuilt"
assert update.outputs["reason"] == "Different sparsity tag"
print(profile)
```

```text
solve profile
sequence 0
  init[shape=(4, 4), nse=10, sparsity_hash=0x60347]
    KLU.analyze
    KLU.factor
  compute
    KLU.solve_with_numeric => (rebuild="none")
  track
  update[shape=(4, 4), nse=10, sparsity_hash=0x9e04a] => (outcome=rebuilt, reason="Different sparsity tag")
    KLU.analyze
    KLU.factor
  compute
    KLU.solve_with_numeric => (rebuild="none")
  track
  release
    KLU.free_numeric
    KLU.free_symbolic
```

## Limitations

- **Full detail needs `splineax.linear_solve`.** While a profile is open,
  `splineax.linear_solve` runs the solver's `compute` directly so the `compute`
  operations (the triangular solve and the iterative-refinement steps) are captured. A
  solve issued through `lineax.linear_solve` (or a `stateful_solve_transform`) still
  records the generic and structural operations (`init`, `update`, `analyze`, `factor`,
  `refactor`, ...), but not the ones inside `compute`.
- **`vmap` over an iterative refinement.** The log is appended through `io_callback`s,
  which are unordered IO effects. A plain solve (and its `grad`, `jacfwd`, and `jacrev`)
  composes with them, but `jax.vmap` over a refinement loop does not, because its
  batched `while_loop` predicate rejects unordered IO. Profile plain solves rather than
  a `vmap` of a refinement.
