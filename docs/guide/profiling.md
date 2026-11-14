# The solve profile

When using a sparse direct solver, every linear solve is part of a larger chain of
operations. The solver analyzes a matrix's sparsity pattern, factorizes it, reuses or
rebuilds that factorization as the values change, solves against it, and when doing a
hybrid solve, additional iterative adjustments are performed. When a solve is slower than
expected, it might be the case that some work is unexpectedly being repeated.
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

To switch profiling off without touching the call sites, decorate with
`@splx.profile_solves(enabled=False)` instead. The function still returns a pair, but the
profile is `None` and no profile is active during the call.

One edge remains, and it comes from JAX compiling once per shape rather than from
anything `profile_solves` gets wrong. If you call the undecorated jitted function directly,
or through a second wrapper made with `enabled=False`, the first call for a shape may
compile without profiling hooks. No later call for that shape will ever be profiled, even
through an enabled wrapper.

## Record order

In a compiled function, a record is written when XLA runs the operation, not when the
function is traced. `profile.records` lists the records in the order in which they were
written.

Each record takes the arrays of its operation and returns them unchanged, and later
operations use the returned arrays. A record is therefore written after the operations
that it depends on and before the operations that depend on it. In a loop, the records of
each iteration are after the records of the previous iteration. When two operations do not
depend on each other, their records can be in either order. For example, the `release` of
a transformed function and the solves in the backward pass of `jax.grad` do not depend on
each other.

When no result of the function depends on an operation, XLA removes the operation, and its
record is removed as well. This is the case for the `init` that `lineax.linear_solve`
calls before a threaded solve, because the transform replaces the state of this `init`.

When `jax.vmap` batches a value in a record, such as `rebuild`, the profile has one record
for each element of the batch. Otherwise the record is written once.

## Unordered records

The records of `init_symbolic` of `Pardiso` and `Spsolve` are side effects. They are
written each time the compiled function runs, also when no result depends on the state.
Their order relative to the other records is not defined. The same is true for the
`release` of `CuDSS` under `jax.jit`. This `release` cannot read the traced handle and
frees nothing, so its record has the note "skipped under jit".

JAX does not support side effects in a `lax.cond` under `jax.vmap`. While a profile is
active, `jax.vmap` therefore raises `NotImplementedError` when a branch of a `lax.cond`
calls one of these operations.

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
