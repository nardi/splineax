# Advanced usage: reusing work across solves

A sparse direct solve does expensive work that does not change between right-hand sides,
or between matrices that share a structure. `splineax` lets you do that work once and
reuse it, through an explicit solver state that you create, update, and thread through
your solves.

The idea is to tell the solver about the operators it will solve, as soon as you know
them, and let it decide what to recompute. You never ask it to factor or refactor by
hand. You hand it an operator, and it reuses whatever it still can.

## The stateful solve API

Every solver in this package exposes the same small API.

- `state = solver.init(operator)` builds a state for an operator.
- `state = solver.update(state, operator)` folds a new operator into an existing state,
  reusing prior work where the two operators allow it.
- `solver.release(state)` says you are done with the state, so any memory it holds may go.
- `state = state.track(solution)` records that a solution depends on the state, so a later
  `release` is ordered after that solve. It is optional, and a no-op for solvers whose
  state holds nothing.

`lineax.linear_solve` does not return an updated state, so on its own these steps read as:

```{.python notest}
state = solver.update(state, operator)
solution = lineax.linear_solve(operator, vector, solver, state=state)
state = state.track(solution)
```

`splineax.linear_solve` does all three for you and returns a `(solution, state)` tuple:

```python
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

import splineax as splx

# KLU and Pardiso require 64-bit mode.
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

operator = splx.BCOOLinearOperator(BCOO.fromdense(dense))
solver = splx.KLU()

# The first call builds a fresh state. Thread it back in to reuse the factorization.
solution, state = splx.linear_solve(operator, b1, solver)
solution, state = splx.linear_solve(operator, b2, solver, state=state)

# Release it once you are done solving.
solver.release(state)
```

With no `state`, `splineax.linear_solve` builds one with `init`. With a `state`, it calls
`update`, so passing the same operator again will cost nothing, and passing a matrix that
shares the structure will reuse the analysis. Either way it tracks the solution before
returning, so the state you get back is safe to release after the loop.

The default solver is `AutoSparseLinearSolver`, which picks a backend for the platform and
precision. Any splineax solver works in its place.

## Reusing an analysis across changing values

Often you know the sparsity pattern before the values, or you solve a family of matrices
that share a pattern. Analyze the pattern once with `init_symbolic`, then `update` folds
in each matrix and reuses that analysis:

```{.python continuation}
# Only the structure matters here, not the values.
sparsity = BCOO.fromdense(dense)

state = solver.init_symbolic(sparsity)
state = solver.update(state, operator)
solution = lx.linear_solve(operator, b1, solver=solver, state=state).value
solver.release(state)
```

`init_symbolic` accepts a `BCOO`, `BCSR`, `BCOOLinearOperator`, `BCSRLinearOperator`,
`SparseJacobianLinearOperator`, `SparseJacobianLinearOperatorColoring`, or
`JacobianColoring`. Only its sparsity pattern is read. For the Jacobian and coloring
forms, the pattern comes from the precomputed coloring, without materialising the Jacobian
numerically.

### Telling the solver two operators share a pattern

For `update` to reuse an analysis, it has to know the new operator has the same structure
as the last one. You assert that with a tag from
[`splineax.sparsity_pattern_tag`][], attached to both operators:

```{.python continuation}
tag = splx.sparsity_pattern_tag(sparsity)
first = splx.BCOOLinearOperator(BCOO.fromdense(dense), tags=tag)
second = splx.BCOOLinearOperator(BCOO.fromdense(2.0 * dense), tags=tag)

state = solver.init(first)
# The shared tag lets update reuse the analysis.
state = solver.update(state, second)
solution = lx.linear_solve(second, b1, solver=solver, state=state).value
solver.release(state)
```

Two operators carrying the same tag are asserted to have exactly the same index arrays,
in the same order. Given a concrete pattern the tag is a content hash, so operators tagged
separately still compare equal when their indices match. With no argument, or under `jit`
where the indices are traced, `sparsity_pattern_tag()` instead returns a marker you thread
onto every operator sharing the pattern. Operators built by one
`SparseJacobianLinearOperatorColoring.operator_at` factory reuse a factorization across
evaluation points automatically, without any tagging.

There is also a `sparse_indices_sorted` tag. Attaching it to an operator asserts its
indices are already row-major sorted, so `Pardiso` and `Spsolve` skip the sort they would
otherwise do.

## Solving inside `jax.jit`

A factorization handle is an ordinary JAX value, not a native object tied to the Python
side, so the whole lifecycle composes inside a jitted function. Build the state, solve,
and release, all under one trace:

```{.python continuation}
@jax.jit
def solve_under_jit(values, b):
    operator = splx.BCOOLinearOperator(
        BCOO((values, sparsity.indices), shape=sparsity.shape, indices_sorted=True)
    )
    solution, state = splx.linear_solve(operator, b, solver)
    solver.release(state)
    return solution.value


x = solve_under_jit(sparsity.data, b1)
```

`state.track` records the solve as a dependency of the state, and `release` consumes that,
so XLA orders the native release after the solve. That holds eagerly and inside one trace,
so there is nothing special to remember here. Use `splineax.linear_solve` and it tracks
for you.

## What each solver reuses

The API is the same across solvers, but what they reuse differs.

`KLU` keeps two handles, a symbolic analysis and a numeric factorization. `init` builds
both. `update` on a matching pattern reuses the analysis and rebuilds the numeric factor
for the new values. `transpose` reuses both and solves the transposed system directly.

`Pardiso` keeps one factorization handle. Under its default weighted matching, an analysis
that ignores the values is not sound, so `init_symbolic` defers the analysis. It records
the pattern only, and the first `update` with real values runs analyze and factor. Later
updates on the same pattern refactor while reusing that analysis.

`Spsolve` has no separate factorization phase, so the reuse API is a set of no-ops for
parity. `update` rebuilds the state, `release` frees nothing, and `track` returns the
state unchanged. Code written against the API runs unchanged on any backend, and
`AutoSparseLinearSolver` forwards to whichever it picked.

## Writing backend-agnostic code

Type a routine against the [`splineax.SparseLinearSolver`][] protocol and let the caller
pick the solver:

```{.python continuation}
def solve_many(
    solver: splx.SparseLinearSolver,
    operator: lx.AbstractLinearOperator,
    right_hand_sides: list[jax.Array],
):
    solution, state = splx.linear_solve(operator, right_hand_sides[0], solver)
    results = [solution.value]
    for b in right_hand_sides[1:]:
        solution, state = splx.linear_solve(operator, b, solver, state=state)
        results.append(solution.value)
    solver.release(state)
    return results


solve_many(splx.AutoSparseLinearSolver(), operator, [b1, b2])
```
