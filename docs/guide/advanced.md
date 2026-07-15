# Advanced usage: separating factorization from solving

A sparse direct solve has two expensive stages:

1. **Symbolic factorization** — analyze the sparsity pattern (which entries fill in).
   Depends only on *where* the nonzeros are, not their values.
2. **Numeric factorization** — compute the actual LU factors. Depends on the values.

If you solve `Ax = b` many times with a fixed matrix, or many matrices that share a
sparsity pattern, you can compute these stages once and reuse them. `splineax` exposes this
through the [`SparseLinearSolver`][splineax.SparseLinearSolver] protocol, implemented by all
three solvers.

!!! note

    Only [`KLU`][splineax.KLU] actually reuses factorizations. [`Spsolve`][splineax.Spsolve]
    implements the same API with **no-op** fallbacks (each solve refactors), so code written
    against the protocol runs unchanged on any backend.
    [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] delegates to whichever it
    picked.

## Reusing a full factorization

Use `solver.factorize(operator)` as a context manager. Inside the block the operator is
factorized once; every `linear_solve` that passes the yielded `state` reuses it.

```python
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

import splineax

# KLU solver requires 64-bit mode:
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
b3 = b1 + 1.0

operator = splineax.BCOOLinearOperator(BCOO.fromdense(dense))
solver = splineax.KLU()

with solver.factorize(operator) as state:
    x1 = lx.linear_solve(operator, b1, solver=solver, state=state).value
    x2 = lx.linear_solve(operator, b2, solver=solver, state=state).value
    # ... reuse `state` for as many right-hand sides as you like.
```

This is equivalent to `solver.init(operator, options).factorize()`.

## Reusing a symbolic factorization

If you know the *sparsity pattern* ahead of time but the values change (for example,
solving a family of matrices with identical structure), pre-analyze the pattern with
`solver.factorize_symbolic(sparsity)`. It yields a *scope* offering two options.

```{.python continuation}
sparsity = BCOO.fromdense(dense)  # only the structure matters here

with solver.factorize_symbolic(sparsity) as scope:
    # Option A: reuse the symbolic analysis, refactor numerically on each solve.
    state = scope.init(operator)
    x = lx.linear_solve(operator, b1, solver=solver, state=state).value

    # Option B: also pre-compute the numeric factorization for full reuse.
    with scope.factorize(operator) as numeric_state:
        x1 = lx.linear_solve(operator, b1, solver=solver, state=numeric_state).value
        x2 = lx.linear_solve(operator, b2, solver=solver, state=numeric_state).value
```

`factorize_symbolic` accepts a `BCOO`, `BCSR`, `BCOOLinearOperator`,
`BCSRLinearOperator`, `SparseJacobianLinearOperator`,
`SparseJacobianLinearOperatorColoring`, or `JacobianColoring`. Only its sparsity
pattern is read. For the Jacobian operator and the two coloring wrappers, the pattern
comes from the precomputed sparsity, without materialising the Jacobian numerically.

## How the states chain

The protocol describes a small family of state types
([`SparseBasicState`][splineax.solvers.SparseBasicState],
[`SparseSymbolicState`][splineax.solvers.SparseSymbolicState],
[`SparseNumericState`][splineax.solvers.SparseNumericState]) and a scope
([`SparseSymbolicScope`][splineax.solvers.SparseSymbolicScope]):

```
solver.init(operator)                                 -> SparseBasicState
       .factorize()                                   -> SparseNumericState   (context manager)

solver.factorize(operator)                            -> SparseNumericState   (context manager)

solver.factorize_symbolic(sparsity)                   -> SparseSymbolicScope  (context manager)
       .init(operator)                                -> SparseSymbolicState
       .factorize(operator)                           -> SparseNumericState   (context manager)
```

Any of these states can be passed as `state=` to `lineax.linear_solve`.

## Writing backend-agnostic code

All three solvers subclass
[`AbstractSparseLinearSolver`][splineax.AbstractSparseLinearSolver] (and so are usable both
with `lineax.linear_solve` and the factorization API). Type a routine against it and let the
caller pick the solver:

```{.python continuation}
from splineax import AbstractSparseLinearSolver


def solve_many(solver: AbstractSparseLinearSolver, operator, right_hand_sides):
    with solver.factorize(operator) as state:
        return [
            lx.linear_solve(operator, b, solver=solver, state=state).value
            for b in right_hand_sides
        ]


# Fast factorization reuse on CPU, plain (re)solves elsewhere, same code:
solve_many(splineax.AutoSparseLinearSolver(), operator, [b1, b2, b3])
```

The [`SparseLinearSolver`][splineax.SparseLinearSolver] protocol describes the same surface
structurally, for when you prefer duck typing or `isinstance` checks.
