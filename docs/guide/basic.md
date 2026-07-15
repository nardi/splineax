# Basic usage

`splineax` follows the standard Lineax workflow: build a linear *operator*, then call
`lineax.linear_solve` with a *solver*. The only difference is that the operator wraps a
sparse array and the solver is a sparse direct solver.

## Solving a system

```python
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

import splineax

# A sparse matrix A and a right-hand side b.
dense = jnp.array(
    [
        [10.0, 2.0, 0.0, 0.0],
        [3.0, 14.0, 5.0, 0.0],
        [0.0, 6.0, 18.0, 9.0],
        [0.0, 0.0, 1.0, 12.0],
    ]
)
operator = splineax.BCOOLinearOperator(BCOO.fromdense(dense))
b = jnp.array([1.0, 2.0, 3.0, 4.0])

solution = lx.linear_solve(operator, b, solver=splineax.AutoSparseLinearSolver())
x = solution.value
```

`solution` is a `lineax.Solution`; the answer is `solution.value` and `solution.result`
reports success or failure, exactly as in plain lineax.

You can swap the solver for any of the three: `splineax.Spsolve()`, `splineax.KLU()`, or
`splineax.AutoSparseLinearSolver()` (see [Solvers](solvers.md)).

!!! note

    These solvers only handle **square, nonsingular** systems. Passing a non-square
    operator raises a `ValueError` from `init`.

## Transposes, batching, and autodiff

Because the operators and solvers are ordinary Lineax components, the usual JAX
transformations work:

```{.python continuation}
import jax

# KLU solver requires 64-bit mode:
jax.config.update("jax_enable_x64", True)

# Solve the transposed system.
solution_T = lx.linear_solve(operator.T, b, solver=splineax.KLU())

# Solve under jit.
solve = jax.jit(lambda v: lx.linear_solve(operator, v, solver=splineax.KLU()).value)
x = solve(b)

# Batch over many right-hand sides with vmap.
bs = jnp.stack([b, b[::-1]])
xs = jax.vmap(solve)(bs)

# Differentiate through the solve (w.r.t. the right-hand side or the matrix entries).
jacobian = jax.jacrev(solve)(b)
```

## Reusing work across right-hand sides

It is common to solve `Ax = b` for the same `A` and many different `b`. Lineax lets you
compute the operator-only part once with `init`, then reuse the resulting `state`:

```{.python continuation}
solver = splineax.KLU()
state = solver.init(operator, options={})

x1 = lx.linear_solve(operator, b, solver=solver, state=state).value
x2 = lx.linear_solve(operator, b[::-1], solver=solver, state=state).value
```

For `KLU` this avoids redoing the matrix bookkeeping on each solve. To go further and reuse
an actual matrix *factorization* across solves, see [Advanced usage](advanced.md).
