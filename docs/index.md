# splineax

Sparse linear operators and direct solvers for
[Lineax](https://github.com/patrick-kidger/lineax).

`splineax` lets you keep a linear system in its native sparse storage
(`jax.experimental.sparse.BCOO` / `BCSR`) instead of densifying it, and solve it with a
sparse *direct* solver that plugs straight into `lineax.linear_solve`.

It provides:

- **Operators** that wrap a sparse array into a `lineax.AbstractLinearOperator`:
  [`BCOOLinearOperator`][splineax.BCOOLinearOperator] and
  [`BCSRLinearOperator`][splineax.BCSRLinearOperator].
- **Solvers**: [`Spsolve`][splineax.Spsolve] (any backend, wraps
  `jax.experimental.sparse.linalg.spsolve`), [`KLU`][splineax.KLU] (CPU-only, wraps the
  SuiteSparse KLU library via `klujax`, with factorization reuse), and
  [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] which picks one based on the
  platform.
- A [`SparseLinearSolver`][splineax.SparseLinearSolver] protocol for separating
  factorization from solving.

## Installation

```bash
pip install git+https://github.com/nardi/splineax.git@v0.1.1
```

## Quick example

```python
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

import splineax

matrix = BCOO.fromdense(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
operator = splineax.BCOOLinearOperator(matrix)
vector = jnp.array([1.0, 2.0])

solution = lx.linear_solve(
    operator, vector, solver=splineax.AutoSparseLinearSolver()
)
print(solution.value)  # [0.2 0.6]
```

## Where to next

- [Basic usage](guide/basic.md): build operators and solve, just like plain lineax.
- [Operators](guide/operators.md): `BCOO` vs `BCSR` and how to construct them.
- [Solvers](guide/solvers.md): what each solver does and when to use it.
- [Advanced usage](guide/advanced.md): reuse a factorization across many solves with the
  `SparseLinearSolver` protocol.

!!! note

    The solvers handle **square, nonsingular** operators only. `KLU` additionally runs on
    **CPU in double precision** only (see [Solvers](guide/solvers.md)); `Spsolve` runs on
    any backend.
