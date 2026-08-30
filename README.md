# splineax

Sparse linear operators and direct solvers for
[Lineax](https://github.com/patrick-kidger/lineax).

`splineax` lets you keep a linear system in its native sparse storage
(`jax.experimental.sparse.BCOO` / `BCSR`) and solve it with a sparse *direct* solver that
plugs straight into `lineax.linear_solve`. It also interfaces with [asdex](https://github.com/adrhill/asdex) for calculating sparse Jacobians and using them as operators.

- **Operators**: `BCOOLinearOperator`, `BCSRLinearOperator`, `SparseJacobianLinearOperator`.
- **Stateful solver protocols**: `StatefulSolver` and `SparseLinearSolver` for writing
  solver-agnostic code that reuses factorizations over many solves and operators.
- **Solver library bindings**: `Spsolve` (any backend), `KLU` (CPU-only, SuiteSparse KLU),
  and `Pardiso` (CPU-only, Intel oneMKL Pardiso, installed as extra). `KLU` and `Pardiso`
  reuse their factorization across solves.
- **Higher-level solvers**: `AutoSparseLinearSolver`, which picks an appropriate solver
  based on platform and settings, and `IterativeRefinement`, which wraps any solver and
  refines its solution to a target residual.
- **Lineax code interop**: `stateful_solve_transform` rewrites a function that calls
  `lineax.linear_solve` so its solves thread a solver state and reuse a factorization.

## Installation

```bash
pip install splineax
```

## Example

Solve a 10000 x 10000 system. As a dense matrix it would need 10^8 entries, but kept
sparse it has only ~3 x 10^4 nonzeros, and the solver never materialises the dense form.

```python
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO

import splineax as splx

n = 10000
np.random.seed(0)

# A large, randomly sparse matrix with a heavy diagonal (so it is invertible).
diagonal_indices = np.stack([np.arange(n), np.arange(n)], axis=1)
off_diagonal_indices = np.unique(np.random.randint(0, n, size=(2 * n, 2)), axis=0)
indices = jnp.concatenate([diagonal_indices, off_diagonal_indices])
values = jnp.concatenate(
    [
        np.full(n, float(n)),
        np.random.uniform(low=-1, high=1, size=off_diagonal_indices.shape[0]),
    ]
)
matrix = BCOO((values, indices), shape=(n, n)).sum_duplicates()

operator = splx.BCOOLinearOperator(matrix)
vectors = [jnp.ones(n), jnp.arange(n) % 2]
solver = splx.AutoSparseLinearSolver()

# Solve once, then thread the returned state back in to reuse the factorization.
solution, state = splx.linear_solve(operator, vectors[0], solver)
assert jnp.allclose(matrix @ solution.value, vectors[0], atol=1e-4)

solution, state = splx.linear_solve(operator, vectors[1], solver, state=state)
assert jnp.allclose(matrix @ solution.value, vectors[1], atol=1e-4)

# Free the factorization when you are done with it.
solver.release(state)
```

## Documentation

Build the docs locally with `uv run mkdocs serve`, or view the [user guide and API reference here](https://nardi.github.io/splineax).
