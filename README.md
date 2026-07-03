# splineax

Sparse linear operators and direct solvers for
[Lineax](https://github.com/patrick-kidger/lineax).

`splineax` lets you keep a linear system in its native sparse storage
(`jax.experimental.sparse.BCOO` / `BCSR`) and solve it with a sparse *direct* solver that
plugs straight into `lineax.linear_solve`.

- **Operators**: `BCOOLinearOperator`, `BCSRLinearOperator`.
- **Solvers**: `Spsolve` (any backend), `KLU` (CPU-only, SuiteSparse, factorization reuse),
  and `AutoSparseLinearSolver` (picks one based on the platform).
- A `SparseLinearSolver` protocol for separating factorization from solving.

## Installation

```bash
pip install git+https://github.com/nardi/splineax.git@v0.1.1
```

## Example

Solve a 10000 x 10000 system. As a dense matrix it would need 10^8 entries, but kept
sparse it has only ~3 x 10^4 nonzeros, and the solver never materialises the dense form.

```python
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO

import splineax

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

operator = splineax.BCOOLinearOperator(matrix)
vectors = [jnp.ones(n), jnp.arange(n) % 2]
solver = splineax.AutoSparseLinearSolver()

# Calculate factorization once...
with solver.factorize(operator) as factorized_state:
    # ...and reuse for multiple solves.
    solution = lx.linear_solve(
        operator, vectors[0], solver=solver, state=factorized_state
    )
    assert jnp.allclose(matrix @ solution.value, vectors[0], atol=1e-4)

    solution = lx.linear_solve(
        operator, vectors[1], solver=solver, state=factorized_state
    )
    assert jnp.allclose(matrix @ solution.value, vectors[1], atol=1e-4)
```

## Documentation

Build the docs locally with `uv run mkdocs serve`, or view the [user guide and API reference here](https://nardi.github.io/splineax).
