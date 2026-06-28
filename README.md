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
pip install splineax
```

## Example

```python
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

import splineax

matrix = BCOO.fromdense(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
operator = splineax.BCOOLinearOperator(matrix)
vector = jnp.array([1.0, 2.0])

solution = lx.linear_solve(operator, vector, solver=splineax.AutoSparseLinearSolver())
print(solution.value)  # [0.2 0.6]
```

## Documentation

Build the docs locally with `uv run mkdocs serve`, or see the user guide and API reference
under [`docs/`](docs/).
