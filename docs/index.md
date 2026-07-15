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
- **Sparse Jacobian operators**:
  [`SparseJacobianLinearOperator`][splineax.SparseJacobianLinearOperator] represents the
  Jacobian of a function sparsely, detecting its sparsity pattern and constructing a coloring automatically (via
  [asdex](https://github.com/adrhill/asdex)), which allows for efficient materialization into a sparse matrix by the solvers.
- **Solvers**: [`Spsolve`][splineax.Spsolve] (any backend, wraps
  `jax.experimental.sparse.linalg.spsolve`), [`KLU`][splineax.KLU] (CPU-only, wraps the
  SuiteSparse KLU library via `klujax`, with factorization reuse), and
  [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] which picks one based on the
  platform.
- A [`SparseLinearSolver`][splineax.SparseLinearSolver] protocol for separating
  factorization from solving.

## Installation

```bash
pip install splineax
```

## Quick example

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

## Where to next

- [Basic usage](guide/basic.md): build operators and solve, just like plain lineax.
- [Operators](guide/operators.md): `BCOO` vs `BCSR`, how to construct them, and details on sparse
  Jacobian operators.
- [Solvers](guide/solvers.md): what each solver does and when to use it.
- [Advanced usage](guide/advanced.md): reuse a factorization across many solves with the
  `SparseLinearSolver` protocol.

!!! note

    The solvers handle **square, nonsingular** operators only. `KLU` additionally runs on
    **CPU in double precision** only (see [Solvers](guide/solvers.md)); `Spsolve` runs on
    any backend.
