# Solvers

`splineax` provides three sparse direct solvers. All implement Lineax's
`AbstractLinearSolver` interface (so they work with `lineax.linear_solve`) and the
[`SparseLinearSolver`][splineax.SparseLinearSolver] protocol (factorization reuse, see
[Advanced usage](advanced.md)). All handle **square, nonsingular** operators only.

| Solver | Backend | Precision | Factorization reuse |
| --- | --- | --- | --- |
| [`Spsolve`][splineax.Spsolve] | any | input dtype | no (no-op fallbacks) |
| [`KLU`][splineax.KLU] | CPU only | float64 / complex128 | yes |
| [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] | any | depends on choice | delegates |

## `Spsolve`

Wraps `jax.experimental.sparse.linalg.spsolve`, which performs a sparse QR factorization
(native on CUDA; on CPU it falls back to `scipy.sparse.linalg.spsolve`). It runs on any
backend.

```python
import splineax

solver = splineax.Spsolve(
    tol=1e-6, reorder=splineax.solvers.ReorderingScheme.SYMRCM
)
```

- `tol`: tolerance used to decide whether the system is singular.
- `reorder`: fill-reducing reordering scheme.

`spsolve` has no batching rule of its own, so `splineax` adds a sequential `vmap` rule;
this means `jax.vmap`, `jax.jacfwd`, and `jax.jacrev` work, looping over the batch.

## `KLU`

Wraps [`klujax`](https://github.com/flaport/klujax), bindings for the SuiteSparse KLU
sparse LU solver. It keeps the operator in coordinate form and supports reusing a symbolic
and/or numeric factorization across many solves (see [Advanced usage](advanced.md)).

```{.python continuation}
solver = splineax.KLU()
```

!!! warning "CPU and double precision only"

    `klujax` wraps a CPU-only library. Importing it (which happens lazily, on the first
    `KLU` solve) **enables JAX's x64 mode and forces the CPU platform globally**.
    `float32` / `complex64` inputs are upcast to `float64` / `complex128`. If you need to
    stay on GPU/TPU, use [`Spsolve`][splineax.Spsolve].

## `AutoSparseLinearSolver`

Picks a solver based on the JAX platform: [`KLU`][splineax.KLU] on CPU (fast direct solve
with factorization reuse), [`Spsolve`][splineax.Spsolve] otherwise. It exposes the same
factorization API as `KLU`, so you can substitute it for `KLU` verbatim; on non-CPU
backends the factorization methods degrade to no-ops via `Spsolve`.

```python
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

import splineax

operator = splineax.BCOOLinearOperator(
    BCOO.fromdense(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
)
solver = splineax.AutoSparseLinearSolver()

# Inspect what it will dispatch to (mirrors lineax.AutoLinearSolver.select_solver).
chosen = solver.select_solver(operator)

# Force a specific platform's choice.
cpu_solver = splineax.AutoSparseLinearSolver(platform="cpu")  # -> KLU
gpu_solver = splineax.AutoSparseLinearSolver(platform="gpu")  # -> Spsolve
```

This is the recommended default when you want portable code that uses `KLU` where it is
available and `Spsolve` elsewhere.
