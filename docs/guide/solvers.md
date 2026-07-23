# Solvers

`splineax` provides four sparse direct solvers. All implement Lineax's
`AbstractLinearSolver` interface (so they work with `lineax.linear_solve`) and the
[`SparseLinearSolver`][splineax.SparseLinearSolver] protocol (factorization reuse, see
[Advanced usage](advanced.md)). All handle **square, nonsingular** operators only.

| Solver | Backend | Precision | Factorization reuse |
| --- | --- | --- | --- |
| [`Spsolve`][splineax.Spsolve] | any | input dtype | no (no-op fallbacks) |
| [`KLU`][splineax.KLU] | CPU only | float64 / complex128 | yes |
| [`Pardiso`][splineax.Pardiso] | CPU only | float64 | yes |
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

    `klujax` wraps a CPU-only library, and does not enable JAX's x64 mode or force the
    CPU platform automatically: `jax_enable_x64` must already be on before you solve
    with `KLU`, or `klujax` raises a clear error. `float32` / `complex64` inputs are
    upcast to `float64` / `complex128`. If you need to stay on GPU/TPU, use
    [`Spsolve`][splineax.Spsolve].

## `Pardiso`

Wraps [`pardiso-mkl-jax`](https://github.com/nardi/pardiso-mkl-jax), bindings for Intel
oneMKL's Pardiso direct sparse solver. Like `KLU`, it keeps the operator in its native
sparse storage and supports reusing a symbolic and/or numeric factorization across many
solves (see [Advanced usage](advanced.md)).

`Pardiso` is an **optional dependency**: install it with

```bash
pip install splineax[pardiso]
```

```{.python notest}
solver = splineax.Pardiso()
```

!!! warning "CPU, real-valued, and double precision only, and requires installation"

    `pardiso_mkl_jax` wraps a CPU-only library and only supports real-valued matrices
    (`float32` inputs are upcast to `float64`, and complex operators raise `TypeError`).
    Like `klujax`, it does not enable JAX's x64 mode automatically, so you must do that
    yourself. `Pardiso()` raises `ImportError` if `pardiso-mkl-jax` isn't installed. Use
    [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] for code that should work
    whether or not it is.

## `AutoSparseLinearSolver`

Picks a solver based on the JAX platform and what's installed: on CPU with x64 enabled,
[`Pardiso`][splineax.Pardiso] if the optional `pardiso-mkl-jax` dependency is installed,
otherwise [`KLU`][splineax.KLU] (both fast direct solves with factorization reuse), and
[`Spsolve`][splineax.Spsolve] otherwise. It exposes the same factorization API as
`Pardiso`/`KLU`, so you can substitute it for either verbatim. On non-CPU backends the
factorization methods degrade to no-ops via `Spsolve`. Since `pardiso_mkl_jax` doesn't
support complex matrices, `Auto` falls back to `KLU` for a complex operator even when
`Pardiso` was otherwise selected.

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
cpu_solver = splineax.AutoSparseLinearSolver(platform="cpu")  # -> Pardiso, or KLU
gpu_solver = splineax.AutoSparseLinearSolver(platform="gpu")  # -> Spsolve
```

This is the recommended default when you want portable code that uses `Pardiso`/`KLU`
where available and `Spsolve` elsewhere.
