# Solvers

`splineax` provides five sparse direct solvers. All implement Lineax's
`AbstractLinearSolver` interface (so they work with `lineax.linear_solve`) and the
[`SparseLinearSolver`][splineax.SparseLinearSolver] protocol (factorization reuse, see
[Advanced usage](advanced.md)). All handle **square, nonsingular** operators only.

| Solver | Backend | Precision | Factorization reuse |
| --- | --- | --- | --- |
| [`Spsolve`][splineax.Spsolve] | any | input dtype | no (no-op fallbacks) |
| [`KLU`][splineax.KLU] | CPU only | float64 / complex128 | yes |
| [`Pardiso`][splineax.Pardiso] | CPU only | float64 | yes |
| [`CuDSS`][splineax.CuDSS] | CUDA GPU only | input dtype (f32/f64/complex) | yes |
| [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] | any | depends on choice | delegates |

## `Spsolve`

Wraps `jax.experimental.sparse.linalg.spsolve`, which performs a sparse QR factorization
(native on CUDA; on CPU it falls back to `scipy.sparse.linalg.spsolve`). It runs on any
backend.

```python
import splineax as splx

solver = splx.Spsolve(
    tol=1e-6, reorder=splx.solvers.ReorderingScheme.SYMRCM
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
solver = splx.KLU()
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
solver = splx.Pardiso()
```

!!! warning "CPU, real-valued, and double precision only, and requires installation"

    `pardiso_mkl_jax` wraps a CPU-only library and only supports real-valued matrices
    (`float32` inputs are upcast to `float64`, and complex operators raise `TypeError`).
    Like `klujax`, it does not enable JAX's x64 mode automatically, so you must do that
    yourself. `Pardiso()` raises `ImportError` if `pardiso-mkl-jax` isn't installed. Use
    [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] for code that should work
    whether or not it is.

## `CuDSS`

Wraps NVIDIA's cuDSS library, a direct sparse solver with an explicit analysis /
factorization / refactorization / solve phase split. It is the only solver in this
package that both runs on GPU **and** keeps real factorization reuse (see
[Advanced usage](advanced.md)). `Spsolve` runs on GPU too, but its factorization
methods are no-ops.

`CuDSS` is an **optional dependency**: install it with

```bash
pip install splineax[cudss]
```

```{.python notest}
solver = splx.CuDSS()
```

!!! warning "CUDA GPU only, and requires installation"

    cuDSS is a CUDA-only library: `CuDSS` raises an error at trace time if solved on
    any other platform. Unlike `KLU`/`Pardiso`, it needs no upcasting: `float32`,
    `float64`, `complex64`, and `complex128` are all supported directly. `CuDSS()`
    raises `ImportError` if the optional dependency isn't installed. Use
    [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] for code that should work
    whether or not it is.

## `AutoSparseLinearSolver`

Picks a solver based on the JAX platform and what's installed: on CPU with x64 enabled,
[`Pardiso`][splineax.Pardiso] if the optional `pardiso-mkl-jax` dependency is installed,
otherwise [`KLU`][splineax.KLU] (both fast direct solves with factorization reuse). On a
CUDA GPU it picks [`CuDSS`][splineax.CuDSS] if its optional dependency is installed,
with no x64 requirement. Everything else gets [`Spsolve`][splineax.Spsolve]. It exposes
the same factorization API as `Pardiso`/`KLU`/`CuDSS`, so you can substitute it for any
of them verbatim. When it dispatches to `Spsolve`, the factorization methods degrade to
no-ops. Since `pardiso_mkl_jax` doesn't support complex matrices, `Auto` falls back to
`KLU` for a complex operator even when `Pardiso` was otherwise selected. `CuDSS` needs
no equivalent fallback, since it supports complex directly.

```python
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

import splineax as splx

operator = splx.BCOOLinearOperator(
    BCOO.fromdense(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
)
solver = splx.AutoSparseLinearSolver()

# Inspect what it will dispatch to (mirrors lineax.AutoLinearSolver.select_solver).
chosen = solver.select_solver(operator)

# Force a specific platform's choice.
cpu_solver = splx.AutoSparseLinearSolver(platform="cpu")  # -> Pardiso, or KLU
gpu_solver = splx.AutoSparseLinearSolver(platform="gpu")  # -> CuDSS if installed, else Spsolve
```

This is the recommended default when you want portable code that uses
`Pardiso`/`KLU`/`CuDSS` where available and `Spsolve` elsewhere.
