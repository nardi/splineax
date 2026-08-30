# Solvers

`splineax` provides four sparse direct solvers, plus
[`IterativeRefinement`][splineax.IterativeRefinement], which wraps any of them to sharpen
a solution. All implement Lineax's `AbstractLinearSolver` interface (so they work with
`lineax.linear_solve`) and the [`SparseLinearSolver`][splineax.SparseLinearSolver] protocol
(factorization reuse, see [Advanced usage](advanced.md)). All handle **square,
nonsingular** operators only.

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

import splineax as splx

operator = splx.BCOOLinearOperator(
    BCOO.fromdense(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
)
solver = splx.AutoSparseLinearSolver()

# Inspect the exact solver it will run (mirrors lineax.AutoLinearSolver.select_solver).
# With refinement on, this is an IterativeRefinement wrapping the chosen direct solver.
chosen = solver.select_solver(operator)

# Force a specific platform's choice.
cpu_solver = splx.AutoSparseLinearSolver(platform="cpu")  # -> Pardiso, or KLU
gpu_solver = splx.AutoSparseLinearSolver(platform="gpu")  # -> Spsolve
```

This is the recommended default when you want portable code that uses `Pardiso`/`KLU`
where available and `Spsolve` elsewhere. By default it also refines every solution with
iterative refinement (see below). Pass `iterative_refinement=False` to solve with the
chosen direct solver alone.

```{.python continuation}
# The direct solve, refined until the residual is small (the default).
refining = splx.AutoSparseLinearSolver()

# The direct solve on its own.
plain = splx.AutoSparseLinearSolver(iterative_refinement=False)

# A looser tolerance and a lower step cap.
tuned = splx.AutoSparseLinearSolver(
    iterative_refinement=splx.IterativeRefinementSettings(tol=1e-8, max_steps=5)
)
```

## `IterativeRefinement`

A direct solve returns `x0 = solve(b)`, accurate to the backend's working precision. When
you need more, iterative refinement improves it. It forms the residual `r = b - A x`,
solves `A dx = r` with the same factorization, and adds the correction `x = x + dx`. Each
step reuses the factorization the wrapped solver already built, so a step costs one
matrix-vector product and one back-substitution, not a new factorization.

`IterativeRefinement` wraps any of the solvers above and drives this loop. It stops once
the relative residual `||b - A x|| <= tol * ||b||` is met, or after `max_steps`
corrections. When it cannot reach the tolerance in time, it returns NaN, so a caller can
tell the solve fell short instead of trusting a solution that never converged.

```{.python continuation}
import lineax as lx

refined = splx.IterativeRefinement(splx.Spsolve(), tol=1e-6, max_steps=10)
solution = lx.linear_solve(operator, jnp.array([1.0, 2.0]), solver=refined)
```

- `tol`: the target relative residual. Defaults to `1e-10`.
- `max_steps`: the maximum number of correction steps before returning NaN. Defaults to
  `10`.

The threshold is floored at machine precision, so a tolerance tighter than the working
precision can reach still reports success rather than returning NaN. A single-precision
solve, for instance, cannot push the relative residual much below `1e-6`, and refinement
will not demand it. The wrapper exposes the same stateful API as the solver it wraps (see
[Advanced usage](advanced.md)), so it reuses factorizations across right-hand sides the
same way.
