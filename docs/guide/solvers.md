# Solvers

`splineax` provides five sparse direct solvers, plus
[`HybridDirectIterative`][splineax.HybridDirectIterative], which pairs any of them with
an iterative solver. All implement Lineax's `AbstractLinearSolver` interface (so they work with
`lineax.linear_solve`) and the [`SparseLinearSolver`][splineax.SparseLinearSolver] protocol
(factorization reuse, see [Stateful solves](stateful.md)). All handle **square,
nonsingular** operators only.

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
and/or numeric factorization across many solves (see [Stateful solves](stateful.md)).

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
solves (see [Stateful solves](stateful.md)).

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

Wraps NVIDIA's cuDSS library, a direct sparse solver with an explicit analysis,
factorization, and solve phase split. It is the only solver in this package that both runs
on GPU and keeps real factorization reuse (see [Stateful solves](stateful.md)). `Spsolve`
runs on GPU too, but its reuse methods are no-ops.

`CuDSS` is an **optional dependency**: install it with

```bash
pip install splineax[cudss]
```

The extra needs Python 3.12 or newer on x86_64 Linux with CUDA 13, since `spineax`, the
cuDSS binding it installs, only publishes wheels for that setup. Anywhere else the install
still succeeds but skips `spineax`, and `CuDSS()` then raises `ImportError`.

```{.python notest}
solver = splx.CuDSS()

# COLAMD reordering, under which `update` reuses the previous pivots.
reusing = splx.CuDSS(reordering=splx.solvers.CuDSSReordering.COLAMD)
```

The `reordering` argument picks cuDSS's fill-reducing reordering. The default suits most
matrices. Under `COLAMD` and `BTF_COLAMD`, cuDSS pivots globally, and an `update` with
new values refactorizes with the previous pivots, which is cheaper than a fresh
factorization. When the reused pivots come out badly scaled for the new values, `update`
factorizes fresh instead, like `KLU` does.

!!! warning "CUDA GPU only, and requires installation"

    cuDSS is a CUDA-only library: `CuDSS` raises an error at trace time if solved on any
    other platform. Unlike `KLU`/`Pardiso`, it needs no upcasting: `float32`, `float64`,
    `complex64`, and `complex128` are all supported directly. `CuDSS()` raises
    `ImportError` if the optional dependency isn't installed. Use
    [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] for code that should work
    whether or not it is.

## `AutoSparseLinearSolver`

Picks a solver based on the JAX platform and what's installed: on CPU with x64 enabled,
[`Pardiso`][splineax.Pardiso] if the optional `pardiso-mkl-jax` dependency is installed,
otherwise [`KLU`][splineax.KLU] (both fast direct solves with factorization reuse). On a
CUDA GPU it picks [`CuDSS`][splineax.CuDSS] if its optional dependency is installed, with
no x64 requirement. Everything else gets [`Spsolve`][splineax.Spsolve]. It exposes the
same factorization API as `Pardiso`/`KLU`/`CuDSS`, so you can substitute it for any of
them verbatim. When it dispatches to `Spsolve`, the factorization methods degrade to
no-ops. Since `pardiso_mkl_jax` doesn't support complex matrices, `Auto` falls back to
`KLU` for a complex operator even when `Pardiso` was otherwise selected. `CuDSS` needs no
equivalent fallback, since it supports complex directly.

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
gpu_solver = splx.AutoSparseLinearSolver(platform="gpu")  # -> CuDSS if installed, else Spsolve
```

This is the recommended default when you want portable code that uses
`Pardiso`/`KLU`/`CuDSS` where available and `Spsolve` elsewhere. By default it also
refines every solution with [iterative refinement](#iterativerefinement). Pass
`iterative=False` to solve with the chosen direct solver alone.

```{.python continuation}
# The direct solve, refined until the residual is small (the default).
refining = splx.AutoSparseLinearSolver()

# The direct solve on its own.
plain = splx.AutoSparseLinearSolver(iterative=False)

# A looser tolerance and a lower step cap.
tuned = splx.AutoSparseLinearSolver(
    iterative=splx.IterativeRefinementSettings(tol=1e-8, max_steps=5)
)
```

## `HybridDirectIterative`

A direct solver factors a matrix once and then solves cheaply, but a new matrix needs a
new factorization. When the matrix changes a little between solves, the factorization of
an earlier matrix still comes close to solving the new problem. This means that it can serve as a good preconditioner for an iterative solver. `HybridDirectIterative` pairs a direct
solver with an iterative solver, and uses the direct solver as the preconditioner for the iterative one.
For sake of efficiency, it only factors a new matrix when the preconditioned iterative solve is not able to (quickly) solve the new problem.

```python
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

import splineax as splx

# KLU requires 64-bit mode.
jax.config.update("jax_enable_x64", True)

dense = jnp.array([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]])
vector = jnp.array([1.0, 2.0, 3.0])
tag = splx.sparsity_pattern_tag(BCOO.fromdense(dense))

solver = splx.HybridDirectIterative(
    splx.KLU(),
    splx.GMRESOptions(),
    reuse_direct=splx.ReuseOptions(max_reuses=20),
)

state = None
for step in range(3):
    # The shared tag says that every operator has the same sparsity pattern.
    operator = splx.BCOOLinearOperator(
        BCOO.fromdense(dense + 0.01 * step * jnp.eye(3)), tags=tag
    )
    solution, state = splx.linear_solve(operator, vector, solver, state=state)
    print(step, solution.stats["reused"], solution.stats["refactored"])
state.release()
```

The first solve will factor the matrix, after which the following ones will try the old factorization first, giving up after a small amount of steps (`max_steps_stale`). If the iteration gets within solution tolerance, it keeps the old
factorization. If it does not converge (in time), it will factor the new matrix and then run the
iterative solver with a higher step cap (`max_steps`). Every solution is checked
against the true residual `||b - A x|| <= tol * ||b||`. A solve that still fails after a
new factorization returns NaN with the result `max_steps_reached`. So reuse can make a solve cheaper, but it
never changes whether a solve is successful. Tuning the iterative and reuse parameters can help to avoid ineffective reuse, where the fallback is triggered more often than not.

The factorization is only reused through `splineax.linear_solve` (or `update_and_compute`,
see [Stateful solves](stateful.md)), and only for operators that share a
[sparsity pattern tag](stateful.md#shared-patterns-between-operators). A plain `update`
always makes a new factorization, and so does a solve that is differentiated.

### Iterative solver options

The `iterative` argument picks the iterative solver, with the options that belong to it.
Every option type has the step caps `max_steps` (for a factorization of the current
matrix) and `max_steps_stale` (for a factorization of an earlier matrix, kept low).

| Options | Method | Operator |
| --- | --- | --- |
| [`RichardsonOptions`][splineax.RichardsonOptions] | Iterative refinement | any |
| [`GMRESOptions`][splineax.GMRESOptions] | GMRES | any |
| [`BiCGStabOptions`][splineax.BiCGStabOptions] | BiCGStab | any |
| [`CGOptions`][splineax.CGOptions] | Conjugate gradients | symmetric positive definite |

`CGOptions` needs the operator to carry `lineax.positive_semidefinite_tag`. The default is
`GMRESOptions`. `RichardsonOptions` repeats the direct solve on the residual, which is
iterative refinement. Each of its steps costs one back-substitution and one matrix-vector
product. The Krylov methods can converge when the old factorization is too far off for
Richardson iteration.

### Reuse options

`reuse_direct` is `True` by default. Pass `False` to factor every new matrix, or a
[`ReuseOptions`][splineax.ReuseOptions] to set the limits. `max_reuses` is the number of
solves in a row that may use the same factorization. `slow_fraction` is the fraction of
`max_steps_stale` that a reuse may use before the next solve makes a new factorization.

Reuse pays off when making a factorization costs much more than a few back-substitutions,
which holds for large matrices. For a small matrix, a new factorization can be cheaper
than the stale solve.

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
solution = lx.linear_solve(operator, vector, solver=refined)
```

- `tol`: the target relative residual. Defaults to `1e-10`.
- `max_steps`: the maximum number of correction steps before returning NaN. Defaults to
  `10`.

The threshold is floored at machine precision, so a tolerance tighter than the working
precision can reach still reports success rather than returning NaN. A single-precision
solve, for instance, cannot push the relative residual much below `1e-6`, and refinement
will not demand it. The wrapper exposes the same stateful API as the solver it wraps (see
[Stateful solves](stateful.md)), so it reuses factorizations across right-hand sides the
same way.

Under the hood, `IterativeRefinement` consists of a
[`HybridDirectIterative`](#hybriddirectiterative) solver with a `Richardson` iteration, and
no reuse of the direct solver state.
