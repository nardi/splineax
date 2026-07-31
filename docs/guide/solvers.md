# Solvers

`splineax` provides four sparse **direct** solvers and one **iterative** one. All implement
Lineax's `AbstractLinearSolver` interface (so they work with `lineax.linear_solve`) and the
[`SparseLinearSolver`][splineax.SparseLinearSolver] protocol (factorization reuse, see
[Advanced usage](advanced.md)). All handle **square** operators only.

| Solver | Kind | Backend | Precision | Factorization reuse |
| --- | --- | --- | --- | --- |
| [`Spsolve`][splineax.Spsolve] | direct | any | input dtype | no (no-op fallbacks) |
| [`KLU`][splineax.KLU] | direct | CPU only | float64 / complex128 | yes |
| [`Pardiso`][splineax.Pardiso] | direct | CPU only | float64 | yes |
| [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] | direct | any | depends on choice | delegates |
| [`BlockJacobiGMRES`][splineax.BlockJacobiGMRES] | iterative | any | input dtype | yes |

The direct solvers require a **nonsingular** operator and return an answer accurate to the
precision of a factorization. `BlockJacobiGMRES` differs on both counts: it iterates to a
tolerance you choose, and it may fail to converge on a hard problem, which it reports rather
than raises. In exchange it is the only one that needs no external library, so it is also the
only one whose whole solve compiles into a single computation and runs unchanged on a GPU or a
TPU.

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

## `BlockJacobiGMRES`

The one iterative solver, and the one that calls no external library. It reorders the matrix to
narrow its band, cuts the reordered range into overlapping blocks, inverts those blocks, and
uses the collection as a preconditioner for restarted GMRES. [The theory
page](../theory/block-jacobi-gmres.md) describes the method and why each stage is there.

```{.python continuation}
solver = splx.BlockJacobiGMRES(rtol=1e-8, atol=1e-8)
```

Because it needs no library and no data-dependent control flow, it is the solver to reach for
when the solve has to happen inside a compiled computation or on an accelerator.

The settings worth knowing about, in the order you would usually try them:

- `rtol` and `atol` set what counts as converged. A solve that does not get there returns an
  unsuccessful `lineax.RESULTS` rather than raising, so check `solution.result` if you passed
  `throw=False`.
- `overlap_fraction` is usually the cheapest way to improve convergence. Raising it widens each
  block over its neighbours, so more of the matrix reaches the preconditioner, at a cost of
  about `1 / (1 - overlap_fraction)`.
- `max_block_size` caps how large a block may get. Inverting the blocks costs on the order of
  `n * max_block_size^2`, so this is the main control on how expensive a numeric factorization
  is. That cost is paid per set of values, which matters if you are solving a sequence of
  related systems.
- `ordering` selects the reordering. The default, `Ordering.RCM`, gives the narrowest band on
  everything measured here. Use `Ordering.NONE` for an operator that is already banded, since
  reordering it only costs time, and `Ordering.SPECTRAL` for a pattern whose graph has very many
  breadth-first levels, where `RCM` becomes expensive.
- `block_inverse` selects how blocks are inverted. `BlockInverse.QR` is cheaper than the
  default `BlockInverse.SVD` but relies on column-pivoted QR, which is unavailable on TPU.

A matrix with rows that have no diagonal entry, a saddle point being the standard example, is
handled automatically rather than needing a different setting. The symbolic stage detects the
pattern and reorders around it: a genuine saddle point gets each such row paired with a matched
ordinary unknown before the blocks are cut, and a matrix whose rows merely happen to be numbered
so their diagonal is hidden gets those rows permuted back onto a populated diagonal instead. Both
are pattern-only, so they cost nothing extra once a symbolic factorization is reused. [The theory
page](../theory/block-jacobi-gmres.md#repairing-an-accidental-diagonal) covers both cases and what
each one does and does not guarantee.

```{.python continuation}
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

# [[F, B^T], [B, 0]]: F a Laplacian, B a discrete divergence, so half the diagonal is
# structurally zero, the shape a saddle point has.
f = jnp.array([[2.0, -1.0], [-1.0, 2.0]])
b = jnp.array([[-1.0, 1.0]])
matrix = jnp.block([[f, b.T], [b, jnp.zeros((1, 1))]])
operator = splx.BCOOLinearOperator(BCOO.fromdense(matrix))

solution = lx.linear_solve(operator, jnp.array([1.0, 0.0, 0.0]), solver=splx.BlockJacobiGMRES())
assert jnp.allclose(matrix @ solution.value, jnp.array([1.0, 0.0, 0.0]), atol=1e-4)
```

!!! note "It is an approximation, and it tells you so"

    The preconditioner keeps only the entries that fall inside some block. Those left out are
    dropped from the preconditioner alone, never from the matrix the iteration multiplies by, so
    discarding them costs iterations rather than accuracy. When the operator is small enough to
    fit inside one block the preconditioner is an exact inverse and the solve converges
    immediately, which is why this solver is also fine on small systems.

    A reported success is always backed by a check of the real residual. GMRES measures its own
    progress on the preconditioned residual, and the two part company when the preconditioner is
    ill-conditioned, so the solver verifies the unpreconditioned residual itself before agreeing
    that the solve converged.

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

# Inspect what it will dispatch to (mirrors lineax.AutoLinearSolver.select_solver).
chosen = solver.select_solver(operator)

# Force a specific platform's choice.
cpu_solver = splx.AutoSparseLinearSolver(platform="cpu")  # -> Pardiso, or KLU
gpu_solver = splx.AutoSparseLinearSolver(platform="gpu")  # -> Spsolve
```

This is the recommended default when you want portable code that uses `Pardiso`/`KLU`
where available and `Spsolve` elsewhere.
