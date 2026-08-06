# Preconditioning: iterative solves for large systems

The direct solvers ([`KLU`][splineax.KLU], [`Pardiso`][splineax.Pardiso],
[`Spsolve`][splineax.Spsolve]) factorize the matrix exactly. That is fast and exact,
but it does not scale to every system: some are too large, too GPU-bound, or too far
from the direct solvers' CPU or double-precision requirements.
[`PreconditionedIterativeSolver`][splineax.PreconditionedIterativeSolver] solves those
with one of lineax's own iterative solvers (`GMRES`, `BiCGStab`, `CG`) instead, made to
converge quickly by rewriting the system first.

## The idea: everything is a system transformation

`A x = b` becomes `(L A R) y = L b`, solved for `y`, then `x = R y` recovers the
original solution. Each rewrite is one `(L, R)` pair, and a preconditioner `Q`
(approximating `(L A R)^-1`) is handed to the Krylov solver to speed up convergence.

| stage | picks | from |
| --- | --- | --- |
| symbolic transform | a permutation | the sparsity pattern alone |
| numeric transform | a scaling | the matrix values |
| preconditioner | an approximate inverse | the (already transformed) matrix |

`Q` never gets multiplied into the matrix. The Krylov solver applies it to residuals
inside the iteration, so materialising `Q A` would defeat the point of preconditioning
in the first place.

`splineax` ships one of each, picked so they compose:
[`AggregationClustering`][splineax.transforms.AggregationClustering] (symbolic) gathers
coupled rows and columns into contiguous blocks by clustering, so that
[`RuizEquilibration`][splineax.transforms.RuizEquilibration] (numeric) scales rows and
columns towards unit magnitude, so that
[`BlockJacobi`][splineax.transforms.BlockJacobi] (the preconditioner) inverts those
blocks accurately, batched into one GPU kernel.
[`compose_transforms`][splineax.transforms.compose_transforms] chains transforms
together; see [Writing your own transform](#writing-your-own-transform) below for the
two protocols this all rests on.

## How this maps onto the direct solvers' staging

This is the same `analyze_symbolic` / `analyze_numeric` split every direct solver uses,
run in lockstep for the transform and the preconditioner:

```
transform.analyze_symbolic(pattern)       -> AnalyzedTransform
          .analyze_numeric(matrix)        -> (matrix, AppliedTransform)

preconditioner.analyze_symbolic(pattern)  -> AnalyzedPreconditioner
               .analyze_numeric(matrix)   -> AbstractLinearOperator  (context manager)
```

`PreconditionedIterativeSolver.analyze_symbolic(sparsity)` runs both `analyze_symbolic`
calls, and the resulting state's own `analyze_numeric()` runs both `analyze_numeric`
calls. That is the whole solver: its tiers *are* the transform's and the
preconditioner's tiers, run together. A preconditioned iterative solve therefore reuses
work exactly the way a direct solve does, through the same `with` blocks described in
[Advanced usage](advanced.md):

```
solver.analyze_symbolic(sparsity)        -> scope   (context manager)
       .init(operator)                   -> state
       .analyze_numeric(operator)        -> numeric state   (context manager)

solver.analyze_numeric(operator)         -> numeric state   (context manager)
```

Pattern-only work (the clustering, and the preconditioner's scatter destinations) runs
once in `analyze_symbolic` and is reused across every matrix sharing that pattern;
value-dependent work (the Ruiz iterations, the block inversion) reruns in
`analyze_numeric` for each one.

## A worked example

[`block_jacobi_solver`][splineax.block_jacobi_solver] wires the three pieces above at
one consistent block size, and picks a lineax solver for you if you don't:

```python
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

import splineax as splx


def laplacian_2d(nx: int, ny: int) -> jnp.ndarray:
    """A 2D 5-point Laplacian: sparse, symmetric positive definite."""
    n = nx * ny
    dense = jnp.zeros((n, n))
    for i in range(nx):
        for j in range(ny):
            k = i * ny + j
            dense = dense.at[k, k].set(4.0)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    dense = dense.at[k, ii * ny + jj].set(-1.0)
    return dense


dense = laplacian_2d(10, 10)
n = dense.shape[0]
matrix = BCOO.fromdense(dense)
operator = splx.BCOOLinearOperator(matrix)

solver = splx.block_jacobi_solver(block_size=8, solver=lx.GMRES(rtol=1e-6, atol=1e-6))
b1 = jnp.ones(n)
solution = lx.linear_solve(operator, b1, solver=solver).value
assert jnp.allclose(matrix @ solution, b1, atol=1e-4)
```

Reusing the analysis across several matrices that share a pattern works exactly like
the direct solvers:

```{.python continuation}
sparsity = matrix  # only the structure matters here

with solver.analyze_symbolic(sparsity) as scope:
    state = scope.init(operator)
    x1 = lx.linear_solve(operator, b1, solver=solver, state=state).value

    other_matrix = 2.0 * dense
    other_operator = splx.BCOOLinearOperator(BCOO.fromdense(other_matrix))
    with scope.analyze_numeric(other_operator) as numeric_state:
        x2 = lx.linear_solve(
            other_operator, b1, solver=solver, state=numeric_state
        ).value

assert jnp.allclose(x1, solution, atol=1e-4)
assert jnp.allclose(other_matrix @ x2, b1, atol=1e-4)
```

## Writing your own transform

A `SystemTransform` and a `Preconditioner` are each two small protocols, structural
rather than base classes: any object with the right methods and fields satisfies them.

```{.python notest}
class SystemTransform(Protocol):
    def analyze_symbolic(self, pattern: MatrixSparsity) -> "AnalyzedTransform": ...


class AnalyzedTransform(Protocol):
    pattern: MatrixSparsity      # the pattern this stage produces
    is_congruence: bool          # whether L == R^T, see below
    def analyze_numeric(self, matrix: BCOO) -> tuple[BCOO, "AppliedTransform"]: ...


class AppliedTransform(Protocol):
    def transform_vector(self, b: Array) -> Array: ...     # applies L
    def recover_solution(self, y: Array) -> Array: ...     # applies R
    def transpose(self) -> "AppliedTransform": ...
    def conj(self) -> "AppliedTransform": ...
```

`Preconditioner` and `AnalyzedPreconditioner` mirror the first two: `analyze_symbolic`
takes a `MatrixSparsity`, and `AnalyzedPreconditioner.analyze_numeric(matrix, tags)`
returns a context manager yielding the `lineax.AbstractLinearOperator` the Krylov
solver conditions on. A plain `NamedTuple` or `eqx.Module` satisfies these by just
declaring the right fields, no inheritance needed. Chain several `SystemTransform`s
with [`compose_transforms`][splineax.transforms.compose_transforms].

`is_congruence` decides whether an operator's lineax tags (symmetry,
positive-semidefiniteness) survive onto the transformed operator:
`PreconditionedIterativeSolver` carries them across only when every transform in the
chain reports `is_congruence = True`, and drops them otherwise. Losing a tag is always
safe (a solver that needs it, like `CG`, will say so); keeping a wrong one is not. Set
`is_congruence = True` only when `L` and `R` are transposes of each other, `L = R^T`
(a permutation applied the same way to rows and columns, or a scaling with equal row
and column factors both do), since only then is `L A R` guaranteed to preserve the
matrix's symmetry and definiteness.

See the [transforms API reference](../api/transforms.md) for the full protocol
reference and the three built-in implementations.
