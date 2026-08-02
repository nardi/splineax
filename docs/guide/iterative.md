# Iterative solvers

The direct solvers in this package factor the matrix. An iterative solver instead
applies it over and over, building up a solution from Krylov subspaces, and how quickly
it gets there depends almost entirely on the *preconditioner*: an approximate inverse
`M`, applied alongside `A`, chosen so the iteration converges in a few steps rather than
thousands.

Lineax already provides `CG`, `BiCGStab` and `GMRES`, and each accepts a preconditioner
through `options["preconditioner"]`. What it does not provide is a way to *build* one.
That is the gap [`PreconditionedIterativeLinearSolver`][splineax.PreconditionedIterativeLinearSolver]
fills, and it is a sparse problem: a good preconditioner is built in two phases --
analyse the pattern once, rebuild cheaply from each new set of values -- which is the
same split [`factorize_symbolic`](advanced.md) already exposes for the direct solvers.

```python
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO

import splineax as splx

jax.config.update("jax_enable_x64", True)


def block_matrix(num_blocks=8, block_size=4):
    """Diagonal blocks spanning many orders of magnitude, weakly coupled."""
    rng = np.random.default_rng(3)
    size = num_blocks * block_size
    dense = np.zeros((size, size))
    for block in range(num_blocks):
        rows = slice(block_size * block, block_size * (block + 1))
        dense[rows, rows] = 10.0 ** (block - num_blocks // 2) * (
            rng.uniform(-1.0, 1.0, (block_size, block_size))
            + 3.0 * np.eye(block_size)
        )
    for i in range(size - block_size):
        dense[i, i + block_size] += 1e-3 * dense[i, i]
    return jnp.asarray(dense)


dense = block_matrix()
operator = splx.BCOOLinearOperator(BCOO.fromdense(dense))
b = jnp.ones(dense.shape[0])

solver = splx.PreconditionedIterativeLinearSolver(
    lx.GMRES(rtol=1e-8, atol=1e-12, max_steps=2000, restart=8),
    splx.BlockJacobi(blocks=4),
    sparsity=operator,
)
solution = lx.linear_solve(operator, b, solver=solver)
assert jnp.allclose(solution.value, jnp.linalg.solve(dense, b))
```

The matrix above is badly scaled -- its blocks span eight orders of magnitude -- and
unpreconditioned GMRES cannot solve it at all within its step budget, while block Jacobi
removes the scaling and converges in a handful of iterations:

```{.python continuation}
bare = lx.linear_solve(
    operator, b, solver=lx.GMRES(rtol=1e-8, atol=1e-12, max_steps=2000, restart=8),
    throw=False,
)
assert bare.result != lx.RESULTS.successful
assert solution.stats["num_steps"] < bare.stats["num_steps"]
```

## Parameters, fulfilled by values or by injected providers

**A preconditioner is parametrized, and each parameter can be fulfilled either by a
fixed value or by an object that generates that value and is injected as a dependency.**

[`BlockJacobi`][splineax.BlockJacobi] needs a partition of the index range into diagonal
blocks. When you know the structure, state it:

```{.python continuation}
fixed = splx.BlockJacobi(blocks=4)             # uniform 4x4 blocks
ragged = splx.BlockJacobi(blocks=(3, 5, 4))    # explicit sizes
assert fixed.blocks == 4
```

When you do not, inject something that works it out from the sparsity pattern:

```{.python continuation}
derived = splx.BlockJacobi(blocks=splx.MaximalCaptureBlockPartitioner())
assert isinstance(derived.blocks, splx.BlockPartitioner)
```

Both are the same field, so the two are interchangeable at the call site. On a matrix
that really is block diagonal, the partitioner derives exactly the partition the fixed
value states, and the resulting preconditioners are identical:

```{.python continuation}
clean = jnp.asarray(np.kron(np.eye(8), np.ones((4, 4)) + 3.0 * np.eye(4)))
clean_operator = splx.BCOOLinearOperator(BCOO.fromdense(clean))


def build(preconditioner, on=clean_operator):
    return preconditioner.symbolic(on).numeric(on).left_operator()


assert jnp.allclose(build(fixed).as_matrix(), build(derived).as_matrix())
```

Providers are typed by **minimal protocols**, never base classes. Anything with a
`partition` method satisfies [`BlockPartitioner`][splineax.BlockPartitioner]:

```{.python continuation}
class EveryOtherIndex:
    """A partitioner that pairs up neighbouring indices."""

    def partition(self, pattern):
        return splx.BlockPartition.uniform(pattern.shape[0], 2)


assert isinstance(EveryOtherIndex(), splx.BlockPartitioner)
assert splx.BlockJacobi(blocks=EveryOtherIndex()).symbolic(operator).partition.sizes[0] == 2
```

Nothing subclasses anything, and a partitioner cannot be injected where a different kind
of parameter is wanted, because the protocols differ.

!!! note "There is no default partition"

    `BlockJacobi()` is an error. A partition guessed on your behalf would silently
    decide how much of the matrix gets discarded, which is exactly the decision you
    should be making.

## What block Jacobi keeps, and what it throws away

`M = blockdiag(A)^-1`. Everything outside the diagonal blocks is **discarded** -- that
is what "block Jacobi" means, not an approximation that gets better with effort. The
preconditioner is good exactly when the discarded coupling is weak.

[`coverage`][splineax.coverage] reports how much survives, so the loss is visible rather
than implied:

```{.python continuation}
partition = splx.BlockPartition.uniform(operator.in_size(), 4)
assert splx.coverage(operator, partition) < 1.0
```

[`MaximalCaptureBlockPartitioner`][splineax.MaximalCaptureBlockPartitioner] chooses
blocks with that in mind. It first finds the coarsest partition that is *exactly* block
diagonal -- the connected components of the pattern -- which by construction captures
everything. Since a single far-off-diagonal entry can fuse the whole matrix into one
component, and inverting an `n x n` "block" is the cost block Jacobi exists to avoid,
components longer than `max_block_size` are cut. Where they are cut is the point of the
name: the cuts go where they discard the fewest entries, not where the cap happens to
fall.

```{.python continuation}
clumped = np.zeros((12, 12))
clumped[0:6, 0:6] = clumped[6:12, 6:12] = 1.0
clumped[0, 11] = 1.0  # one entry fuses both clumps into a single component
clumped_operator = splx.BCOOLinearOperator(BCOO.fromdense(jnp.asarray(clumped)))

partitioner = splx.MaximalCaptureBlockPartitioner(max_block_size=8)
sizes = splx.BlockJacobi(blocks=partitioner).symbolic(clumped_operator).partition.sizes
assert sizes == (6, 6)  # cut between the clumps, not at the cap
```

## Inverting the blocks

| `factorization` | what it does |
| --- | --- |
| `"svd"` | pseudo-inverse. Rank-revealing, and finite even for a singular block. **The default.** |
| `"lu"` | plain inverse. Cheapest, but gives `inf`/`nan` on a singular block with no error to show for it. |
| `"qr"` | rank-revealing QR with column pivoting. Truncates deficient directions rather than inverting them, so it stays finite --- though the result is a one-sided inverse on the kept subspace, not the Moore-Penrose pseudo-inverse `"svd"` computes. CPU and GPU only: JAX does not implement pivoting on TPU, where this raises. |
| `"auto"` | `"lu"` if `assume_nonsingular`, otherwise `"svd"`. |

All three are batched over the whole block stack, so inverting is a single XLA call
however many blocks there are.

The default is `"svd"` because of the failure mode of `"lu"`: a singular block yields
`nan`, the Krylov solver then returns `nan`, and the result still reports success.
Nothing announces it. Where the pattern *proves* a block is singular -- a block with a
structurally empty row or column, once the out-of-block coupling is discarded -- that is
caught up front instead:

```{.python continuation}
empty = np.zeros((4, 4))
empty[0, 0] = empty[2, 2] = empty[3, 3] = 1.0  # index 1 has no entries at all
empty_operator = splx.BCOOLinearOperator(BCOO.fromdense(jnp.asarray(empty)))

try:
    splx.BlockJacobi(blocks=2, factorization="lu").symbolic(empty_operator)
except ValueError as error:
    assert "structurally singular" in str(error)
```

`assume_nonsingular=True` asserts that every block is invertible, letting `"auto"` take
the cheap route. It is checked against the pattern: if a block is structurally singular
the assertion is provably false and raises rather than being taken on trust.

## Transforms are global

A reordering applies to the whole system or not at all. It is not a parameter of any one
preconditioner, and nesting it inside a block partitioner would make that object
responsible for far more than deriving its own parameter. So transforms are an ordered
sequence on the **solver**, applied before the preconditioner's parameters are resolved:

```{.python continuation}
solver = splx.PreconditionedIterativeLinearSolver(
    lx.GMRES(rtol=1e-8, atol=1e-12, max_steps=2000, restart=8),
    splx.BlockJacobi(blocks=splx.MaximalCaptureBlockPartitioner(max_block_size=8)),
    transforms=(splx.ReverseCuthillMcKee(),),
    sparsity=operator,
)
reordered = lx.linear_solve(operator, b, solver=solver)
assert jnp.allclose(reordered.value, jnp.linalg.solve(dense, b))
```

The solver permutes the right-hand side and un-permutes the solution, so the reordering
is invisible from outside. The preconditioner only ever sees the pattern the transforms
produced -- [`ReverseCuthillMcKee`][splineax.ReverseCuthillMcKee] narrows the bandwidth,
and the partitioner then analyses the narrowed pattern, without either knowing the other
exists.

That decoupling is what lets the two compose in either combination. A stage is anything
satisfying [`SymbolicTransform`][splineax.SymbolicTransform] (derivable from the pattern,
like a reordering or a bipartite matching) or
[`NumericTransform`][splineax.NumericTransform] (needing the values, like equilibration).

!!! note "Tags have to survive the transform"

    `lineax.CG` refuses an operator not tagged positive- or negative-semidefinite, and
    the operator it sees is the *transformed* one. A symmetric permutation preserves
    symmetry and definiteness, so those tags carry over; structural tags (diagonal,
    triangular, tridiagonal) never do. An unsymmetric transform -- a row permutation
    from a matching -- drops definiteness too, which is correct, and means such a
    combination belongs with `GMRES` or `BiCGStab` rather than `CG`.

## Which side, and which method

Each lineax Krylov method applies its preconditioner in exactly one way, so the choice is
not yours to make:

| Solver | Side | Also requires |
| --- | --- | --- |
| `lineax.GMRES` | left -- `M A x = M b` | |
| `lineax.BiCGStab` | right -- `A M y = b`, `x = M y` | |
| `lineax.CG` | split across both | a positive definite operator *and* preconditioner |

A preconditioner declares which sides it supports, so a mismatch is caught when the
solver is **constructed** rather than at the first solve:

```{.python continuation}
class LeftOnly(splx.BlockJacobi):
    @property
    def sides(self):
        return frozenset({splx.Side.LEFT})


try:
    splx.PreconditionedIterativeLinearSolver(lx.BiCGStab(rtol=1e-6, atol=1e-6), LeftOnly(blocks=4))
except ValueError as error:
    assert "does not support: RIGHT" in str(error)
```

`BlockJacobi` supports both: `M` is a single operator, applicable either way round.
They are separate methods on the numeric tier because that is not true in general -- a
split incomplete factorisation would supply `L^-1` on the left and `U^-1` on the right.

For `CG`, the operator must be tagged definite, and the preconditioner inherits that tag
automatically (every principal submatrix of a definite matrix is definite, and so is its
inverse):

```{.python continuation}
spd = dense @ dense.T + jnp.eye(dense.shape[0])
spd_operator = splx.BCOOLinearOperator(
    BCOO.fromdense(spd), lx.positive_semidefinite_tag
)
cg_solver = splx.PreconditionedIterativeLinearSolver(
    lx.CG(rtol=1e-8, atol=1e-12, max_steps=2000),
    splx.BlockJacobi(blocks=4),
    sparsity=spd_operator,
)
cg_solution = lx.linear_solve(spd_operator, b, solver=cg_solver)
assert jnp.allclose(cg_solution.value, jnp.linalg.solve(spd, b), atol=1e-6)
```

## Why `sparsity` is given up front

Analysing a pattern -- detecting blocks, computing an ordering -- happens in Python, on
concrete index values. `lineax.linear_solve` is itself jit-wrapped, so by the time it
calls `init` the operator's indices are *tracers*, and no host-side analysis is possible
there even outside any `jax.jit` of your own.

Passing `sparsity` does the analysis once, when the solver is built. That is also where
you want it: it is then reused by every solve. Two other routes analyse outside the
trace and so need no `sparsity` -- calling `init` yourself:

```{.python continuation}
unprepared = splx.PreconditionedIterativeLinearSolver(
    lx.GMRES(rtol=1e-8, atol=1e-12, max_steps=2000, restart=8),
    splx.BlockJacobi(blocks=4),
)
state = unprepared.init(operator, {})
assert jnp.allclose(
    lx.linear_solve(operator, b, solver=unprepared, state=state).value,
    jnp.linalg.solve(dense, b),
)
```

or opening a symbolic scope, which is the right choice when the same pattern is solved
with many different sets of values:

```{.python continuation}
with unprepared.factorize_symbolic(operator) as scope:
    for scale in (1.0, 7.0):
        scaled = splx.BCOOLinearOperator(BCOO.fromdense(scale * dense))
        value, result, _ = unprepared.compute(scope.init(scaled), b, {})
        assert result == lx.RESULTS.successful
        assert jnp.allclose(value, jnp.linalg.solve(scale * dense, b))
```

This is where the two-phase split pays. The reordering, the block detection and the
scatter indices are computed once per *pattern*; each `scope.init` redoes only the
scatter-add and the batched inverse. Omitting all three routes raises an error naming
them, rather than silently falling back to something slower.

## Sparse-specific notes

- Only square operators, as elsewhere in this package.
- The operator must be sparse (`BCOOLinearOperator`, `BCSRLinearOperator`, or a
  `SparseJacobianLinearOperator`): the preconditioner needs a sparsity pattern, and a
  dense operator has nothing to analyse.
- The preconditioner is a
  [`BlockDiagonalLinearOperator`][splineax.BlockDiagonalLinearOperator], which stays
  sparse through lineax's `linearise`, so a preconditioned solve never materialises an
  `n x n` array.
- `result` and `stats` pass through untouched, so `RESULTS.max_steps_reached`,
  breakdown, `stats["num_steps"]` and `throw=True` behave exactly as they do with the
  bare lineax solver.
- Supplying `options["preconditioner"]` yourself raises rather than being silently
  overwritten.
- Applied to a *transposed* solve, the preconditioning side swaps, inherited from lineax
  as-is. Moot for a symmetric preconditioner such as `BlockJacobi`.
- Do not reach for this through
  [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver]: direct and iterative
  solvers are not interchangeable, and the choice is yours to make.
