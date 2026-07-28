# Advanced usage: separating factorization from solving

A sparse direct solve has two expensive stages:

1. **Symbolic factorization** — analyze the sparsity pattern (which entries fill in).
   Depends only on *where* the nonzeros are, not their values.
2. **Numeric factorization** — compute the actual LU factors. Depends on the values.

If you solve `Ax = b` many times with a fixed matrix, or many matrices that share a
sparsity pattern, you can compute these stages once and reuse them. `splineax` exposes this
through the [`SparseLinearSolver`][splineax.SparseLinearSolver] protocol, which every solver
implements.

!!! note

    [`KLU`][splineax.KLU] and [`Pardiso`][splineax.Pardiso] really do reuse factorizations.
    [`Spsolve`][splineax.Spsolve] implements the same API with **no-op** fallbacks (each
    solve refactors), so code written against the protocol runs unchanged on any backend.
    [`AutoSparseLinearSolver`][splineax.AutoSparseLinearSolver] delegates to whichever it
    picked.

## Reusing a full factorization

Use `solver.factorize(operator)` as a context manager. Inside the block the operator is
factorized once; every `linear_solve` that passes the yielded `state` reuses it.

```python
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

import splineax as splx

# KLU solver requires 64-bit mode:
jax.config.update("jax_enable_x64", True)

dense = jnp.array(
    [
        [10.0, 2.0, 0.0, 0.0],
        [3.0, 14.0, 5.0, 0.0],
        [0.0, 6.0, 18.0, 9.0],
        [0.0, 0.0, 1.0, 12.0],
    ]
)
b1 = jnp.array([1.0, 2.0, 3.0, 4.0])
b2 = b1[::-1]
b3 = b1 + 1.0

operator = splx.BCOOLinearOperator(BCOO.fromdense(dense))
solver = splx.KLU()

with solver.factorize(operator) as state:
    x1 = lx.linear_solve(operator, b1, solver=solver, state=state).value
    x2 = lx.linear_solve(operator, b2, solver=solver, state=state).value
    # ... reuse `state` for as many right-hand sides as you like.
```

This is equivalent to `solver.init(operator, options).factorize()`.

## Reusing a symbolic factorization

If you know the *sparsity pattern* ahead of time but the values change (for example,
solving a family of matrices with identical structure), pre-analyze the pattern with
`solver.factorize_symbolic(sparsity)`. It yields a *scope* offering two options.

```{.python continuation}
sparsity = BCOO.fromdense(dense)  # only the structure matters here

with solver.factorize_symbolic(sparsity) as scope:
    # Option A: reuse the symbolic analysis, refactor numerically on each solve.
    state = scope.init(operator)
    x = lx.linear_solve(operator, b1, solver=solver, state=state).value

    # Option B: also pre-compute the numeric factorization for full reuse.
    with scope.factorize(operator) as numeric_state:
        x1 = lx.linear_solve(operator, b1, solver=solver, state=numeric_state).value
        x2 = lx.linear_solve(operator, b2, solver=solver, state=numeric_state).value
```

`factorize_symbolic` accepts a `BCOO`, `BCSR`, `BCOOLinearOperator`,
`BCSRLinearOperator`, `SparseJacobianLinearOperator`, `SparseFunctionLinearOperator`,
`SparseJacobianLinearOperatorColoring`, or `JacobianColoring`. Only its sparsity
pattern is read. For the two AD operators and the two coloring wrappers, the pattern
comes from the precomputed sparsity, without materialising the matrix numerically.

Besides the sparse operators, a scope's `init` accepts a dense
`lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator` too. It is rebuilt as
its sparse analogue against the sparsity the scope was opened with, and then
materialised, so it costs one evaluation per color rather than one per column or row. A
scope opened from a coloring hands that coloring over directly. One opened from a plain
matrix colors that matrix instead, and does so lazily, only when such an operator
actually arrives.

The scope itself, not just the state it produces, can be passed into a jitted function
that builds the operator inside and calls `scope.init` again, so the analysis is reused
across solves whose values are only known under the trace.

## Bundling the scope into a solver

A scope is only usable together with the solver it came from, so the two normally
travel as a pair: every call site takes both, and every `linear_solve` repeats
`solver=solver, state=scope.init(operator)`. Passing `as_solver=True` to
`factorize_symbolic` collapses that pair into a single object, a
[`SymbolicScopedSparseLinearSolver`][splineax.SymbolicScopedSparseLinearSolver]:

```{.python continuation}
with solver.factorize_symbolic(sparsity, as_solver=True) as scoped_solver:
    x1 = lx.linear_solve(operator, b1, solver=scoped_solver).value
    x2 = lx.linear_solve(operator, b2, solver=scoped_solver).value
```

It is an ordinary `lineax.AbstractLinearSolver`, so it goes wherever the solver it came
from goes, and no `state=` is needed: its `init` *is* the scope's `init`, so
`lineax.linear_solve` builds a state that reuses the symbolic analysis by itself. That
also means it is not a general-purpose solver. It only solves operators sharing the
sparsity pattern its scope was opened with, which is exactly the guarantee that makes
it safe to hand to code that knows nothing about scopes:

```{.python continuation}
def solve_all(solver, operator, right_hand_sides):
    # Knows nothing about factorization reuse, yet reuses the analysis.
    return [lx.linear_solve(operator, b, solver=solver).value for b in right_hand_sides]


with solver.factorize_symbolic(sparsity, as_solver=True) as scoped_solver:
    xs = solve_all(scoped_solver, operator, [b1, b2, b3])
```

The numeric tier is reachable the same way as on the scope itself:
`scoped_solver.factorize(operator)` is `scope.factorize(operator)`, yielding a state to
pass as `state=` for full reuse across right-hand sides. Like the scope it wraps, the
scoped solver is only valid inside its `with` block: the factorization is freed on
exit.

## Solving fully inside `jax.jit`

A factorization handle is an ordinary JAX array value rather than a native object tied
to the Python side, so `solver.factorize_symbolic(...)` may be opened *and* closed
entirely inside a jitted function too, not just called on from outside it:

```{.python continuation}
@jax.jit
def solve_under_jit(values, b):
    operator = splx.BCOOLinearOperator(
        BCOO((values, sparsity.indices), shape=sparsity.shape)
    )
    with solver.factorize_symbolic(operator) as scope:
        state = scope.init(operator)
        return splx.linear_solve(operator, b, solver, state=state).value


x = solve_under_jit(sparsity.data, b1)
```

Use [`splineax.linear_solve`][splineax.linear_solve] here, not `lineax.linear_solve`
directly, whenever the whole `with` block above is itself traced. `lineax.linear_solve`
stages the solve into a trace nested inside the one being built for `solve_under_jit`,
so its result is invisible to the scope's cleanup, which runs once the `with` block
ends, back in the outer trace. `splineax.linear_solve` calls `lineax.linear_solve` and
then additionally registers its result from the outer trace, so the cleanup can order
itself after the solve, which matters for a solver (`Pardiso`) whose native release must
not run before every solve that used the handle has finished. Calling
`lineax.linear_solve` directly here instead raises a `RuntimeError` explaining why, on
any solver that owns a handle. Everywhere else, when the scope is opened outside the
jitted function it is solved within (as throughout the rest of this page), either
function works the same: there is no nested trace for a result to go missing from.

A scoped solver may be opened and closed under one `jax.jit` call in the same way, and
carries no state to pass along:

```{.python continuation}
@jax.jit
def solve_scoped_under_jit(values, b):
    operator = splx.BCOOLinearOperator(
        BCOO((values, sparsity.indices), shape=sparsity.shape)
    )
    with solver.factorize_symbolic(operator, as_solver=True) as scoped_solver:
        return splx.linear_solve(operator, b, scoped_solver).value


x = solve_scoped_under_jit(sparsity.data, b1)
```

`splineax.linear_solve` is required here for the same reason, and does a little more
work for it: with no `state=` given it runs the scoped solver's `init` itself, so that
there is a state to register the solve against at all.

## How the states chain

The protocol describes a small family of state types
([`SparseBasicState`][splineax.solvers.SparseBasicState],
[`SparseSymbolicState`][splineax.solvers.SparseSymbolicState],
[`SparseNumericState`][splineax.solvers.SparseNumericState]) and a scope
([`SparseSymbolicScope`][splineax.solvers.SparseSymbolicScope]):

```
solver.init(operator)                                 -> SparseBasicState
       .factorize()                                   -> SparseNumericState   (context manager)

solver.factorize(operator)                            -> SparseNumericState   (context manager)

solver.factorize_symbolic(sparsity)                   -> SparseSymbolicScope  (context manager)
       .init(operator)                                -> SparseSymbolicState
       .factorize(operator)                           -> SparseNumericState   (context manager)

solver.factorize_symbolic(sparsity, as_solver=True)   -> SymbolicScopedSparseLinearSolver
                                                                               (context manager)
       .init(operator)                                -> SparseSymbolicState
       .factorize(operator)                           -> SparseNumericState   (context manager)
```

Any of these states can be passed as `state=` to `lineax.linear_solve`. The last one is
the scope's two methods again, on an object that is itself a solver, so its states also
get built for you when it is passed as `solver=` without a `state=`.

## Writing backend-agnostic code

All three solvers subclass
[`AbstractSparseLinearSolver`][splineax.AbstractSparseLinearSolver] (and so are usable both
with `lineax.linear_solve` and the factorization API). Type a routine against it and let the
caller pick the solver:

```{.python continuation}
from splineax import AbstractSparseLinearSolver


def solve_many(solver: AbstractSparseLinearSolver, operator, right_hand_sides):
    with solver.factorize(operator) as state:
        return [
            lx.linear_solve(operator, b, solver=solver, state=state).value
            for b in right_hand_sides
        ]


# Fast factorization reuse on CPU, plain (re)solves elsewhere, same code:
solve_many(splx.AutoSparseLinearSolver(), operator, [b1, b2, b3])
```

The [`SparseLinearSolver`][splineax.SparseLinearSolver] protocol describes the same surface
structurally, for when you prefer duck typing or `isinstance` checks.
