# Transforming existing Lineax code

Many complicated numerical algorithms are built around linear solves at their
core. These range from root-finding and second-order minimization algorithms, to
differential equation solvers and statistical model fitting procedures built on
top of them. If such algorithms are written using Lineax, they might be able to
accept a splineax solver, but will not be able to make use of the stateful API,
since this requires changing how the solver is used. This means that they have
to be rewritten to benefit of any factorization reuse.

To make this process easier and more adaptable,
`splineax.stateful_solve_transform` provides a function transformation that
takes a "naive" function calling `lineax.linear_solve` multiple times and
threads a solver state through it. The internal solves then reuse a
factorization, without having to change algorithm code.

This builds on the explicit stateful solve API from
[Stateful solves](stateful.md). It is recommended to become familiar with that
API first, in order to understand the changes the transform makes to your
function.

## Usage

Start with a function that calls `lineax.linear_solve`. When you give this to
the transform it runs the function, and each time it encounters a
`lineax.linear_solve` it surrounds it `init`/`update` and `track` calls,
equivalent to `splx.linear_solve`. The first solve builds the state with `init`,
and every later solve folds the operator in with `update`, so a matrix that
shares the pattern reuses the analysis.

```python
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO

import splineax as splx

# KLU needs 64-bit mode.
jax.config.update("jax_enable_x64", True)

dense = jnp.array(
    [
        [10.0, 2.0, 0.0, 0.0],
        [3.0, 14.0, 5.0, 0.0],
        [0.0, 6.0, 18.0, 9.0],
        [0.0, 0.0, 1.0, 12.0],
    ]
)
pattern = BCOO.fromdense(dense)
indices = pattern.indices
tag = splx.sparsity_pattern_tag(pattern)

values = pattern.data
b1 = jnp.array([1.0, 2.0, 3.0, 4.0])
b2 = b1[::-1]

def solve_twice(values, first, second):
    operator = splx.BCOOLinearOperator(
        BCOO((values, indices), shape=(4, 4)), tags=tag
    )
    x1 = lx.linear_solve(operator, first, splx.KLU()).value
    x2 = lx.linear_solve(operator, second, splx.KLU()).value
    return x1 + x2
```

Here is a function written against plain lineax. It solves one matrix against two
right-hand sides, unaware that anything is being reused.

```{.python continuation}
solve_twice_stateful = splx.stateful_solve_transform(solve_twice)
result = solve_twice_stateful(values, b1, b2)

assert jnp.allclose(result, solve_twice(values, b1, b2))
```

The wrapped `solve_twice_stateful` returns the same answer as `solve_twice`. The difference is in how the underlying solver is called the second solve refactors the shared analysis instead of building a new one, roughly equivalent to the following:

```{.python continuation}
def solve_twice_stateful_explicit(values, first, second):
    operator = splx.BCOOLinearOperator(
        BCOO((values, indices), shape=(4, 4)), tags=tag
    )
    x1, state = splx.linear_solve(operator, first, splx.KLU()).value
    x2, state = splx.linear_solve(operator, second, splx.KLU(), state=state).value
    splx.KLU().release(state)
    return x1 + x2
```

Reuse depends on the solver recognising that the two operators share a pattern. That is what
the `sparsity_pattern_tag` above asserts. Without a tag the two solves are treated as
unrelated matrices and each one analyzes from scratch. Operators built from a
[`splineax.SparseJacobianLinearOperator`][] carry such a tag automatically, from their
coloring, so a Jacobian solved at many points needs no tagging.

## Reuse across a loop

A loop that solves every iteration should reuse one analysis for the whole loop.
To achieve this, the transform threads the state through the loop carry.

```{.python continuation}
def iterate(values, b):
    operator = splx.BCOOLinearOperator(
        BCOO((values, indices), shape=(4, 4)), tags=tag
    )

    def step(x, _):
        x = lx.linear_solve(operator, b + 0.1 * x, splx.KLU()).value
        return x, None

    final, _ = jax.lax.scan(step, b, xs=None, length=5)
    return final


run = splx.stateful_solve_transform(iterate)
final = run(values, b1)

assert jnp.allclose(final, iterate(values, b1))
```

This function solves only inside the loop, with no solve beforehand. The transform unrolls the
first iteration to create the state, then carries it through the rest, so you do not need
to seed anything by hand. A `lax.while_loop` works the same way.

## Initial and final solver states

By default the wrapped function returns the output alone, and the transform releases the
state it built once the function returns. Pass `return_final_state=True` to get the state
back and release it yourself.

```{.python continuation}
keep = splx.stateful_solve_transform(solve_twice, return_final_state=True)
output, state = keep(values, b1, b2)
splx.KLU().release(state)
```

You can also seed a call with a state through the `state` keyword, which lets you thread one
state across successive calls. Passing a state makes the wrapped function return the pair by
default, the same as `return_final_state=True`.

## Transforming only specific solve calls

The `filter_solver` argument decides which solves are threaded. Its default is the
`StatefulSolver` protocol, so only the sparse solvers are threaded and a plain dense
Lineax solver passes through untouched.

```{.python continuation}
def mixed(matrix, b):
    return lx.linear_solve(lx.MatrixLinearOperator(matrix), b, lx.LU()).value


out = splx.stateful_solve_transform(mixed)(dense, b1)

assert jnp.allclose(out, jnp.linalg.solve(dense, b1))
```

You can pass a solver class, which is matched with `isinstance`, or a predicate
`(solver) -> bool` callable for finer control.

## Composition with jit, vmap, and grad

The wrapped function is an ordinary JAX computation, so the usual transformations work. An
outer `jit` compiles the threaded solves, `grad` differentiates through them, and `vmap`
batches over operators and right-hand sides while one non-batched state is shared.

```{.python continuation}
compiled = jax.jit(solve_twice_stateful)(values, b1, b2)
gradient = jax.grad(lambda values: jnp.sum(solve_twice_stateful(values, b1, b2) ** 2))(values)
batched = jax.vmap(
    lambda bb: solve_twice_stateful(values, bb, b2)
)(jnp.stack([b1, b2]))
```

## Limitations

The transform threads a single state and covers most control flow. A few cases have limits,
listed here.

- **Reuse needs a shared pattern.** Threading a state does not by itself reuse a
  factorization. The solver reuses one only when it can tell two operators share a pattern,
  which comes from a [`splineax.sparsity_pattern_tag`][]. You have to make sure the operators get this tag, either by passing it through yourself or because the operators are generated from a sparse Jacobian calculation.
- **`lax.cond` needs an initial state.** A solve inside a `cond` branch is threaded only when
  a solve before the `cond` has already created the state, since the untaken branch has to
  return a matching state. A first solve reached only inside a `cond` raises.
- **Custom differentiation raises by default.** A matched solve inside a `custom_jvp` or
  `custom_vjp` raises, since the state cannot cross the custom rule. Set
  `pass_through_custom_diff=True` to let such a solve run without threading, so it works but
  does not reuse a factorization.
- **Multiple solve families may not work.** The transform threads one state, so a single loop that
  interleaves two different solvers or patterns may not work, or perform poorly because factorizations are never reused. You might be able to use `filter_solver` and to separate them and apply the transform multiple times.
- **An explicit `state` argument to `linear_solve` is overridden.** If the wrapped function passes its own
  `state` to a threaded `lineax.linear_solve`, the transform ignores it and substitutes the
  state it threads. A solve the filter skips keeps its explicit state.
- **Loop states share one structure.** A loop carry has a fixed structure, so a state carried
  through a `scan` or `while_loop` must keep one pytree shape. A state from `init_symbolic`
  differs, and the transform normalizes it by unrolling. A solver whose `update` changes the
  structure raises a clear error.
