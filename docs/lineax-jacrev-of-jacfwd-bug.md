# Lineax bug report: `jacrev(jacfwd)` of a linear solve fails when the operator tangent's matvec output has a leading batch dimension

## Summary

Taking a reverse-mode derivative of a forward-mode derivative of a
`lineax.linear_solve` call raises a cotangent shape error, when the linear
operator's matvec produces a tangent output whose leaf shape differs from the
primal matvec output's shape. A `jax.experimental.sparse.BCOO` matvec is the
natural way to hit this, since its JVP rules batch the tangent values with a
leading `[1, ...]` dimension.

All other second-order compositions (`jacfwd∘jacfwd`, `jacrev∘jacrev`,
`jacfwd∘jacrev`) work, and both first-order derivatives work. Only
`jacrev∘jacfwd` fails.

## Versions

- lineax 0.1.10
- jax 0.10.1
- equinox (bundled with lineax's dependencies)

## Minimal reproduction (pure lineax, no splineax)

```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import lineax as lx
from jax.experimental.sparse import BCOO

D = jnp.array([[10.0, 2.0, 0.0], [3.0, 14.0, 5.0], [0.0, 6.0, 18.0]])
b = jnp.array([1.0, 2.0, 3.0])
sp = BCOO.fromdense(D)
indices, shape = sp.indices, sp.shape
vals = sp.data

def f(v):
    # A linear operator whose matvec is a BCOO product. Differentiating the
    # matvec wrt the values pads the tangent with a leading [1, ...] batch
    # dimension, so the tangent matvec output shape [1, 3] differs from the
    # primal matvec output shape [3].
    matrix = BCOO((v, indices), shape=shape)
    op = lx.FunctionLinearOperator(
        lambda x: matrix @ x, jax.ShapeDtypeStruct((3,), jnp.float64)
    )
    sol = lx.linear_solve(op, b, lx.LU())
    return jnp.sum(sol.value**2)

print(jax.jacfwd(f)(vals).shape)              # ok: (7,)
print(jax.jacrev(f)(vals).shape)              # ok: (7,)
print(jax.jacfwd(jax.jacfwd(f))(vals).shape)  # ok: (7, 7)
print(jax.jacrev(jax.jacrev(f))(vals).shape)  # ok: (7, 7)
print(jax.jacfwd(jax.jacrev(f))(vals).shape)  # ok: (7, 7)
try:
    jax.jacrev(jax.jacfwd(f))(vals)
except ValueError as e:
    # The expected failure, printed verbatim for the report.
    print(e)
```

The last line raises:

```
ValueError: Expected cotangent type float64[1,7] for primal type float64[1,7], but got float64[7,7]
```

## What triggers it

- A `linear_solve` whose operator is differentiated (a tangent on the matrix).
- An operator `mv` whose tangent output has a different leaf shape than the
  primal output. `BCOO.__matmul__`'s JVP batches the values with a leading
  `[1, ...]` dimension, so `jax.jvp` of the matvec returns a `[1, 3]` tangent
  next to a `[3, 3]` primal. A dense `MatrixLinearOperator` whose matvec is a
  plain `@` does not trigger it: all six compositions pass.
- The composition `jacrev(jacfwd(...))`. Forward-then-reverse. The other
  second-order compositions pass.

## Diagnosis

`_linear_solve_jvp` (`lineax/_solve.py`) builds the tangent right-hand side
`b' - A'x` using a `TangentLinearOperator` matvec. `TangentLinearOperator.mv`
runs `eqx.filter_jvp` of the operator's `mv` and materialises the tangent
output, keeping whatever leading batch dimension the operator's JVP produced.
So the staged `-A'x` term enters the jaxpr as a leaf with a `[1, 7]` shape
(inside the solve, the tangent values leaf is `[1, 7]`), while the primal
matvec output leaf is `[7]`-shaped.

Under `jacfwd` alone everything runs forward, so the shape discrepancy is
invisible. Under `jacrev(jacfwd)`, JAX's backward pass walks the staged jaxpr
equation by equation and accumulates cotangents into `GradAccum` cells keyed
by the primal aval. `jax._src.interpreters.ad.ct_check` then rejects the
cotangent: the primal aval is `float64[1,7]` (the padded tangent leaf), and
the incoming cotangent is `float64[7,7]`, batched over the outer reverse
basis. The check only compares shapes, not the semantics of the leading
batch dimension, so it raises.

The underlying disagreement is between two batching conventions: the operator's
JVP pads the tangent with a leading LHS batch dim, while the outer `jacrev`
delivers cotangents with a trailing basis dim. Lineax's transpose
(`_linear_solve_transpose`, `materialise_zeros=True`) expects them to match.

## Suggested fix direction

Either:

- materialise the tangent matvec output to drop the padding dimension inside
  `TangentLinearOperator.mv` / `_linear_solve_jvp` (so the tangent term has
  the same leaf shape as the primal matvec output), or
- reshape the cotangent in the transpose rule to match the padded primal aval
  before accumulating.

## Impact on splineax

splineax's sparse operators (`BCOOLinearOperator`, `BCSRLinearOperator`) all
matvec through a `BCOO`/`BCSR` product, so every second-order
`jacrev∘jacfwd` derivative of a splineax solve hits this. It is tracked as a
strict `xfail` in `tests/unit/test_factorization_reuse.py`
(`test_second_order_jacrev_of_jacfwd_wrt_values`): if lineax fixes it, the
xfail turns into a failure and we get told.

The bug is not caused by splineax's stateful threading or its custom JVP
rule: the pure-lineax reproduction above contains no splineax code.
