# Operators

An *operator* wraps a sparse array into a `lineax.AbstractLinearOperator`, so it can be used
with `lineax.linear_solve` and the rest of the Lineax ecosystem. `splineax` provides one
operator per JAX sparse format.

| Operator | Wraps | Storage |
| --- | --- | --- |
| [`BCOOLinearOperator`][splineax.BCOOLinearOperator] | `jax.experimental.sparse.BCOO` | coordinate (row, col, value) |
| [`BCSRLinearOperator`][splineax.BCSRLinearOperator] | `jax.experimental.sparse.BCSR` | compressed sparse row |

Both wrap a two-dimensional sparse array of shape `(a, b)` and define matrix-vector
products the usual way: `mv` takes a vector of shape `(b,)` and returns one of shape
`(a,)`.

## Constructing operators

```python
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR

import splineax as splx

dense = jnp.array([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]])

bcoo_operator = splx.BCOOLinearOperator(BCOO.fromdense(dense))
bcsr_operator = splx.BCSRLinearOperator(BCSR.fromdense(dense))
```

You can build the underlying `BCOO` / `BCSR` however you like (`fromdense`, or directly
from `(data, indices)` / `(data, indices, indptr)`); the operator just stores it. Integer
matrices are upcast to floating point, since a linear solve needs an inexact dtype.

## Supported operations

Both operators implement the standard `AbstractLinearOperator` surface:

- `operator.mv(vector)` — sparse matrix-vector product.
- `operator.as_matrix()` — densify to an ordinary array (handy for debugging or comparison).
- `operator.T` / `operator.transpose()` — the transpose, kept sparse.
- `operator.in_structure()` / `operator.out_structure()` — shape/dtype metadata.
- Conjugation, used by Lineax when differentiating complex solves.

## Tags

Like other Lineax operators, you can attach `tags` describing structural properties
(symmetry, positive-definiteness, and so on):

```{.python continuation}
import lineax as lx

operator = splx.BCOOLinearOperator(
    BCOO.fromdense(dense), tags=lx.symmetric_tag
)
```

!!! warning

    Tags are **unchecked**. If you tag a matrix with a property it does not have, you may
    get incorrect results. Only tag matrices you are sure about.

## Jacobian operators

A `lineax.JacobianLinearOperator` represents the Jacobian `d(fn)/dx` of a function at a
point without forming the matrix. If you give it a sparsity tag, the splineax solvers will
turn it into a sparse `BCOO` matrix with [asdex](https://github.com/adrhill/asdex). The tag
holds the Jacobian's sparsity pattern and a coloring of it. A coloring splits the columns
(or rows) into groups whose nonzero elements never overlap, so the matrix costs one JVP (or
VJP) per color instead of one per column or row. For example, a function that operates
elementwise has a diagonal Jacobian. All of its columns fit in one color, so a single JVP
computes the whole matrix.

```python
import jax
import jax.numpy as jnp
import lineax as lx

import splineax as splx

# The KLU solver requires 64-bit mode.
jax.config.update("jax_enable_x64", True)


def residual(y, args):
    return 3.0 * y + y**2 + 0.5 * jnp.roll(y, 1) * y


y0 = jnp.linspace(0.5, 1.5, 5)

# Detect the sparsity pattern of `residual` and color it.
tag = splx.sparsity_coloring_tag(residual, y0)
operator = lx.JacobianLinearOperator(residual, y0, tags=tag)

b = jnp.arange(1.0, 6.0)
solution = lx.linear_solve(operator, b, solver=splx.KLU()).value
```

### Sparse materialisation

The solvers call [`materialise_as_bcoo`][splineax.materialise_as_bcoo] to do the
conversion. You can call it yourself when you want the matrix:

```{.python continuation}
jacobian = splx.materialise_as_bcoo(operator).matrix
```

The entries of the `BCOO` come out in the same order as the indices in the tag.

### Pattern tags and coloring tags

There are two ways to create a tag that indicates an operator follows a certain sparsity
pattern:

- [`sparsity_pattern_tag`][splineax.sparsity_pattern_tag] holds only the pattern. The
  first conversion colors it and caches the coloring on the tag, and later conversions
  reuse that coloring.
- [`sparsity_coloring_tag`][splineax.sparsity_coloring_tag] colors the pattern right away.
  It takes a known pattern (a `BCOO`, a `BCSR`, a sparse operator, an
  `asdex.SparsityPattern`, a dense boolean mask, or an `asdex.ColoredPattern`), or a
  function and a point, and then it detects the pattern first. Only the shape and dtype of
  the point matter, so a `jax.ShapeDtypeStruct` works as well.

Both kinds of tag compare equal when their patterns match, so operators that carry either
one share a factorization (see [Stateful solves](stateful.md)). The Jacobian of `residual`
has a diagonal, a subdiagonal and one corner entry, and a tag of that known pattern equals
the detected tag:

```{.python continuation}
from jax.experimental.sparse import BCOO

pattern = BCOO.fromdense(jnp.eye(5) + jnp.eye(5, k=-1) + jnp.eye(5, k=4))
assert splx.sparsity_pattern_tag(pattern) == tag
```

### Coloring direction

With `jac="fwd"` the tag colors columns and the conversion uses JVPs. With `jac="bwd"` it
colors rows and uses VJPs. These names match the `jac` argument of
`lineax.JacobianLinearOperator`. A tag that holds a coloring always uses that coloring.
Without one, the conversion follows the operator's own `jac`, and asdex picks the direction
that needs fewer colors when `jac` is None. A function that only defines a
`jax.custom_vjp` has no forward derivative, so give its operator `jac="bwd"`:

```{.python continuation}
row_tag = splx.sparsity_coloring_tag(residual, y0, jac="bwd")
row_operator = lx.JacobianLinearOperator(residual, y0, tags=row_tag, jac="bwd")
```

### Function operators

A `lineax.FunctionLinearOperator` with a tag converts the same way. This operator is meant
to wrap a linear function, which means its matrix representation is exactly the Jacobian
of `mv` (at any point). This also means a function operator of the JVP of `residual` has
the same indices and values as the Jacobian operator above:

```{.python continuation}
def residual_jvp(tangent):
    _, output_tangent = jax.jvp(lambda y: residual(y, None), (y0,), (tangent,))
    return output_tangent


function_operator = lx.FunctionLinearOperator(
    residual_jvp, jax.ShapeDtypeStruct(y0.shape, y0.dtype), tags=tag
)
function_jacobian = splx.materialise_as_bcoo(function_operator).matrix
assert jnp.array_equal(function_jacobian.indices, jacobian.indices)
assert jnp.allclose(function_jacobian.data, jacobian.data)
```

`lineax.linearise` of a tagged Jacobian operator returns a function operator with the same
tags, so it converts in the same way.

### Reuse across points

The sparsity pattern depends on the computation graph of the function and not on the
point. Create the tag once and use it at every point, for example in a Newton iteration.
Tags are static data of an operator, so a jitted function compiles once for all
operators that share a tag. The shared tag also lets the solver's `update` reuse its
analysis.

```{.python continuation}
import equinox as eqx


@eqx.filter_jit
def newton_step(point, state):
    step_operator = lx.JacobianLinearOperator(residual, point, tags=tag)
    solution, state = splx.linear_solve(
        step_operator, residual(point, None), splx.KLU(), state=state
    )
    return point - solution.value, state


y, state = newton_step(y0, None)
for _ in range(3):
    y, state = newton_step(y, state)
state.release()
```

### Transposed operators

When an operator is transposed, the tags get transformed with it. The transpose of a
tagged `lineax.JacobianLinearOperator` is a `lineax.FunctionLinearOperator` of the VJP,
and it carries the tag of the transposed pattern. A coloring carries over with its
direction swapped, because a column coloring of a matrix is a row coloring of its
transpose. `BCOOLinearOperator` and `BCSRLinearOperator` transpose their tags in the same
way.

```{.python continuation}
transposed = splx.materialise_as_bcoo(operator.T).matrix
assert jnp.allclose(transposed.todense(), jacobian.todense().T)
```

### Limits

- Only real dtypes are supported.
- Pytree inputs and outputs are raveled in leaf order into one vector each.
- To use a tag to compute a sparse Jacobian, it has to be built from concrete indices. A
  tag created under `jax.jit` from traced indices, or by `sparsity_pattern_tag()` with no
  argument, carries only an id, and a solver will reject it for a Jacobian or function
  operator.

## BCOO or BCSR?

Either format solves correctly with any of the solvers, so the choice is mostly about which
format your data already lives in.

- [`Spsolve`][splineax.Spsolve] and [`Pardiso`][splineax.Pardiso] internally need CSR with
  sorted column indices, and will convert/sort a `BCOO` (or an unsorted `BCSR`) for you.
  Doing so on every solve is wasted work if the same operator is reused, so an unsorted
  matrix triggers a [`PerformanceWarning`][splineax.PerformanceWarning]. For a `BCOO`,
  call `.sort_indices()` once yourself to avoid both the cost and the warning.
- [`KLU`][splineax.KLU] consumes coordinate triples and is agnostic to index order; it
  converts a `BCSR` to `BCOO` internally.

If you already have a `BCSR`, use `BCSRLinearOperator`; otherwise `BCOOLinearOperator` is a
fine default.
