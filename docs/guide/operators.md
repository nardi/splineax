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

[`SparseJacobianLinearOperator`][splineax.SparseJacobianLinearOperator] is the sparse
analogue of `lineax.JacobianLinearOperator`: given a function `fn(x, args)` and a point
`x`, it represents the Jacobian `d(fn)/dx` at that point.

The Jacobian's sparsity
pattern and a matching coloring (basically, a partitioning into column/row sets with disjoint nonzero elements) are computed once at construction, using
[asdex](https://github.com/adrhill/asdex), so materialising the Jacobian costs one
JVP or VJP per color rather than one per column or row. For example, a function that operates elementwise will have a diagonal Jacobian, which can be colored with a single color (since all columns have disjoint nonzero elements), and therefore computed with a single JVP.

```python
import jax.numpy as jnp

import splineax as splx


def residual(y, args):
    return 3.0 * y + y**2 + 0.5 * jnp.roll(y, 1) * y


y0 = jnp.linspace(0.5, 1.5, 5)
operator = splx.SparseJacobianLinearOperator(residual, y0)

# One JVP per color, decompressed into a sparse BCOO matrix.
jacobian = operator.as_bcoo()
```

The operator can be handed straight to any splineax solver, which internally reads the
matrix off it with `as_bcoo`:

```{.python continuation}
import jax
import lineax as lx

# KLU solver requires 64-bit mode:
jax.config.update("jax_enable_x64", True)

b = jnp.arange(1.0, 6.0)
solution = lx.linear_solve(operator, b, solver=splx.KLU()).value
```

Three construction paths are available, from least to most precomputed:

- Pass only `fn` and `x`: the sparsity pattern is detected automatically.
- Pass `sparsity=` (an `asdex.SparsityPattern`, a dense boolean mask, or a `BCOO`):
  detection is skipped and only the coloring is computed.
- Pass `coloring=` (an `asdex.ColoredPattern` or a
  [`JacobianColoring`][splineax.JacobianColoring]): both steps are skipped.

The optional `mode=` argument selects `"fwd"` (column coloring, JVPs) or `"bwd"` (row
coloring, VJPs), spelled as lineax and JAX spell it. Only real dtypes are supported, and
a complex one is rejected at construction.

### Pytree inputs and outputs

As in lineax, `fn` may take and return arrays of any shape, or pytrees of them:

```{.python continuation}
def coupled(state, args):
    return {
        "position": state["velocity"] * 0.1 + state["position"] ** 2,
        "velocity": jnp.sin(state["velocity"]) + state["position"],
    }


state = {"position": jnp.linspace(0.0, 1.0, 4), "velocity": jnp.ones(4)}
coupled_operator = splx.SparseJacobianLinearOperator(coupled, state)

# Structures are the ones `fn` has, and `mv` maps pytrees to pytrees.
assert coupled_operator.in_structure() == jax.eval_shape(lambda: state)
tangent = jax.tree.map(jnp.ones_like, state)
product = coupled_operator.mv(tangent)
```

Everything handed to asdex is raveled first, so the sparsity pattern and `as_bcoo` are
over the flattened input and output, and stay two-dimensional. That is what the solvers
consume, and a solve returns a solution shaped like `x`:

```{.python continuation}
rhs = coupled(state, None)
solution = splx.linear_solve(coupled_operator, rhs, solver=splx.KLU()).value
assert jax.tree.structure(solution) == jax.tree.structure(state)
```

`lineax.materialise` is the one operation that cannot carry those structures, since a
[`BCOOLinearOperator`][splineax.BCOOLinearOperator] holds only a matrix. Rather than
dropping them silently it raises, and points at `as_bcoo` instead.

### Converting a dense Jacobian operator

An existing `lineax.JacobianLinearOperator` can be converted with
[`from_jacobian_operator`][splineax.SparseJacobianLinearOperator.from_jacobian_operator],
which carries over its function, point, arguments and tags:

```{.python continuation}
dense_operator = lx.JacobianLinearOperator(residual, y0)
sparse_operator = splx.SparseJacobianLinearOperator.from_jacobian_operator(dense_operator)
```

The `sparsity`, `coloring` and `mode` arguments of the constructor are accepted here
too, so the same three construction paths are available. Left out, the sparsity is
detected, using the operator's own `jac` as the mode.

### Reusing a coloring across points

Sparsity and coloring depend only on the computation graph of `fn`, not on the values
of `x`. When the Jacobian is needed at many points (a Newton iteration, for example),
detect once with
[`SparseJacobianLinearOperatorColoring`][splineax.SparseJacobianLinearOperatorColoring]
and build cheap operators from it:

```{.python continuation}
coloring = splx.SparseJacobianLinearOperatorColoring.detect(residual, y0)

for step in range(3):
    step_operator = coloring.operator_at(y0)
    y0 = y0 - lx.linear_solve(step_operator, residual(y0, None), solver=splx.KLU()).value
```

All operators built from one `SparseJacobianLinearOperatorColoring` share their pytree
structure, so a jitted function accepting them (including the solve above) compiles only
once.

### Carrying a coloring into a jitted function

The coloring itself is a pytree, wrapped as a
[`JacobianColoring`][splineax.JacobianColoring]. It carries no function or point, so it
can be created once (host-side) and passed as an argument into a jitted function that
builds the operator internally. Any two colorings of the same sparsity pattern share a
pytree structure, so regenerating the coloring from scratch does not trigger a
recompile:

```{.python continuation}
import equinox as eqx


@eqx.filter_jit
def newton_step(coloring, point, b):
    operator = splx.SparseJacobianLinearOperator(residual, point, coloring=coloring)
    return point - lx.linear_solve(operator, b, solver=splx.KLU()).value


coloring = splx.JacobianColoring.detect(residual, y0)
y0 = newton_step(coloring, y0, residual(y0, None))
```

A [`JacobianColoring`][splineax.JacobianColoring] is created either by detection with
`JacobianColoring.detect(fn, x)` or, when the sparsity is already known, by
`JacobianColoring.from_sparsity(sparsity)`. To bind a coloring to a function and reuse
it across points, hand it to
`SparseJacobianLinearOperatorColoring.from_jacobian_coloring(coloring, fn, x)`.

`KLU.factorize_symbolic` (see [the advanced guide](advanced.md)) also accepts a
`SparseJacobianLinearOperator`, a `SparseJacobianLinearOperatorColoring`, or a
`JacobianColoring` directly, reading the indices from the precomputed pattern without
materialising the Jacobian numerically.

## Function operators

[`SparseFunctionLinearOperator`][splineax.SparseFunctionLinearOperator] is the sparse
analogue of `lineax.FunctionLinearOperator`. Where the Jacobian operator takes a
function and a point, this one takes a function that is *already linear*, together with
the structure of its input:

```{.python continuation}
_, jvp_of_residual = jax.linearize(lambda y: residual(y, None), y0)

function_operator = splx.SparseFunctionLinearOperator(
    jvp_of_residual, jax.eval_shape(lambda: y0)
)
```

Sparsity detection, coloring and materialisation work exactly as they do for the
Jacobian operator, and the `sparsity=`, `coloring=` and `mode=` arguments mean the same
things. The one difference is that there is no evaluation point to speak of: a linear
map has the same matrix everywhere, so the operator traces at the zeros of the input
structure.

The two operators are two views of the same thing, and they agree on it. Linearising a
function and wrapping the result gives the same operator as taking its Jacobian
directly:

```{.python continuation}
jacobian_operator = splx.SparseJacobianLinearOperator(residual, y0)
assert jnp.allclose(function_operator.as_matrix(), jacobian_operator.as_matrix())
```

That is also what `lineax.linearise` returns when handed a
[`SparseJacobianLinearOperator`][splineax.SparseJacobianLinearOperator]. It caches the
primal pass with `jax.linearize` and passes the coloring on, so the result still
materialises sparsely:

```{.python continuation}
linearised = lx.linearise(jacobian_operator)
assert isinstance(linearised, splx.SparseFunctionLinearOperator)
```

Linearity is unchecked. A function that is not linear will give wrong answers rather
than an error, since the operator materialises its Jacobian at a single point and takes
that to be the whole matrix.

### Converting a dense function operator

An existing `lineax.FunctionLinearOperator` can be converted with
[`from_function_operator`][splineax.SparseFunctionLinearOperator.from_function_operator],
which carries over its function, input structure and tags:

```{.python continuation}
dense_function_operator = lx.FunctionLinearOperator(
    jvp_of_residual, jax.eval_shape(lambda: y0)
)
sparse_function_operator = splx.SparseFunctionLinearOperator.from_function_operator(
    dense_function_operator
)
```

As with the Jacobian operator, the `sparsity`, `coloring` and `mode` arguments are
accepted here too, and a symbolic scope will do this conversion itself when a dense
`lineax.FunctionLinearOperator` reaches its `init`.

## BCOO or BCSR?

Either format solves correctly with any of the solvers, so the choice is mostly about which
format your data already lives in.

- [`Spsolve`][splineax.Spsolve] internally needs CSR with sorted column indices, and will
  convert/sort a `BCOO` (or an unsorted `BCSR`) for you.
- [`KLU`][splineax.KLU] consumes coordinate triples and is agnostic to index order; it
  converts a `BCSR` to `BCOO` internally.

If you already have a `BCSR`, use `BCSRLinearOperator`; otherwise `BCOOLinearOperator` is a
fine default.
