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

import splineax

dense = jnp.array([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]])

bcoo_operator = splineax.BCOOLinearOperator(BCOO.fromdense(dense))
bcsr_operator = splineax.BCSRLinearOperator(BCSR.fromdense(dense))
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

```python
import lineax as lx

operator = splineax.BCOOLinearOperator(
    BCOO.fromdense(dense), tags=lx.symmetric_tag
)
```

!!! warning

    Tags are **unchecked**. If you tag a matrix with a property it does not have, you may
    get incorrect results. Only tag matrices you are sure about.

## BCOO or BCSR?

Either format solves correctly with any of the solvers, so the choice is mostly about which
format your data already lives in.

- [`Spsolve`][splineax.Spsolve] internally needs CSR with sorted column indices, and will
  convert/sort a `BCOO` (or an unsorted `BCSR`) for you.
- [`KLU`][splineax.KLU] consumes coordinate triples and is agnostic to index order; it
  converts a `BCSR` to `BCOO` internally.

If you already have a `BCSR`, use `BCSRLinearOperator`; otherwise `BCOOLinearOperator` is a
fine default.
