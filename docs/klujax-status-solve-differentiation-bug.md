# klujax bug report: the `_with_status` solve primitives have no JVP or transpose rules

## Summary

klujax 0.5.0.post7 ships two families of solve primitives against a numeric
factorization: `solve_with_numeric` / `tsolve_with_numeric`, and their
status-reporting variants `solve_with_numeric_with_status` /
`tsolve_with_numeric_with_status`, which also return a per-handle `RebuildReason`.
The plain family registers differentiation rules:

```text
ad.primitive_jvps[solve_with_numeric_f64] = solve_with_numeric_f64_value_and_jvp
ad.primitive_transposes[solve_with_numeric_f64] = solve_with_numeric_f64_transpose
```

The status family registers neither a JVP nor a transpose rule. Any
differentiation that reaches the status primitive directly therefore fails:

- `jax.jvp` / `jax.grad` of a function calling `solve_with_numeric_with_status`
  raises `NotImplementedError: Differentiation rule for
  'solve_with_numeric_status_f64' not implemented`.
- The transpose direction raises the analogous
  `Transpose rule (for reverse-mode differentiation) for
  'solve_with_numeric_status_f64' not implemented`.

## Versions

- klujax 0.5.0.post7
- jax 0.10.1

## Minimal reproduction

```python
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import klujax

Ai = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
Aj = jnp.array([0, 1, 0, 1, 1, 2], dtype=jnp.int32)
Ax = jnp.array([10.0, 2.0, 3.0, 14.0, 6.0, 18.0])
b = jnp.array([1.0, 2.0, 3.0])

symbol = klujax.analyze(Ai, Aj, 3)
numeric = klujax.factor(Ai, Aj, Ax, symbol)


def f_plain(b):
    x = klujax.solve_with_numeric(numeric, b, symbol)
    return jnp.sum(x**2)


def f_status(b):
    x, _rebuild = klujax.solve_with_numeric_with_status(numeric, b, symbol)
    return jnp.sum(x**2)


print(jax.grad(f_plain)(b))  # ok

try:
    jax.grad(f_status)(b)
except NotImplementedError as e:
    print(e)  # Differentiation rule for 'solve_with_numeric_status_f64' not implemented
```

## Why it matters

The status return is the whole point of the variant: a caller wants the rebuild
reason and the solution from the same call. Today that caller cannot also
differentiate, even though the mathematical operation is identical to the plain
variant: the solve is linear in `b`, so its JVP and transpose are the same solves
against the same factorization, and the rebuild flag is a constant of the
differentiation (its tangent is zero).

## Suggested rules

Both rules can delegate to the plain family, which already has correct JVP and
transpose rules:

- JVP: bind the status primitive for the primal (so the reported rebuild flag
  stays honest), bind the plain primitive for the tangent of the solution, and
  return a zero tangent for the rebuild output.
- Transpose: the rebuild output receives no cotangent (or a zero one), so the
  rule can reuse `solve_with_numeric_transpose` directly, passing the cotangent
  of the solution through.

The same applies to the `tsolve_with_numeric_status_*` pair.

## Workaround used by splineax

splineax's solves under `solve_trace` go through lineax's `linear_solve`
primitive, whose own JVP absorbs any differentiation before it reaches the
klujax primitives, so derivatives work. Code that calls the status primitives
directly under differentiation has no workaround today.
