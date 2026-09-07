# Feature request: return the numeric token from solves

## Summary

`solve_with_numeric` and `tsolve_with_numeric` return only the solution, so a later
in-place `refactor` on the same cache slot has no data dependency on the solve. Under
`jit`, XLA is free to run the write before the read, and the earlier solve then reads the
later matrix. Returning the `NumericToken` from the read ops fixes this by giving the
write a real operand to wait on.

`refactor_and_solve` already returns `(x, NumericToken)`, so this makes the standalone
read consistent with the fused one.

## The ordering problem

A solve reads the numeric slot. An in-place `refactor` writes it. Both take the token
produced by `factor`, and neither takes anything produced by the other, so they are
siblings in the dataflow graph with nothing sequencing them.

Tracing this function shows it directly:

```{.python notest}
n0 = klujax.factor(row, col, values1, symbol)
x1 = klujax.solve_with_numeric(n0, b1, symbol)   # reads the slot, wants values1
n1 = klujax.refactor(row, col, values2, n0, symbol)  # writes the slot, in place
x2 = klujax.solve_with_numeric(n1, b2, symbol)
```

In the jaxpr, `solve_with_numeric` and `refactor` take the same `u64[1]` operand, which is
`factor`'s output. `x1` does not appear anywhere in `refactor`'s transitive inputs. In the
optimized HLO the `refactor_f64` custom call takes `%ffi_call.2` directly, which is the
`factor_f64` call, confirming there is no edge from the solve.

Measured on a chain of solves through one slot with a `free_numeric` at the end, across
matrix sizes 3 to 10 and chain lengths 2 to 6 over 50 seeds: wrong results in 1500 of 1500
configurations. The earlier solves return the later matrix's solution.

## What does not fix it

`NumericToken.track` writes its ordering witness into `n_dependent_solutions`. The native
calls read `.handle`, which is `.id`, so the witness never reaches them. In the traced
jaxpr the `0.0 * x` term is computed and then bound to a dropped variable.

Moving the witness onto `.id` does not work either. XLA's algebraic simplifier folds the
multiply by zero into a constant, so `refactor`'s operand goes back to being `factor`'s
raw output. This is visible in the optimized HLO.

`jax.lax.optimization_barrier` is unreliable here. On the CPU backend its ordering can be
overridden by copy insertion unless `--xla_cpu_copy_insertion_use_region_analysis=true` is
set, which is off by default (jax-ml/jax#25399). Measured, it held on a simple two-call
chain and failed in 120 of 180 configurations once the call path included the
`refactor_with_status` and `rcond` branch.

`custom_call_has_side_effect` does not cover this. `refactor_f64` already carries the flag.
It orders the refactor against other effectful calls, not against a pure
`solve_with_numeric_f64` read.

## Proposed change

Have the read ops consume and return the token:

```{.python notest}
solve_with_numeric(numeric, b, symbolic)  -> (x, NumericToken)
tsolve_with_numeric(numeric, b, symbolic) -> (x, NumericToken)
```

The returned token carries the same cache id and the same contents. It is a distinct value
in the graph, which is the point, because later ops thread it and XLA then cannot hoist a
write above a read.

`rcond` and `condest` also read the factorization and would want the same treatment.

Threading then looks like this, with every slot access on one chain:

```{.python notest}
n0     = klujax.factor(row, col, values1, symbol)
x1, n1 = klujax.solve_with_numeric(n0, b1, symbol)
n2     = klujax.refactor(row, col, values2, n1, symbol)   # ordered after x1
x2, n3 = klujax.solve_with_numeric(n2, b2, symbol)
```

## A version on the token

Along with the above, a version field on `NumericToken` that the native side stamps would
help. Every write, meaning `factor` and `refactor`, produces a new version. Reads pass the
version through unchanged. The native call compares the token's version against the
version the slot currently holds and reports a mismatch instead of solving.

JAX cannot stop a caller from using one token twice, so the threading discipline is not
enforceable at the type level. The version turns that mistake into a reported error rather
than a wrong answer.

## Why this matters for reverse mode

Under `jax.grad`, the adjoint of an earlier solve needs that solve's factorization. With
one slot reused in place, a later `refactor` has already overwritten it, so the adjoint
solves against the wrong matrix and the gradient for the earlier operator is wrong.

Fixing that means the backward pass refactors back to the earlier values before its
`tsolve`, and the refactor-back has to be ordered against the adjoint reads. Autodiff
generates those adjoint calls independently, with no channel between them, so the ordering
has to travel on something the native call actually consumes. The token chain is that
channel. Nothing available in pure Python reaches it, which is what the section above
shows.

## Compatibility

This changes the return arity of two public functions, so it is breaking. Two options that
avoid a hard break:

- A keyword, for example `solve_with_numeric(..., return_token=True)`.
- Separate functions following the existing `refactor_with_status` naming.

Either is fine for our use. The important part is that the token comes back so it can be
threaded into the next call.
