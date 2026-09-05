# Feature request: return the factorization token from solves

## Summary

`solve_stateful` returns only the solution, so a later `factor` on the same handle has no
data dependency on the solve. Under `jit`, XLA can run the numeric factorization before
the solve that still needs the previous one, and that solve then returns the wrong answer.
Returning the `FactorizationToken` from the solve gives the factorization a real operand to
wait on.

## The ordering problem

`solve_stateful` reads the factorization held by the handle. `factor` overwrites it. Both
take the token from `analyze`, and neither takes anything produced by the other, so nothing
sequences them.

```{.python notest}
tok  = pardiso.analyze(indptr, indices, values1, matrix_type=mt)
tok  = pardiso.factor(tok, indptr, indices, values1, matrix_type=mt)
x1   = pardiso.solve_stateful(tok, indptr, indices, values1, b1, matrix_type=mt)
tok2 = pardiso.factor(tok, indptr, indices, values2, matrix_type=mt)  # overwrites
x2   = pardiso.solve_stateful(tok2, indptr, indices, values2, b2, matrix_type=mt)
```

`x1` is not in the transitive inputs of the second `factor`, so the write can be scheduled
before the read. We see the same failure in the equivalent `klujax` code, where a chain of
reused solves returns the last matrix's solution for every earlier solve in every one of
1500 tested configurations.

## What does not fix it

`FactorizationToken.track` puts its ordering witness in `n_dependent_solutions`. The native
calls take `token.id`, so the witness never reaches them. It is computed and then dropped.

Moving the witness onto `id` does not survive compilation. XLA's algebraic simplifier folds
a multiply by zero into a constant, so the operand reverts to the original id.

`jax.lax.optimization_barrier` is unreliable on the CPU backend, where copy insertion can
override its ordering unless a non-default flag is set (jax-ml/jax#25399).

## Proposed change

Have the solve consume and return the token:

```{.python notest}
solve_stateful(token, indptr, indices, values, right_hand_side, ...) -> (x, FactorizationToken)
```

Same handle, same contents, a distinct value in the graph so later calls can thread it.
Every access to the handle then sits on one chain and the compiler cannot reorder them.

`factor_and_solve_stateful` would want the same treatment, since it both writes and reads.

There is precedent for controlling return arity with a keyword in this library already,
in `return_diagnostics`, so `return_token=True` would fit if a hard break is unwanted.

## A version on the token

A version field on `FactorizationToken`, stamped natively, changed by every `factor` or
`reanalyze` and passed through unchanged by solves, would let a caller tell whether the
handle still holds what a given token expects. The native call can compare it against the
handle's current version and report a mismatch rather than solving.

This matters more here than in a library that can hold several factorizations at once.
With a single factorization per handle, a caller that needs an earlier factorization has to
rebuild it, and it needs to know when that is necessary. The version answers that. It also
catches a caller threading one token into two writes, which JAX cannot prevent
structurally.

## Why this matters for reverse mode

Under `jax.grad`, the adjoint of an earlier solve needs that solve's factorization. Because
the handle holds one factorization, a later `factor` has already replaced it, so the
adjoint solves the wrong matrix and the gradient for the earlier operator comes out wrong.

The backward pass therefore has to refactor back to the earlier values before its transposed
solve, and that refactor-back must be ordered against the adjoint reads. Autodiff emits
those adjoint calls independently with no channel between them, so the ordering has to
travel on a value the native call consumes. The token chain is that channel.

## Secondary request: several factorizations per analysis

MKL PARDISO takes `maxfct`, the number of factors with identical sparsity structure to keep
in memory at once, and `mnum`, which selects the one to use. Neither is exposed here, so a
caller cannot hold two factorizations that share one analysis, and reusing an earlier
factorization always costs a rebuild.

Exposing them would let a caller keep several factorizations alive and skip the
refactor-back in the backward pass entirely. The tradeoff is memory, and `maxfct` is fixed
when the handle is set up, so a caller has to know the count in advance and cannot grow it
inside a `scan` or `while_loop` with a dynamic trip count. That makes it a useful fast path
rather than a replacement for the token threading above.
