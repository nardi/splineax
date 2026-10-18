import equinox as eqx
from jaxtyping import Array, Bool, Integer


class SolverDiagnostics(eqx.Module):
    """What a solver reports about the factorization it holds.

    Every field is `None` when the solver does not report it, and an array otherwise. A
    `None` field is part of the pytree structure, so the diagnostics of one state keep the
    same structure inside a `jax.lax.while_loop`, `jax.lax.cond` or `jax.lax.scan`. A solver
    that reports nothing holds `SolverDiagnostics()`.

    The diagnostics of a state are `state.diagnostics`. A solve also returns them as
    `solution.stats["diagnostics"]`, which is the way to read them inside
    `splineax.stateful_solve_transform`, where the final state is only returned at the end.
    """

    positive_definite: Bool[Array, ""] | None = None
    """Whether the matrix is positive definite, which means every pivot is positive and none
    was perturbed. A singular matrix whose zero pivot was replaced by a small number is
    not reported as definite."""

    positive: Integer[Array, ""] | None = None
    """The number of positive eigenvalues of the factored matrix."""

    negative: Integer[Array, ""] | None = None
    """The number of negative eigenvalues of the factored matrix."""

    perturbed_pivots: Integer[Array, ""] | None = None
    """The number of pivots the solver replaced by a small number, which is nonzero for a
    singular or nearly singular matrix."""
