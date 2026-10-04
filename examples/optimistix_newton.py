"""Newton solve of a sparse nonlinear system with `optimistix`, solved by `KLU`.

The system is the one-dimensional Bratu problem on `n` grid points. Its Jacobian is
tridiagonal, so a sparse direct solver factorizes it cheaply. The script solves it twice and
prints a solve profile for each run.

1. A plain `optimistix.root_find`. A sparsity tag is passed through its `tags` argument, so
   the operator that `optimistix` builds for each Newton step is a tagged
   `lineax.FunctionLinearOperator`, and `KLU` turns it into a `BCOO` with the tag's
   coloring. Every Newton step builds a fresh factorization.
2. The same function, wrapped in `splineax.stateful_solve_transform`. The transform threads
   a solver state through the Newton loop, so the first step analyzes and factorizes and
   every later step only refactorizes.

Run it with `uv run --with optimistix python examples/optimistix_newton.py`.
"""

import functools
from collections.abc import Callable
from typing import Any

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import Array, Float

import splineax as splx

# `KLU` factorizes in double precision.
jax.config.update("jax_enable_x64", True)

NUM_POINTS = 1000
GRID_SPACING = 1.0 / (NUM_POINTS + 1)
REACTION_RATE = 1.0


def residual(solution: Float[Array, " n"], reaction_rate: float) -> Float[Array, " n"]:
    """The Bratu residual `u'' + reaction_rate * exp(u)`, with `u = 0` at both ends."""
    padded = jnp.pad(solution, 1)
    second_difference = padded[:-2] - 2.0 * solution + padded[2:]
    return second_difference / GRID_SPACING**2 + reaction_rate * jnp.exp(solution)


class PlainLoopAdjoint(optx.AbstractAdjoint):
    """Runs the Newton loop as a plain `jax.lax.while_loop`, with no custom derivative rule.

    The adjoints that `optimistix` ships wrap the loop in a `jax.custom_jvp`
    (`optimistix.ImplicitAdjoint`) or a `jax.custom_vjp`
    (`optimistix.RecursiveCheckpointAdjoint`). `stateful_solve_transform` cannot thread a
    solver state through either of them. This adjoint runs the same loop without one. It
    gives up reverse-mode differentiation through the solve, and forward mode still works.
    """

    def apply(
        self,
        primal_fn: Callable[..., Any],
        rewrite_fn: Callable[..., Any],
        inputs: Any,
        tags: frozenset[object],
    ) -> Any:
        del rewrite_fn, tags
        plain_while_loop = functools.partial(eqxi.while_loop, kind="lax")
        return primal_fn(inputs + (plain_while_loop,))


def count_operations(profile: splx.SolveProfile) -> dict[str, int]:
    """Count how often each solver operation ran in `profile`."""
    counts: dict[str, int] = {}
    for record in profile.records:
        counts[record.operation] = counts.get(record.operation, 0) + 1
    return counts


def main() -> None:
    initial_guess = jnp.zeros(NUM_POINTS)

    # Detecting the pattern and coloring it happens once, outside of `jax.jit`. The tag
    # holds both, so no solver has to detect or color anything while it runs.
    tag = splx.sparsity_coloring_tag(residual, initial_guess, REACTION_RATE)
    assert tag.coloring is not None
    print(
        f"Jacobian: {NUM_POINTS} x {NUM_POINTS} with {len(tag.indices)} nonzeros, "
        f"colored with {tag.coloring.num_colors} colors "
        f"({tag.coloring.num_colors} JVPs per Jacobian instead of {NUM_POINTS})."
    )

    newton = optx.Newton(rtol=1e-10, atol=1e-10, linear_solver=splx.KLU())

    def solve(
        guess: Float[Array, " n"], reaction_rate: float
    ) -> optx.Solution[Float[Array, " n"], None]:
        return optx.root_find(
            residual,
            newton,
            guess,
            reaction_rate,
            tags=frozenset({tag}),
            adjoint=PlainLoopAdjoint(),
        )

    plain = splx.profile_solves(eqx.filter_jit(solve))
    threaded = splx.profile_solves(eqx.filter_jit(splx.stateful_solve_transform(solve)))

    plain_solution, plain_profile = plain(initial_guess, REACTION_RATE)
    threaded_solution, threaded_profile = threaded(initial_guess, REACTION_RATE)

    for label, solution, profile in [
        ("plain optimistix.root_find", plain_solution, plain_profile),
        ("stateful_solve_transform of it", threaded_solution, threaded_profile),
    ]:
        error = float(jnp.max(jnp.abs(residual(solution.value, REACTION_RATE))))
        print(f"\n{label}")
        print(f"  Newton steps: {int(solution.stats['num_steps'])}")
        print(f"  largest residual: {error:.2e}")
        print(f"  solver operations: {count_operations(profile)}")

    difference = float(jnp.max(jnp.abs(plain_solution.value - threaded_solution.value)))
    print(f"\nlargest difference between the two solutions: {difference:.2e}")

    print("\nsolve profile of the transformed run:")
    print(threaded_profile)


if __name__ == "__main__":
    main()
