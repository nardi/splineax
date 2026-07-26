"""Every benchmark parameter, in one place, overridable by environment variable.

Each name below is read from `SPLINEAX_BENCH_<NAME>`, so a cheap smoke run needs no code
edit:

    SPLINEAX_BENCH_SIZES=50,150 SPLINEAX_BENCH_FAMILIES=band uv run pytest benchmarks

`environs` does the parsing and type coercion. `resolved()` dumps the whole set for
recording in the benchmark metadata, so a result file always says what produced it.
"""

import os

from environs import Env

# Treat an empty value as unset. The CI workflow passes blank `workflow_dispatch` inputs
# through as empty strings, and an empty string is not a valid value for any parameter
# here, so it should fall back to the default rather than fail to parse.
for _name in [
    name
    for name, value in os.environ.items()
    if name.startswith("SPLINEAX_BENCH_") and not value.strip()
]:
    del os.environ[_name]

env = Env(prefix="SPLINEAX_BENCH_")

# Log-spaced, six points. Six is about the minimum for a credible power-law fit, and
# 10000 is the ceiling because unstructured fill-in at that size is already the dominant
# cost of a full run.
SIZES: tuple[int, ...] = tuple(
    env.list("SIZES", [50, 150, 500, 1500, 5000, 10000], subcast=int)
)

# See `matrices.py` for what each family is and why it is here.
FAMILIES: tuple[str, ...] = tuple(
    env.list("FAMILIES", ["band", "grid2d", "graph", "random"])
)

# Distinct matrices per (family, size). More than one widens the fitted confidence
# interval honestly, at linear cost in runtime.
REPLICATES: int = env.int("REPLICATES", 1)

# 1 exercises the one-off path, 100 the batched path. See the README table for what the
# (mode, n_rhs) combinations each mean, which is not always the obvious thing.
N_RHS: tuple[int, ...] = tuple(env.list("N_RHS", [1, 100], subcast=int))

# Factorization-reuse tier. Determines what is set up outside the timer.
MODES: tuple[str, ...] = tuple(env.list("MODES", ["none", "symbolic", "numeric"]))

# `auto` is excluded on purpose: it dispatches to one of these three, so benchmarking it
# would silently duplicate whichever one it picks.
SOLVERS: tuple[str, ...] = tuple(env.list("SOLVERS", ["spsolve", "klu", "pardiso"]))

# How multiple right-hand sides are batched. "vmap" prefers `jax.vmap` and falls back to
# `jax.lax.map` for the pairs that reject it (Pardiso symbolic and numeric, see
# `batching.py`). "lax_map" forces the sequential form everywhere, which makes all
# solvers directly comparable at the cost of hiding klujax's native batching rule.
BATCHING: str = env.str("BATCHING", "vmap")

# Half-width of the `band` family: 2 gives 5 diagonals, so 5 nonzeros per row.
BAND_HALFWIDTH: int = env.int("BAND_HALFWIDTH", 2)

# Nonzeros per row for the `random` family. 6 rather than 5 keeps it visibly the densest
# and least structured of the four.
RANDOM_NNZ_PER_ROW: int = env.int("RANDOM_NNZ_PER_ROW", 6)

# Degree of the `graph` family. 4 gives exactly 5 nonzeros per row including the
# diagonal, matching `band` and `grid2d`, so those three differ only in topology. Must
# be even, see `matrices.py`.
GRAPH_DEGREE: int = env.int("GRAPH_DEGREE", 4)

# Double-edge swaps per vertex when randomizing the `graph` circulant seed. 10 is well
# past the point where the result stops looking like a circulant.
GRAPH_SWAP_FACTOR: int = env.int("GRAPH_SWAP_FACTOR", 10)

# Mixed into every matrix seed, so a whole run can be re-randomized at once.
SEED_BASE: int = env.int("SEED_BASE", 20260726)

# Sizes below this are dominated by fixed dispatch overhead, which biases the fitted
# exponent downward, so the fit ignores them. They are still measured and plotted.
FIT_MIN_N: int = env.int("FIT_MIN_N", 500)

# Relative residual a warm-up solve must achieve before its benchmark is allowed to
# report a timing. No solver here detects singularity, so this is the only thing
# standing between a bad matrix and a plausible-looking number.
RESIDUAL_TOL: float = env.float("RESIDUAL_TOL", 1e-8)


def resolved() -> dict[str, object]:
    """Every parameter actually in effect, for recording in benchmark metadata."""
    return {
        "sizes": list(SIZES),
        "families": list(FAMILIES),
        "replicates": REPLICATES,
        "n_rhs": list(N_RHS),
        "modes": list(MODES),
        "solvers": list(SOLVERS),
        "batching": BATCHING,
        "band_halfwidth": BAND_HALFWIDTH,
        "random_nnz_per_row": RANDOM_NNZ_PER_ROW,
        "graph_degree": GRAPH_DEGREE,
        "graph_swap_factor": GRAPH_SWAP_FACTOR,
        "seed_base": SEED_BASE,
        "fit_min_n": FIT_MIN_N,
        "residual_tol": RESIDUAL_TOL,
    }
