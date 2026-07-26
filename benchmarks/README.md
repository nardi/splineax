# Solver benchmarks

How the three sparse direct solvers (`Spsolve`, `KLU`, `Pardiso`) scale across matrix
families, sizes, factorization-reuse tiers, and right-hand-side counts. Produces plots and
an empirical `t ~ n**p` exponent per configuration.

These are **not** part of the default test run. `testpaths` in `pyproject.toml` covers only
`tests`, `docs` and `README.md`, so the benchmarks run only when this path is given
explicitly.

## Running them

```
uv sync --locked --extra pardiso --group bench

uv run pytest benchmarks \
    --benchmark-json=benchmarks/results/bench.json \
    --benchmark-warmup=on --benchmark-min-rounds=3 --benchmark-disable-gc \
    --benchmark-sort=name --benchmark-columns=min,median,mean,stddev,rounds

uv run python benchmarks/plot_results.py benchmarks/results/bench.json
```

That is the same command the CI workflow runs, so local and CI results are comparable.
Round counts stay autocalibrated: `--benchmark-min-rounds` only lowers the floor from 5.

A full default run takes roughly 20 to 50 minutes. The dominant cell is `Spsolve` with 100
right-hand sides on `random` at n=10000, which is 100 unstructured factorizations per
round for reasons explained under [Spsolve and multiple right-hand
sides](#spsolve-and-multiple-right-hand-sides). Trim while iterating:

```
SPLINEAX_BENCH_SIZES=50,150 SPLINEAX_BENCH_FAMILIES=band \
    uv run pytest benchmarks -q --benchmark-min-rounds=1
```

## Configuration

Every parameter is a named constant in [config.py](config.py), overridable by environment
variable, so no run needs a code edit. Each is read from `SPLINEAX_BENCH_<NAME>`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SIZES` | `50,150,500,1500,5000,10000` | Nominal matrix sizes, log-spaced. Six points is about the minimum for a credible power-law fit. |
| `FAMILIES` | `band,grid2d,graph,random` | Which sparsity patterns to generate. |
| `REPLICATES` | `1` | Distinct matrices per (family, size). Above 1 widens the fitted interval honestly. |
| `N_RHS` | `1,100` | Right-hand sides per solve. |
| `MODES` | `none,symbolic,numeric` | Factorization-reuse tier. |
| `SOLVERS` | `spsolve,klu,pardiso` | Solvers to measure. `auto` is excluded because it only dispatches to these. |
| `BATCHING` | `vmap` | How multiple right-hand sides are batched. See [Batching](#batching). |
| `BAND_HALFWIDTH` | `2` | Half-width of the `band` family, so 5 diagonals. |
| `RANDOM_NNZ_PER_ROW` | `6` | Nonzeros per row for the `random` family. |
| `GRAPH_DEGREE` | `4` | Degree of the `graph` family. Must be even. |
| `GRAPH_SWAP_FACTOR` | `10` | Double-edge swaps per vertex when randomizing the `graph` seed. |
| `SEED_BASE` | `20260726` | Mixed into every matrix seed, so a run can be re-randomized wholesale. |
| `FIT_MIN_N` | `500` | Sizes below this are excluded from the fit. See [The fit](#the-fit). |
| `RESIDUAL_TOL` | `1e-8` | Relative residual a warm-up solve must hit before its timing is reported. |

## Matrix families

All four are square, real, float64, coalesced, and strictly diagonally dominant. Dominance
is not cosmetic: no solver here detects singularity and `RESULTS.successful` is returned
unconditionally, so an ill-conditioned matrix would silently benchmark garbage. Values are
asymmetric even where the pattern is symmetric, because `Pardiso` is hardcoded to
`REAL_NONSYMMETRIC` and no solver exploits numeric symmetry.

| Family | Pattern | nnz/row | Why it is here |
| --- | --- | --- | --- |
| `band` | banded, half-width 2 | 5 | The spline and collocation case this library exists for. Should fit near p=1. |
| `grid2d` | 5-point stencil on a square grid | 5 | Structured 2D, moderate fill-in. Rounds `n` to a perfect square. |
| `graph` | random connected 4-regular graph | 5 | An expander, so no small vertex separators. The honest hard case. |
| `random` | uniformly random off-diagonals | 6 | Unstructured, and the only family with an asymmetric pattern. |

Two properties make the comparison meaningful:

**Constant nonzeros per row.** Every family holds nnz per row fixed as `n` grows, so `nnz`
is proportional to `n`. Without that, a single-variable `t ~ n**p` fit would be measuring
changing density rather than fill-in and ordering quality.

**`band`, `grid2d` and `graph` all have exactly 5 nonzeros per row**, so at a given size
they have the same `nnz` and differ *only* in topology. That makes comparing their
exponents a controlled experiment. (`band` and `grid2d` lose a few entries at boundaries,
`O(1)` and `O(sqrt(n))` respectively, so the match is very close rather than exact.)

### Why `graph` is 4-regular rather than Erdos-Renyi

An Erdos-Renyi graph needs mean degree around `ln(n)`, roughly 9 at n=10000, to be
connected. At a sparse degree of 3 or 4 it reliably fragments into a giant component plus
isolated vertices, and a reducible matrix factorizes block by block and scales differently.

Exact regularity also pins the coloring count. splineax's coloring is an `asdex` column or
row coloring, where `num_colors` is the number of JVPs or VJPs one Jacobian materialisation
costs. Column coloring is distance-2, meaning two columns conflict when they share a row,
so `num_colors` is bounded below by the maximum nonzeros per row and above by roughly its
square. Bounded degree therefore gives a color count independent of `n`, and *exact*
regularity keeps it from drifting with local structure. `max_nnz_per_row` and
`max_nnz_per_col` are recorded per benchmark so this is checkable in a result file rather
than assumed.

The construction is a circulant seed, which is regular and connected by construction,
randomized by double-edge swaps. A swap replaces `(a,b)` and `(c,d)` with `(a,c)` and
`(b,d)`, leaving every degree untouched, so the result stays exactly regular however long
it runs. Connectivity is asserted afterwards.

`random` is the one family whose pattern is asymmetric. Its max nonzeros per row is fixed
at 6 by construction, but its max per *column* grows slowly with `n` (12 at n=50 rising to
16 at n=5000), because a fixed count per row leaves the per-column count Poisson
distributed. That asymmetry is what stops a fill-reducing ordering from working on the
pattern directly, since it has to use `A + A.T` instead.

## What is inside the timer

The reuse mode decides what is set up in a fixture, outside the measurement:

| Mode | Set up outside the timer | Timed |
| --- | --- | --- |
| `none` | nothing | symbolic analysis, numeric factorization and solve, every call |
| `symbolic` | `solver.factorize_symbolic(sparsity)` scope | numeric factorization and solve |
| `numeric` | `solver.factorize(operator)` | the solve alone |

So each curve answers a different question:

| Curve | KLU and Pardiso stages per timed call | The question |
| --- | --- | --- |
| `none`, 1 RHS | analyze, factor, solve | cost of a cold one-off solve |
| `none`, 100 RHS | 100 x (analyze, factor, solve) | worst case, naive code in a loop |
| `symbolic`, 1 RHS | factor, solve | new values, known pattern |
| `symbolic`, 100 RHS | 100 x (factor, solve) | 100 matrices sharing a pattern |
| `numeric`, 1 RHS | solve | the triangular solves alone |
| `numeric`, 100 RHS | 100 x solve | one matrix, many right-hand sides, the best case |

The ratio of `none` to `numeric` at 100 right-hand sides is the headline number for whether
the reuse API is worth using. The `numeric` exponent is the one that should sit closest to
linear.

## Three results that look like harness bugs and are not

### `symbolic` with 100 right-hand sides is 100 factorizations

In that tier the numeric refactorization is fused into each *solve*, not done once when the
scope is entered: KLU calls `solve_with_symbol` and Pardiso `factor_and_solve_stateful` per
solve. So `symbolic` at 100 right-hand sides honestly measures the workload that tier
exists for, which is **100 different matrices sharing one sparsity pattern, one right-hand
side each**. It is not measuring one matrix with 100 right-hand sides. That is the `numeric`
tier. Read the `symbolic` line as a different question rather than as a pathology.

### `Spsolve`'s three modes time about the same

`Spsolve`'s reuse tiers are no-ops, so all three modes do the same work. Their timings
agreeing is a sanity check on the harness, not a redundancy. The only thing `Spsolve` would
otherwise "reuse" is the CSR index sort, and since it is fed a sorted BCSR already (see
below) even that is near-free.

### `Pardiso` at 100 right-hand sides is sequential

See [Batching](#batching).

## Operator formats

Each solver is fed the format it stores internally, so it pays no conversion. That table is
verified against the source, not assumed, and it is not one format per solver:

| Solver | Mode | Native | Cost of the other format |
| --- | --- | --- | --- |
| `KLU` | all | BCOO | BCSR input pays a `to_bcoo()` |
| `Spsolve` | all | BCSR | BCOO input pays a `BCSR.from_bcoo` |
| `Pardiso` | `none`, `numeric` | BCSR | BCOO input pays a `BCSR.from_bcoo` |
| `Pardiso` | `symbolic` | **BCOO** | BCSR input round-trips BCSR to BCOO to BCSR |

`Pardiso` is the exception because its symbolic scope's `init` always converts through BCOO,
while its plain `init` takes a sorted BCSR directly. `input_format` is recorded per
benchmark, so a plot can never silently compare a converting configuration against a
non-converting one.

The native operator is built **in a fixture, outside the timer**. A caller already holds
their matrix in some format, so treating construction as setup is what makes the solvers
comparable.

**Measured caveat: the format barely matters here, and the table is a principled default
rather than a measured optimisation.** Feeding each pair the other format changes the timing
by only a few percent, in both directions, which is within run-to-run noise. The likely
reason is the interaction with running everything inside `jit`: the operator is closed over
as a constant, so XLA can fold the index conversion into compilation instead of paying it on
every call. Conversion cost therefore mostly lands in the warm-up, which is untimed. The
table is still what the suite uses, since it costs nothing and keeps the comparison
principled, but do not read these numbers as quantifying conversion overhead. Measuring that
would need the conversion moved inside the timed region deliberately.

**Caveat: input is always sorted.** The generators coalesce every matrix, which leaves the
BCOO index-sorted, and sorted input is exactly what lets the BCSR path skip a conversion.
These numbers therefore describe the best case for `Spsolve` and `Pardiso`. They say nothing
about unsorted input.

**Caveat: reordering is not a controlled variable.** `Spsolve` is constructed with its
defaults (`tol=1e-6`, `reorder=SYMRCM`). On CPU `spsolve` falls back to
`scipy.sparse.linalg.spsolve`, which may ignore both, so this suite does not claim to
benchmark reordering schemes.

## Batching

One right-hand side is a direct call. Many are batched, and the strategy is not fully free.

`jax.vmap` works for seven of the nine (solver, mode) pairs. It raises for `Pardiso` in
`symbolic` and `numeric` mode, because those go through an `ffi_call` whose `vmap_method` is
not one of the batchable ones, and that is set inside `pardiso_mkl_jax` rather than
anywhere reachable from here. Those two pairs fall back to `jax.lax.map`, which is
scan-based, so compile time stays flat in the right-hand-side count.

This means the default `BATCHING=vmap` produces a mix: most series are batched, two are
sequential. Each benchmark records `batching` and `batching_forced`, and the plots append
`[sequential]` to the affected legend entries. For a strictly like-for-like comparison
across solvers, set `SPLINEAX_BENCH_BATCHING=lax_map`, which forces the sequential form
everywhere at the cost of hiding klujax's native batching rule.

### `Spsolve` and multiple right-hand sides

`Spsolve` wraps `spsolve` in `jax.custom_batching.sequential_vmap`, so batching over 100
right-hand sides performs **100 full factorizations**, even though the CPU path is
`scipy.sparse.linalg.spsolve`, which handles a matrix right-hand side with a single one.
This dominates the runtime of a full benchmark run and is visible directly as the gap
between `Spsolve`/`none`/100 RHS and `KLU`/`numeric`/100 RHS.

## Timing hygiene

- **x64 is enabled at import time** in [conftest.py](conftest.py), before any JAX array
  exists. `KLU` and `Pardiso` require it and do not enable it themselves.
- **Every timed call runs inside `jax.jit`**, and compilation is excluded by triggering it
  in an untimed warm-up. The compile cache size is recorded before the measured loop and
  asserted unchanged afterwards, so a silent recompilation fails the benchmark rather than
  inflating it.
- **The factorization scope stays open across every timed call.** Exiting a `factorize` or
  `factorize_symbolic` block frees the native handle, so the scope is entered on a
  `contextlib.ExitStack` in a fixture and unwound only at teardown.
- **State is closed over, not passed as an argument.** A scope opened while tracing gets an
  id token that changes the jit cache key, so scopes are opened eagerly outside `jit`.
- **Results are blocked on inside the timed callable.** JAX dispatches asynchronously, so
  without `jax.block_until_ready` the benchmark would measure dispatch instead of compute.
- **A residual gate runs before any timing.** The warm-up solve must reach a relative
  residual below `RESIDUAL_TOL` or the benchmark fails, so no timing is ever reported for a
  wrong answer.

## Outputs

`plot_results.py` writes PNG and SVG per figure, plus two tables:

- `scaling_<family>.png` -- log-log median time against size, rows are modes and columns
  are right-hand-side counts, with the fitted power law overlaid and the exponent in the
  legend.
- `amortized_<family>.png` -- time per right-hand side, showing what reuse plus batching
  buys per solve.
- `speedup_<family>.png` -- `symbolic` and `numeric` speedup against the `none` baseline.
- `scaling.csv` and `scaling.md` -- the fitted exponent per (solver, mode, RHS, family),
  with R squared, a 95% interval, and the number of points fitted.

### The fit

Least squares of `log10(t)` on `log10(n)`, restricted to `n >= FIT_MIN_N`. Smaller sizes
are dominated by fixed dispatch overhead, which drags the exponent downward, so they are
measured and plotted but excluded from the fit. Medians are used rather than means, because
pytest-benchmark's mean is skewed by the occasional GC or OS outlier. With `REPLICATES` above
1 the fit uses the individual points rather than per-size means, so replicate scatter widens
the interval honestly. An exponent is reported only when at least 3 points survive the cut,
and otherwise the table says `n/a` with a reason.

Fitting per family is the point. `band` should land near p=1 and `graph` noticeably higher.
Collapsing the families together would hide exactly that.

Missing combinations are tolerated throughout. Skipped benchmarks (`Pardiso` absent) and
failed ones simply have no JSON entry, and the script prints a summary of what is missing
rather than assuming a full grid.

## A warning about absolute numbers

The CI workflow runs on shared GitHub runners, where absolute timings are unstable. **These
numbers are for shape and ratios, not for tracking regressions.** Do not wire them into a
pass/fail gate.
