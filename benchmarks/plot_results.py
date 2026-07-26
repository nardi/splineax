"""Turn a pytest-benchmark JSON file into plots and a fitted scaling table.

    uv run python benchmarks/plot_results.py benchmarks/results/bench.json

Not named `test_*`, so pytest never collects it. Writes PNG and SVG per figure plus
`scaling.csv` and `scaling.md` into the JSON file's directory, or `--out` if given.

Everything here tolerates holes. Skipped benchmarks (`Pardiso` absent) and failed ones
simply have no entry in the JSON, so results are grouped by what is present and missing
combinations are reported rather than assumed away.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from scipy import stats  # noqa: E402

# First three categorical slots of the reference palette, which are the three that
# validate on all pairs in both light and dark mode. Markers vary too, so solver identity
# is never carried by color alone.
SOLVER_STYLE: dict[str, tuple[str, str]] = {
    "spsolve": ("#2a78d6", "o"),
    "klu": ("#eb6834", "s"),
    "pardiso": ("#1baf7a", "^"),
}
SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6a63"
MODE_ORDER = ("none", "symbolic", "numeric")


@dataclass(frozen=True)
class Record:
    """One benchmark's median time plus the metadata needed to place it on a plot."""

    solver: str
    mode: str
    n_rhs: int
    family: str
    n: int
    nnz: int
    median: float
    batching: str
    batching_forced: bool


@dataclass(frozen=True)
class Fit:
    """A power-law fit of `t ~ n**exponent` over one series."""

    exponent: float
    r_squared: float
    ci_low: float
    ci_high: float
    points: int
    note: str = ""

    @property
    def ok(self) -> bool:
        return not self.note


def load(path: Path) -> tuple[list[Record], dict[str, Any]]:
    """Read the benchmark JSON into records, skipping anything without our metadata."""
    payload = json.loads(path.read_text())
    records = []
    for entry in payload.get("benchmarks", []):
        info = entry.get("extra_info") or {}
        if "solver" not in info:
            continue
        records.append(
            Record(
                solver=info["solver"],
                mode=info["mode"],
                n_rhs=int(info["n_rhs"]),
                family=info["family"],
                n=int(info["actual_n"]),
                nnz=int(info["nnz"]),
                median=float(entry["stats"]["median"]),
                batching=info.get("batching", "?"),
                batching_forced=bool(info.get("batching_forced", False)),
            )
        )
    return records, payload.get("machine_info", {})


def fit_power_law(sizes: Iterable[int], times: Iterable[float], min_n: int) -> Fit:
    """Least-squares fit of `log10(t)` on `log10(n)`, restricted to `n >= min_n`.

    Small sizes are dominated by fixed dispatch overhead, which drags the exponent down,
    so they are measured and plotted but excluded here. Fits on the individual points
    rather than per-size means, so replicate scatter widens the interval honestly.
    """
    pairs = [(n, t) for n, t in zip(sizes, times) if n >= min_n and t > 0]
    if len(pairs) < 3:
        return Fit(
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            len(pairs),
            note=f"only {len(pairs)} points at n >= {min_n}, need 3",
        )

    x = np.log10([n for n, _ in pairs])
    y = np.log10([t for _, t in pairs])
    result = stats.linregress(x, y)

    # 95% interval on the slope, two-sided, with the usual n-2 degrees of freedom.
    degrees = len(pairs) - 2
    half_width = stats.t.ppf(0.975, degrees) * result.stderr
    return Fit(
        exponent=float(result.slope),
        r_squared=float(result.rvalue**2),
        ci_low=float(result.slope - half_width),
        ci_high=float(result.slope + half_width),
        points=len(pairs),
    )


def _series(records: list[Record], **filters: Any) -> list[Record]:
    """Records matching every filter, sorted by size."""
    picked = [
        r
        for r in records
        if all(getattr(r, key) == value for key, value in filters.items())
    ]
    return sorted(picked, key=lambda r: r.n)


def _style_axes(
    axes: plt.Axes, xlabel: str, ylabel: str, ticks: Iterable[int] = ()
) -> None:
    """Recessive grid and spines, so the data reads first.

    `ticks` replaces matplotlib's default log ticks with one label per measured size.
    The defaults put minor labels like `2 x 10^3` next to major ones and they collide.
    """
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    axes.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    axes.grid(True, which="major", color="#e5e4de", linewidth=0.8)
    axes.tick_params(colors=INK_MUTED, labelsize=8)

    sizes = sorted(set(ticks))
    if sizes:
        axes.set_xticks(sizes)
        axes.set_xticklabels([str(size) for size in sizes], fontsize=8)
        # Silence the minor ticks entirely, otherwise they relabel over the majors.
        axes.xaxis.set_minor_formatter(NullFormatter())
        axes.tick_params(axis="x", which="minor", length=0)

    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#d5d4ce")


def _save(figure: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    written = []
    for suffix in ("png", "svg"):
        path = out_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        written.append(path)
    plt.close(figure)
    return written


def _context_line(machine: dict[str, Any], records: list[Record]) -> str:
    """Machine context, since absolute timings mean nothing without it."""
    cpu = (machine.get("cpu") or {}).get("brand_raw", "unknown CPU")
    backends = {r.batching for r in records}
    return f"{cpu} · batching: {', '.join(sorted(backends))}"


def plot_scaling(
    records: list[Record], out_dir: Path, machine: dict[str, Any], min_n: int
) -> tuple[list[Path], dict[tuple[str, str, int, str], Fit]]:
    """Log-log time against size, one figure per family, rows modes and columns n_rhs."""
    written: list[Path] = []
    fits: dict[tuple[str, str, int, str], Fit] = {}

    families = sorted({r.family for r in records})
    rhs_values = sorted({r.n_rhs for r in records})
    modes = [m for m in MODE_ORDER if m in {r.mode for r in records}]

    for family in families:
        figure, grid = plt.subplots(
            len(modes),
            len(rhs_values),
            figsize=(5.2 * len(rhs_values), 3.6 * len(modes)),
            squeeze=False,
            facecolor=SURFACE,
        )
        for row, mode in enumerate(modes):
            for col, n_rhs in enumerate(rhs_values):
                axes = grid[row][col]
                axes.set_facecolor(SURFACE)
                for solver, (color, marker) in SOLVER_STYLE.items():
                    series = _series(
                        records, solver=solver, mode=mode, n_rhs=n_rhs, family=family
                    )
                    if not series:
                        continue
                    sizes = [r.n for r in series]
                    times = [r.median for r in series]
                    fit = fit_power_law(sizes, times, min_n)
                    fits[(solver, mode, n_rhs, family)] = fit

                    label = solver
                    if fit.ok:
                        label = f"{solver} (p={fit.exponent:.2f})"
                    if any(r.batching_forced for r in series):
                        label += " [sequential]"

                    axes.plot(
                        sizes,
                        times,
                        color=color,
                        marker=marker,
                        markersize=5,
                        linewidth=2,
                        label=label,
                        markeredgecolor=SURFACE,
                        markeredgewidth=0.8,
                    )
                    if fit.ok:
                        # Fitted power law over the fitted range only, so the dashed line
                        # never implies the small sizes were part of the fit.
                        span = np.array([n for n in sizes if n >= min_n], dtype=float)
                        intercept = (
                            np.log10(
                                [t for n, t in zip(sizes, times) if n >= min_n]
                            ).mean()
                            - fit.exponent * np.log10(span).mean()
                        )
                        axes.plot(
                            span,
                            10 ** (intercept + fit.exponent * np.log10(span)),
                            color=color,
                            linewidth=1,
                            linestyle="--",
                            alpha=0.55,
                        )
                family_sizes = sorted({r.n for r in records if r.family == family})
                _style_axes(axes, "matrix size n", "median time (s)", family_sizes)
                axes.set_title(
                    f"{mode}, {n_rhs} RHS", color=INK, fontsize=10, loc="left"
                )
                if axes.get_legend_handles_labels()[0]:
                    axes.legend(
                        frameon=False, fontsize=8, labelcolor=INK, loc="upper left"
                    )

        figure.suptitle(
            f"Solver scaling, {family} family",
            color=INK,
            fontsize=13,
            x=0.01,
            ha="left",
        )
        figure.text(
            0.01,
            -0.01,
            _context_line(machine, records),
            color=INK_MUTED,
            fontsize=8,
            ha="left",
        )
        figure.tight_layout()
        written += _save(figure, out_dir, f"scaling_{family}")
    return written, fits


def plot_amortized(
    records: list[Record], out_dir: Path, machine: dict[str, Any]
) -> list[Path]:
    """Cost per right-hand side, showing what reuse plus batching buys per solve."""
    written: list[Path] = []
    families = sorted({r.family for r in records})
    rhs_values = sorted({r.n_rhs for r in records})
    modes = [m for m in MODE_ORDER if m in {r.mode for r in records}]
    # Dashed for the batched count, solid for a single right-hand side.
    styles = {k: ("-" if k == min(rhs_values) else "--") for k in rhs_values}

    for family in families:
        figure, grid = plt.subplots(
            1,
            len(modes),
            figsize=(5.0 * len(modes), 3.6),
            squeeze=False,
            facecolor=SURFACE,
        )
        for col, mode in enumerate(modes):
            axes = grid[0][col]
            axes.set_facecolor(SURFACE)
            for solver, (color, marker) in SOLVER_STYLE.items():
                for n_rhs in rhs_values:
                    series = _series(
                        records, solver=solver, mode=mode, n_rhs=n_rhs, family=family
                    )
                    if not series:
                        continue
                    axes.plot(
                        [r.n for r in series],
                        [r.median / n_rhs for r in series],
                        color=color,
                        marker=marker,
                        markersize=5,
                        linewidth=2,
                        linestyle=styles[n_rhs],
                        label=f"{solver}, {n_rhs} RHS",
                        markeredgecolor=SURFACE,
                        markeredgewidth=0.8,
                    )
            family_sizes = sorted({r.n for r in records if r.family == family})
            _style_axes(axes, "matrix size n", "median time per RHS (s)", family_sizes)
            axes.set_title(mode, color=INK, fontsize=10, loc="left")
            if axes.get_legend_handles_labels()[0]:
                axes.legend(frameon=False, fontsize=7, labelcolor=INK, loc="upper left")

        figure.suptitle(
            f"Amortized cost per right-hand side, {family} family",
            color=INK,
            fontsize=13,
            x=0.01,
            ha="left",
        )
        figure.text(
            0.01,
            -0.03,
            _context_line(machine, records),
            color=INK_MUTED,
            fontsize=8,
            ha="left",
        )
        figure.tight_layout()
        written += _save(figure, out_dir, f"amortized_{family}")
    return written


def plot_speedup(records: list[Record], out_dir: Path) -> list[Path]:
    """Reuse speedup against the `none` baseline, grouped bars per size."""
    written: list[Path] = []
    families = sorted({r.family for r in records})
    rhs_values = sorted({r.n_rhs for r in records})
    reuse_modes = [m for m in ("symbolic", "numeric") if m in {r.mode for r in records}]
    if not reuse_modes:
        return written

    for family in families:
        figure, grid = plt.subplots(
            1,
            len(rhs_values),
            figsize=(5.6 * len(rhs_values), 3.6),
            squeeze=False,
            facecolor=SURFACE,
        )
        for col, n_rhs in enumerate(rhs_values):
            axes = grid[0][col]
            axes.set_facecolor(SURFACE)
            sizes = sorted({r.n for r in records if r.family == family})
            labels: list[str] = []
            positions: list[float] = []
            group = 0.0
            for solver, (color, _) in SOLVER_STYLE.items():
                baseline = {
                    r.n: r.median
                    for r in _series(
                        records, solver=solver, mode="none", n_rhs=n_rhs, family=family
                    )
                }
                if not baseline:
                    continue
                for offset, mode in enumerate(reuse_modes):
                    values, spots = [], []
                    for index, size in enumerate(sizes):
                        match = _series(
                            records,
                            solver=solver,
                            mode=mode,
                            n_rhs=n_rhs,
                            family=family,
                            n=size,
                        )
                        if not match or size not in baseline:
                            continue
                        values.append(baseline[size] / match[0].median)
                        spots.append(group + index * 0.28 * len(reuse_modes) * 1.6)
                    if not values:
                        continue
                    axes.bar(
                        [s + offset * 0.26 for s in spots],
                        values,
                        width=0.24,
                        color=color,
                        alpha=1.0 if mode == "numeric" else 0.55,
                        label=f"{solver}, {mode}",
                        edgecolor=SURFACE,
                        linewidth=2,
                    )
                    if mode == reuse_modes[0]:
                        # Label every solver's group, not just the first: each group
                        # repeats the same sizes at its own offset, and skipping the
                        # later ones leaves them unlabelled.
                        labels += [str(size) for size in sizes[: len(values)]]
                        positions += [s + 0.13 for s in spots]
                group += 0.28 * len(reuse_modes) * 1.6 * len(sizes) + 0.6

            axes.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle=":")
            axes.set_yscale("log")
            axes.set_ylabel("speedup vs none (x)", color=INK_MUTED, fontsize=9)
            axes.set_xlabel(
                "matrix size n, grouped by solver", color=INK_MUTED, fontsize=9
            )
            if positions:
                axes.set_xticks(positions)
                axes.set_xticklabels(labels, fontsize=7)
            axes.grid(True, axis="y", color="#e5e4de", linewidth=0.8)
            axes.tick_params(colors=INK_MUTED, labelsize=8)
            for side in ("top", "right"):
                axes.spines[side].set_visible(False)
            axes.set_title(f"{n_rhs} RHS", color=INK, fontsize=10, loc="left")
            if axes.get_legend_handles_labels()[0]:
                # Above the plot area, so it cannot sit on top of a tall bar.
                axes.legend(
                    frameon=False,
                    fontsize=7,
                    labelcolor=INK,
                    ncol=len(SOLVER_STYLE),
                    loc="lower center",
                    bbox_to_anchor=(0.5, 1.02),
                )

        figure.suptitle(
            f"Factorization reuse speedup, {family} family",
            color=INK,
            fontsize=13,
            x=0.01,
            ha="left",
        )
        figure.tight_layout()
        written += _save(figure, out_dir, f"speedup_{family}")
    return written


def write_table(
    fits: dict[tuple[str, str, int, str], Fit], out_dir: Path, min_n: int
) -> list[Path]:
    """Fitted exponents as CSV and markdown, which is also the table view the palette's
    contrast relief rule requires."""
    rows = sorted(fits.items(), key=lambda item: item[0])

    csv_path = out_dir / "scaling.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "solver",
                "mode",
                "n_rhs",
                "family",
                "exponent",
                "r_squared",
                "ci_low",
                "ci_high",
                "points",
                "note",
            ]
        )
        for (solver, mode, n_rhs, family), fit in rows:
            writer.writerow(
                [
                    solver,
                    mode,
                    n_rhs,
                    family,
                    "" if math.isnan(fit.exponent) else f"{fit.exponent:.4f}",
                    "" if math.isnan(fit.r_squared) else f"{fit.r_squared:.4f}",
                    "" if math.isnan(fit.ci_low) else f"{fit.ci_low:.4f}",
                    "" if math.isnan(fit.ci_high) else f"{fit.ci_high:.4f}",
                    fit.points,
                    fit.note,
                ]
            )

    md_path = out_dir / "scaling.md"
    lines = [
        f"# Fitted scaling exponents (`t ~ n**p`, fitted on `n >= {min_n}`)",
        "",
        "| solver | mode | RHS | family | p | 95% CI | R2 | points |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (solver, mode, n_rhs, family), fit in rows:
        if fit.ok:
            exponent = f"{fit.exponent:.2f}"
            interval = f"{fit.ci_low:.2f} to {fit.ci_high:.2f}"
            r_squared = f"{fit.r_squared:.3f}"
        else:
            exponent = interval = r_squared = "n/a"
        lines.append(
            f"| {solver} | {mode} | {n_rhs} | {family} | {exponent} | {interval} "
            f"| {r_squared} | {fit.points} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return [csv_path, md_path]


def report_gaps(records: list[Record]) -> list[str]:
    """Combinations absent from the JSON, so a hole is stated rather than inferred."""
    present = {(r.solver, r.mode, r.n_rhs, r.family, r.n) for r in records}
    solvers = sorted({r.solver for r in records})
    modes = sorted({r.mode for r in records})
    rhs_values = sorted({r.n_rhs for r in records})
    families = sorted({r.family for r in records})
    sizes_by_family = defaultdict(set)
    for record in records:
        sizes_by_family[record.family].add(record.n)

    missing = [
        f"{solver}/{mode}/rhs{n_rhs}/{family}/n{n}"
        for solver in solvers
        for mode in modes
        for n_rhs in rhs_values
        for family in families
        for n in sorted(sizes_by_family[family])
        if (solver, mode, n_rhs, family, n) not in present
    ]
    return missing


def _default_fit_min_n() -> int:
    """The suite's `FIT_MIN_N`, importable whether this runs as a script or a module."""
    try:
        from .config import FIT_MIN_N
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from benchmarks.config import FIT_MIN_N
    return FIT_MIN_N


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file", type=Path, help="pytest-benchmark --benchmark-json"
    )
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--fit-min-n",
        type=int,
        default=None,
        help="ignore sizes below this in the fit (default: the suite's FIT_MIN_N)",
    )
    args = parser.parse_args()

    min_n = args.fit_min_n
    if min_n is None:
        min_n = _default_fit_min_n()

    records, machine = load(args.json_file)
    if not records:
        raise SystemExit(
            f"no benchmark records with solver metadata in {args.json_file}"
        )

    out_dir = args.out or args.json_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    written, fits = plot_scaling(records, out_dir, machine, min_n)
    written += plot_amortized(records, out_dir, machine)
    written += plot_speedup(records, out_dir)
    written += write_table(fits, out_dir, min_n)

    print(f"read {len(records)} benchmarks from {args.json_file}")
    for path in written:
        print(f"  wrote {path}")

    missing = report_gaps(records)
    if missing:
        print(f"\n{len(missing)} combinations absent from the JSON:")
        for label in missing[:20]:
            print(f"  {label}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


if __name__ == "__main__":
    main()
