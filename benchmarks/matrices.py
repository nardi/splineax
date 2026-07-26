"""Deterministic sparse matrix generators for the solver benchmarks.

Four families, so a size is never characterised by a single sparsity pattern. Fill-in,
and therefore scaling, differs sharply between them:

| Family   | Pattern                             | Nonzeros per row     |
| -------- | ----------------------------------- | -------------------- |
| `band`   | banded, half-width `BAND_HALFWIDTH` | 5 (default)          |
| `grid2d` | 5-point stencil on a square grid    | 5                    |
| `graph`  | random `GRAPH_DEGREE`-regular graph | 5 (default)          |
| `random` | uniformly random off-diagonals      | `RANDOM_NNZ_PER_ROW` |

At the defaults `band`, `grid2d` and `graph` all have exactly 5 nonzeros per row, so they
have the same `nnz` at a given size and differ *only* in topology. That is what makes
comparing their fitted exponents a controlled experiment rather than a density artefact.

Every family holds nonzeros per row constant as `n` grows, so `nnz` is proportional to
`n`. Without that, a single-variable `t ~ n**p` fit would measure changing density rather
than fill-in and ordering quality.

Every matrix is square, real, float64, coalesced and strictly diagonally dominant. The
dominance is not cosmetic: no solver here detects singularity and `RESULTS.successful` is
returned unconditionally, so an ill-conditioned matrix would silently benchmark garbage.
Values are asymmetric even where the pattern is symmetric, because `Pardiso` is hardcoded
to `REAL_NONSYMMETRIC` and no solver exploits numeric symmetry.
"""

import zlib
from dataclasses import dataclass
from functools import cached_property

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.experimental.sparse import BCOO
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import config


def _entropy(*parts: object) -> tuple[int, ...]:
    """Stable integer seed material from arbitrary labels.

    `hash()` is not usable here: Python randomizes string hashing per process, so seeding
    from it would make a matrix differ between runs despite an identical spec.
    """
    return tuple(
        part if isinstance(part, int) else zlib.crc32(str(part).encode())
        for part in parts
    )


@dataclass(frozen=True)
class MatrixSpec:
    """Identifies one matrix. Frozen and hashable so it can key a fixture or a cache."""

    family: str
    n: int
    """Nominal size. The actual size may differ, see `Matrix.n`."""
    replicate: int

    @property
    def id(self) -> str:
        """Short label used in pytest node ids, e.g. `band-n1500-r0`."""
        return f"{self.family}-n{self.n}-r{self.replicate}"

    def streams(self) -> tuple[np.random.Generator, np.random.Generator]:
        """Independent generators for the sparsity pattern and for the values.

        Two separate streams, so changing how the pattern is drawn cannot silently change
        the values as well.
        """
        root = np.random.SeedSequence(
            _entropy(config.SEED_BASE, self.family, self.n, self.replicate)
        )
        structure, values = root.spawn(2)
        return np.random.default_rng(structure), np.random.default_rng(values)


@dataclass(frozen=True)
class Matrix:
    """A generated matrix and the metadata the plots and fits need."""

    spec: MatrixSpec
    bcoo: BCOO

    @property
    def n(self) -> int:
        """Actual size, which `grid2d` rounds to a perfect square. This, not the nominal
        size, is the x-axis of every plot and fit."""
        return int(self.bcoo.shape[0])

    @property
    def nnz(self) -> int:
        return int(self.bcoo.nse)

    @cached_property
    def max_nnz_per_row(self) -> int:
        """Lower bound on the number of colors a distance-2 (Jacobian column) coloring
        needs, so the property that keeps coloring cost from confounding the fitted
        exponent. Constant in `n` for all four families, and equal to 5 for the three
        with 5 nonzeros per row."""
        rows = np.asarray(self.bcoo.indices[:, 0])
        return int(np.bincount(rows, minlength=self.n).max())

    @cached_property
    def max_nnz_per_col(self) -> int:
        """Companion to `max_nnz_per_row`, and the one that separates the families.

        Equal to `max_nnz_per_row` for the three symmetric families. For `random` it
        grows slowly with `n` (12 at n=50 rising to 16 at n=5000), because a fixed count
        per row leaves the per-column count Poisson-distributed. That asymmetry is
        exactly what stops a fill-reducing ordering from working on the pattern directly.
        """
        cols = np.asarray(self.bcoo.indices[:, 1])
        return int(np.bincount(cols, minlength=self.n).max())

    @cached_property
    def n_components(self) -> int:
        """Connected components of the sparsity pattern, treated as undirected. 1 for
        every family here. A reducible matrix would factorize block by block and scale
        differently, so this is asserted at generation time rather than just observed."""
        indices = np.asarray(self.bcoo.indices)
        pattern = coo_matrix(
            (np.ones(self.nnz), (indices[:, 0], indices[:, 1])), shape=(self.n, self.n)
        )
        return int(connected_components(pattern, directed=False)[0])


def _band_edges(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Off-diagonals of a banded pattern, both orientations, so the pattern is
    symmetric."""
    rows, cols = [], []
    for offset in range(-config.BAND_HALFWIDTH, config.BAND_HALFWIDTH + 1):
        if offset == 0:
            continue
        index = np.arange(max(0, -offset), min(n, n - offset))
        rows.append(index)
        cols.append(index + offset)
    return np.concatenate(rows), np.concatenate(cols)


def _grid2d_edges(side: int) -> tuple[np.ndarray, np.ndarray]:
    """Off-diagonals of a 5-point stencil on a `side` by `side` grid."""
    index = np.arange(side * side).reshape(side, side)
    pairs = [
        (index[:, :-1].ravel(), index[:, 1:].ravel()),
        (index[:, 1:].ravel(), index[:, :-1].ravel()),
        (index[:-1, :].ravel(), index[1:, :].ravel()),
        (index[1:, :].ravel(), index[:-1, :].ravel()),
    ]
    return (
        np.concatenate([rows for rows, _ in pairs]),
        np.concatenate([cols for _, cols in pairs]),
    )


def _regular_graph_edges(n: int, degree: int, rng: np.random.Generator) -> np.ndarray:
    """Undirected edges of a random connected `degree`-regular graph on `n` vertices.

    Starts from a circulant graph, which is regular and connected by construction, then
    randomizes it with double-edge swaps. A swap replaces edges `(a,b)` and `(c,d)` with
    `(a,c)` and `(b,d)`, leaving every degree untouched, so the result stays exactly
    regular however long it runs. Exact regularity is what pins the coloring count, see
    the README.

    Returns an `(n * degree / 2, 2)` array of canonical `(min, max)` pairs.
    """
    if degree % 2 != 0:
        # An odd degree needs an extra perfect matching, which only exists for even `n`.
        # Even degrees keep the circulant seed trivial, so odd is simply refused.
        raise ValueError(f"GRAPH_DEGREE must be even, got {degree}")
    if n <= degree:
        raise ValueError(f"GRAPH_DEGREE ({degree}) must be below the size ({n})")

    for _ in range(8):
        # Circulant seed: every vertex joined to its `degree / 2` nearest neighbours on
        # each side, modulo `n`.
        edges = {
            (min(i, (i + step) % n), max(i, (i + step) % n))
            for i in range(n)
            for step in range(1, degree // 2 + 1)
        }
        edge_list = list(edges)

        # Draw all randomness up front: the loop runs `GRAPH_SWAP_FACTOR * n` times, and
        # per-iteration numpy calls would dominate the generation cost at n=10000.
        swaps = config.GRAPH_SWAP_FACTOR * n
        picks = rng.integers(len(edge_list), size=(swaps, 2))
        flips = rng.random(swaps) < 0.5

        for (i, j), flip in zip(picks, flips):
            if i == j:
                continue
            a, b = edge_list[i]
            # Keep the canonical pair to remove: flipping below reorders `c` and `d`, and
            # the set only ever holds `(min, max)`, so discarding the flipped tuple would
            # silently no-op and leak an extra edge.
            stored = edge_list[j]
            c, d = stored
            if flip:
                c, d = d, c
            # Reject any swap making a self-loop or a duplicate edge, either of which
            # would break exact regularity.
            if len({a, b, c, d}) < 4:
                continue
            first = (min(a, c), max(a, c))
            second = (min(b, d), max(b, d))
            if first in edges or second in edges:
                continue
            edges.discard((a, b))
            edges.discard(stored)
            edges.add(first)
            edges.add(second)
            edge_list[i] = first
            edge_list[j] = second

        result = np.array(sorted(edges), dtype=np.int64)

        # Swaps can in principle disconnect the graph. Retry rather than silently
        # benchmark a reducible matrix. At degree 4 this effectively never triggers.
        pattern = coo_matrix(
            (np.ones(len(result)), (result[:, 0], result[:, 1])), shape=(n, n)
        )
        if connected_components(pattern, directed=False)[0] == 1:
            return result

    raise RuntimeError(
        f"could not build a connected {degree}-regular graph on {n} vertices"
    )


def _random_edges(
    n: int, per_row: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """`per_row` distinct random off-diagonal columns in every row.

    Asymmetric by construction, unlike the other three families, so a fill-reducing
    ordering cannot work on the pattern directly and has to use `A + A.T` instead.
    """
    rows = np.repeat(np.arange(n), per_row)
    cols = np.empty(n * per_row, dtype=np.int64)
    for i in range(n):
        # Draw from the `n - 1` non-diagonal columns, then shift past the diagonal. This
        # is a bijection onto the valid columns, so the drawn columns stay distinct and
        # the row keeps exactly `per_row` entries.
        choices = rng.choice(n - 1, size=per_row, replace=False)
        cols[i * per_row : (i + 1) * per_row] = choices + (choices >= i)
    return rows, cols


def _pattern(
    spec: MatrixSpec, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, int]:
    """Off-diagonal (row, col) pairs for a family, plus the actual matrix size."""
    if spec.family == "band":
        return (*_band_edges(spec.n), spec.n)

    if spec.family == "grid2d":
        # Round to a perfect square so the stencil is well defined. Callers read the
        # actual size back off the generated matrix.
        side = max(2, round(np.sqrt(spec.n)))
        return (*_grid2d_edges(side), side * side)

    if spec.family == "graph":
        edges = _regular_graph_edges(spec.n, config.GRAPH_DEGREE, rng)
        rows = np.concatenate([edges[:, 0], edges[:, 1]])
        cols = np.concatenate([edges[:, 1], edges[:, 0]])
        return rows, cols, spec.n

    if spec.family == "random":
        per_row = min(config.RANDOM_NNZ_PER_ROW - 1, spec.n - 1)
        return (*_random_edges(spec.n, per_row, rng), spec.n)

    raise ValueError(
        f"unknown family {spec.family!r}, expected one of {config.FAMILIES}"
    )


def generate(spec: MatrixSpec) -> Matrix:
    """Build the matrix for `spec`, asserting the invariants the benchmarks rely on."""
    structure_rng, value_rng = spec.streams()
    rows, cols, n = _pattern(spec, structure_rng)
    values = value_rng.uniform(0.5, 1.5, size=rows.shape[0])

    # Strict diagonal dominance: every diagonal entry exceeds its row's off-diagonal
    # absolute sum, making the matrix nonsingular and well conditioned.
    diagonal = 1.0 + np.bincount(rows, weights=np.abs(values), minlength=n)

    all_rows = np.concatenate([rows, np.arange(n)])
    all_cols = np.concatenate([cols, np.arange(n)])
    all_values = np.concatenate([values, diagonal])

    indices = jnp.asarray(np.stack([all_rows, all_cols], axis=1), dtype=jnp.int32)
    # `sum_duplicates` coalesces and sorts the indices. Sorted input is what lets the
    # BCSR path in `Spsolve` and `Pardiso` skip a conversion, see the README.
    bcoo = BCOO(
        (jnp.asarray(all_values, dtype=jnp.float64), indices), shape=(n, n)
    ).sum_duplicates()

    matrix = Matrix(spec=spec, bcoo=bcoo)
    assert matrix.n_components == 1, (
        f"{spec.id} is reducible ({matrix.n_components} components), which would "
        "factorize block by block and scale differently"
    )
    if spec.family == "graph":
        assert matrix.max_nnz_per_row == config.GRAPH_DEGREE + 1, (
            f"{spec.id} is not {config.GRAPH_DEGREE}-regular: max nonzeros per row is "
            f"{matrix.max_nnz_per_row}"
        )
    return matrix


def generate_rhs(n: int, n_rhs: int) -> Array:
    """Right-hand side block for a size, shaped `(n,)` for one and `(n_rhs, n)` for many.

    Keyed on size and count only, deliberately not on family, solver or mode, so every
    configuration at a given size solves the identical system.
    """
    rng = np.random.default_rng(
        np.random.SeedSequence(_entropy(config.SEED_BASE, "rhs", n, n_rhs))
    )
    shape = (n,) if n_rhs == 1 else (n_rhs, n)
    return jnp.asarray(rng.normal(size=shape), dtype=jnp.float64)
