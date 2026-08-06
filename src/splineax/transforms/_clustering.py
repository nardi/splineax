"""Parallel pairwise aggregation clustering, a symbolic `SystemTransform`.

The GPU-native analogue of PABLO (Notay & Napov's pairwise aggregation): PABLO grows
blocks one at a time from a queue, which is inherently sequential. This does the same
job with fixed-shape, fixed-round-count array ops that stay entirely in XLA, at the
cost of accepting whatever cluster sizes fall out rather than hand-tuning each one.

Each round, every current cluster proposes to merge with its lowest-id eligible
neighbour (one not already at the block size cap); a proposal becomes a merge only
when it is mutual (a handshake, so no cluster is claimed by two others in the same
round). `ceil(log2(block_size))` rounds are enough for a cluster to at least double in
size each round it still has room to grow, which is why the round count is fixed by
`block_size` alone. The result is a label per vertex; sorting by label is the
permutation that gathers each cluster into a contiguous block.
"""

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Array, Integer

from splineax.operators._pattern import MatrixSparsity
from splineax.transforms._permutation import AppliedPermutation
from splineax.transforms._protocols import AppliedTransform

# Default block size is clamped to this range. The lower bound keeps tiny blocks from
# defeating the point of batched inversion; the upper bound caps the storage a padded
# `(nblocks, block_size, block_size)` array costs (`n * block_size` numbers) and the
# cost of the per-block `block_size x block_size` inverse.
_MIN_BLOCK_SIZE = 4
_MAX_BLOCK_SIZE = 128


def _default_block_size(nse: int, n: int) -> int:
    """Nearest power of two to the average nonzeros per row, clamped to a sane range.

    A matrix with roughly `k` nonzeros per row typically couples each row to about `k`
    neighbours, so blocks sized near `k` are the natural unit to capture that coupling
    without being needlessly larger than the local structure calls for.
    """
    average = max(nse / n, 1.0)
    power_of_two = 2 ** round(math.log2(average))
    return min(max(power_of_two, _MIN_BLOCK_SIZE), _MAX_BLOCK_SIZE)


def _aggregate_clusters(
    rows: Integer[Array, " nse"], cols: Integer[Array, " nse"], n: int, block_size: int
) -> Integer[Array, " n"]:
    """Runs the pairwise aggregation rounds, returning a cluster label per vertex."""
    off_diagonal = rows != cols
    # Symmetrise: an edge in either direction of the (possibly one-sided) input
    # pattern counts as an undirected coupling. Diagonal entries are routed to a
    # harmless self-loop (0, 0) rather than dropped, keeping every array's shape
    # static; `eligible` below discards them (`a == b`) exactly as it would discard
    # a genuine self-loop.
    rows_masked = jnp.where(off_diagonal, rows, 0)
    cols_masked = jnp.where(off_diagonal, cols, 0)
    src = jnp.concatenate([rows_masked, cols_masked])
    dst = jnp.concatenate([cols_masked, rows_masked])

    label = jnp.arange(n, dtype=jnp.int32)
    size = jnp.ones(n, dtype=jnp.int32)
    all_ids = jnp.arange(n, dtype=jnp.int32)
    size_sentinel = jnp.int32(block_size + 1)  # larger than any eligible neighbour size
    id_sentinel = jnp.int32(n)  # larger than any real vertex id

    num_rounds = math.ceil(math.log2(block_size)) if block_size > 1 else 0
    for _ in range(num_rounds):
        a = label[src]
        b = label[dst]
        eligible = (a != b) & (size[a] + size[b] <= block_size)

        # Rank neighbours by their current cluster size first, vertex id second: a
        # cluster grows by absorbing its smallest eligible neighbour, so no single
        # cluster snowballs just for having the lowest id. Two segment_min passes,
        # each padded with an explicit sentinel candidate per segment so a cluster
        # touching no eligible edge this round resolves to "no proposal" rather than
        # depending on the reduction's fill value for an empty segment.
        segment_ids = jnp.concatenate([a, all_ids])
        size_candidates = jnp.concatenate(
            [jnp.where(eligible, size[b], size_sentinel), jnp.full(n, size_sentinel)]
        )
        best_size = jax.ops.segment_min(size_candidates, segment_ids, num_segments=n)

        achieves_best = eligible & (size[b] == best_size[a])
        id_candidates = jnp.concatenate(
            [jnp.where(achieves_best, b, id_sentinel), jnp.full(n, id_sentinel)]
        )
        proposal = jax.ops.segment_min(id_candidates, segment_ids, num_segments=n)

        has_proposal = proposal < id_sentinel
        # Clamp before the lookup: an out-of-range `proposal` is never selected below
        # (`has_proposal` is False there), the clamp only keeps the gather in bounds.
        proposed_back = proposal[jnp.minimum(proposal, n - 1)]
        mutual = has_proposal & (proposed_back == all_ids)
        partner = jnp.where(mutual, proposal, all_ids)
        new_rep = jnp.minimum(all_ids, partner)

        size = jnp.zeros(n, dtype=jnp.int32).at[new_rep].add(size)
        label = new_rep[label]

    return label


class _ClusteringPlan(NamedTuple):
    perm: Integer[Array, " n"]
    sort_order: Integer[Array, " nse"]
    pattern: MatrixSparsity
    is_congruence: bool

    def analyze_numeric(self, matrix: BCOO) -> tuple[BCOO, AppliedTransform]:
        new_data = matrix.data[self.sort_order]
        new_indices = jnp.stack([self.pattern.rows, self.pattern.cols], axis=1)
        new_matrix = BCOO((new_data, new_indices), shape=self.pattern.shape)
        return new_matrix, AppliedPermutation(self.perm, self.perm)


class AggregationClustering(eqx.Module):
    """Symbolic transform that permutes rows and columns to gather coupled entries
    into contiguous diagonal blocks, by parallel pairwise aggregation.

    A symmetric permutation (`P^T A P`), so it is always a congruence: it moves entries
    around without changing the matrix's conditioning. Meant to run ahead of a
    block-structured preconditioner like `BlockJacobi`, so that a fixed-size cut after
    permuting actually captures real coupling instead of arbitrary rows.
    """

    block_size: int | None = eqx.field(static=True, default=None)

    def analyze_symbolic(self, pattern: MatrixSparsity) -> _ClusteringPlan:
        n, _ = pattern.shape
        nse = pattern.rows.shape[0]
        block_size = (
            self.block_size
            if self.block_size is not None
            else _default_block_size(nse, n)
        )

        label = _aggregate_clusters(pattern.rows, pattern.cols, n, block_size)
        perm = jnp.argsort(label, stable=True)
        inv_perm = jnp.argsort(perm)

        new_rows_unsorted = inv_perm[pattern.rows]
        new_cols_unsorted = inv_perm[pattern.cols]
        # Primary key last: sorts by row, then by column within a row, matching the
        # canonical order the rest of the package expects from a coalesced `BCOO`.
        sort_order = jnp.lexsort((new_cols_unsorted, new_rows_unsorted))
        new_pattern = MatrixSparsity(
            new_rows_unsorted[sort_order], new_cols_unsorted[sort_order], pattern.shape
        )

        return _ClusteringPlan(perm, sort_order, new_pattern, is_congruence=True)


AggregationClustering.__init__.__doc__ = """**Arguments:**

- `block_size`: the target cluster size. Clusters may end up smaller (a vertex with no
    eligible merge partner stays alone) or, rarely, larger (a merge can overshoot the
    cap by one when both sides are one below it), so this is a target, not a hard
    limit. If `None`, a size is derived from the pattern: the nearest power of two to
    the average nonzeros per row, clamped to `[4, 128]`.
"""
