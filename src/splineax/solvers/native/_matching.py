"""Maximum bipartite matching between the rows and columns of a sparsity pattern.

Computed from a COO index pair alone, the same input `_ordering.py` works from, so this
too is pattern-only and reusable across every matrix sharing a pattern. Unlike the
reordering, the graph here is bipartite rather than symmetrized: a row vertex and a
column vertex for every index, joined whenever the corresponding entry is stored.

See the theory page in `docs/theory/block-jacobi-gmres.md` for what a matching is used
for and the reasoning behind the algorithm below.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Integer

# Sentinel for a BFS distance not yet reached. Larger than any real distance, so an
# unvisited vertex sorts last and `dist == _UNVISITED` is the "not yet reached" test.
_UNVISITED = jnp.iinfo(jnp.int32).max


def _greedy_init(
    rows: Integer[Array, " nse"], cols: Integer[Array, " nse"], size: int
) -> tuple[Integer[Array, " n"], Integer[Array, " n"]]:
    """A matching built by repeated propose-and-accept.

    Each free row proposes to its lowest-index available column, and each column
    accepts its lowest-index proposer. Rows that lose retry with whatever column is
    next available once the round's acceptances free up the graph. This is a warm
    start rather than the final answer: `_augment` brings whatever it leaves unmatched
    up to a true maximum regardless of how good a start this made, so the tie-break
    only affects how much work is left for that stage, never correctness.
    """
    positions = jnp.arange(size, dtype=jnp.int32)

    def unfinished(state: tuple[Array, Array]) -> Bool[Array, ""]:
        mrow, mcol = state
        free_edge = (mrow[rows] == -1) & (mcol[cols] == -1)
        return jnp.any(free_edge)

    def step(state: tuple[Array, Array]) -> tuple[Array, Array]:
        mrow, mcol = state
        free_edge = (mrow[rows] == -1) & (mcol[cols] == -1)

        # Each free row's lowest-index free column, or `size` if it has none.
        proposal = jax.ops.segment_min(
            jnp.where(free_edge, cols, size), rows, num_segments=size
        )
        has_proposal = proposal < size

        # Each proposed-to column's lowest-index proposer, via a reduction grouped by
        # the proposed column, with rows proposing nothing sent to a trash group.
        target = jnp.where(has_proposal, proposal, size)
        best_row = jax.ops.segment_min(
            jnp.where(has_proposal, positions, size), target, num_segments=size + 1
        )
        accepted = has_proposal & (best_row[proposal] == positions)

        new_mrow = jnp.where(accepted, proposal, mrow)
        new_mcol = mcol.at[jnp.where(accepted, proposal, size)].set(
            positions, mode="drop"
        )
        return new_mrow, new_mcol

    init = (jnp.full(size, -1, dtype=jnp.int32), jnp.full(size, -1, dtype=jnp.int32))
    return jax.lax.while_loop(unfinished, step, init)


def _bfs_phase(
    rows: Integer[Array, " nse"],
    cols: Integer[Array, " nse"],
    size: int,
    mrow: Integer[Array, " n"],
    mcol: Integer[Array, " n"],
) -> tuple[Integer[Array, " n"], Integer[Array, " n"], Integer[Array, ""]]:
    """Layer the graph by breadth-first search from the free rows, stopping at the
    first level holding a free column.

    Returns `(dist_col, parent_col, target_level)`. `dist_col[j]` is the level at
    which column `j` was reached, `parent_col[j]` the lowest-index row that reached
    it there, and `target_level` the level of the first free column found, or
    `_UNVISITED` if none is reachable, meaning the matching passed in is already
    maximum.

    A round advances the frontier two levels at once: forward from a row level to the
    column level it reaches, then backward from every *matched* newly reached column
    to its row through the matching edge, which is the only edge a column takes
    backward. That second step is skipped once a free column is found, since a
    shortest path stops there and levels beyond it are never needed.
    """
    dist_row = jnp.where(mrow == -1, 0, _UNVISITED).astype(jnp.int32)
    dist_col0 = jnp.full(size, _UNVISITED, dtype=jnp.int32)
    parent_col0 = jnp.full(size, -1, dtype=jnp.int32)
    frontier0 = dist_row == 0
    target0 = jnp.int32(_UNVISITED)

    def unfinished(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> Bool[Array, ""]:
        _, _, _, frontier, target = state
        return (target == _UNVISITED) & jnp.any(frontier)

    def step(
        state: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        dist_row, dist_col, parent_col, frontier, target = state
        cur_level = jnp.min(jnp.where(frontier, dist_row, _UNVISITED))
        from_frontier = frontier[rows]

        # Step out: every column newly reachable this round, and the lowest-index
        # frontier row reaching it, in one pass over the edge list each.
        cand_dist = jnp.where(from_frontier, cur_level + 1, _UNVISITED)
        new_col_dist = jax.ops.segment_min(cand_dist, cols, num_segments=size)
        newly_reached_col = (dist_col == _UNVISITED) & (new_col_dist < _UNVISITED)

        cand_row = jnp.where(from_frontier, rows, _UNVISITED)
        best_parent = jax.ops.segment_min(cand_row, cols, num_segments=size)

        new_parent_col = jnp.where(newly_reached_col, best_parent, parent_col)
        new_dist_col = jnp.where(newly_reached_col, new_col_dist, dist_col)

        free_newly = newly_reached_col & (mcol == -1)
        found = jnp.any(free_newly)
        new_target = jnp.where(found, cur_level + 1, target)

        # Step back: matched newly reached columns hand the search on to the one row
        # each is matched to, deterministically, since a matching is injective.
        matched_newly = newly_reached_col & (mcol != -1)
        row_target = jnp.where(matched_newly, mcol, size)
        dist_row_cand = (
            jnp.full(size, _UNVISITED, dtype=jnp.int32)
            .at[row_target]
            .set(jnp.where(matched_newly, new_dist_col + 1, _UNVISITED), mode="drop")
        )
        newly_reached_row = (dist_row == _UNVISITED) & (dist_row_cand < _UNVISITED)
        new_dist_row = jnp.where(newly_reached_row, dist_row_cand, dist_row)

        return new_dist_row, new_dist_col, new_parent_col, newly_reached_row, new_target

    init = (dist_row, dist_col0, parent_col0, frontier0, target0)
    _, dist_col, parent_col, _, target = jax.lax.while_loop(unfinished, step, init)
    return dist_col, parent_col, target


def _augment(
    mrow: Integer[Array, " n"],
    mcol: Integer[Array, " n"],
    dist_col: Integer[Array, " n"],
    parent_col: Integer[Array, " n"],
    target_level: Integer[Array, ""],
    size: int,
) -> tuple[Integer[Array, " n"], Integer[Array, " n"]]:
    """Realize a maximal set of vertex-disjoint shortest augmenting paths, walking
    backward one level at a time from the free columns `_bfs_phase` found.

    Each active column claims its recorded parent row, with `size` conflicting claims
    on the same row resolved by keeping the lowest-index column and abandoning the
    rest. A claimed row's own former match, if it had one, becomes the next column to
    walk from. A row already claimed earlier in the same walk cannot be claimed again,
    since `parent_col` holds no alternative for a column to retry, so such a column is
    abandoned instead, and its stale match is cleared rather than left dangling.

    Abandoning a column leaves its augmenting path unrealized this phase rather than
    wrong: it stays free, and `_bfs_phase` will find it again, at this level or a
    later one, for as long as a shortest augmenting path to it still exists. What this
    guarantees is only that a nonempty phase strictly grows the matching. It need not
    grow it by every shortest path available, and how many phases a pattern needs is
    therefore an empirical question, not a bound derived here.
    """
    positions = jnp.arange(size, dtype=jnp.int32)
    mrow_start = mrow

    active0 = (dist_col == target_level) & (mcol == -1)
    used_row0 = jnp.zeros(size, dtype=bool)

    def unfinished(state: tuple[Array, Array, Array, Array]) -> Bool[Array, ""]:
        *_, active = state
        return jnp.any(active)

    def step(
        state: tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array]:
        mrow, mcol, used_row, active = state

        claimable = active & ~used_row[jnp.where(active, parent_col, 0)]
        claim_row = jnp.where(claimable, parent_col, size)
        best_col = jax.ops.segment_min(
            jnp.where(claimable, positions, size), claim_row, num_segments=size + 1
        )
        survive = claimable & (best_col[claim_row] == positions)
        abandoned = active & ~survive

        mcol = mcol.at[jnp.where(abandoned, positions, size)].set(-1, mode="drop")
        mrow = mrow.at[jnp.where(survive, parent_col, size)].set(positions, mode="drop")
        mcol = mcol.at[jnp.where(survive, positions, size)].set(parent_col, mode="drop")
        used_row = used_row.at[jnp.where(survive, parent_col, size)].set(
            True, mode="drop"
        )

        old_partner = mrow_start[jnp.where(survive, parent_col, 0)]
        continues = survive & (old_partner >= 0)
        new_active = (
            jnp.zeros(size, dtype=bool)
            .at[jnp.where(continues, old_partner, size)]
            .set(True, mode="drop")
        )
        return mrow, mcol, used_row, new_active

    mrow, mcol, _, _ = jax.lax.while_loop(
        unfinished, step, (mrow, mcol, used_row0, active0)
    )
    return mrow, mcol


def matching(
    rows: Integer[Array, " nse"], cols: Integer[Array, " nse"], size: int
) -> tuple[Integer[Array, " n"], Integer[Array, ""]]:
    """A maximum matching between the rows and columns of a sparsity pattern, by
    Hopcroft-Karp: a greedy warm start, then repeated shortest-augmenting-path phases
    until none remain.

    Returns `(partner, matched)`. `partner[i]` is the column row `i` is matched to, or
    `-1` if row `i` is unmatched. `matched` is the number of matched pairs, which by
    the Frobenius-König theorem is the pattern's structural rank. A perfect matching
    (`matched == size`) is not guaranteed: a pattern whose determinant is identically
    zero has none, whatever values it is later given.
    """
    mrow, mcol = _greedy_init(rows, cols, size)

    def unfinished(state: tuple[Array, Array, Array]) -> Bool[Array, ""]:
        *_, done = state
        return jnp.logical_not(done)

    def step(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        mrow, mcol, _ = state
        dist_col, parent_col, target = _bfs_phase(rows, cols, size, mrow, mcol)
        found = target != _UNVISITED
        new_mrow, new_mcol = jax.lax.cond(
            found,
            lambda: _augment(mrow, mcol, dist_col, parent_col, target, size),
            lambda: (mrow, mcol),
        )
        return new_mrow, new_mcol, jnp.logical_not(found)

    mrow, mcol, _ = jax.lax.while_loop(unfinished, step, (mrow, mcol, jnp.array(False)))
    return mrow, jnp.sum(mrow >= 0).astype(jnp.int32)
