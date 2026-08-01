"""Bandwidth-reducing symmetric permutations, computed from a COO index pair.

A symmetric permutation `P A P^T` leaves the solution of `A x = b` unchanged and only
moves where the nonzeros sit, so it can be chosen purely to improve structure. Both
heuristics here treat the sparsity pattern as a graph and work directly on the stored
index vectors: those already form an edge list, so vertex degrees, frontier expansion and
Laplacian products are all segment reductions over them, with no adjacency structure built.

See the theory page in `docs/theory/block-jacobi-gmres.md` for why these orderings help and
what the block preconditioner does with the result.
"""

from enum import IntEnum

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard
from jaxtyping import Array, Bool, Inexact, Integer

# Marks a vertex the search has not numbered yet. Larger than any real position, so
# unnumbered vertices sort last and `rank == _UNRANKED` is the "still to do" test.
_UNRANKED = jnp.iinfo(jnp.int32).max

# Below this size the Laplacian is small enough to diagonalize densely, which is exact and
# free of the convergence risk an iterative eigensolver carries. Above it the dense matrix
# would be the largest array in the solver, so LOBPCG takes over.
_DENSE_EIGH_LIMIT = 256

# Laplacian eigenvectors requested. One spans the null space and orders nothing, so this
# leaves three ordering candidates to score against each other.
_SPECTRAL_VECTORS = 4


class Ordering(IntEnum):
    """Which bandwidth-reducing permutation to apply before blocking."""

    NONE = 0
    """Leave the ordering alone. Right for operators that are already narrowly banded."""
    SPECTRAL = 1
    """Sort by an eigenvector of the graph Laplacian. One sort and one eigensolve, so the
    cost does not depend on the shape of the graph."""
    RCM = 2
    """Reverse Cuthill-McKee. Deterministic, but costs one pass per breadth-first level, so
    it is a poor fit for path-like graphs where the level count grows with the size."""


def bandwidth(
    rows: Integer[Array, " nse"], cols: Integer[Array, " nse"]
) -> Integer[Array, ""]:
    """The largest distance from the diagonal at which the pattern has a nonzero."""
    return jnp.max(jnp.abs(rows - cols), initial=0)


def order(
    rows: Integer[Array, " nse"],
    cols: Integer[Array, " nse"],
    size: int,
    ordering: Ordering,
) -> Integer[Array, " n"]:
    """Compute a permutation of `0..size-1` that reduces the pattern's bandwidth.

    The result is in new-position order: entry `k` is the original index that ends up at
    position `k`. Reordering a vector is therefore an indexing by the result, while
    relabeling stored indices needs `inverse_permutation` of it.
    """
    match ordering:
        case Ordering.NONE:
            return jnp.arange(size, dtype=jnp.int32)
        case Ordering.SPECTRAL:
            return _spectral(rows, cols, size)
        case Ordering.RCM:
            return _reverse_cuthill_mckee(rows, cols, size)
        case _:
            raise ValueError(
                "`ordering` must be one of `Ordering.NONE`, `Ordering.SPECTRAL` or "
                f"`Ordering.RCM`; got {ordering!r}."
            )


def inverse_permutation(perm: Integer[Array, " n"]) -> Integer[Array, " n"]:
    """The inverse of `perm`, mapping an original index to its new position."""
    size = perm.shape[0]
    positions = jnp.arange(size, dtype=jnp.int32)
    return jnp.zeros(size, dtype=jnp.int32).at[perm].set(positions)


def _undirected_edges(
    rows: Integer[Array, " nse"], cols: Integer[Array, " nse"]
) -> tuple[Integer[Array, " 2nse"], Integer[Array, " 2nse"], Bool[Array, " 2nse"]]:
    """The pattern as an undirected edge list, plus a mask dropping the diagonal.

    Reading every stored entry in both directions symmetrizes the pattern, which is what
    both orderings need and what the pattern of `A + A^T` would give. Self-loops carry no
    adjacency information, so the mask excludes them.
    """
    source = jnp.concatenate([rows, cols])
    target = jnp.concatenate([cols, rows])
    return source, target, source != target


def _degrees(
    source: Integer[Array, " 2nse"], off_diagonal: Bool[Array, " 2nse"], size: int
) -> Integer[Array, " n"]:
    """Vertex degrees, counting incident off-diagonal entries.

    A duplicated entry in the pattern inflates a degree. That only perturbs a tie-break, so
    coalescing first is not worth the sort it would cost.
    """
    counts = off_diagonal.astype(jnp.int32)
    return jax.ops.segment_sum(counts, source, num_segments=size)


def _reverse_cuthill_mckee(
    rows: Integer[Array, " nse"], cols: Integer[Array, " nse"], size: int
) -> Integer[Array, " n"]:
    """Number the vertices by breadth-first level, then reverse.

    An edge can only join vertices in the same or adjacent levels, so the bandwidth is
    bounded by the largest number of vertices in two consecutive levels: thin levels give a
    narrow band. Within a level, vertices are ordered by the position of their
    lowest-numbered neighbor in the previous level and then by degree, which keeps siblings
    together and is what separates Cuthill-McKee from a plain breadth-first numbering.

    Ordering by neighbor *position* rather than neighbor index is what makes the result
    independent of how the unknowns happened to be numbered to begin with. It costs one sort
    per level, since a level's positions are only known once the previous level has been
    numbered.
    """
    source, target, off_diagonal = _undirected_edges(rows, cols)
    degree = _degrees(source, off_diagonal, size)
    positions = jnp.arange(size, dtype=jnp.int32)

    def unfinished(state: tuple[Array, Array, Array]) -> Bool[Array, ""]:
        rank, _, _ = state
        return jnp.any(rank == _UNRANKED)

    def step(state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        rank, frontier, numbered = state
        unranked = rank == _UNRANKED

        # Expand the frontier: a vertex is reached when some incident edge starts in it.
        from_frontier = jnp.where(off_diagonal, frontier[source], False)
        reached = (
            jax.ops.segment_max(
                from_frontier.astype(jnp.int32), target, num_segments=size
            )
            > 0
        )
        newly_reached = reached & unranked
        # Position of the earliest-numbered frontier vertex adjacent to each vertex.
        parent_rank = jax.ops.segment_min(
            jnp.where(from_frontier, rank[source], _UNRANKED),
            target,
            num_segments=size,
        )

        def grow() -> tuple[Array, Array, Array]:
            # Sorting with the untouched vertices keyed to the sentinel brings the newly
            # reached ones to the front, in the order they should be numbered.
            ranking = jnp.lexsort(
                (
                    jnp.where(newly_reached, degree, _UNRANKED),
                    jnp.where(newly_reached, parent_rank, _UNRANKED),
                )
            )
            # Cast because `sum` widens to int64 under x64, while the counter stays int32.
            count = jnp.sum(newly_reached).astype(jnp.int32)
            assigned = jnp.where(positions < count, numbered + positions, rank[ranking])
            return rank.at[ranking].set(assigned), newly_reached, numbered + count

        def seed() -> tuple[Array, Array, Array]:
            # Nothing was reached, so this is either the first step or a finished component.
            # Start the next one at an unnumbered vertex of least degree.
            start = jnp.argmin(jnp.where(unranked, degree, _UNRANKED))
            return (
                rank.at[start].set(numbered),
                jnp.zeros_like(frontier).at[start].set(True),
                numbered + 1,
            )

        return jax.lax.cond(jnp.any(newly_reached), grow, seed)

    initial = (
        jnp.full(size, _UNRANKED, dtype=jnp.int32),
        jnp.zeros(size, dtype=bool),
        jnp.int32(0),
    )
    # Every step numbers at least one vertex, so `size` steps always suffice.
    rank, _, _ = jax.lax.while_loop(unfinished, step, initial)

    # Reversal leaves the bandwidth alone but shrinks the envelope.
    return jnp.argsort(rank)[::-1].astype(jnp.int32)


def _spectral(
    rows: Integer[Array, " nse"], cols: Integer[Array, " nse"], size: int
) -> Integer[Array, " n"]:
    """Sort by an eigenvector of the graph Laplacian belonging to a small eigenvalue.

    The Fiedler vector, belonging to the second smallest eigenvalue, is the smoothest
    non-constant function on the graph, so adjacent vertices take similar values and sorting
    by those values places connected vertices close together.

    Several eigenvectors are scored rather than just the Fiedler vector, because the second
    eigenvalue need not be simple: on a grid, symmetry between the two axes makes it exactly
    double, and on a disconnected graph zero repeats once per component. In either case no
    single vector is distinguished and an eigensolver may return any vector from the
    eigenspace, some of which order badly. The identity is scored alongside them, which
    bounds the result: reordering can then never come out worse than leaving the pattern
    alone.
    """
    identity = jnp.arange(size, dtype=jnp.float32)
    if size <= 2:
        return jnp.arange(size, dtype=jnp.int32)

    source, target, off_diagonal = _undirected_edges(rows, cols)
    degree = _degrees(source, off_diagonal, size)
    eigenvectors = _small_laplacian_eigenvectors(
        source, target, off_diagonal, degree, size
    )
    candidates = jnp.concatenate([identity[:, None], eigenvectors], axis=1)

    def score(vector: Inexact[Array, " n"]) -> tuple[Array, Array]:
        perm = jnp.argsort(vector).astype(jnp.int32)
        relabeled = inverse_permutation(perm)
        return perm, bandwidth(relabeled[rows], relabeled[cols])

    permutations, bandwidths = jax.vmap(score, in_axes=1)(candidates)
    return permutations[jnp.argmin(bandwidths)]


def _small_laplacian_eigenvectors(
    source: Integer[Array, " 2nse"],
    target: Integer[Array, " 2nse"],
    off_diagonal: Bool[Array, " 2nse"],
    degree: Integer[Array, " n"],
    size: int,
) -> Inexact[Array, "n vectors"]:
    """Eigenvectors of the graph Laplacian for its smallest nonzero eigenvalues.

    The constant vector spans the null space of a connected Laplacian and orders nothing, so
    it is dropped and the remaining columns are returned as ordering candidates.
    """
    wanted = min(_SPECTRAL_VECTORS, size)
    diagonal = degree.astype(jnp.float32)

    if size <= _DENSE_EIGH_LIMIT:
        adjacency = (
            jnp.zeros((size, size), dtype=jnp.float32)
            .at[source, target]
            .set(jnp.where(off_diagonal, 1.0, 0.0))
        )
        # `eigh` returns ascending eigenvalues, so the first column is the constant one.
        _, vectors = jnp.linalg.eigh(jnp.diag(diagonal) - adjacency)
        return vectors[:, 1:wanted]

    def laplacian_product(block: Inexact[Array, "n k"]) -> Inexact[Array, "n k"]:
        neighbor_sum = jax.ops.segment_sum(
            jnp.where(off_diagonal[:, None], block[target], 0.0),
            source,
            num_segments=size,
        )
        return diagonal[:, None] * block - neighbor_sum

    # LOBPCG finds the *largest* eigenvalues, so the Laplacian is reflected about a shift
    # exceeding its spectral radius. The largest eigenvalues of the reflection belong to the
    # smallest of the Laplacian, with the eigenvectors unchanged.
    shift = 2.0 * jnp.max(diagonal)

    def reflected(block: Inexact[Array, "n k"]) -> Inexact[Array, "n k"]:
        return shift * block - laplacian_product(block)

    # A fixed key keeps the ordering reproducible. A random start is generically full rank,
    # which is all LOBPCG asks of it.
    start = jax.random.normal(jax.random.key(0), (size, wanted), dtype=jnp.float32)
    _, vectors, _ = lobpcg_standard(reflected, start)
    return vectors[:, 1:wanted]
