"""Geometry of the overlapping block partition, and the choice of block size.

The reordered index range is cut into equal-size intervals that overlap their neighbours by
a fixed fraction. Each interval indexes one diagonal block of the reordered matrix, and each
row is assigned a single owning interval so that applying the collection of block inverses
sums to a well-defined operator rather than double-counting the overlaps.

Everything here depends on the sparsity pattern alone, never on the values, which is what
lets one analysis serve every matrix sharing a pattern. See the theory page in
`docs/theory/block-jacobi-gmres.md`.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Integer

# Block sizes considered when the caller does not fix one. Geometric rather than exhaustive:
# capture changes slowly with the block size, so a fine sweep would cost passes over the
# pattern without changing the answer.
_CANDIDATE_BLOCK_SIZES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def geometry(
    size: int, block_size: int, overlap_fraction: float
) -> tuple[int, int, int]:
    """Resolve the partition's shape to `(block_size, stride, num_blocks)`.

    The block size is clamped to `size`, since a block wider than the matrix would only be
    padding. Blocks start at multiples of the stride, except that the last one is pulled back
    to end exactly at `size`. That keeps every block full, so no padded rows exist anywhere
    and the block array never contains positions that have to be masked out later.
    """
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError(
            f"`overlap_fraction` must be in [0, 1); got {overlap_fraction}."
        )
    if block_size < 1:
        raise ValueError(f"`block_size` must be at least 1; got {block_size}.")

    resolved = min(block_size, size)
    stride = max(1, resolved - int(resolved * overlap_fraction))
    if resolved >= size:
        return resolved, stride, 1
    num_blocks = -(-(size - resolved) // stride) + 1
    return resolved, stride, num_blocks


def block_starts(
    size: int, block_size: int, overlap_fraction: float
) -> Integer[Array, " num_blocks"]:
    """The first row index of each block."""
    resolved, stride, num_blocks = geometry(size, block_size, overlap_fraction)
    offsets = jnp.arange(num_blocks, dtype=jnp.int32) * stride
    return jnp.minimum(offsets, size - resolved)


def max_blocks_per_entry(size: int, block_size: int, overlap_fraction: float) -> int:
    """How many blocks can contain a single entry, at most.

    A block spanning `b` rows and starts advancing by `stride` would give `ceil(b / stride)`,
    but pulling the last block back leaves the final starts closer together than that, so the
    true figure can be one higher. It is counted here instead of predicted, since the starts
    follow from the size and the overlap alone and so are known without touching the pattern.

    An entry is in block `j` when `start_j` lies in `(lower - b, lower]`. That count only
    rises as `lower` crosses a start, so checking each start finds the maximum.
    """
    resolved, stride, num_blocks = geometry(size, block_size, overlap_fraction)
    starts = np.minimum(np.arange(num_blocks) * stride, size - resolved)
    above = np.searchsorted(starts, starts, side="right")
    below = np.searchsorted(starts, starts - resolved, side="right")
    return int(np.max(above - below))


def partition(
    size: int, block_size: int, overlap_fraction: float
) -> tuple[Integer[Array, "num_blocks b"], Bool[Array, "num_blocks b"]]:
    """The rows each block covers, and which of them it owns.

    Returns `(gather_index, core_mask)`. `gather_index[k, j]` is the row at position `j` of
    block `k`, so reading a vector at those indices assembles every block's subvector at
    once. `core_mask[k, j]` marks the rows block `k` is responsible for writing back.

    Ownership is by stride rather than by block, so a row belongs to whichever block would
    contain it if none were pulled back. Every row then has exactly one owner, and that
    owner always covers it.
    """
    resolved, stride, num_blocks = geometry(size, block_size, overlap_fraction)
    starts = block_starts(size, block_size, overlap_fraction)
    gather_index = starts[:, None] + jnp.arange(resolved, dtype=jnp.int32)[None, :]
    owner = jnp.clip(gather_index // stride, 0, num_blocks - 1)
    core_mask = owner == jnp.arange(num_blocks, dtype=jnp.int32)[:, None]
    return gather_index, core_mask


def _containing_block(
    lower: Integer[Array, " nse"],
    size: int,
    resolved: int,
    stride: int,
    num_blocks: int,
) -> Integer[Array, " nse"]:
    """The highest-numbered block starting at or before `lower`.

    Since every block is the same width, a later start always reaches at least as far, so
    this is the single block most likely to contain an entry. Pulling the last block back
    compresses the final few starts together, which is why saturation is tested for directly
    rather than inferred from the stride.
    """
    saturated = jnp.int32(num_blocks - 1)
    stepped = (lower // stride).astype(jnp.int32)
    highest = jnp.where(size - resolved <= lower, saturated, stepped)
    return jnp.clip(highest, 0, saturated)


def capture_fraction(
    rows: Integer[Array, " nse"],
    cols: Integer[Array, " nse"],
    size: int,
    block_size: int,
    overlap_fraction: float,
) -> Array:
    """The proportion of stored entries that fall inside some block.

    Measured on the pattern rather than estimated from its bandwidth. Aligned blocks capture
    an entry at offset `d` only about `1 - d/b` of the time, so any estimate from the offset
    distribution alone is optimistic, and overlap changes the relationship again.

    The entries this leaves out are dropped from the preconditioner only. The iteration
    multiplies by the full reordered matrix, so they cost convergence rather than accuracy.
    """
    resolved, stride, num_blocks = geometry(size, block_size, overlap_fraction)
    lower = jnp.minimum(rows, cols)
    upper = jnp.maximum(rows, cols)
    start = jnp.minimum(
        _containing_block(lower, size, resolved, stride, num_blocks) * stride,
        size - resolved,
    )
    return jnp.mean(upper < start + resolved)


def choose_block_size(
    rows: Integer[Array, " nse"],
    cols: Integer[Array, " nse"],
    size: int,
    max_block_size: int,
    overlap_fraction: float,
    capture_target: float,
) -> tuple[int, float]:
    """Pick the cheapest block size whose capture reaches `capture_target`.

    Storing and applying the inverses both cost `num_blocks * b^2`, which grows with the
    block size, so the cheapest adequate partition is wanted rather than the smallest block.
    The two differ only for small matrices: when the whole matrix fits one block, that block
    captures everything and costs less than two blocks large enough to reach the same target,
    so it wins. The preconditioner is then the exact inverse and the iteration converges at
    once.

    Returns `(block_size, captured)`. Both are Python scalars because the block size sets
    array shapes, so this needs a sparsity pattern whose indices are known values rather
    than placeholders.
    """
    if not 0.0 < capture_target <= 1.0:
        raise ValueError(f"`capture_target` must be in (0, 1]; got {capture_target}.")
    largest = min(max_block_size, size)
    candidates = sorted(
        {min(b, largest) for b in _CANDIDATE_BLOCK_SIZES if b <= largest} | {largest}
    )
    # Cost depends only on the geometry, so it is known without touching the pattern.
    costs = jnp.asarray(
        [geometry(size, b, overlap_fraction)[2] * b * b for b in candidates],
        dtype=jnp.float32,
    )
    captured = jnp.stack(
        [capture_fraction(rows, cols, size, b, overlap_fraction) for b in candidates]
    )

    adequate = captured >= capture_target
    ranked = jnp.argmin(jnp.where(adequate, costs, jnp.inf))
    # Nothing adequate means the pattern is too spread out for any permitted block, so take
    # the largest and let the capture diagnostic report how little was kept.
    chosen = jnp.where(jnp.any(adequate), ranked, len(candidates) - 1)

    try:
        index = int(chosen)
    except jax.errors.TracerBoolConversionError as error:
        raise _traced_pattern_error() from error
    except jax.errors.ConcretizationTypeError as error:
        raise _traced_pattern_error() from error
    return candidates[index], float(captured[index])


def _traced_pattern_error() -> ValueError:
    return ValueError(
        "Choosing a block size requires a sparsity pattern with known indices, because "
        "the block size determines array shapes and so cannot be a traced value. This "
        "happens when the pattern's indices are themselves traced, for example under a "
        "`jax.vmap` over indices. Pass `block_size=` explicitly to skip the choice, which "
        "makes the whole solver traceable."
    )


def block_destinations(
    rows: Integer[Array, " nse"],
    cols: Integer[Array, " nse"],
    size: int,
    block_size: int,
    overlap_fraction: float,
) -> Integer[Array, "num_dest nse"]:
    """Where each stored entry lands in the flattened `(num_blocks, b, b)` block array.

    With overlap an entry can belong to several blocks and must appear in each, since every
    block is a genuine submatrix of the reordered matrix. One row of the result is produced
    per block that could contain an entry, as counted by `max_blocks_per_entry`. Blocks
    containing a given entry are consecutive and end at the highest one starting at or before
    it, so the candidates are found by walking back from there.

    Entries that a candidate does not contain, including entries no block covers at all, are
    sent one position past the end of the block array. Scattering into an array with that one
    extra slot and then discarding it is what drops them, so nothing has to be tested for
    per entry when the blocks are assembled.
    """
    resolved, stride, num_blocks = geometry(size, block_size, overlap_fraction)
    lower = jnp.minimum(rows, cols)
    upper = jnp.maximum(rows, cols)
    highest = _containing_block(lower, size, resolved, stride, num_blocks)
    discard = jnp.int32(num_blocks * resolved * resolved)

    def destination(back: int) -> Integer[Array, " nse"]:
        block = highest - back
        start = jnp.minimum(block * stride, size - resolved)
        inside = (block >= 0) & (start <= lower) & (upper < start + resolved)
        flat = block * resolved * resolved + (rows - start) * resolved + (cols - start)
        return jnp.where(inside, flat.astype(jnp.int32), discard)

    candidates = max_blocks_per_entry(size, block_size, overlap_fraction)
    return jnp.stack([destination(back) for back in range(candidates)])
