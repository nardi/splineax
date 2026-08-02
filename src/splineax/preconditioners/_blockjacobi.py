"""Block Jacobi preconditioning: invert the diagonal blocks, discard everything else."""

from functools import lru_cache
from typing import Literal, Sequence, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import qr as _qr
from jax.scipy.linalg import solve_triangular
from jaxtyping import Array, Inexact, Integer
from lineax import (
    AbstractLinearOperator,
    is_negative_semidefinite,
    is_positive_semidefinite,
)
from lineax._tags import positive_semidefinite_tag

from splineax._partition import BlockPartition, _block_of_index
from splineax._pattern import ConcreteCooPattern, CooPattern, _Sparsity, as_coo_pattern
from splineax.operators._blockdiag import BlockDiagonalLinearOperator
from splineax.preconditioners._partition import BlockPartitioner, captured_fraction
from splineax.preconditioners._preconditioner import Side
from splineax.preconditioners._transform import _as_bcoo_matrix

_Factorization = Literal["auto", "lu", "qr", "svd"]


@lru_cache(maxsize=None)
def _padding_identity(sizes: tuple[int, ...], dtype: np.dtype) -> np.ndarray | None:
    """An identity filling the unused corner of every short block.

    Every block is stored at the width of the largest, so a shorter one leaves a
    trailing square of zeros. Left alone that makes the stored block exactly singular
    and its inverse meaningless; putting an identity there instead makes the stored
    block `[[A, 0], [0, I]]`, whose inverse is `[[A^-1, 0], [0, I]]` -- so the real
    part inverts as if the padding were not there. `None` when the partition is uniform,
    so the whole thing costs nothing in the common case.
    """
    width = max(sizes)
    if len(set(sizes)) == 1:
        return None
    padded = np.arange(width)[None, :] >= np.array(sizes)[:, None]
    identity = padded[:, :, None] & np.eye(width, dtype=bool)[None, :, :]
    return identity.astype(dtype)


def _scatter_indices(sizes: tuple[int, ...], pattern: ConcreteCooPattern) -> np.ndarray:
    """Where each stored entry lands in the flattened block stack.

    Entries falling outside every block -- the coupling block Jacobi throws away -- are
    sent to a sentinel slot one past the end of the buffer, which is then dropped. An
    explicit sentinel rather than `mode="drop"`, whose exact semantics vary between
    primitives while a negative index would silently wrap around to a real slot.
    """
    row = pattern.rows.astype(np.int64)
    col = pattern.cols.astype(np.int64)
    block_of, start_of = _block_of_index(sizes)
    width = max(sizes)
    sentinel = len(sizes) * width * width
    inside = block_of[row] == block_of[col]
    slot = (
        block_of[row] * width * width
        + (row - start_of[row]) * width
        + (col - start_of[col])
    )
    return np.where(inside, slot, sentinel).astype(np.int64)


def structurally_singular_blocks(
    pattern: ConcreteCooPattern, partition: BlockPartition
) -> np.ndarray:
    """Which blocks are singular for want of entries, whatever the values are.

    Counts only the entries that will actually land inside each block. A block with a
    structurally empty row or column has an identically zero line once the out-of-block
    coupling is discarded, so it is exactly singular -- a certainty derived from the
    pattern, not an estimate.

    `BlockJacobi` derives this itself rather than asking the partitioner for it, which
    is what lets `BlockPartitioner` stay a one-method protocol.
    """
    block_of, start_of = _block_of_index(partition.sizes)
    width = partition.max_block_size
    slots = partition.num_blocks * width
    rows, cols = pattern.rows.astype(np.int64), pattern.cols.astype(np.int64)
    inside = block_of[rows] == block_of[cols]
    rows, cols = rows[inside], cols[inside]
    local_row = block_of[rows] * width + (rows - start_of[rows])
    local_col = block_of[cols] * width + (cols - start_of[cols])
    row_filled = np.bincount(local_row, minlength=slots) > 0
    col_filled = np.bincount(local_col, minlength=slots) > 0
    # Only positions inside a block's real extent count; the padding is an identity.
    real = np.arange(width)[None, :] < np.array(partition.sizes)[:, None]
    filled = (row_filled & col_filled).reshape(partition.num_blocks, width)
    return np.any(real & ~filled, axis=1)


class _BlockJacobiNumeric(eqx.Module):
    """A built block Jacobi preconditioner: the inverted blocks, ready for lineax."""

    inverse: BlockDiagonalLinearOperator

    def left_operator(self) -> AbstractLinearOperator:
        return self.inverse

    def right_operator(self) -> AbstractLinearOperator:
        # Block Jacobi is one operator applied either way round, so both sides are the
        # same `M`. They are separate methods because that is not true in general: a
        # split incomplete factorisation supplies `L^-1` here and `U^-1` on the left.
        return self.inverse


class _BlockJacobiSymbolic(eqx.Module):
    """Block Jacobi with its partition resolved, awaiting values."""

    partition: BlockPartition = eqx.field(static=True)
    scatter: Integer[Array, " nse"]
    factorization: _Factorization = eqx.field(static=True)
    rcond: float | None = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)
    nse: int = eqx.field(static=True)

    @property
    def sides(self) -> frozenset[Side]:
        return frozenset({Side.LEFT, Side.RIGHT})

    def numeric(self, operator: AbstractLinearOperator) -> _BlockJacobiNumeric:
        """Build the preconditioner for an operator with this symbolic's pattern."""
        matrix = _as_bcoo_matrix(operator)
        if matrix.nse != self.nse:
            raise ValueError(
                f"This preconditioner was built for a pattern with {self.nse} stored "
                f"entries, but the operator has {matrix.nse}. The symbolic phase can "
                "only be reused for operators with the very same sparsity pattern, in "
                "the same index order."
            )
        # Matching the operator's dtype exactly, not just its kind: lineax's
        # `preconditioner_and_y0` compares structures including dtype, and `pinv`/`inv`
        # would otherwise be free to promote.
        dtype = operator.in_structure().dtype
        partition = self.partition
        width = partition.max_block_size
        flat = jnp.zeros(partition.num_blocks * width * width + 1, dtype=dtype)
        # `.add`, not `.set`: an uncoalesced matrix stores an entry more than once, and
        # its value is the sum of those. The final slot is the out-of-block sentinel.
        flat = flat.at[self.scatter].add(matrix.data.astype(dtype))
        blocks = flat[:-1].reshape(partition.num_blocks, width, width)
        padding = _padding_identity(partition.sizes, np.dtype(dtype))
        if padding is not None:
            blocks = blocks + jnp.asarray(padding)

        tags = self.tags
        if is_positive_semidefinite(operator) or is_negative_semidefinite(operator):
            # Every principal submatrix of a definite matrix is definite, and so is the
            # inverse, so the preconditioner inherits the tag `lineax.CG` demands. It
            # demands *positive* definiteness specifically, and takes the
            # preconditioner as given while negating the operator itself, so a negative
            # definite system needs its blocks negated here to match.
            if is_negative_semidefinite(operator):
                blocks = -blocks
            tags = tags | {positive_semidefinite_tag}

        inverse = _invert_blocks(blocks, self.factorization, self.rcond)
        return _BlockJacobiNumeric(
            BlockDiagonalLinearOperator(inverse.astype(dtype), partition, tags)
        )


def _invert_blocks(
    blocks: Inexact[Array, "n w w"], factorization: _Factorization, rcond: float | None
) -> Inexact[Array, "n w w"]:
    """Invert every block at once. All three routes are batched, so this is a single
    XLA call however many blocks there are."""
    match factorization:
        case "lu":
            return jnp.linalg.inv(blocks)
        case "svd":
            return jnp.linalg.pinv(blocks, rtol=rcond)
        case "qr":
            return _pivoted_qr_inverse(blocks, rcond)
        case _:  # pragma: no cover - resolved before reaching here
            raise ValueError(f"Unknown factorization {factorization!r}.")


def _pivoted_qr_inverse(
    blocks: Inexact[Array, "n w w"], rcond: float | None
) -> Inexact[Array, "n w w"]:
    """A rank-revealing inverse via QR with column pivoting.

    `A[:, p] = Q R` with `|diag(R)|` non-increasing, so a rank deficiency shows up as a
    tail of tiny diagonal entries and `A^-1 = P R^-1 Q^H`. Directions below the
    tolerance are truncated away rather than amplified: the trailing block of `R` is
    replaced by an identity (which decouples it from the rows that are kept, so back
    substitution is unaffected) and the corresponding rows of the result are zeroed.
    """
    if jax.default_backend() == "tpu":
        raise NotImplementedError(
            "`BlockJacobi(factorization='qr')` needs QR with column pivoting, which "
            "JAX implements on CPU and GPU only. Use `factorization='svd'` (also "
            "rank-revealing) or `'lu'` on TPU."
        )
    width = blocks.shape[-1]
    # `pivoting=True` makes this a 3-tuple; the overload is not narrowable statically.
    q, r, pivots = cast(tuple[Array, Array, Array], _qr(blocks, pivoting=True))
    diagonal = jnp.abs(jnp.diagonal(r, axis1=-2, axis2=-1))
    tolerance = (
        jnp.finfo(blocks.dtype).eps * width if rcond is None else rcond
    ) * diagonal[..., :1]
    keep = diagonal > tolerance
    identity = jnp.eye(width, dtype=blocks.dtype)
    truncated = jnp.where(keep[..., :, None] & keep[..., None, :], r, identity)
    solved = solve_triangular(truncated, jnp.swapaxes(q.conj(), -1, -2), lower=False)
    solved = jnp.where(keep[..., :, None], solved, 0)
    # `A[:, p] = Q R` means row `p[k]` of `A^-1` is row `k` of `R^-1 Q^H`.
    rows = jnp.arange(blocks.shape[0])[:, None]
    return jnp.zeros_like(solved).at[rows, pivots].set(solved)


class BlockJacobi(eqx.Module):
    """Preconditions by inverting the matrix's diagonal blocks.

    The preconditioner is `M = blockdiag(A)^-1`: the matrix is cut into contiguous
    diagonal blocks, everything outside them is discarded, and what remains is
    inverted. It is a good preconditioner exactly when the discarded coupling is weak,
    which is why it pairs so naturally with a bandwidth-reducing reordering -- see the
    solver's `transforms`.

    **The block partition is a parameter, and can be fulfilled two ways.** Give it a
    value directly, when you know the structure:

    ```python
    import splineax as splx

    fixed = splx.BlockJacobi(blocks=4)                 # uniform 4x4 blocks
    ragged = splx.BlockJacobi(blocks=(3, 5, 4))        # explicit sizes
    assert fixed.blocks == 4
    ```

    Or inject an object that derives one from the sparsity pattern, when you do not:

    ```python
    derived = splx.BlockJacobi(blocks=splx.MaximalCaptureBlockPartitioner())
    assert isinstance(derived.blocks, splx.BlockPartitioner)
    ```

    Anything satisfying the one-method `BlockPartitioner` protocol works there; nothing
    needs to subclass anything.

    !!! note

        There is no default. A partition guessed on your behalf would silently decide
        how much of the matrix gets thrown away, so `blocks` must be given.
    """

    blocks: BlockPartition | int | Sequence[int] | BlockPartitioner = eqx.field(
        static=True
    )
    factorization: _Factorization = eqx.field(default="svd", static=True)
    assume_nonsingular: bool = eqx.field(default=False, static=True)
    rcond: float | None = eqx.field(default=None, static=True)
    tags: frozenset[object] = eqx.field(default=frozenset(), static=True)

    def __check_init__(self):
        if self.factorization not in ("auto", "lu", "qr", "svd"):
            raise ValueError(
                "`factorization` must be one of 'auto', 'lu', 'qr' or 'svd'; got "
                f"{self.factorization!r}."
            )

    @property
    def sides(self) -> frozenset[Side]:
        """Both: `M` is a single operator, applicable on either side."""
        return frozenset({Side.LEFT, Side.RIGHT})

    def symbolic(self, sparsity: _Sparsity | CooPattern) -> _BlockJacobiSymbolic:
        """Resolve the block partition against a sparsity pattern.

        Everything derivable from the pattern alone happens here: the partition, the
        scatter map from stored entries to block slots, whether any block is
        structurally singular, and which factorization that implies.
        """
        pattern = (
            sparsity
            if isinstance(sparsity, CooPattern)
            else as_coo_pattern(sparsity, "`BlockJacobi.symbolic`")
        )
        concrete = pattern.concrete("`BlockJacobi`")
        if concrete.shape[0] != concrete.shape[1]:
            raise ValueError(
                f"`BlockJacobi` requires a square pattern; got shape {concrete.shape}."
            )
        partition = self._resolve_partition(pattern, concrete)
        singular = structurally_singular_blocks(concrete, partition)
        factorization = self._resolve_factorization(singular)
        scatter = _scatter_indices(partition.sizes, concrete)
        return _BlockJacobiSymbolic(
            partition=partition,
            scatter=jnp.asarray(scatter),
            factorization=factorization,
            rcond=self.rcond,
            tags=self.tags,
            nse=concrete.nse,
        )

    def _resolve_partition(
        self, pattern: CooPattern, concrete: ConcreteCooPattern
    ) -> BlockPartition:
        """Fulfil the `blocks` parameter, from a value or from an injected provider."""
        size = concrete.shape[0]
        match self.blocks:
            case BlockPartition() as partition:
                if partition.size != size:
                    raise ValueError(
                        f"`blocks` partitions {partition.size} indices, but the system "
                        f"has size {size}."
                    )
                return partition
            case int() as block_size:
                return BlockPartition.uniform(size, block_size)
            case BlockPartitioner() as partitioner:
                return partitioner.partition(pattern)
            case sizes if isinstance(sizes, Sequence):
                return BlockPartition.from_sizes(sizes, size)
            case other:
                raise TypeError(
                    "`blocks` must be a `BlockPartition`, an `int` block size, a "
                    "sequence of block sizes, or an object satisfying the "
                    f"`BlockPartitioner` protocol; got {type(other).__name__}."
                )

    def _resolve_factorization(self, singular: np.ndarray) -> _Factorization:
        """Turn `'auto'` into a concrete choice, and refuse a provably wrong one."""
        any_singular = bool(np.any(singular))
        if any_singular and self.assume_nonsingular:
            offenders = np.flatnonzero(singular)[:8].tolist()
            raise ValueError(
                "`assume_nonsingular=True`, but blocks "
                f"{offenders} are singular whatever the values are: once the "
                "out-of-block coupling is discarded they have an all-zero row or "
                "column."
            )
        if self.factorization == "lu" and any_singular:
            offenders = np.flatnonzero(singular)[:8].tolist()
            raise ValueError(
                f"`factorization='lu'` cannot invert blocks {offenders}, which are "
                "structurally singular: once the out-of-block coupling is discarded "
                "they have an all-zero row or column. Use 'svd' (the default), which "
                "takes a pseudo-inverse instead, or choose a partition that keeps "
                "those rows and columns coupled."
            )
        if self.factorization == "auto":
            return "lu" if self.assume_nonsingular else "svd"
        return self.factorization


BlockJacobi.__init__.__doc__ = """**Arguments:**

- `blocks`: the block partition. Either a value -- a `BlockPartition`, an `int` giving
    a uniform block size, or a sequence of block sizes summing to the system size -- or
    an object satisfying the `BlockPartitioner` protocol, which derives one from the
    sparsity pattern. Required.
- `factorization`: how to invert each block.

    | value | meaning |
    | --- | --- |
    | `"svd"` | pseudo-inverse. Rank-revealing, and finite even for a singular block. The default. |
    | `"lu"` | plain inverse. Cheapest, but yields `inf`/`nan` for a singular block, which no error will announce. Refused up front if a block is *structurally* singular. |
    | `"qr"` | rank-revealing QR with column pivoting: deficient directions are truncated, giving a one-sided inverse on the kept subspace rather than the pseudo-inverse. CPU and GPU only -- JAX does not implement pivoting on TPU, where this raises. |
    | `"auto"` | `"lu"` if `assume_nonsingular`, otherwise `"svd"`. |

- `assume_nonsingular`: assert that every block is invertible, which lets `"auto"`
    choose the cheap route. Checked against the pattern: if a block is structurally
    singular the assertion is provably false and this raises. Defaults to `False`.
- `rcond`: relative tolerance below which a block's singular values (for `"svd"`) or
    pivots (for `"qr"`) are treated as zero. `None` uses the backend default.
- `tags`: any additional lineax tags to attach to the resulting preconditioner.
    Positive-definiteness is inferred from the operator and need not be given.
"""


def coverage(sparsity: _Sparsity | CooPattern, partition: BlockPartition) -> float:
    """The fraction of a pattern's entries that a partition captures.

    Everything else is discarded when the preconditioner is built. `1.0` means the
    matrix is exactly block diagonal for this partition, so the preconditioner is an
    exact inverse.
    """
    pattern = (
        sparsity
        if isinstance(sparsity, CooPattern)
        else as_coo_pattern(sparsity, "`coverage`")
    )
    return captured_fraction(pattern.concrete("`coverage`"), partition)
