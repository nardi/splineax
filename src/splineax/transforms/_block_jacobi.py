"""Block Jacobi, a conditioning transform (`Preconditioner`).

Cuts the (already permuted) matrix into fixed-size contiguous diagonal blocks and
inverts each one, batched in a single `jnp.linalg.inv` call. The preconditioner
operator is one reshape, one batched matrix-vector product (`einsum`), and a reshape
back: no Python-level loop over blocks, so it is one fused kernel on GPU.
"""

from contextlib import AbstractContextManager, contextmanager
from typing import Iterator, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Array, Inexact, Integer
from lineax import AbstractLinearOperator, FunctionLinearOperator
from lineax._tags import positive_semidefinite_tag

from splineax.operators._pattern import MatrixSparsity
from splineax.transforms._clustering import _default_block_size


def _block_mv(
    inverse_blocks: Inexact[Array, "nblocks block_size block_size"],
    n: int,
    block_size: int,
    vector: Inexact[Array, " n"],
) -> Inexact[Array, " n"]:
    nblocks = inverse_blocks.shape[0]
    padded = jnp.zeros(nblocks * block_size, dtype=vector.dtype).at[:n].set(vector)
    result = jnp.einsum(
        "kij,kj->ki", inverse_blocks, padded.reshape(nblocks, block_size)
    )
    return result.reshape(-1)[:n]


class _BlockJacobiPlan(NamedTuple):
    # Flat destination per nonzero into a raveled (nblocks, block_size, block_size)
    # array; entries outside a diagonal block point one past the end, dropped by
    # `.at[destination].add(..., mode="drop")`, the same out-of-range-segment trick
    # `operators/_operations.py`'s `_bcoo_band` uses for `segment_sum`.
    destination: Integer[Array, " nse"]
    n: int
    block_size: int
    nblocks: int
    regularization: float

    def analyze_numeric(
        self, matrix: BCOO, tags: frozenset[object]
    ) -> AbstractContextManager[AbstractLinearOperator]:
        return _block_jacobi_operator(
            matrix,
            self.destination,
            self.n,
            self.block_size,
            self.nblocks,
            self.regularization,
            tags,
        )


@contextmanager
def _block_jacobi_operator(
    matrix: BCOO,
    destination: Integer[Array, " nse"],
    n: int,
    block_size: int,
    nblocks: int,
    regularization: float,
    tags: frozenset[object],
) -> Iterator[AbstractLinearOperator]:
    total = nblocks * block_size * block_size
    blocks = (
        jnp.zeros(total, dtype=matrix.dtype)
        .at[destination]
        .add(matrix.data, mode="drop")
        .reshape(nblocks, block_size, block_size)
    )

    # Every diagonal slot gets `regularization` added to its own (real) value. A slot
    # past the true matrix size (only possible in the last block, when `n` isn't a
    # multiple of `block_size`) never received any scattered value, so it is exactly
    # 0 here; setting it to exactly 1 turns that padding into an identity row rather
    # than an all-zero (singular) one.
    global_index = jnp.arange(nblocks)[:, None] * block_size + jnp.arange(block_size)
    is_padding = global_index >= n
    diagonal_add = jnp.where(is_padding, 1.0, regularization)
    eye = jnp.eye(block_size, dtype=blocks.dtype)
    blocks = blocks + diagonal_add[:, :, None] * eye

    inverse_blocks = jnp.linalg.inv(blocks)
    finite = jnp.all(jnp.isfinite(inverse_blocks), axis=(1, 2))
    inverse_blocks = jnp.where(finite[:, None, None], inverse_blocks, eye)
    # A numeric analysis is never meant to be differentiated through: the
    # preconditioner only changes how fast a Krylov iteration converges, never what it
    # converges to, and lineax already treats a solver's state as a constant when
    # differentiating a solve built from it. Stopping the gradient here, rather than
    # relying on that alone, avoids computing (and discarding) the backward pass
    # through `jnp.linalg.inv` and the scatter above whenever this runs inside an
    # active `jax.grad`.
    inverse_blocks = jax.lax.stop_gradient(inverse_blocks)

    partial_mv = eqx.Partial(_block_mv, inverse_blocks, n, block_size)
    structure = jax.ShapeDtypeStruct((n,), matrix.dtype)
    operator_tags = (
        (positive_semidefinite_tag,) if positive_semidefinite_tag in tags else ()
    )
    yield FunctionLinearOperator(
        partial_mv, structure, tags=operator_tags, closure_convert=False
    )


class BlockJacobi(eqx.Module):
    """Conditioning transform that inverts fixed-size contiguous diagonal blocks.

    Meant to run after a symbolic transform that gathers coupled entries onto the
    diagonal (e.g. `AggregationClustering`), so a fixed-size cut actually captures
    real structure. Entries outside a diagonal block are dropped: this is a block
    Jacobi preconditioner, not an incomplete factorization.
    """

    block_size: int | None = eqx.field(static=True, default=None)
    regularization: float = eqx.field(static=True, default=0.0)

    def analyze_symbolic(self, pattern: MatrixSparsity) -> _BlockJacobiPlan:
        n, _ = pattern.shape
        block_size = (
            self.block_size
            if self.block_size is not None
            else _default_block_size(pattern.rows.shape[0], n)
        )
        nblocks = -(-n // block_size)  # ceil division

        block_row = pattern.rows // block_size
        block_col = pattern.cols // block_size
        same_block = block_row == block_col
        local_row = pattern.rows % block_size
        local_col = pattern.cols % block_size
        flat = block_row * block_size * block_size + local_row * block_size + local_col
        out_of_range = nblocks * block_size * block_size
        destination = jnp.where(same_block, flat, out_of_range)

        return _BlockJacobiPlan(
            destination, n, block_size, nblocks, self.regularization
        )


BlockJacobi.__init__.__doc__ = """**Arguments:**

- `block_size`: size of each diagonal block. If `None`, a size is derived from the
    pattern the same way `AggregationClustering` does, so the two agree by
    construction when neither is given explicitly.
- `regularization`: added to every diagonal entry before inverting each block, for
    numerical stability. Defaults to `0.0`, using the matrix's own diagonal as-is.
"""
