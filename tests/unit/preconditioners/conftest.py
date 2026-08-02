"""Reference matrices shared by the preconditioner test suites."""

import numpy as np


def block_diagonal_matrix(
    block_sizes: tuple[int, ...] = (4, 4, 4), seed: int = 0
) -> np.ndarray:
    """An exactly block-diagonal matrix with well-conditioned blocks.

    The preconditioner is an *exact* inverse for this one, which is what makes it the
    right reference: any discrepancy is a bug rather than the approximation working as
    designed.
    """
    rng = np.random.default_rng(seed)
    size = sum(block_sizes)
    dense = np.zeros((size, size))
    start = 0
    for block_size in block_sizes:
        block = rng.uniform(-1.0, 1.0, (block_size, block_size))
        block += (block_size + 2) * np.eye(block_size)
        dense[start : start + block_size, start : start + block_size] = block
        start += block_size
    return dense


def with_coupling(dense: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Add entries outside the diagonal blocks, which block Jacobi discards."""
    coupled = dense.copy()
    size = coupled.shape[0]
    for i in range(size - 4):
        coupled[i, i + 4] += strength
        coupled[i + 4, i] += strength
    return coupled
