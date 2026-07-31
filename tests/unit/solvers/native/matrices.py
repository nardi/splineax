"""Matrix families for the convergence suite.

Each generator returns a dense reference matrix, so that `numpy.linalg.solve` can supply the
answer to compare against. They are dense only as test fixtures: the solver receives them as
sparse operators.

The families are chosen to span the ways a block preconditioner can be defeated. Each breaks a
different assumption: that the matrix is banded, that it is symmetric, that it is real, that its
diagonal blocks are invertible, that its entries are comparably scaled, or that its nonzeros can
be brought near the diagonal at all. Only two of the nine actually defeat it.

`DIFFICULTY` records which is which, because the two groups deserve different assertions. For
the tractable ones a small residual is required. For the hard ones it is not, since a
preconditioner that captures little of a hard matrix legitimately fails to converge. What is
required of those is that the failure is reported rather than disguised.

The labels are set from measurement rather than from expectation, and two of them are not what
they look like. `arrow` cannot be helped by any reordering, since a dense row spans the matrix
whatever the numbering, and its capture fraction sits around a third; it nonetheless converges
to machine precision, because its diagonal dominates. `badly_scaled` is the opposite: its
capture fraction is above nine tenths and it still fails, because capture counts nonzeros
without weighing them, so a well-covered matrix can be badly served. Between them they are the
reason capture should not be read as a quality score.
"""

from typing import Callable, Literal

import numpy as np

Family = Callable[[int], np.ndarray]


def one_dimensional_laplacian(size: int) -> np.ndarray:
    """The second difference operator. Tridiagonal, symmetric positive definite.

    Already minimally banded, so it is the case where reordering has nothing to do and the
    preconditioner should work almost perfectly.
    """
    return np.eye(size) * 2.0 - np.eye(size, k=1) - np.eye(size, k=-1)


def grid_laplacian(size: int) -> np.ndarray:
    """The five-point Laplacian on the largest square grid that fits `size`.

    Its natural bandwidth is one grid line wide, so unlike the tridiagonal case above there is
    real work for the reordering, and its condition number grows with the grid.
    """
    side = max(2, int(np.sqrt(size)))
    identity = np.eye(side)
    line = np.eye(side) * 4.0 - np.eye(side, k=1) - np.eye(side, k=-1)
    coupling = np.eye(side, k=1) + np.eye(side, k=-1)
    return np.kron(identity, line) - np.kron(coupling, identity)


def diagonally_dominant(size: int, dominance: float = 4.0, seed: int = 0) -> np.ndarray:
    """A scattered symmetric pattern with a tunable diagonal.

    Its nonzeros are spread across the whole matrix rather than banded, so the reordering
    cannot narrow it much and the preconditioner has to earn its keep from the diagonal.
    Lowering `dominance` walks it towards singularity.
    """
    generator = np.random.default_rng(seed)
    density = min(0.4, 8.0 / size)
    mask = generator.random((size, size)) < density
    entries = mask * generator.normal(size=(size, size))
    symmetric = entries + entries.T
    return symmetric + np.eye(size) * dominance * np.abs(symmetric).sum(axis=1).max()


def convection_diffusion(size: int, peclet: float = 8.0) -> np.ndarray:
    """An upwind-differenced convection-diffusion operator. Non-symmetric.

    `peclet` is the mesh Peclet number, setting how far convection dominates diffusion.
    Raising it makes the matrix progressively less symmetric and the iteration progressively
    harder, which is what this family is for.

    Written in the form scaled by the square of the mesh spacing, so its entries stay of order
    one. Without that the entries grow with the size, and a solver's absolute tolerance then
    becomes unreachable for reasons that have nothing to do with the solver.
    """
    return (
        np.eye(size) * (2.0 + peclet)
        + np.eye(size, k=-1) * (-1.0 - peclet)
        + np.eye(size, k=1) * -1.0
    )


def shifted_laplacian(size: int, shift: float = 0.5) -> np.ndarray:
    """A complex, indefinite shift of the grid Laplacian, as in a Helmholtz problem.

    Exercises complex arithmetic through every stage, and the shift moves eigenvalues off the
    real axis so the spectrum no longer clusters where an unshifted Laplacian's does.
    """
    real = grid_laplacian(size)
    return real.astype(np.complex128) - 1j * shift * np.eye(real.shape[0])


def saddle_point(size: int, seed: int = 0) -> np.ndarray:
    """A `[[0, B], [B^T, -C]]` block system, so half the diagonal is structurally zero.

    Indefinite, and the reason the block inverses have to be rank-revealing: a block sitting
    over the zero half has no inverse at all. Mirrors `_zero_diagonal_saddle_point` in the
    parent conftest, generated here so it can be produced at any size.

    `B` here is random and roughly as dense as the pattern allows, which makes the Schur
    complement `B F^{-1} B^T` a globally coupled matrix with no locality for any block method,
    matching-informed grouping included, to capture. That is a property of this matrix, not a
    limitation the grouping happens to have: see `divergence_saddle_point` below for a saddle
    point with the local structure real ones have, which the grouping is measured against.
    """
    half = max(2, size // 2)
    generator = np.random.default_rng(seed)
    block = generator.normal(size=(half, half)) * (
        generator.random((half, half)) < min(0.5, 6.0 / half)
    )
    # Keep the off-diagonal block nonsingular, or the whole system is singular.
    block = block + np.eye(half) * 2.0
    lower_right = np.diag(generator.uniform(0.5, 1.5, size=half))
    return np.block([[np.zeros((half, half)), block], [block.T, -lower_right]])


def divergence_saddle_point(size: int) -> np.ndarray:
    """An `[[F, B^T], [B, 0]]` block system shaped like a real saddle point rather than an
    adversarial one: `F` a tridiagonal Laplacian, `B` a rectangular discrete divergence
    coupling each constraint to three neighbouring ordinary unknowns.

    Unlike `saddle_point` above, whose random `B` has no locality, a constraint here couples
    only to unknowns near it, which is what lets a block method capture it at all. This is the
    family the matching-informed grouping in `_block_jacobi.py` is for: pairing each constraint
    with a matched ordinary unknown makes the block holding both of them invertible, in place
    of the zero block a plain reordering would otherwise leave there.
    """
    ordinary = max(3, (2 * size) // 3)
    constraints = max(1, ordinary // 3)
    stiffness = np.eye(ordinary) * 2.0 - np.eye(ordinary, k=1) - np.eye(ordinary, k=-1)
    divergence = np.zeros((constraints, ordinary))
    for row in range(constraints):
        divergence[row, 3 * row] = -1.0
        divergence[row, 3 * row + 1] = 1.0
        divergence[row, min(3 * row + 2, ordinary - 1)] += 0.5
    return np.block(
        [
            [stiffness, divergence.T],
            [divergence, np.zeros((constraints, constraints))],
        ]
    )


def badly_scaled(size: int, decades: float = 4.0, seed: int = 1) -> np.ndarray:
    """The grid Laplacian with its rows scaled over many orders of magnitude.

    Row scaling leaves the solution unchanged but wrecks any judgement made by comparing
    entries, which is what the capture heuristic does when it counts nonzeros without weighing
    them. This is the family that shows what that costs.
    """
    matrix = grid_laplacian(size)
    generator = np.random.default_rng(seed)
    scale = 10.0 ** generator.uniform(-decades, decades, size=matrix.shape[0])
    return matrix * scale[:, None]


def arrow(size: int, seed: int = 2) -> np.ndarray:
    """A diagonal matrix with a few dense rows and columns.

    No permutation can narrow this: a dense row spans the whole matrix whatever the numbering.
    It is the family where the reordering has nothing to offer and the capture fraction stays
    low however the blocks are chosen, so it bounds what the method can do.
    """
    generator = np.random.default_rng(seed)
    matrix = np.eye(size) * size
    for index in (0, 1, size - 2, size - 1):
        matrix[index, :] += generator.normal(size=size)
        matrix[:, index] += generator.normal(size=size)
    return matrix


Difficulty = Literal["tractable", "hard"]

FAMILIES: dict[str, Family] = {
    "1d_laplacian": one_dimensional_laplacian,
    "grid_laplacian": grid_laplacian,
    "diagonally_dominant": diagonally_dominant,
    "convection_diffusion": convection_diffusion,
    "shifted_laplacian": shifted_laplacian,
    "saddle_point": saddle_point,
    "divergence_saddle_point": divergence_saddle_point,
    "badly_scaled": badly_scaled,
    "arrow": arrow,
}

DIFFICULTY: dict[str, Difficulty] = {
    "1d_laplacian": "tractable",
    "grid_laplacian": "tractable",
    "diagonally_dominant": "tractable",
    "convection_diffusion": "tractable",
    "shifted_laplacian": "tractable",
    "saddle_point": "hard",
    "divergence_saddle_point": "tractable",
    "badly_scaled": "hard",
    "arrow": "tractable",
}

# Complex families need a complex right-hand side to exercise the complex path properly.
COMPLEX_FAMILIES = frozenset({"shifted_laplacian"})
