"""Numerical convergence and stability of `BlockJacobiGMRES` across matrix families.

The other suites check that the pieces compute what they claim. This one asks whether the
solver is usable, by running it over the families in [matrices.py](matrices.py) and over the
settings a caller might reasonably choose.

Three properties are checked, and the distinction between them is the point of the file.

*Consistency* is required of every family without exception: if the solve reports success, the
unpreconditioned residual must actually be small. GMRES measures convergence on the
preconditioned residual, so this is not automatic, and it is the property a caller depends on
most since there is no other way to tell a good answer from a bad one.

*Finiteness* is likewise required everywhere. A singular block must never turn into infinities.

*A small residual* is required only of the tractable families. Demanding it of the hard ones
would be asserting that a weak preconditioner works, which it does not, and pinning a failure
threshold would only record today's behaviour.

Note that the tractable families are judged on the relative residual rather than on the success
flag. Lineax's criterion is a max norm scaled by the right-hand side, so whether it is met
depends on the absolute scale of the matrix, which varies across these families by orders of
magnitude. The relative residual is the property that means the same thing everywhere.
"""

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from splineax import BCOOLinearOperator, BlockInverse, BlockJacobiGMRES, Ordering

from .matrices import COMPLEX_FAMILIES, DIFFICULTY, FAMILIES, convection_diffusion

# Tolerance to ask the solver for, and the relative residual a tractable family must reach.
# The residual bound is looser than the request because the request is a scaled max norm, and
# because single precision on an ill-conditioned family cannot do much better.
TOLERANCE = {np.float64: 1e-8, np.float32: 1e-4}
RESIDUAL_BOUND = {np.float64: 1e-8, np.float32: 1e-3}
# What counts as small enough to back a reported success, per precision.
CONSISTENCY_BOUND = {np.float64: 1e-6, np.float32: 1e-2}

PRECISIONS = [np.float64, np.float32]
# Two families broad enough to be worth sweeping settings over: one banded and symmetric, one
# scattered and not.
REPRESENTATIVE = ["grid_laplacian", "diagonally_dominant"]


def _dense(family: str, size: int, dtype: type) -> np.ndarray:
    """The family's reference matrix at the requested precision."""
    matrix = FAMILIES[family](size)
    if family in COMPLEX_FAMILIES:
        complex_dtype = np.complex128 if dtype is np.float64 else np.complex64
        return matrix.astype(complex_dtype)
    return matrix.astype(dtype)


def _right_hand_side(matrix: np.ndarray, seed: int = 3) -> jnp.ndarray:
    generator = np.random.default_rng(seed)
    size = matrix.shape[0]
    vector = generator.normal(size=size)
    if np.iscomplexobj(matrix):
        vector = vector + 1j * generator.normal(size=size)
    return jnp.asarray(vector.astype(matrix.dtype))


def _solve(
    matrix: np.ndarray, vector: jnp.ndarray, dtype: type, **settings: object
) -> tuple[bool, float, bool]:
    """Solve and report `(converged, relative_residual, finite)`."""
    tolerance = TOLERANCE[dtype]
    solver = BlockJacobiGMRES(rtol=tolerance, atol=tolerance, **settings)  # type: ignore[arg-type]
    operator = BCOOLinearOperator(BCOO.fromdense(jnp.asarray(matrix)))
    solution = lx.linear_solve(operator, vector, solver=solver, throw=False)

    value = np.asarray(solution.value)
    finite = bool(np.all(np.isfinite(value)))
    if not finite:
        return False, float("inf"), False
    residual = matrix @ value - np.asarray(vector)
    relative = float(np.linalg.norm(residual) / np.linalg.norm(np.asarray(vector)))
    return solution.result == lx.RESULTS.successful, relative, finite


def _check(
    family: str, dtype: type, converged: bool, relative: float, finite: bool
) -> None:
    """Apply the three properties described in the module docstring."""
    assert finite, f"{family} produced a non-finite solution"
    if converged:
        assert relative < CONSISTENCY_BOUND[dtype], (
            f"{family} reported success at relative residual {relative:.2e}"
        )
    if DIFFICULTY[family] == "tractable":
        assert relative < RESIDUAL_BOUND[dtype], (
            f"{family} reached only relative residual {relative:.2e}"
        )


@pytest.mark.parametrize("dtype", PRECISIONS, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("family", list(FAMILIES), ids=list(FAMILIES))
def test_every_family(family: str, dtype: type, enable_x64: None) -> None:
    """Run every family at both precisions with the solver's default settings.

    This is the breadth check. Single precision is included because it halves the exponent
    range available to the block inverses, and a preconditioner that only works in double
    precision would be a poor default.
    """
    matrix = _dense(family, 512, dtype)
    _check(family, dtype, *_solve(matrix, _right_hand_side(matrix), dtype))


@pytest.mark.parametrize("block_inverse", list(BlockInverse), ids=lambda m: m.name)
@pytest.mark.parametrize("ordering", list(Ordering), ids=lambda o: o.name)
@pytest.mark.parametrize("family", REPRESENTATIVE)
def test_every_ordering_and_inversion_route(
    family: str, ordering: Ordering, block_inverse: BlockInverse, enable_x64: None
) -> None:
    """No combination of ordering and inversion route may diverge where the default converges.

    The two are swept together because they interact: the ordering decides what the blocks
    contain, and the inversion route decides how faithfully those blocks are inverted.
    """
    matrix = _dense(family, 512, np.float64)
    _check(
        family,
        np.float64,
        *_solve(
            matrix,
            _right_hand_side(matrix),
            np.float64,
            ordering=ordering,
            block_inverse=block_inverse,
        ),
    )


@pytest.mark.parametrize("size", [64, 512, 2048])
@pytest.mark.parametrize("family", ["1d_laplacian", "grid_laplacian"])
def test_scales_with_size(family: str, size: int, enable_x64: None) -> None:
    """Convergence must hold as the system grows.

    The block size is capped, so a larger matrix means proportionally more blocks and a weaker
    preconditioner. Sweeping the size is what catches a method that only works when the blocks
    cover most of the matrix.
    """
    matrix = _dense(family, size, np.float64)
    _check(family, np.float64, *_solve(matrix, _right_hand_side(matrix), np.float64))


@pytest.mark.parametrize("peclet", [1.0, 4.0, 8.0, 32.0, 128.0])
def test_survives_growing_asymmetry(peclet: float, enable_x64: None) -> None:
    """Sweep a convection-diffusion operator from nearly symmetric to strongly convective.

    GMRES needs no symmetry, and neither does the block preconditioner, so a small residual is
    required across the whole sweep rather than only at the symmetric end.
    """
    matrix = convection_diffusion(512, peclet=peclet)
    converged, relative, finite = _solve(matrix, _right_hand_side(matrix), np.float64)
    assert finite
    if converged:
        assert relative < CONSISTENCY_BOUND[np.float64]
    assert relative < RESIDUAL_BOUND[np.float64]


@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
def test_stable_across_overlap(overlap: float, enable_x64: None) -> None:
    """Every overlap fraction must still converge.

    Iteration counts are deliberately not asserted to fall as the overlap rises. More overlap
    captures more of the matrix and so should help, but the relationship is not monotone in
    practice and pinning it would make this suite fail for reasons that are not defects.
    """
    matrix = _dense("grid_laplacian", 512, np.float64)
    _check(
        "grid_laplacian",
        np.float64,
        *_solve(
            matrix,
            _right_hand_side(matrix),
            np.float64,
            overlap_fraction=overlap,
        ),
    )


@pytest.mark.parametrize("capture_target", [0.5, 0.8, 0.95])
def test_stable_across_capture_target(capture_target: float, enable_x64: None) -> None:
    """Asking the block size heuristic for more or less coverage must not break the solve."""
    matrix = _dense("grid_laplacian", 512, np.float64)
    _check(
        "grid_laplacian",
        np.float64,
        *_solve(
            matrix,
            _right_hand_side(matrix),
            np.float64,
            capture_target=capture_target,
        ),
    )


def test_low_capture_does_not_imply_poor_convergence(enable_x64: None) -> None:
    """An arrow matrix cannot be narrowed by any reordering, so its capture fraction stays low,
    and it converges to machine precision anyway.

    Worth pinning because it is the counterexample to reading the capture fraction as a quality
    score. What the preconditioner misses matters only in proportion to how much the iteration
    needed it.
    """
    matrix = _dense("arrow", 512, np.float64)
    solver = BlockJacobiGMRES()
    operator = BCOOLinearOperator(BCOO.fromdense(jnp.asarray(matrix)))
    captured = solver.init(operator, {}).analysis.captured

    _, relative, _ = _solve(matrix, _right_hand_side(matrix), np.float64)
    assert captured < 0.5
    assert relative < 1e-12


def test_high_capture_does_not_guarantee_convergence(enable_x64: None) -> None:
    """The other half of the same point. A badly row-scaled matrix is well covered by its
    blocks and still fails to converge, because coverage is counted without weighing entries.

    Only finiteness and consistency are asserted, since the failure is the expected outcome.
    """
    matrix = _dense("badly_scaled", 512, np.float64)
    solver = BlockJacobiGMRES()
    operator = BCOOLinearOperator(BCOO.fromdense(jnp.asarray(matrix)))
    captured = solver.init(operator, {}).analysis.captured

    converged, relative, finite = _solve(matrix, _right_hand_side(matrix), np.float64)
    assert captured > 0.8
    assert finite
    if converged:
        assert relative < CONSISTENCY_BOUND[np.float64]
