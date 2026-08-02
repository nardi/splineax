"""Tests for `PreconditionedIterativeLinearSolver`.

The suite that matters most is the pair of convergence tests: an assertion that the
answer is right only shows the solve worked, not that the preconditioner reached lineax
at all. Comparing iteration counts against the same unpreconditioned solve is the only
check that actually proves it --- and it is what would catch a regression in the
`linearise` registration, which would otherwise densify the preconditioner silently and
still give the right answer.
"""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import splineax as splx
from splineax._pattern import as_coo_pattern
from splineax.preconditioners._transform import SymmetricPermutation


def scaled_block_matrix(num_blocks: int = 8, block_size: int = 4) -> np.ndarray:
    """Diagonal blocks spanning many orders of magnitude, weakly coupled.

    Unpreconditioned Krylov methods struggle badly on this --- the scaling alone gives
    it a condition number around 1e7 --- while block Jacobi removes the scaling
    entirely. That gap is what makes the iteration-count comparison meaningful.
    """
    rng = np.random.default_rng(3)
    size = num_blocks * block_size
    dense = np.zeros((size, size))
    for block in range(num_blocks):
        scale = 10.0 ** (block - num_blocks // 2)
        dense[
            block_size * block : block_size * (block + 1),
            block_size * block : block_size * (block + 1),
        ] = scale * (
            rng.uniform(-1.0, 1.0, (block_size, block_size)) + 3.0 * np.eye(block_size)
        )
    for i in range(size - block_size):
        dense[i, i + block_size] += 1e-3 * dense[i, i]
    return dense


def operator_of(dense: np.ndarray, tags: object = ()) -> splx.BCOOLinearOperator:
    return splx.BCOOLinearOperator(BCOO.fromdense(jnp.asarray(dense)), tags)


def gmres() -> lx.GMRES:
    """The settings every solve below shares, tight enough to need preconditioning."""
    return lx.GMRES(rtol=1e-8, atol=1e-12, max_steps=2000, restart=8)


@pytest.fixture
def x64():
    """Iterative tolerances this tight are only meaningful in double precision."""
    with jax.enable_x64(True):
        yield


def test_preconditioning_converges_where_the_bare_solver_does_not(x64: None):
    """The headline: the same GMRES, with and without the preconditioner.

    Unpreconditioned it exhausts its step budget; preconditioned it converges in a
    handful. Nothing but the preconditioner actually reaching lineax explains that.
    """
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    reference = jnp.linalg.solve(jnp.asarray(dense), b)
    bare = lx.linear_solve(operator, b, solver=gmres(), throw=False)
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(), splx.BlockJacobi(blocks=4), sparsity=operator
    )
    preconditioned = lx.linear_solve(operator, b, solver=solver)

    assert preconditioned.result == lx.RESULTS.successful
    assert np.allclose(np.asarray(preconditioned.value), np.asarray(reference))
    assert int(preconditioned.stats["num_steps"]) < int(bare.stats["num_steps"])


def test_bicgstab_solves(x64: None):
    """The right-preconditioned method works too."""
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    solver = splx.PreconditionedIterativeLinearSolver(
        lx.BiCGStab(rtol=1e-8, atol=1e-12, max_steps=2000),
        splx.BlockJacobi(blocks=4),
        sparsity=operator,
    )
    solution = lx.linear_solve(operator, b, solver=solver)
    assert np.allclose(
        np.asarray(solution.value), np.asarray(jnp.linalg.solve(jnp.asarray(dense), b))
    )


def test_cg_solves_a_definite_system(x64: None):
    """`CG` needs both sides and a definite preconditioner; it gets both."""
    dense = scaled_block_matrix()
    spd = dense @ dense.T + np.eye(dense.shape[0])
    operator = operator_of(spd, lx.positive_semidefinite_tag)
    b = jnp.ones(spd.shape[0])
    solver = splx.PreconditionedIterativeLinearSolver(
        lx.CG(rtol=1e-8, atol=1e-12, max_steps=2000),
        splx.BlockJacobi(blocks=4),
        sparsity=operator,
    )
    solution = lx.linear_solve(operator, b, solver=solver)
    assert np.allclose(
        np.asarray(solution.value),
        np.asarray(jnp.linalg.solve(jnp.asarray(spd), b)),
        atol=1e-6,
    )


class _LeftOnly(splx.BlockJacobi):
    """A preconditioner offering only left application, for the sidedness checks."""

    @property
    def sides(self) -> frozenset[splx.Side]:
        return frozenset({splx.Side.LEFT})


@pytest.mark.parametrize("solver_class", [lx.BiCGStab, lx.CG], ids=["bicgstab", "cg"])
def test_missing_side_is_caught_at_construction(solver_class: type):
    """A preconditioner that cannot supply the needed side fails immediately.

    At construction rather than at the first solve, which is the whole reason the spec
    tier declares its sides.
    """
    with pytest.raises(ValueError, match="does not support: RIGHT"):
        splx.PreconditionedIterativeLinearSolver(
            solver_class(rtol=1e-6, atol=1e-6), _LeftOnly(blocks=4)
        )


def test_gmres_accepts_a_left_only_preconditioner():
    """The mirror of the previous test: `GMRES` only needs the left side."""
    solver = splx.PreconditionedIterativeLinearSolver(
        lx.GMRES(rtol=1e-6, atol=1e-6), _LeftOnly(blocks=4)
    )
    assert solver.preconditioner.sides == frozenset({splx.Side.LEFT})


def test_direct_solvers_are_refused():
    """There is nothing to precondition in a direct solve."""
    with pytest.raises(TypeError, match="supports `lineax.CG`"):
        splx.PreconditionedIterativeLinearSolver(
            lx.LU(),  # ty: ignore[invalid-argument-type]
            splx.BlockJacobi(blocks=4),
        )


def test_a_user_supplied_preconditioner_is_refused(x64: None):
    """Silently overwriting the caller's option would be worse than refusing it."""
    operator = operator_of(scaled_block_matrix())
    solver = splx.PreconditionedIterativeLinearSolver(
        lx.GMRES(rtol=1e-6, atol=1e-6), splx.BlockJacobi(blocks=4), sparsity=operator
    )
    with pytest.raises(ValueError, match="builds the preconditioner itself"):
        solver.init(operator, {"preconditioner": operator})


def test_non_square_operators_are_refused():
    """The square-only contract, as elsewhere in this package."""
    operator = operator_of(np.ones((2, 3)))
    solver = splx.PreconditionedIterativeLinearSolver(
        lx.GMRES(rtol=1e-6, atol=1e-6), splx.BlockJacobi(blocks=1)
    )
    with pytest.raises(ValueError, match="square matrices"):
        solver.init(operator, {})


def test_a_traced_pattern_raises_with_the_three_routes_named(x64: None):
    """Without `sparsity`, `lineax.linear_solve` cannot analyze, and says what to do."""
    operator = operator_of(scaled_block_matrix())
    solver = splx.PreconditionedIterativeLinearSolver(
        lx.GMRES(rtol=1e-6, atol=1e-6), splx.BlockJacobi(blocks=4)
    )
    with pytest.raises(ValueError, match="Build the solver with `sparsity=`"):
        lx.linear_solve(operator, jnp.ones(operator.in_size()), solver=solver)


def test_init_outside_the_trace_needs_no_sparsity(x64: None):
    """Calling `init` yourself analyzes on concrete indices, so it just works."""
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=4),
    )
    state = solver.init(operator, {})
    solution = lx.linear_solve(operator, b, solver=solver, state=state)
    assert np.allclose(
        np.asarray(solution.value), np.asarray(jnp.linalg.solve(jnp.asarray(dense), b))
    )


def test_y0_is_honoured(x64: None):
    """`y0` is given in untransformed coordinates and moved across for us."""
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    reference = jnp.linalg.solve(jnp.asarray(dense), b)
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=4),
        sparsity=operator,
    )
    solution = lx.linear_solve(operator, b, solver=solver, options={"y0": reference})
    assert np.allclose(np.asarray(solution.value), np.asarray(reference))


def test_transposed_solve(x64: None):
    """Transposing delegates, and the transform transposes with it."""
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=4),
        sparsity=operator,
    )
    state = solver.init(operator, {})
    transposed_state, options = solver.transpose(state, {})
    value, _, _ = solver.compute(transposed_state, b, options)
    assert np.allclose(
        np.asarray(value), np.asarray(jnp.linalg.solve(jnp.asarray(dense).T, b))
    )


def test_complex_systems_solve(x64: None):
    """The complex path works end to end."""
    dense = scaled_block_matrix(num_blocks=4) * (1.0 + 0.2j)
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0], dtype=jnp.complex128)
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=4),
        sparsity=operator,
    )
    solution = lx.linear_solve(operator, b, solver=solver)
    assert np.allclose(
        np.asarray(solution.value), np.asarray(jnp.linalg.solve(jnp.asarray(dense), b))
    )


def test_solves_under_jit_and_grad(x64: None):
    """The solver is a pytree and the analysis is static, so both transforms work."""
    dense = jnp.asarray(scaled_block_matrix(num_blocks=4))
    pattern_operator = operator_of(np.asarray(dense))
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=4),
        sparsity=pattern_operator,
    )
    b = jnp.ones(dense.shape[0])

    # Indices outside the trace, values inside: the usual sparse-in-jit arrangement,
    # and the one the pre-analysed `sparsity` is designed for.
    matrix = BCOO.fromdense(dense)
    indices = matrix.indices

    @jax.jit
    def solve(values: jax.Array) -> jax.Array:
        operator = splx.BCOOLinearOperator(BCOO((values, indices), shape=dense.shape))
        return lx.linear_solve(operator, b, solver=solver).value

    values = matrix.data
    assert np.allclose(
        np.asarray(solve(values)), np.asarray(jnp.linalg.solve(dense, b))
    )
    gradient = jax.grad(lambda v: jnp.sum(solve(v)))(values)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_factorize_symbolic_reuses_the_analysis_across_values(x64: None):
    """The reason the symbolic tier exists: one analysis, many sets of values."""
    dense = scaled_block_matrix()
    pattern = BCOO.fromdense(jnp.asarray(dense))
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=4),
    )
    b = jnp.ones(dense.shape[0])
    with solver.factorize_symbolic(pattern) as scope:
        for scale in (1.0, 7.0):
            operator = operator_of(scale * dense)
            state = scope.init(operator)
            value, result, _ = solver.compute(state, b, {})
            assert result == lx.RESULTS.successful
            assert np.allclose(
                np.asarray(value),
                np.asarray(jnp.linalg.solve(jnp.asarray(scale * dense), b)),
            )


def test_factorize_symbolic_as_solver(x64: None):
    """The scoped-solver form works as it does for the direct solvers."""
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=4),
    )
    with solver.factorize_symbolic(operator, as_solver=True) as scoped:
        solution = lx.linear_solve(operator, b, solver=scoped)
    assert np.allclose(
        np.asarray(solution.value), np.asarray(jnp.linalg.solve(jnp.asarray(dense), b))
    )


def test_factorize_yields_a_reusable_numeric_state(x64: None):
    """`factorize` is a no-op wrapper here: building `M` already is the factorization."""
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=4),
        sparsity=operator,
    )
    with solver.factorize(operator) as state:
        value, _, _ = solver.compute(cast(Any, state), b, {})
    assert np.allclose(
        np.asarray(value), np.asarray(jnp.linalg.solve(jnp.asarray(dense), b))
    )


class _RecordingPartitioner:
    """A partitioner that remembers the pattern it was given."""

    def __init__(self):
        self.seen = None

    def partition(self, pattern):
        self.seen = pattern.concrete("test")
        return splx.BlockPartition.uniform(pattern.shape[0], 4)


class _ReversePermutation:
    """A stub transform stage that reverses the index order."""

    def symbolic(self, pattern):
        order = np.arange(pattern.shape[0])[::-1].copy()
        return SymmetricPermutation.from_order(order)


def test_transforms_run_before_the_preconditioner_sees_the_pattern(x64: None):
    """The composability guarantee: the provider analyses the *transformed* pattern.

    The stage reverses the index order, so the pattern the partitioner receives must be
    the reversed one --- and the answer must still come back in the caller's
    coordinates, untransformed.
    """
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    partitioner = _RecordingPartitioner()
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=partitioner),
        transforms=(_ReversePermutation(),),
        sparsity=operator,
    )

    original = as_coo_pattern(operator, "test").concrete("test")
    size = dense.shape[0]
    assert partitioner.seen is not None
    assert np.array_equal(
        np.sort(np.asarray(partitioner.seen.rows)),
        np.sort(size - 1 - np.asarray(original.rows)),
    )

    solution = lx.linear_solve(operator, b, solver=solver)
    assert np.allclose(
        np.asarray(solution.value), np.asarray(jnp.linalg.solve(jnp.asarray(dense), b))
    )


def test_reverse_cuthill_mckee_composes_with_the_partitioner(x64: None):
    """A real reordering in front of a real partitioner still gives the right answer."""
    dense = scaled_block_matrix()
    operator = operator_of(dense)
    b = jnp.ones(dense.shape[0])
    solver = splx.PreconditionedIterativeLinearSolver(
        gmres(),
        splx.BlockJacobi(blocks=splx.MaximalCaptureBlockPartitioner(max_block_size=8)),
        transforms=(splx.ReverseCuthillMcKee(),),
        sparsity=operator,
    )
    solution = lx.linear_solve(operator, b, solver=solver)
    assert solution.result == lx.RESULTS.successful
    assert np.allclose(
        np.asarray(solution.value), np.asarray(jnp.linalg.solve(jnp.asarray(dense), b))
    )


def test_a_stage_satisfying_neither_transform_protocol_is_refused():
    """A stage that would never be applied is an error, not a silent no-op."""
    with pytest.raises(TypeError, match="neither a `SymbolicTransform`"):
        splx.PreconditionedIterativeLinearSolver(
            gmres(),
            splx.BlockJacobi(blocks=4),
            transforms=(object(),),  # ty: ignore[invalid-argument-type]
        )
