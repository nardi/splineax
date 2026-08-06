"""Test suite for `PreconditionedIterativeSolver` and `block_jacobi_solver`.

Covers the same factorization-reuse contract as the direct solvers
([test_factorization.py](test_factorization.py)) plus what's specific to a
preconditioned iterative solve: it must agree with a direct solve, it must actually
benefit from the preconditioner, and its autodiff must match a direct solver's exactly
(the preconditioner is a constant, never part of the derivative). The transform algebra
itself is covered in `tests/unit/transforms/test_transforms.py`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from splineax import (
    KLU,
    BCOOLinearOperator,
    PreconditionedIterativeSolver,
    SymbolicScopedSparseLinearSolver,
    block_jacobi_solver,
)
from splineax.transforms import AggregationClustering, BlockJacobi, RuizEquilibration
from splineax.transforms._compose import _ComposedTransform


def _laplacian_2d(nx: int, ny: int) -> np.ndarray:
    n = nx * ny
    dense = np.zeros((n, n))

    def index(i: int, j: int) -> int:
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            k = index(i, j)
            dense[k, k] = 4.0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < nx and 0 <= jj < ny:
                    dense[k, index(ii, jj)] = -1.0
    return dense


LAPLACIAN: np.ndarray = _laplacian_2d(6, 6)
N: int = LAPLACIAN.shape[0]
RIGHT_HAND_SIDE: jax.Array = jnp.arange(N, dtype=jnp.float32) + 1.0


@pytest.fixture
def operator() -> BCOOLinearOperator:
    return BCOOLinearOperator(BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32)))


@pytest.fixture
def solver() -> PreconditionedIterativeSolver:
    return block_jacobi_solver(block_size=4, solver=lx.GMRES(rtol=1e-5, atol=1e-5))


@pytest.fixture
def expected() -> jax.Array:
    return jnp.linalg.solve(jnp.array(LAPLACIAN, dtype=jnp.float32), RIGHT_HAND_SIDE)


# ---------------------------------------------------------------------------
# Correctness and benefit
# ---------------------------------------------------------------------------


def test_solve_matches_dense_solve(operator, solver, expected) -> None:
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert jnp.allclose(solution, expected, atol=1e-3)


def test_preconditioner_converges_where_plain_gmres_stagnates() -> None:
    """At a tolerance tight enough that plain GMRES on this (larger) matrix stalls
    out, the same GMRES conditioned by `block_jacobi_solver` still converges cleanly,
    in far fewer steps. This is the entire point of preconditioning, so it is worth
    pinning directly rather than only checking the easy case.

    Needs a bigger grid than the rest of this module's `N = 36` matrix: that one is
    small and well-conditioned enough that plain GMRES has no trouble with it even at
    a tight tolerance, which would not demonstrate anything.
    """
    dense = _laplacian_2d(12, 12)
    n = dense.shape[0]
    operator = BCOOLinearOperator(BCOO.fromdense(jnp.array(dense, dtype=jnp.float32)))
    b = jnp.ones(n, dtype=jnp.float32)
    expected = jnp.linalg.solve(jnp.array(dense, dtype=jnp.float32), b)
    tight_gmres = lx.GMRES(rtol=1e-6, atol=1e-6, max_steps=200)

    plain_solution = lx.linear_solve(operator, b, solver=tight_gmres, throw=False)
    assert plain_solution.result != lx.RESULTS.successful

    preconditioned = block_jacobi_solver(block_size=8, solver=tight_gmres)
    preconditioned_solution = lx.linear_solve(
        operator, b, solver=preconditioned, throw=False
    )
    assert preconditioned_solution.result == lx.RESULTS.successful
    assert jnp.allclose(preconditioned_solution.value, expected, atol=1e-3)
    assert (
        preconditioned_solution.stats["num_steps"] < plain_solution.stats["num_steps"]
    )


def test_requires_square_operator(solver) -> None:
    wide = BCOOLinearOperator(BCOO.fromdense(jnp.ones((2, 3))))
    with pytest.raises(ValueError, match="square"):
        solver.init(wide, {})


def test_no_default_configuration() -> None:
    """`transform`/`preconditioner`/`solver` are all required: there is no implicit
    default trio, only `block_jacobi_solver` as an explicit, documented choice."""
    with pytest.raises(TypeError):
        PreconditionedIterativeSolver()  # type: ignore


# ---------------------------------------------------------------------------
# Factorization-reuse contract: mirrors test_factorization.py's suite, but for a single
# solver rather than a fixture cross-product, since there is exactly one
# `PreconditionedIterativeSolver`.
# ---------------------------------------------------------------------------


def test_analyze_numeric_reuses_state_across_vectors(
    operator, solver, expected
) -> None:
    second_rhs = RIGHT_HAND_SIDE[::-1]
    with solver.analyze_numeric(operator) as state:
        solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=solver, state=state
        )
        second_solution = lx.linear_solve(
            operator, second_rhs, solver=solver, state=state
        )
    assert jnp.allclose(solution.value, expected, atol=1e-3)
    second_expected = jnp.linalg.solve(
        jnp.array(LAPLACIAN, dtype=jnp.float32), second_rhs
    )
    assert jnp.allclose(second_solution.value, second_expected, atol=1e-3)


def test_analyze_symbolic_solves_correctly(operator, solver, expected) -> None:
    sparsity = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    with solver.analyze_symbolic(sparsity) as scope:
        symbolic_state = scope.init(operator)
        symbolic_solution = lx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver=solver, state=symbolic_state
        )
        with scope.analyze_numeric(operator) as numeric_state:
            numeric_solution = lx.linear_solve(
                operator, RIGHT_HAND_SIDE, solver=solver, state=numeric_state
            )
    assert jnp.allclose(symbolic_solution.value, expected, atol=1e-3)
    assert jnp.allclose(numeric_solution.value, expected, atol=1e-3)


def test_as_solver_solves_correctly(operator, solver, expected) -> None:
    sparsity = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    with solver.analyze_symbolic(sparsity, as_solver=True) as scoped_solver:
        assert isinstance(scoped_solver, SymbolicScopedSparseLinearSolver)
        solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=scoped_solver)
    assert jnp.allclose(solution.value, expected, atol=1e-3)


def test_numeric_state_solve_under_jit(operator, solver, expected) -> None:
    @eqx.filter_jit
    def run(solver, state, b):
        return lx.linear_solve(operator, b, solver=solver, state=state).value

    with solver.analyze_numeric(operator) as state:
        solution = run(solver, state, RIGHT_HAND_SIDE)
    assert jnp.allclose(solution, expected, atol=1e-3)


def test_symbolic_scope_solve_under_jit(operator, solver, expected) -> None:
    sparsity = BCOO.fromdense(jnp.array(LAPLACIAN, dtype=jnp.float32))
    indices, shape = sparsity.indices, sparsity.shape

    @eqx.filter_jit
    def run(scope, data, b):
        op = BCOOLinearOperator(BCOO((data, indices), shape=shape))
        state = scope.init(op)
        return lx.linear_solve(op, b, solver=solver, state=state).value

    with solver.analyze_symbolic(sparsity) as scope:
        solution = run(scope, sparsity.data, RIGHT_HAND_SIDE)
    assert jnp.allclose(solution, expected, atol=1e-3)


def test_transpose_of_numeric_state_solves_transposed(
    operator, solver, expected
) -> None:
    # The Laplacian is symmetric, so the transposed solve must match the original.
    with solver.analyze_numeric(operator) as state:
        transposed_state, _ = solver.transpose(state, options={})
        solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0]
    assert jnp.allclose(solution, expected, atol=1e-3)


# ---------------------------------------------------------------------------
# Autodiff: must match a direct solver's gradient exactly (the preconditioner and
# transform are constants, never differentiated through).
# ---------------------------------------------------------------------------


def test_grad_matches_klu(enable_x64: None) -> None:
    indices = BCOO.fromdense(jnp.array(LAPLACIAN)).indices
    b = jnp.ones(N, dtype=jnp.float64)
    solver = block_jacobi_solver(block_size=4, solver=lx.GMRES(rtol=1e-10, atol=1e-10))

    def solve_with(values, solver):
        matrix = BCOO((values, indices), shape=(N, N))
        op = BCOOLinearOperator(matrix)
        return jnp.sum(lx.linear_solve(op, b, solver=solver).value)

    values = jnp.array(LAPLACIAN, dtype=jnp.float64)[indices[:, 0], indices[:, 1]]

    grad_precond = jax.grad(solve_with)(values, solver)
    grad_klu = jax.grad(solve_with)(values, KLU())
    assert jnp.allclose(grad_precond, grad_klu, atol=1e-6)


def test_jacfwd_matches_klu(enable_x64: None) -> None:
    indices = BCOO.fromdense(jnp.array(LAPLACIAN)).indices
    b = jnp.ones(N, dtype=jnp.float64)
    solver = block_jacobi_solver(block_size=4, solver=lx.GMRES(rtol=1e-10, atol=1e-10))

    def solve_with(values, solver):
        matrix = BCOO((values, indices), shape=(N, N))
        op = BCOOLinearOperator(matrix)
        return jnp.sum(lx.linear_solve(op, b, solver=solver).value)

    values = jnp.array(LAPLACIAN, dtype=jnp.float64)[indices[:, 0], indices[:, 1]]

    jac_precond = jax.jacfwd(solve_with)(values, solver)
    jac_klu = jax.jacfwd(solve_with)(values, KLU())
    assert jnp.allclose(jac_precond, jac_klu, atol=1e-6)


# ---------------------------------------------------------------------------
# block_jacobi_solver
# ---------------------------------------------------------------------------


def test_block_jacobi_solver_wires_matching_block_sizes() -> None:
    solver = block_jacobi_solver(block_size=8)
    assert isinstance(solver.preconditioner, BlockJacobi)
    assert solver.preconditioner.block_size == 8
    # The clustering is the first stage of the composed transform.
    assert isinstance(solver.transform, _ComposedTransform)
    clustering = solver.transform.stages[0]
    assert isinstance(clustering, AggregationClustering)
    assert clustering.block_size == 8


def test_block_jacobi_solver_symmetric_equilibration_for_cg() -> None:
    default = block_jacobi_solver()
    assert isinstance(default.transform, _ComposedTransform)
    equilibration = default.transform.stages[1]
    assert isinstance(equilibration, RuizEquilibration)
    assert not equilibration.symmetric

    for_cg = block_jacobi_solver(solver=lx.CG(rtol=1e-6, atol=1e-6))
    assert isinstance(for_cg.transform, _ComposedTransform)
    equilibration_cg = for_cg.transform.stages[1]
    assert isinstance(equilibration_cg, RuizEquilibration)
    assert equilibration_cg.symmetric
