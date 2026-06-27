"""Shared test suite for the sparse direct solvers.

Both `Spsolve` (wrapping `jax.experimental.sparse.linalg.spsolve`) and `KLU` (wrapping
`klujax`) must behave identically against either sparse operator format. The `solver`
fixture is parametrised over the two solvers and the `make_operator` fixture over
`["bcoo", "bcsr"]` (both in [conftest.py](conftest.py)), so every test below runs for the full
solver x format cross-product, with the dense reference matrix as the source of truth.

KLU-specific behaviour (the `factorize()` context manager and its handle lifecycle) lives
in [test_klu.py](test_klu.py).
"""

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO, BCSR

from splineax import BCOOLinearOperator, BCSRLinearOperator

from .conftest import (
    COMPLEX_MATRIX,
    COMPLEX_RIGHT_HAND_SIDE,
    RIGHT_HAND_SIDE,
    SQUARE_MATRIX,
    WIDE_MATRIX,
    OperatorFactory,
)


def test_solve_matches_numpy(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """The core proof that the solver works for both operator formats: the solution must
    match `numpy.linalg.solve` against the dense reference matrix."""
    operator = make_operator(SQUARE_MATRIX)
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_solve_matches_dense_lu(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """Cross-check against Lineax's own dense `LU` path on the same operator, so that any
    discrepancy is attributable to the solver rather than to the reference."""
    operator = make_operator(SQUARE_MATRIX)
    sparse_solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    lu_solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=lx.LU()).value
    assert jnp.allclose(sparse_solution, lu_solution, atol=1e-5)


def test_result_is_successful(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """A well-posed solve must report `RESULTS.successful`, confirming `compute` reports
    success (and that the no-throw path returns a usable solution)."""
    operator = make_operator(SQUARE_MATRIX)
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver)
    assert solution.result == lx.RESULTS.successful


def test_reuses_state_across_vectors(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """`init` computes on just the operator; `compute` then solves for a given vector.
    Re-using a single `state` across multiple right-hand sides must give correct
    solutions, exercising that init/compute separation."""
    operator = make_operator(SQUARE_MATRIX)
    state = solver.init(operator, options={})
    # Match the operator dtype: a `KLU` solve enables x64 globally, after which a freshly
    # created array would default to float64 and mismatch the (float32) operator.
    second_rhs = jnp.array([4.0, 3.0, 2.0, 1.0]).astype(RIGHT_HAND_SIDE.dtype)
    for rhs in (RIGHT_HAND_SIDE, second_rhs):
        solution = lx.linear_solve(operator, rhs, solver=solver, state=state).value
        expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(rhs))
        assert jnp.allclose(solution, expected, atol=1e-5)


def test_transpose_solve(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """Solving against `operator.T` must recover the transposed system's solution,
    exercising the solver's `transpose` state path."""
    operator = make_operator(SQUARE_MATRIX)
    solution = lx.linear_solve(operator.T, RIGHT_HAND_SIDE, solver=solver).value
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_complex_solve(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """Solving a complex system must match `numpy.linalg.solve`, exercising the
    `complex128` path of each backend."""
    operator = make_operator(COMPLEX_MATRIX)
    solution = lx.linear_solve(operator, COMPLEX_RIGHT_HAND_SIDE, solver=solver).value
    expected = jnp.linalg.solve(
        np.asarray(COMPLEX_MATRIX), np.asarray(COMPLEX_RIGHT_HAND_SIDE)
    )
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_non_square_raises(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """Both solvers only handle square systems, so initialising on a non-square operator
    must fail loudly rather than producing nonsense."""
    operator = make_operator(WIDE_MATRIX)
    with pytest.raises(ValueError):
        solver.init(operator, options={})


def _unsorted_bcoo(dense_matrix: jax.Array) -> BCOO:
    """A coalesced `BCOO` whose stored entries are deliberately *not* in sorted order,
    by reversing the entry order of the canonical (sorted) representation."""
    canonical = BCOO.fromdense(dense_matrix)
    reversed_bcoo = BCOO(
        (canonical.data[::-1], canonical.indices[::-1]), shape=canonical.shape
    )
    assert not reversed_bcoo.indices_sorted
    return reversed_bcoo


@pytest.mark.parametrize("fmt", ["bcoo", "bcsr"])
def test_solve_with_unsorted_indices(fmt: str, solver: lx.AbstractLinearSolver) -> None:
    """A coalesced but *unsorted* operator (in either format) must still solve correctly:
    `Spsolve` canonicalises the index order in `init`, while `KLU` is order-agnostic."""
    unsorted_bcoo = _unsorted_bcoo(SQUARE_MATRIX)
    if fmt == "bcoo":
        operator = BCOOLinearOperator(unsorted_bcoo)
    else:
        # `BCSR.from_bcoo` sorts, so build an unsorted `BCSR` by reversing the column
        # order *within each row* of the sorted one (preserving `indptr`, hence the same
        # matrix, but with non-ascending column indices inside each row).
        sorted_bcsr = BCSR.from_bcoo(unsorted_bcoo.sort_indices())
        indptr = np.asarray(sorted_bcsr.indptr)
        data = np.array(sorted_bcsr.data)
        indices = np.array(sorted_bcsr.indices)
        for start, end in zip(indptr[:-1], indptr[1:]):
            data[start:end] = data[start:end][::-1]
            indices[start:end] = indices[start:end][::-1]
        unsorted_bcsr = BCSR(
            (jnp.asarray(data), jnp.asarray(indices), sorted_bcsr.indptr),
            shape=sorted_bcsr.shape,
        )
        assert not unsorted_bcsr.indices_sorted
        operator = BCSRLinearOperator(unsorted_bcsr)

    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_solve_under_jit(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """The solve must be traceable: wrapping it in `jax.jit` (so the backend runs inside a
    compiled computation) must still give the right answer."""
    operator = make_operator(SQUARE_MATRIX)

    @jax.jit
    def solve(b: jax.Array) -> jax.Array:
        return lx.linear_solve(operator, b, solver=solver).value

    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solve(RIGHT_HAND_SIDE), expected, atol=1e-5)


def test_solve_under_vmap(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """`jax.vmap` over a stack of right-hand sides must solve each one, exercising each
    backend's batching rule (`Spsolve`'s custom sequential rule; `klujax`'s native one)."""
    operator = make_operator(SQUARE_MATRIX)
    right_hand_sides = jnp.stack([RIGHT_HAND_SIDE, RIGHT_HAND_SIDE[::-1]])

    def solve(b: jax.Array) -> jax.Array:
        return lx.linear_solve(operator, b, solver=solver).value

    solutions = jax.vmap(solve)(right_hand_sides)
    expected = np.stack(
        [
            np.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(b))
            for b in right_hand_sides
        ]
    )
    assert jnp.allclose(solutions, expected, atol=1e-5)


def test_differentiable_wrt_vector(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """Forward- and reverse-mode AD through the solve w.r.t. the right-hand side must
    yield the analytic Jacobian `d/db (A^-1 b) = A^-1`. The reverse path additionally
    exercises the solver's `transpose` method (used by Lineax's backward pass)."""
    operator = make_operator(SQUARE_MATRIX)

    def solve(b: jax.Array) -> jax.Array:
        return lx.linear_solve(operator, b, solver=solver).value

    expected_jacobian = jnp.linalg.inv(SQUARE_MATRIX)
    assert jnp.allclose(
        jax.jacfwd(solve)(RIGHT_HAND_SIDE), expected_jacobian, atol=1e-5
    )
    assert jnp.allclose(
        jax.jacrev(solve)(RIGHT_HAND_SIDE), expected_jacobian, atol=1e-5
    )


@pytest.mark.parametrize("fmt", ["bcoo", "bcsr"])
def test_differentiable_wrt_matrix(fmt: str, solver: lx.AbstractLinearSolver) -> None:
    """AD w.r.t. the matrix entries must match Lineax's own dense `LU` path differentiated
    the same way. The operator is rebuilt from a differentiable `data` vector (with fixed
    sparsity), so the Jacobian flows through the sparse solve; both `jax.jacfwd` and
    `jax.jacrev` are checked against the dense reference."""
    canonical = BCOO.fromdense(SQUARE_MATRIX)
    data0, indices, shape = canonical.data, canonical.indices, canonical.shape

    def solve_sparse(data: jax.Array) -> jax.Array:
        bcoo = BCOO((data, indices), shape=shape)
        if fmt == "bcoo":
            operator = BCOOLinearOperator(bcoo)
        else:
            operator = BCSRLinearOperator(BCSR.from_bcoo(bcoo))
        return lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value

    def solve_dense(data: jax.Array) -> jax.Array:
        dense = BCOO((data, indices), shape=shape).todense()
        operator = lx.MatrixLinearOperator(dense)
        return lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=lx.LU()).value

    reference_jacobian = jax.jacrev(solve_dense)(data0)
    assert jnp.allclose(jax.jacfwd(solve_sparse)(data0), reference_jacobian, atol=1e-5)
    assert jnp.allclose(jax.jacrev(solve_sparse)(data0), reference_jacobian, atol=1e-5)
