"""Differentiability tests for the sparse solvers.

Every solve path must be differentiable w.r.t. both the right-hand side vector (rhs)
and the sparse matrix entries (lhs), in both forward mode (`jax.jacfwd`) and reverse
mode (`jax.jacrev`). The basic stateless path is covered in `test_solvers.py`; this
module adds coverage for:

- factorization-reuse paths (`factorize`, `factorize_symbolic`, `as_solver`)
- `splineax.linear_solve` (needed for full-jit `factorize_symbolic` scopes)
- JIT + AD composition (`jacfwd`/`jacrev` of a jitted solve)

All tests are parametrised over the `solver` fixture (spsolve, klu, pardiso, auto) and,
where applicable, the `make_operator` fixture (bcoo, bcsr) from `conftest.py`.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO

import splineax as splx
from splineax import (
    AbstractSparseLinearSolver,
    BCOOLinearOperator,
    BCSRLinearOperator,
    SparseJacobianLinearOperator,
)

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


def _dense_reference_jacobian_wrt_vector() -> jax.Array:
    """d/db (A^-1 b) = A^-1."""
    return jnp.linalg.inv(SQUARE_MATRIX)


def _dense_reference_jacobian_wrt_data() -> jax.Array:
    """Jacobian of the solve w.r.t. the sparse data entries, computed through the dense
    LU path as the ground truth."""
    canonical = BCOO.fromdense(SQUARE_MATRIX)

    def solve_dense(data: jax.Array) -> jax.Array:
        dense = BCOO((data, canonical.indices), shape=canonical.shape).todense()
        operator = lx.MatrixLinearOperator(dense)
        return lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=lx.LU()).value

    return jax.jacrev(solve_dense)(canonical.data)


# ---------------------------------------------------------------------------
# 1. Factorize (numeric) — AD w.r.t. vector
# ---------------------------------------------------------------------------


class TestFactorizeDifferentiability:
    """AD through a solve that reuses a pre-factorized (numeric) state.

    The factorization is performed eagerly, outside the differentiated function,
    matching the intended usage pattern: factorise once, solve (and differentiate
    through the solve) many times.
    """

    def test_wrt_vector_jacfwd(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)
        expected = _dense_reference_jacobian_wrt_vector()

        with solver.factorize(operator) as state:

            def solve(b: jax.Array) -> jax.Array:
                return lx.linear_solve(
                    operator, b, solver=solver, state=state
                ).value

            assert jnp.allclose(
                jax.jacfwd(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5
            )

    def test_wrt_vector_jacrev(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)
        expected = _dense_reference_jacobian_wrt_vector()

        with solver.factorize(operator) as state:

            def solve(b: jax.Array) -> jax.Array:
                return lx.linear_solve(
                    operator, b, solver=solver, state=state
                ).value

            assert jnp.allclose(
                jax.jacrev(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5
            )


# ---------------------------------------------------------------------------
# 2. Factorize symbolic — AD w.r.t. vector and matrix data
# ---------------------------------------------------------------------------


class TestFactorizeSymbolicDifferentiability:
    """AD through a solve using a symbolic-scope state.

    For the vector tests, both the symbolic scope and its derived state are built
    eagerly, outside the differentiated function. For the matrix tests the operator is
    rebuilt from differentiated data inside the traced function, and the scope's `init`
    derives a fresh state from it, exercising the path where the numeric values change
    but the sparsity pattern is reused.
    """

    def test_wrt_vector_jacfwd(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)
        sparsity = BCOO.fromdense(SQUARE_MATRIX)
        expected = _dense_reference_jacobian_wrt_vector()

        with solver.factorize_symbolic(sparsity) as scope:
            state = scope.init(operator)

            def solve(b: jax.Array) -> jax.Array:
                return lx.linear_solve(
                    operator, b, solver=solver, state=state
                ).value

            assert jnp.allclose(
                jax.jacfwd(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5
            )

    def test_wrt_vector_jacrev(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)
        sparsity = BCOO.fromdense(SQUARE_MATRIX)
        expected = _dense_reference_jacobian_wrt_vector()

        with solver.factorize_symbolic(sparsity) as scope:
            state = scope.init(operator)

            def solve(b: jax.Array) -> jax.Array:
                return lx.linear_solve(
                    operator, b, solver=solver, state=state
                ).value

            assert jnp.allclose(
                jax.jacrev(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5
            )

    def test_wrt_matrix_jacfwd(self, solver: AbstractSparseLinearSolver) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape
        expected = _dense_reference_jacobian_wrt_data()

        with solver.factorize_symbolic(canonical) as scope:

            def solve(data: jax.Array) -> jax.Array:
                bcoo = BCOO((data, indices), shape=shape)
                operator = BCOOLinearOperator(bcoo)
                state = scope.init(operator)
                return lx.linear_solve(
                    operator, RIGHT_HAND_SIDE, solver=solver, state=state
                ).value

            assert jnp.allclose(jax.jacfwd(solve)(data0), expected, atol=1e-5)

    def test_wrt_matrix_jacrev(self, solver: AbstractSparseLinearSolver) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape
        expected = _dense_reference_jacobian_wrt_data()

        with solver.factorize_symbolic(canonical) as scope:

            def solve(data: jax.Array) -> jax.Array:
                bcoo = BCOO((data, indices), shape=shape)
                operator = BCOOLinearOperator(bcoo)
                state = scope.init(operator)
                return lx.linear_solve(
                    operator, RIGHT_HAND_SIDE, solver=solver, state=state
                ).value

            assert jnp.allclose(jax.jacrev(solve)(data0), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. as_solver (SymbolicScopedSparseLinearSolver) — AD w.r.t. vector and matrix data
# ---------------------------------------------------------------------------


class TestAsSolverDifferentiability:
    """AD through a solve using the scoped solver from
    `factorize_symbolic(as_solver=True)`.

    The scoped solver is built eagerly and the differentiated function only performs the
    solve. For the matrix tests, the operator is rebuilt from differentiated data inside
    the traced function, so the scoped solver's `init` (which derives a state from the
    scope) sees the new values while reusing the symbolic factorization.
    """

    def test_wrt_vector_jacfwd(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)
        sparsity = BCOO.fromdense(SQUARE_MATRIX)
        expected = _dense_reference_jacobian_wrt_vector()

        with solver.factorize_symbolic(sparsity, as_solver=True) as scoped:

            def solve(b: jax.Array) -> jax.Array:
                return lx.linear_solve(operator, b, solver=scoped).value

            assert jnp.allclose(
                jax.jacfwd(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5
            )

    def test_wrt_vector_jacrev(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)
        sparsity = BCOO.fromdense(SQUARE_MATRIX)
        expected = _dense_reference_jacobian_wrt_vector()

        with solver.factorize_symbolic(sparsity, as_solver=True) as scoped:

            def solve(b: jax.Array) -> jax.Array:
                return lx.linear_solve(operator, b, solver=scoped).value

            assert jnp.allclose(
                jax.jacrev(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5
            )

    def test_wrt_matrix_jacfwd(self, solver: AbstractSparseLinearSolver) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape
        expected = _dense_reference_jacobian_wrt_data()

        with solver.factorize_symbolic(canonical, as_solver=True) as scoped:

            def solve(data: jax.Array) -> jax.Array:
                bcoo = BCOO((data, indices), shape=shape)
                operator = BCOOLinearOperator(bcoo)
                return lx.linear_solve(
                    operator, RIGHT_HAND_SIDE, solver=scoped
                ).value

            assert jnp.allclose(jax.jacfwd(solve)(data0), expected, atol=1e-5)

    def test_wrt_matrix_jacrev(self, solver: AbstractSparseLinearSolver) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape
        expected = _dense_reference_jacobian_wrt_data()

        with solver.factorize_symbolic(canonical, as_solver=True) as scoped:

            def solve(data: jax.Array) -> jax.Array:
                bcoo = BCOO((data, indices), shape=shape)
                operator = BCOOLinearOperator(bcoo)
                return lx.linear_solve(
                    operator, RIGHT_HAND_SIDE, solver=scoped
                ).value

            assert jnp.allclose(jax.jacrev(solve)(data0), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. splineax.linear_solve — AD w.r.t. vector and matrix data
# ---------------------------------------------------------------------------


class TestSplineaxLinearSolveDifferentiability:
    """AD through `splineax.linear_solve`, the wrapper needed for full-jit
    `factorize_symbolic` scopes. Tested here on the basic (no factorization) path to
    confirm it is a drop-in for `lineax.linear_solve` under both AD modes."""

    def test_wrt_vector_jacfwd(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)

        def solve(b: jax.Array) -> jax.Array:
            return splx.linear_solve(operator, b, solver).value

        expected = _dense_reference_jacobian_wrt_vector()
        assert jnp.allclose(jax.jacfwd(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5)

    def test_wrt_vector_jacrev(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)

        def solve(b: jax.Array) -> jax.Array:
            return splx.linear_solve(operator, b, solver).value

        expected = _dense_reference_jacobian_wrt_vector()
        assert jnp.allclose(jax.jacrev(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5)

    def test_wrt_matrix_jacfwd(self, solver: AbstractSparseLinearSolver) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape

        def solve(data: jax.Array) -> jax.Array:
            bcoo = BCOO((data, indices), shape=shape)
            operator = BCOOLinearOperator(bcoo)
            return splx.linear_solve(operator, RIGHT_HAND_SIDE, solver).value

        expected = _dense_reference_jacobian_wrt_data()
        assert jnp.allclose(jax.jacfwd(solve)(data0), expected, atol=1e-5)

    def test_wrt_matrix_jacrev(self, solver: AbstractSparseLinearSolver) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape

        def solve(data: jax.Array) -> jax.Array:
            bcoo = BCOO((data, indices), shape=shape)
            operator = BCOOLinearOperator(bcoo)
            return splx.linear_solve(operator, RIGHT_HAND_SIDE, solver).value

        expected = _dense_reference_jacobian_wrt_data()
        assert jnp.allclose(jax.jacrev(solve)(data0), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. splineax.linear_solve with as_solver under JIT — AD w.r.t. matrix data
# ---------------------------------------------------------------------------


class TestSplineaxLinearSolveAsSolverDifferentiability:
    """AD through `splineax.linear_solve` with a scoped solver under JIT, which is the
    full-jit `factorize_symbolic` path that `splineax.linear_solve` exists to support."""

    def test_wrt_matrix_jacfwd(self, solver: AbstractSparseLinearSolver) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape
        expected = _dense_reference_jacobian_wrt_data()

        with solver.factorize_symbolic(canonical, as_solver=True) as scoped:

            def solve(data: jax.Array) -> jax.Array:
                bcoo = BCOO((data, indices), shape=shape, indices_sorted=True)
                operator = BCOOLinearOperator(bcoo)
                return splx.linear_solve(operator, RIGHT_HAND_SIDE, scoped).value

            assert jnp.allclose(jax.jacfwd(solve)(data0), expected, atol=1e-5)

    def test_wrt_matrix_jacrev(self, solver: AbstractSparseLinearSolver) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape
        expected = _dense_reference_jacobian_wrt_data()

        with solver.factorize_symbolic(canonical, as_solver=True) as scoped:

            def solve(data: jax.Array) -> jax.Array:
                bcoo = BCOO((data, indices), shape=shape, indices_sorted=True)
                operator = BCOOLinearOperator(bcoo)
                return splx.linear_solve(operator, RIGHT_HAND_SIDE, scoped).value

            assert jnp.allclose(jax.jacrev(solve)(data0), expected, atol=1e-5)

    def test_wrt_vector_jacfwd(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)
        sparsity = BCOO.fromdense(SQUARE_MATRIX)
        expected = _dense_reference_jacobian_wrt_vector()

        with solver.factorize_symbolic(sparsity, as_solver=True) as scoped:

            def solve(b: jax.Array) -> jax.Array:
                return splx.linear_solve(operator, b, scoped).value

            assert jnp.allclose(
                jax.jacfwd(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5
            )

    def test_wrt_vector_jacrev(
        self,
        make_operator: OperatorFactory,
        solver: AbstractSparseLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)
        sparsity = BCOO.fromdense(SQUARE_MATRIX)
        expected = _dense_reference_jacobian_wrt_vector()

        with solver.factorize_symbolic(sparsity, as_solver=True) as scoped:

            def solve(b: jax.Array) -> jax.Array:
                return splx.linear_solve(operator, b, scoped).value

            assert jnp.allclose(
                jax.jacrev(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5
            )


# ---------------------------------------------------------------------------
# 6. JIT + AD composition
# ---------------------------------------------------------------------------


class TestJitAdComposition:
    """AD of a jitted solve, ensuring JIT compilation does not break differentiability."""

    def test_wrt_vector_jacfwd_of_jit(
        self,
        make_operator: OperatorFactory,
        solver: lx.AbstractLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)

        @eqx.filter_jit
        def solve(b: jax.Array) -> jax.Array:
            return lx.linear_solve(operator, b, solver=solver).value

        expected = _dense_reference_jacobian_wrt_vector()
        assert jnp.allclose(jax.jacfwd(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5)

    def test_wrt_vector_jacrev_of_jit(
        self,
        make_operator: OperatorFactory,
        solver: lx.AbstractLinearSolver,
    ) -> None:
        operator = make_operator(SQUARE_MATRIX)

        @eqx.filter_jit
        def solve(b: jax.Array) -> jax.Array:
            return lx.linear_solve(operator, b, solver=solver).value

        expected = _dense_reference_jacobian_wrt_vector()
        assert jnp.allclose(jax.jacrev(solve)(RIGHT_HAND_SIDE), expected, atol=1e-5)

    def test_wrt_matrix_jacfwd_of_jit(
        self, solver: lx.AbstractLinearSolver
    ) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape

        @eqx.filter_jit
        def solve(data: jax.Array) -> jax.Array:
            bcoo = BCOO((data, indices), shape=shape)
            operator = BCOOLinearOperator(bcoo)
            return lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value

        expected = _dense_reference_jacobian_wrt_data()
        assert jnp.allclose(jax.jacfwd(solve)(data0), expected, atol=1e-5)

    def test_wrt_matrix_jacrev_of_jit(
        self, solver: lx.AbstractLinearSolver
    ) -> None:
        canonical = BCOO.fromdense(SQUARE_MATRIX)
        data0, indices, shape = canonical.data, canonical.indices, canonical.shape

        @eqx.filter_jit
        def solve(data: jax.Array) -> jax.Array:
            bcoo = BCOO((data, indices), shape=shape)
            operator = BCOOLinearOperator(bcoo)
            return lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value

        expected = _dense_reference_jacobian_wrt_data()
        assert jnp.allclose(jax.jacrev(solve)(data0), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. SparseJacobianLinearOperator — nested AD through the solve
# ---------------------------------------------------------------------------


def _square_function(x: jax.Array, args: object) -> jax.Array:
    """A square nonlinear map with an invertible banded Jacobian."""
    del args
    return 3.0 * x + x**2 + 0.5 * jnp.roll(x, 1) * x


_JACOBIAN_POINT = jnp.linspace(0.5, 1.5, 5)
_JACOBIAN_RHS = jnp.arange(1.0, 6.0)


class TestSparseJacobianOperatorDifferentiability:
    """AD of a sparse-Jacobian solve w.r.t. the evaluation point and the rhs.

    Differentiating `J(x)^{-1} b` w.r.t. `x` nests two levels of AD: the outer
    `jacfwd`/`jacrev` and the inner JVP/VJP that the `SparseJacobianLinearOperator`
    uses to materialise `J(x)`. The reference is the same solve through lineax's dense
    `JacobianLinearOperator` + `LU`.
    """

    @staticmethod
    def _dense_solve(x: jax.Array, b: jax.Array) -> jax.Array:
        operator = lx.JacobianLinearOperator(_square_function, x, args=None)
        return lx.linear_solve(operator, b, solver=lx.LU()).value

    @staticmethod
    def _sparse_solve(
        x: jax.Array, b: jax.Array, solver: lx.AbstractLinearSolver
    ) -> jax.Array:
        operator = SparseJacobianLinearOperator(_square_function, x)
        return lx.linear_solve(operator, b, solver=solver).value

    def test_wrt_point_jacfwd(self, solver: AbstractSparseLinearSolver) -> None:
        def solve_sparse(x: jax.Array) -> jax.Array:
            return self._sparse_solve(x, _JACOBIAN_RHS, solver)

        def solve_dense(x: jax.Array) -> jax.Array:
            return self._dense_solve(x, _JACOBIAN_RHS)

        expected = jax.jacfwd(solve_dense)(_JACOBIAN_POINT)
        result = jax.jacfwd(solve_sparse)(_JACOBIAN_POINT)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_wrt_point_jacrev(self, solver: AbstractSparseLinearSolver) -> None:
        def solve_sparse(x: jax.Array) -> jax.Array:
            return self._sparse_solve(x, _JACOBIAN_RHS, solver)

        def solve_dense(x: jax.Array) -> jax.Array:
            return self._dense_solve(x, _JACOBIAN_RHS)

        expected = jax.jacrev(solve_dense)(_JACOBIAN_POINT)
        result = jax.jacrev(solve_sparse)(_JACOBIAN_POINT)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_wrt_rhs_jacfwd(self, solver: AbstractSparseLinearSolver) -> None:
        def solve_sparse(b: jax.Array) -> jax.Array:
            return self._sparse_solve(_JACOBIAN_POINT, b, solver)

        def solve_dense(b: jax.Array) -> jax.Array:
            return self._dense_solve(_JACOBIAN_POINT, b)

        expected = jax.jacfwd(solve_dense)(_JACOBIAN_RHS)
        result = jax.jacfwd(solve_sparse)(_JACOBIAN_RHS)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_wrt_rhs_jacrev(self, solver: AbstractSparseLinearSolver) -> None:
        def solve_sparse(b: jax.Array) -> jax.Array:
            return self._sparse_solve(_JACOBIAN_POINT, b, solver)

        def solve_dense(b: jax.Array) -> jax.Array:
            return self._dense_solve(_JACOBIAN_POINT, b)

        expected = jax.jacrev(solve_dense)(_JACOBIAN_RHS)
        result = jax.jacrev(solve_sparse)(_JACOBIAN_RHS)
        assert jnp.allclose(result, expected, atol=1e-5)
