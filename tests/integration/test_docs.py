"""Tests replicating the code examples in the docs, to ensure they don't break
on code changes."""

# ruff: noqa: F841


def test_index():
    ### Quick example ###
    import jax.numpy as jnp
    import lineax as lx
    import numpy as np
    from jax.experimental.sparse import BCOO

    import splineax

    n = 10000
    np.random.seed(0)

    # A large, randomly sparse matrix with a heavy diagonal (so it is invertible).
    diagonal_indices = np.stack([np.arange(n), np.arange(n)], axis=1)
    off_diagonal_indices = np.unique(np.random.randint(0, n, size=(2 * n, 2)), axis=0)
    indices = jnp.concatenate([diagonal_indices, off_diagonal_indices])
    values = jnp.concatenate(
        [
            np.full(n, float(n)),
            np.random.uniform(low=-1, high=1, size=off_diagonal_indices.shape[0]),
        ]
    )
    matrix = BCOO((values, indices), shape=(n, n)).sum_duplicates()

    operator = splineax.BCOOLinearOperator(matrix)
    vectors = [jnp.ones(n), jnp.arange(n) % 2]
    solver = splineax.AutoSparseLinearSolver()

    # Calculate factorization once...
    with solver.factorize(operator) as factorized_state:
        # ...and reuse for multiple solves.
        solution = lx.linear_solve(
            operator, vectors[0], solver=solver, state=factorized_state
        )
        assert jnp.allclose(matrix @ solution.value, vectors[0], atol=1e-4)

        solution = lx.linear_solve(
            operator, vectors[1], solver=solver, state=factorized_state
        )
        assert jnp.allclose(matrix @ solution.value, vectors[1], atol=1e-4)


def test_basic_usage():
    ### Solving a system ###
    import jax.numpy as jnp
    import lineax as lx
    from jax.experimental.sparse import BCOO

    import splineax

    # A sparse matrix A and a right-hand side b.
    dense = jnp.array(
        [
            [10.0, 2.0, 0.0, 0.0],
            [3.0, 14.0, 5.0, 0.0],
            [0.0, 6.0, 18.0, 9.0],
            [0.0, 0.0, 1.0, 12.0],
        ]
    )
    operator = splineax.BCOOLinearOperator(BCOO.fromdense(dense))
    b = jnp.array([1.0, 2.0, 3.0, 4.0])

    solution = lx.linear_solve(operator, b, solver=splineax.AutoSparseLinearSolver())
    x = solution.value

    ### Transposes, batching, and autodiff ###
    import jax

    # Solve the transposed system.
    solution_T = lx.linear_solve(operator.T, b, solver=splineax.KLU())

    # Solve under jit.
    solve = jax.jit(lambda v: lx.linear_solve(operator, v, solver=splineax.KLU()).value)
    x = solve(b)

    # Batch over many right-hand sides with vmap.
    bs = jnp.stack([b, b[::-1]])
    xs = jax.vmap(solve)(bs)

    # Differentiate through the solve (w.r.t. the right-hand side or the matrix entries).
    jacobian = jax.jacrev(solve)(b)

    ### Reusing work across right-hand sides ###
    solver = splineax.KLU()
    state = solver.init(operator, options={})

    x1 = lx.linear_solve(operator, b, solver=solver, state=state).value
    x2 = lx.linear_solve(operator, b[::-1], solver=solver, state=state).value


def test_operators():
    ### Constructing operators ###
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO, BCSR

    import splineax

    dense = jnp.array([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]])

    bcoo_operator = splineax.BCOOLinearOperator(BCOO.fromdense(dense))
    bcsr_operator = splineax.BCSRLinearOperator(BCSR.fromdense(dense))

    ### Tags ###
    import lineax as lx

    operator = splineax.BCOOLinearOperator(BCOO.fromdense(dense), tags=lx.symmetric_tag)


def test_solvers():
    ### Spsolve ###
    import splineax

    solver = splineax.Spsolve(
        tol=1e-6, reorder=splineax.solvers.ReorderingScheme.SYMRCM
    )

    ### KLU ###
    solver = splineax.KLU()

    ### AutoSparseLinearSolver ###
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

    import splineax

    operator = splineax.BCOOLinearOperator(
        BCOO.fromdense(jnp.array([[2.0, 1.0], [1.0, 3.0]]))
    )
    solver = splineax.AutoSparseLinearSolver()

    # Inspect what it will dispatch to (mirrors lineax.AutoLinearSolver.select_solver).
    chosen = solver.select_solver(operator)

    # Force a specific platform's choice.
    cpu_solver = splineax.AutoSparseLinearSolver(platform="cpu")  # -> KLU
    gpu_solver = splineax.AutoSparseLinearSolver(platform="gpu")  # -> Spsolve


def test_advanced():
    ### Reusing a full factorization ###
    import jax.numpy as jnp
    import lineax as lx
    from jax.experimental.sparse import BCOO

    import splineax

    dense = jnp.array(
        [
            [10.0, 2.0, 0.0, 0.0],
            [3.0, 14.0, 5.0, 0.0],
            [0.0, 6.0, 18.0, 9.0],
            [0.0, 0.0, 1.0, 12.0],
        ]
    )
    b1 = jnp.array([1.0, 2.0, 3.0, 4.0])
    b2 = b1[::-1]
    b3 = b1 + 1.0

    operator = splineax.BCOOLinearOperator(BCOO.fromdense(dense))
    solver = splineax.KLU()

    with solver.factorize(operator) as state:
        x1 = lx.linear_solve(operator, b1, solver=solver, state=state).value
        x2 = lx.linear_solve(operator, b2, solver=solver, state=state).value
        # ... reuse `state` for as many right-hand sides as you like.

    ### Reusing a symbolic factorization ###
    sparsity = BCOO.fromdense(dense)  # only the structure matters here

    with solver.factorize_symbolic(sparsity) as scope:
        # Option A: reuse the symbolic analysis, refactor numerically on each solve.
        state = scope.init(operator)
        x = lx.linear_solve(operator, b1, solver=solver, state=state).value

        # Option B: also pre-compute the numeric factorization for full reuse.
        with scope.factorize(operator) as numeric_state:
            x1 = lx.linear_solve(operator, b1, solver=solver, state=numeric_state).value
            x2 = lx.linear_solve(operator, b2, solver=solver, state=numeric_state).value

    ### Writing backend-agnostic code ###
    from splineax import AbstractSparseLinearSolver

    def solve_many(solver: AbstractSparseLinearSolver, operator, right_hand_sides):
        with solver.factorize(operator) as state:
            return [
                lx.linear_solve(operator, b, solver=solver, state=state).value
                for b in right_hand_sides
            ]

    # Fast factorization reuse on CPU, plain (re)solves elsewhere, same code:
    solve_many(splineax.AutoSparseLinearSolver(), operator, [b1, b2, b3])
