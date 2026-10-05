"""Solves through tagged `lineax.JacobianLinearOperator`s and `FunctionLinearOperator`s.

Every direct solver turns such an operator into a `BCOO` with the tag's coloring. The
reference is a dense numpy solve against `jax.jacfwd`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

import splineax as splx
from splineax import KLU, sparsity_coloring_tag


def square_function(point: jax.Array, args: object) -> jax.Array:
    """A square nonlinear map with an invertible banded Jacobian."""
    del args
    return 3.0 * point + point**2 + 0.5 * jnp.roll(point, 1) * point


SQUARE_POINT = jnp.linspace(0.5, 1.5, 5)
RIGHT_HAND_SIDE = jnp.arange(1.0, 6.0)


def dense_solve(point: jax.Array, transpose: bool = False) -> np.ndarray:
    jacobian = np.asarray(
        jax.jacfwd(lambda input_point: square_function(input_point, None))(point),
        dtype=np.float64,
    )
    if transpose:
        jacobian = jacobian.T
    return np.linalg.solve(jacobian, np.asarray(RIGHT_HAND_SIDE, dtype=np.float64))


def jvp_operator(point: jax.Array, tag: object) -> lx.FunctionLinearOperator:
    """A `lineax.FunctionLinearOperator` of the JVP of `square_function` at `point`."""

    def jvp_of_function(tangent: jax.Array) -> jax.Array:
        _, output_tangent = jax.jvp(
            lambda input_point: square_function(input_point, None), (point,), (tangent,)
        )
        return output_tangent

    return lx.FunctionLinearOperator(
        jvp_of_function, jax.ShapeDtypeStruct(point.shape, point.dtype), tags=tag
    )


def test_jacobian_operator_solve_matches_numpy(solver) -> None:
    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)
    point = SQUARE_POINT.astype(jnp.float64)
    operator = lx.JacobianLinearOperator(square_function, point, tags=tag)
    solution, _ = splx.linear_solve(
        operator, RIGHT_HAND_SIDE.astype(jnp.float64), solver
    )
    np.testing.assert_allclose(solution.value, dense_solve(point), atol=1e-6)


def test_function_operator_solve_matches_numpy(solver) -> None:
    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)
    point = SQUARE_POINT.astype(jnp.float64)
    solution, _ = splx.linear_solve(
        jvp_operator(point, tag), RIGHT_HAND_SIDE.astype(jnp.float64), solver
    )
    np.testing.assert_allclose(solution.value, dense_solve(point), atol=1e-6)


def test_transposed_operator_solve_matches_numpy(solver) -> None:
    """The transpose carries the transposed tag, so it solves with `A^T`."""
    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)
    point = SQUARE_POINT.astype(jnp.float64)
    operator = lx.JacobianLinearOperator(square_function, point, tags=tag)
    solution, _ = splx.linear_solve(
        operator.transpose(), RIGHT_HAND_SIDE.astype(jnp.float64), solver
    )
    np.testing.assert_allclose(
        solution.value, dense_solve(point, transpose=True), atol=1e-6
    )


@pytest.mark.parametrize("source", ["tag", "operator"])
def test_init_symbolic_then_update_across_points(solver, source: str) -> None:
    """`init_symbolic` reads the pattern from a tag or a tagged operator, and `update`
    folds in operators at new points."""
    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)
    first_point = SQUARE_POINT.astype(jnp.float64)
    first = lx.JacobianLinearOperator(square_function, first_point, tags=tag)
    state = solver.init_symbolic(tag if source == "tag" else first)
    for point in (first_point, first_point + 0.25):
        operator = lx.JacobianLinearOperator(square_function, point, tags=tag)
        solution, state = splx.linear_solve(
            operator, RIGHT_HAND_SIDE.astype(jnp.float64), solver, state=state
        )
        np.testing.assert_allclose(solution.value, dense_solve(point), atol=1e-6)
    if hasattr(state, "release"):
        state.release()


@pytest.mark.cpu_only
def test_klu_reuses_the_analysis_across_points(
    enable_x64: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operators that share a tag reuse one symbolic analysis."""
    import klujax

    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)
    analyze_calls: list[bool] = []
    original_analyze = klujax.analyze

    def spy_analyze(*args, **kwargs):
        analyze_calls.append(True)
        return original_analyze(*args, **kwargs)

    monkeypatch.setattr(klujax, "analyze", spy_analyze)
    state = None
    for shift in (0.0, 0.25, 0.5):
        point = SQUARE_POINT.astype(jnp.float64) + shift
        operator = lx.JacobianLinearOperator(square_function, point, tags=tag)
        solution, state = splx.linear_solve(
            operator, RIGHT_HAND_SIDE.astype(jnp.float64), KLU(), state=state
        )
        np.testing.assert_allclose(solution.value, dense_solve(point), atol=1e-6)
    assert len(analyze_calls) == 1


def test_untagged_lineax_operator_is_rejected(solver) -> None:
    operator = lx.JacobianLinearOperator(square_function, SQUARE_POINT)
    with pytest.raises(TypeError, match="sparsity-pattern tag"):
        solver.init(operator, {})


def test_gradient_through_a_tagged_jacobian_solve(solver) -> None:
    """The custom JVP of `splineax.linear_solve` carries the function inside a stored
    `lineax.JacobianLinearOperator` through `jax.grad` and `jax.jit`."""
    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)
    right_hand_side = RIGHT_HAND_SIDE.astype(jnp.float64)

    def loss(point: jax.Array, tagged_solver) -> jax.Array:
        operator = lx.JacobianLinearOperator(square_function, point, tags=tag)
        solution, _ = splx.linear_solve(operator, right_hand_side, tagged_solver)
        return jnp.sum(solution.value**2)

    def dense_loss(point: jax.Array) -> jax.Array:
        jacobian = jax.jacfwd(lambda input_point: square_function(input_point, None))(
            point
        )
        return jnp.sum(jnp.linalg.solve(jacobian, right_hand_side) ** 2)

    point = SQUARE_POINT.astype(jnp.float64)
    gradient = jax.jit(jax.grad(loss), static_argnums=1)(point, solver)
    np.testing.assert_allclose(gradient, jax.grad(dense_loss)(point), rtol=1e-6)


def test_state_holds_only_arrays(solver) -> None:
    """A state built from a tagged `lineax.JacobianLinearOperator` stores the sparse matrix
    it materialises to. The function inside the Jacobian operator is not an array, so a
    state that kept it could not be carried through a loop."""
    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)
    point = SQUARE_POINT.astype(jnp.float64)
    operator = lx.JacobianLinearOperator(square_function, point, tags=tag)
    state = solver.init(operator, {})
    assert all(eqx.is_array(leaf) for leaf in jax.tree.leaves(state))
    if hasattr(state, "release"):
        state.release()


def tagged_operator_at(
    point: jax.Array, operator_kind: str, tag: object
) -> lx.AbstractLinearOperator:
    """Build a tagged Jacobian operator, or a function operator of its linearisation."""
    if operator_kind == "jacobian":
        return lx.JacobianLinearOperator(square_function, point, tags=tag)
    _, linearised = jax.linearize(
        lambda input_point: square_function(input_point, None), point
    )
    return lx.FunctionLinearOperator(
        linearised, jax.eval_shape(lambda: point), tags=tag
    )


def run_loop(loop_kind: str, step, initial: jax.Array) -> jax.Array:
    """Run `step` four times from `initial`, in a `scan` or a `while_loop`."""
    if loop_kind == "scan":
        return jax.lax.scan(
            lambda value, _: (step(value), None), initial, None, length=4
        )[0]
    return jax.lax.while_loop(
        lambda carry: carry[0] < 4,
        lambda carry: (carry[0] + 1, step(carry[1])),
        (0, initial),
    )[1]


@pytest.mark.parametrize("loop_kind", ["scan", "while_loop"])
@pytest.mark.parametrize("operator_kind", ["jacobian", "function"])
def test_transform_threads_a_tagged_operator_through_a_loop(
    solver, operator_kind: str, loop_kind: str
) -> None:
    """`stateful_solve_transform` carries a solver state through a loop whose solves use a
    tagged Jacobian or function operator, and the result matches the plain loop."""
    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)

    def newton_step(point: jax.Array) -> jax.Array:
        operator = tagged_operator_at(point, operator_kind, tag)
        step = lx.linear_solve(operator, square_function(point, None), solver).value
        return point - step

    def newton(initial: jax.Array) -> jax.Array:
        return run_loop(loop_kind, newton_step, initial)

    initial = SQUARE_POINT.astype(jnp.float64)
    expected = newton(initial)
    threaded = splx.stateful_solve_transform(newton)
    np.testing.assert_allclose(threaded(initial), expected, rtol=1e-10)
    np.testing.assert_allclose(jax.jit(threaded)(initial), expected, rtol=1e-10)


@pytest.mark.cpu_only
def test_transform_threads_a_tagged_operator_through_a_hybrid_loop(
    enable_x64: None,
) -> None:
    """The state of `HybridDirectIterative` keeps the operator it solves, so it stores the
    sparse equivalent of a tagged operator as well."""
    tag = sparsity_coloring_tag(square_function, SQUARE_POINT)
    hybrid = splx.HybridDirectIterative(
        KLU(), splx.RichardsonOptions(max_steps_stale=8)
    )

    def newton_step(point: jax.Array) -> jax.Array:
        operator = tagged_operator_at(point, "jacobian", tag)
        step = lx.linear_solve(operator, square_function(point, None), hybrid).value
        return point - step

    def newton(initial: jax.Array) -> jax.Array:
        return run_loop("scan", newton_step, initial)

    initial = SQUARE_POINT.astype(jnp.float64)
    # The threaded state reuses a factorization that is stale, and refines it iteratively.
    # The plain loop factorizes every step, so the two differ slightly at the tolerance.
    np.testing.assert_allclose(
        jax.jit(splx.stateful_solve_transform(newton))(initial),
        newton(initial),
        atol=1e-10,
    )
