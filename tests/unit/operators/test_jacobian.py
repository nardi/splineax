"""Test suite for the sparse Jacobian operator and its coloring wrappers.

The classes under test are `SparseJacobianLinearOperator`, the function-agnostic
`JacobianColoring`, and the function-bound `SparseJacobianLinearOperatorColoring`.

The reference for correctness is always the dense Jacobian computed with
`jax.jacfwd`. Two reference functions are used: an elementwise map (whose Jacobian
is diagonal, the simplest sparsity) and a banded coupling (whose Jacobian needs a
non-trivial coloring). Beyond numerical agreement, the suite checks the properties
that make the operator practical: jit cache stability across evaluation points and
across freshly regenerated colorings, sparse materialisation through
`lineax.materialise` and `lineax.linearise`, and end-to-end solves through every
splineax sparse solver, including the KLU symbolic factorization reuse path and a
coloring passed as an argument into a jitted solve.
"""

import asdex
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from splineax import (
    KLU,
    AutoSparseLinearSolver,
    BCOOLinearOperator,
    JacobianColoring,
    SparseJacobianLinearOperator,
    SparseJacobianLinearOperatorColoring,
    Spsolve,
)


def elementwise_function(x: jax.Array, args: object) -> jax.Array:
    """An elementwise map, whose Jacobian is diagonal."""
    del args
    return jnp.sin(x) + x**2


def banded_function(x: jax.Array, args: object) -> jax.Array:
    """A nearest-neighbour coupling, whose (rectangular) Jacobian is banded and
    needs more than one color."""
    del args
    return jnp.diff(x) * x[:-1] + jnp.cos(x[1:])


def square_function(x: jax.Array, args: object) -> jax.Array:
    """A square nonlinear map with an invertible banded Jacobian, for solver
    integration tests."""
    del args
    return 3.0 * x + x**2 + 0.5 * jnp.roll(x, 1) * x


EVALUATION_POINT = jnp.linspace(1.0, 2.0, 6)
SQUARE_POINT = jnp.linspace(0.5, 1.5, 5)
RIGHT_HAND_SIDE = jnp.arange(1.0, 6.0)


def dense_jacobian(fn, x: jax.Array) -> jax.Array:
    return jax.jacfwd(lambda point: fn(point, None))(x)


@pytest.mark.parametrize("fn", [elementwise_function, banded_function])
def test_materialisation_matches_dense_jacobian(fn) -> None:
    """`as_matrix` and `as_bcoo().todense()` must both reproduce the dense
    Jacobian, since every solver consumes the operator through these paths."""
    operator = SparseJacobianLinearOperator(fn, EVALUATION_POINT)
    expected = dense_jacobian(fn, EVALUATION_POINT)
    assert jnp.allclose(operator.as_matrix(), expected)
    assert jnp.allclose(operator.as_bcoo().todense(), expected)


@pytest.mark.parametrize("fn", [elementwise_function, banded_function])
def test_mv_matches_dense_jacobian(fn) -> None:
    """`mv` must equal a dense Jacobian-vector product, and must remain correct
    under jit, which is how lineax's solvers invoke it."""
    operator = SparseJacobianLinearOperator(fn, EVALUATION_POINT)
    expected = dense_jacobian(fn, EVALUATION_POINT)
    vector = jnp.arange(1.0, 7.0)
    assert jnp.allclose(operator.mv(vector), expected @ vector)
    jitted_mv = eqx.filter_jit(lambda op, v: op.mv(v))
    assert jnp.allclose(jitted_mv(operator, vector), expected @ vector)


def test_transpose_stays_sparse_and_matches_dense() -> None:
    """`transpose()` must return another `SparseJacobianLinearOperator` (not a
    dense fallback) whose products and materialisation match the dense transpose,
    and transposing twice must recover the original behaviour."""
    operator = SparseJacobianLinearOperator(banded_function, EVALUATION_POINT)
    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    transposed = operator.transpose()

    assert isinstance(transposed, SparseJacobianLinearOperator)
    covector = jnp.arange(1.0, 6.0)
    assert jnp.allclose(transposed.mv(covector), expected.T @ covector)
    assert jnp.allclose(transposed.as_matrix(), expected.T)
    assert jnp.allclose(transposed.as_bcoo().todense(), expected.T)
    assert jnp.allclose(transposed.transpose().as_matrix(), expected)


def test_structures_swap_under_transpose() -> None:
    """The banded function maps 6 inputs to 5 outputs, so the structures are
    distinguishable and must swap under transposition."""
    operator = SparseJacobianLinearOperator(banded_function, EVALUATION_POINT)
    transposed = operator.transpose()
    dtype = EVALUATION_POINT.dtype
    assert operator.in_structure() == jax.ShapeDtypeStruct((6,), dtype)
    assert operator.out_structure() == jax.ShapeDtypeStruct((5,), dtype)
    assert transposed.in_structure() == jax.ShapeDtypeStruct((5,), dtype)
    assert transposed.out_structure() == jax.ShapeDtypeStruct((6,), dtype)


def test_construction_paths_agree() -> None:
    """Automatic detection, a caller-supplied sparsity pattern, and a
    caller-supplied coloring must all produce operators with identical Jacobians,
    since they are alternative entry points to the same precomputation."""

    def function_of_point(point):
        return banded_function(point, None)

    from_detection = SparseJacobianLinearOperator(banded_function, EVALUATION_POINT)
    known_sparsity = asdex.jacobian_sparsity(function_of_point, EVALUATION_POINT)
    from_sparsity = SparseJacobianLinearOperator(
        banded_function, EVALUATION_POINT, sparsity=known_sparsity
    )
    known_coloring = asdex.jacobian_coloring(function_of_point, EVALUATION_POINT)
    from_coloring = SparseJacobianLinearOperator(
        banded_function, EVALUATION_POINT, coloring=known_coloring
    )

    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    assert jnp.allclose(from_detection.as_matrix(), expected)
    assert jnp.allclose(from_sparsity.as_matrix(), expected)
    assert jnp.allclose(from_coloring.as_matrix(), expected)


@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_mode_is_forwarded(mode) -> None:
    """The `mode` argument selects column versus row coloring in asdex. Both
    modes must be accepted and give the correct Jacobian."""
    operator = SparseJacobianLinearOperator(
        banded_function, EVALUATION_POINT, mode=mode
    )
    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    assert jnp.allclose(operator.as_matrix(), expected)


def test_coloring_object_matches_direct_construction() -> None:
    """`SparseJacobianLinearOperatorColoring.detect(...).operator_at(x)` must behave
    identically to constructing the operator directly, since it is only a
    precomputation cache."""
    coloring = SparseJacobianLinearOperatorColoring.detect(
        banded_function, EVALUATION_POINT
    )
    from_coloring_object = coloring.operator_at(EVALUATION_POINT)
    direct = SparseJacobianLinearOperator(banded_function, EVALUATION_POINT)
    assert jnp.allclose(from_coloring_object.as_matrix(), direct.as_matrix())

    other_point = EVALUATION_POINT + 1.0
    assert jnp.allclose(
        coloring.operator_at(other_point).as_matrix(),
        dense_jacobian(banded_function, other_point),
    )


def test_coloring_object_from_sparsity_and_abstract_point() -> None:
    """`from_sparsity` must skip detection but still color and key correctly, and
    both constructors must accept a `jax.ShapeDtypeStruct` in place of a concrete
    point, since only shapes and dtypes matter structurally."""
    abstract_point = jax.ShapeDtypeStruct(
        EVALUATION_POINT.shape, EVALUATION_POINT.dtype
    )
    known_sparsity = asdex.jacobian_sparsity(
        lambda point: banded_function(point, None), EVALUATION_POINT
    )
    coloring = SparseJacobianLinearOperatorColoring.from_sparsity(
        banded_function, abstract_point, known_sparsity
    )
    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    assert jnp.allclose(coloring.operator_at(EVALUATION_POINT).as_matrix(), expected)

    detected = SparseJacobianLinearOperatorColoring.detect(
        banded_function, abstract_point
    )
    assert jnp.allclose(detected.operator_at(EVALUATION_POINT).as_matrix(), expected)


def test_jit_cache_is_stable_across_points_and_transposes() -> None:
    """Operators built from one `SparseJacobianLinearOperatorColoring` at different
    points, and an operator transposed twice, must share a pytree structure, so a
    jitted function accepting them compiles exactly once. This is the property that
    makes the precomputed coloring worthwhile inside Newton-style loops."""
    coloring = SparseJacobianLinearOperatorColoring.detect(
        banded_function, EVALUATION_POINT
    )
    trace_log: list[bool] = []

    @eqx.filter_jit
    def apply(operator, vector):
        trace_log.append(True)
        return operator.mv(vector)

    vector = jnp.arange(1.0, 7.0)
    apply(coloring.operator_at(EVALUATION_POINT), vector)
    apply(coloring.operator_at(EVALUATION_POINT + 1.0), vector)
    assert len(trace_log) == 1, "a second operator from the same coloring retraced"

    operator = coloring.operator_at(EVALUATION_POINT)
    apply(operator.transpose().transpose(), vector)
    assert len(trace_log) == 1, "a double transpose changed the pytree structure"


def test_materialise_returns_bcoo_operator() -> None:
    """`lineax.materialise` is how the sparse solvers consume the operator, so it
    must produce a `BCOOLinearOperator` holding the correct Jacobian."""
    operator = SparseJacobianLinearOperator(banded_function, EVALUATION_POINT)
    materialised = lx.materialise(operator)
    assert isinstance(materialised, BCOOLinearOperator)
    assert jnp.allclose(
        materialised.as_matrix(), dense_jacobian(banded_function, EVALUATION_POINT)
    )


def test_linearise_caches_primal_and_stays_sparse() -> None:
    """`lineax.linearise` must return another `SparseJacobianLinearOperator` (so
    it can still be sparsely materialised) with identical products and matrix."""
    operator = SparseJacobianLinearOperator(banded_function, EVALUATION_POINT)
    linearised = lx.linearise(operator)
    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    assert isinstance(linearised, SparseJacobianLinearOperator)
    vector = jnp.arange(1.0, 7.0)
    assert jnp.allclose(linearised.mv(vector), expected @ vector)
    assert jnp.allclose(linearised.as_matrix(), expected)


def test_tags_drive_property_predicates() -> None:
    """Tags must flow through to the lineax structural predicates, including on
    the transpose, and an untagged operator must report no properties."""
    plain = SparseJacobianLinearOperator(elementwise_function, EVALUATION_POINT)
    assert lx.is_symmetric(plain) is False
    assert lx.is_diagonal(plain) is False
    assert lx.is_positive_semidefinite(plain) is False

    tagged = SparseJacobianLinearOperator(
        elementwise_function, EVALUATION_POINT, tags=lx.symmetric_tag
    )
    assert lx.is_symmetric(tagged) is True
    # A symmetric operator transposes to itself.
    assert tagged.transpose() is tagged


def test_complex_point_is_rejected() -> None:
    """The operator is scoped to real dtypes (`jax.vjp` is only the true
    transpose for holomorphic functions), so a complex point must fail loudly."""
    with pytest.raises(TypeError, match="real dtypes"):
        SparseJacobianLinearOperator(elementwise_function, jnp.array([1.0 + 2.0j, 3.0]))


def test_non_1d_point_is_rejected() -> None:
    """The operator models a two-dimensional Jacobian of a one-dimensional map,
    so a matrix-valued point must be rejected."""
    with pytest.raises(ValueError, match="one-dimensional"):
        SparseJacobianLinearOperator(elementwise_function, jnp.ones((2, 3)))


def test_conflicting_precomputation_arguments_are_rejected() -> None:
    """Passing both `coloring` and `sparsity`, or a `coloring` of the wrong type,
    must raise instead of silently ignoring one of them."""

    def function_of_point(point):
        return banded_function(point, None)

    known_sparsity = asdex.jacobian_sparsity(function_of_point, EVALUATION_POINT)
    known_coloring = asdex.jacobian_coloring(function_of_point, EVALUATION_POINT)
    with pytest.raises(TypeError, match="at most one"):
        SparseJacobianLinearOperator(
            banded_function,
            EVALUATION_POINT,
            sparsity=known_sparsity,
            coloring=known_coloring,
        )
    with pytest.raises(TypeError, match="where `coloring` must be"):
        SparseJacobianLinearOperator(
            banded_function,
            EVALUATION_POINT,
            coloring=known_sparsity,  # type: ignore
        )


@pytest.mark.parametrize(
    "solver", [KLU(), Spsolve(), AutoSparseLinearSolver()], ids=type
)
def test_linear_solve_matches_numpy(solver, enable_x64: None) -> None:
    """End-to-end integration proof: handing the Jacobian operator straight to
    each splineax solver must reproduce the dense solve. This exercises the
    `materialise` recursion inside every solver's `init`."""
    operator = SparseJacobianLinearOperator(square_function, SQUARE_POINT)
    expected = np.linalg.solve(
        np.asarray(dense_jacobian(square_function, SQUARE_POINT), dtype=np.float64),
        np.asarray(RIGHT_HAND_SIDE, dtype=np.float64),
    )
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert np.allclose(np.asarray(solution), expected, atol=1e-5)


def test_factorize_symbolic_round_trip(enable_x64: None) -> None:
    """`KLU.factorize_symbolic` must accept the operator, a bound
    `SparseJacobianLinearOperatorColoring`, and a bare `JacobianColoring`, deriving
    the indices host-side from the stored sparsity pattern in each case. Solving
    through the resulting scope must match the dense solve, which fails if the
    pattern's index order ever disagrees with the order `asdex` emits when the
    Jacobian is later materialised."""
    solver = KLU()
    operator_coloring = SparseJacobianLinearOperatorColoring.detect(
        square_function, SQUARE_POINT
    )
    operator = operator_coloring.operator_at(SQUARE_POINT)
    bare_coloring = JacobianColoring.detect(square_function, SQUARE_POINT)
    expected = np.linalg.solve(
        np.asarray(dense_jacobian(square_function, SQUARE_POINT), dtype=np.float64),
        np.asarray(RIGHT_HAND_SIDE, dtype=np.float64),
    )

    for sparsity_source in (operator, operator_coloring, bare_coloring):
        with solver.factorize_symbolic(sparsity_source) as scope:
            state = scope.init(operator)
            solution = lx.linear_solve(
                operator, RIGHT_HAND_SIDE, solver=solver, state=state
            ).value
        assert np.allclose(np.asarray(solution), expected, atol=1e-5)


def test_jacobian_coloring_creation_paths_match_dense() -> None:
    """A `JacobianColoring` built by detection or from a known sparsity pattern must,
    through either entry point (the operator's `coloring` argument, or binding with
    `from_jacobian_coloring` and calling `operator_at`), reproduce the dense
    Jacobian."""
    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    known_sparsity = asdex.jacobian_sparsity(
        lambda point: banded_function(point, None), EVALUATION_POINT
    )

    detected_coloring = JacobianColoring.detect(banded_function, EVALUATION_POINT)
    from_sparsity_coloring = JacobianColoring.from_sparsity(known_sparsity)

    for jacobian_coloring in (detected_coloring, from_sparsity_coloring):
        # Entry point one: hand the coloring straight to the operator.
        direct_operator = SparseJacobianLinearOperator(
            banded_function, EVALUATION_POINT, coloring=jacobian_coloring
        )
        assert jnp.allclose(direct_operator.as_matrix(), expected)

        # Entry point two: bind the coloring to the function, then build at a point.
        bound_coloring = SparseJacobianLinearOperatorColoring.from_jacobian_coloring(
            jacobian_coloring, banded_function, EVALUATION_POINT
        )
        assert jnp.allclose(
            bound_coloring.operator_at(EVALUATION_POINT).as_matrix(), expected
        )


def test_from_jacobian_coloring_matches_direct_construction() -> None:
    """Binding a detected `JacobianColoring` to a function with
    `from_jacobian_coloring` and building an operator at a point must match direct
    construction, since it only reuses the precomputed coloring."""
    jacobian_coloring = JacobianColoring.detect(banded_function, EVALUATION_POINT)
    bound = SparseJacobianLinearOperatorColoring.from_jacobian_coloring(
        jacobian_coloring, banded_function, EVALUATION_POINT
    )
    direct = SparseJacobianLinearOperator(banded_function, EVALUATION_POINT)
    assert jnp.allclose(
        bound.operator_at(EVALUATION_POINT).as_matrix(), direct.as_matrix()
    )


def test_jacobian_coloring_through_jit_and_solver(enable_x64: None) -> None:
    """A `JacobianColoring` is passed as an argument into a
    jitted function, which builds a `SparseJacobianLinearOperator` from it and
    solves with KLU. Because `asdex.ColoredPattern` is a pytree whose treedef
    depends only on the sparsity structure, regenerating the coloring from
    scratch (a fresh object with fresh arrays) and calling again must not
    retrace the jitted function."""
    known_sparsity = asdex.jacobian_sparsity(
        lambda point: square_function(point, None), SQUARE_POINT
    )
    expected = np.linalg.solve(
        np.asarray(dense_jacobian(square_function, SQUARE_POINT), dtype=np.float64),
        np.asarray(RIGHT_HAND_SIDE, dtype=np.float64),
    )
    trace_log: list[bool] = []

    @eqx.filter_jit
    def solve(coloring, point, right_hand_side):
        trace_log.append(True)
        operator = SparseJacobianLinearOperator(
            square_function, point, coloring=coloring
        )
        return lx.linear_solve(operator, right_hand_side, solver=KLU()).value

    # First call, with a coloring built from the known sparsity pattern.
    first_coloring = JacobianColoring.from_sparsity(known_sparsity)
    first_solution = solve(first_coloring, SQUARE_POINT, RIGHT_HAND_SIDE)
    assert np.allclose(np.asarray(first_solution), expected, atol=1e-5)
    assert len(trace_log) == 1

    # Regenerate the coloring from scratch (a fresh object) and solve at a different
    # point. The freshly built coloring must reuse the compiled function, so the
    # trace count stays at one.
    other_point = SQUARE_POINT + 0.25
    other_expected = np.linalg.solve(
        np.asarray(dense_jacobian(square_function, other_point), dtype=np.float64),
        np.asarray(RIGHT_HAND_SIDE, dtype=np.float64),
    )
    regenerated_coloring = JacobianColoring.from_sparsity(known_sparsity)
    regenerated_solution = solve(regenerated_coloring, other_point, RIGHT_HAND_SIDE)
    assert np.allclose(np.asarray(regenerated_solution), other_expected, atol=1e-5)
    assert len(trace_log) == 1, "a regenerated coloring retraced the jitted solve"

    # A coloring produced by detection rather than from a known pattern must share the
    # same treedef too, so it also reuses the compiled function.
    detected_coloring = JacobianColoring.detect(square_function, SQUARE_POINT)
    detected_solution = solve(detected_coloring, SQUARE_POINT, RIGHT_HAND_SIDE)
    assert np.allclose(np.asarray(detected_solution), expected, atol=1e-5)
    assert len(trace_log) == 1, "a detection-built coloring retraced the jitted solve"
