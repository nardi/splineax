"""Test suite for the sparse function operator.

The class under test is `SparseFunctionLinearOperator`, the sparse analogue of
`lineax.FunctionLinearOperator`. Its reference is twofold. Numerically it is the dense
Jacobian computed with `jax.jacfwd`, as in
[the Jacobian operator tests](test_jacobian.py). Structurally it is
`SparseJacobianLinearOperator` itself: wrapping a linearised function here must give
the same operator as taking the Jacobian of that function directly, which is the whole
point of having the pair. That equivalence is checked head-on in
`test_matches_the_jacobian_operator`, and the tests after it cover the operations the
solvers and lineax reach for.

The reference functions are shared with the Jacobian operator tests, and linearised
with `jax.linearize` to obtain the linear maps this operator wraps.
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
    SparseFunctionLinearOperator,
    SparseJacobianLinearOperator,
    Spsolve,
)

from .test_jacobian import (
    EVALUATION_POINT,
    RIGHT_HAND_SIDE,
    SQUARE_POINT,
    banded_function,
    dense_jacobian,
    elementwise_function,
    square_function,
)


def linearize(fn, x):
    """The linear map `d(fn)/dx` at `x`, and the structure of its input.

    This is the pairing `SparseFunctionLinearOperator` takes, and `jax.linearize` is
    how a caller most often comes by one.
    """
    _, jvp_of_fn = jax.linearize(lambda point: fn(point, None), x)
    return jvp_of_fn, jax.eval_shape(lambda: x)


def operator_for(fn, x, **kwargs) -> SparseFunctionLinearOperator:
    """The sparse function operator for the Jacobian of `fn` at `x`."""
    return SparseFunctionLinearOperator(*linearize(fn, x), **kwargs)


# Every partial derivative of `pytree_function` is nonzero here. At a point where one
# of them vanishes, linearising first folds it away and detects a (still correct, but)
# sparser pattern than differentiating the original function does, which would make the
# index comparison in `test_matches_the_jacobian_operator` fail for the right reason.
PYTREE_POINT = {
    "position": jnp.linspace(1.0, 2.0, 4),
    "velocity": jnp.linspace(1.0, 2.0, 4),
}


def pytree_function(state, args: object):
    """A coupled map between dictionaries, to check that pytree structures survive."""
    del args
    return {
        "position": state["velocity"] * 0.1 + state["position"] ** 2,
        "velocity": jnp.sin(state["velocity"]) + state["position"],
    }


@pytest.mark.parametrize(
    "fn, x",
    [
        (elementwise_function, EVALUATION_POINT),
        (banded_function, EVALUATION_POINT),
        (pytree_function, PYTREE_POINT),
    ],
    ids=["elementwise", "banded", "pytree"],
)
def test_matches_the_jacobian_operator(fn, x) -> None:
    """The load-bearing property of this operator: wrapping a linearised function must
    give exactly what taking the Jacobian directly gives. Everything is compared, since
    the two are meant to be interchangeable: the materialised matrix down to its sparse
    indices, the products both ways round, and the structures."""
    function_operator = operator_for(fn, x)
    jacobian_operator = SparseJacobianLinearOperator(fn, x)

    assert jnp.allclose(function_operator.as_matrix(), jacobian_operator.as_matrix())
    function_bcoo = function_operator.as_bcoo()
    jacobian_bcoo = jacobian_operator.as_bcoo()
    assert jnp.array_equal(function_bcoo.indices, jacobian_bcoo.indices)
    assert jnp.allclose(function_bcoo.data, jacobian_bcoo.data)

    assert function_operator.in_structure() == jacobian_operator.in_structure()
    assert function_operator.out_structure() == jacobian_operator.out_structure()

    vector = jax.tree.map(jnp.ones_like, function_operator.in_structure())
    assert _trees_allclose(function_operator.mv(vector), jacobian_operator.mv(vector))

    covector = jax.tree.map(jnp.ones_like, function_operator.out_structure())
    transposed = function_operator.transpose()
    assert jnp.allclose(
        transposed.as_matrix(), jacobian_operator.transpose().as_matrix()
    )
    assert _trees_allclose(
        transposed.mv(covector), jacobian_operator.transpose().mv(covector)
    )


def _trees_allclose(left, right) -> bool:
    return all(jax.tree.leaves(jax.tree.map(jnp.allclose, left, right)))


@pytest.mark.parametrize("fn", [elementwise_function, banded_function])
def test_materialisation_matches_dense_jacobian(fn) -> None:
    """`as_matrix` and `as_bcoo().todense()` must both reproduce the dense Jacobian of
    the function that was linearised, since every solver consumes the operator through
    these paths."""
    operator = operator_for(fn, EVALUATION_POINT)
    expected = dense_jacobian(fn, EVALUATION_POINT)
    assert jnp.allclose(operator.as_matrix(), expected)
    assert jnp.allclose(operator.as_bcoo().todense(), expected)


@pytest.mark.parametrize("fn", [elementwise_function, banded_function])
def test_mv_matches_dense_product(fn) -> None:
    """`mv` must equal a dense matrix-vector product, and must remain correct under
    jit, which is how lineax's solvers invoke it."""
    operator = operator_for(fn, EVALUATION_POINT)
    expected = dense_jacobian(fn, EVALUATION_POINT)
    vector = jnp.arange(1.0, 7.0)
    assert jnp.allclose(operator.mv(vector), expected @ vector)
    jitted_mv = eqx.filter_jit(lambda op, v: op.mv(v))
    assert jnp.allclose(jitted_mv(operator, vector), expected @ vector)


def test_transpose_stays_sparse_and_matches_dense() -> None:
    """`transpose()` must return another `SparseFunctionLinearOperator` (not a dense
    fallback) whose products and materialisation match the dense transpose, and
    transposing twice must recover the original behavior."""
    operator = operator_for(banded_function, EVALUATION_POINT)
    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    transposed = operator.transpose()

    assert isinstance(transposed, SparseFunctionLinearOperator)
    covector = jnp.arange(1.0, 6.0)
    assert jnp.allclose(transposed.mv(covector), expected.T @ covector)
    assert jnp.allclose(transposed.as_matrix(), expected.T)
    assert jnp.allclose(transposed.as_bcoo().todense(), expected.T)
    assert jnp.allclose(transposed.transpose().as_matrix(), expected)


def test_structures_swap_under_transpose() -> None:
    """The banded function maps 6 inputs to 5 outputs, so the structures are
    distinguishable and must swap under transposition."""
    operator = operator_for(banded_function, EVALUATION_POINT)
    transposed = operator.transpose()
    dtype = EVALUATION_POINT.dtype
    assert operator.in_structure() == jax.ShapeDtypeStruct((6,), dtype)
    assert operator.out_structure() == jax.ShapeDtypeStruct((5,), dtype)
    assert transposed.in_structure() == jax.ShapeDtypeStruct((5,), dtype)
    assert transposed.out_structure() == jax.ShapeDtypeStruct((6,), dtype)


def test_construction_paths_agree() -> None:
    """Automatic detection, a caller-supplied sparsity pattern, and a caller-supplied
    coloring must all produce operators with identical matrices, since they are
    alternative entry points to the same precomputation."""
    linear_fn, structure = linearize(banded_function, EVALUATION_POINT)
    zeros = jnp.zeros(EVALUATION_POINT.shape, EVALUATION_POINT.dtype)

    from_detection = SparseFunctionLinearOperator(linear_fn, structure)
    known_sparsity = asdex.jacobian_sparsity(linear_fn, zeros)
    from_sparsity = SparseFunctionLinearOperator(
        linear_fn, structure, sparsity=known_sparsity
    )
    known_coloring = asdex.jacobian_coloring(linear_fn, zeros)
    from_coloring = SparseFunctionLinearOperator(
        linear_fn, structure, coloring=known_coloring
    )
    wrapped_coloring = SparseFunctionLinearOperator(
        linear_fn, structure, coloring=JacobianColoring(known_coloring)
    )

    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    for operator in (from_detection, from_sparsity, from_coloring, wrapped_coloring):
        assert jnp.allclose(operator.as_matrix(), expected)


@pytest.mark.parametrize("mode", ["fwd", "bwd"])
def test_mode_is_forwarded(mode) -> None:
    """The `mode` argument selects column versus row coloring. Both modes must be
    accepted, must reach asdex under its own spelling, and must give the correct
    matrix."""
    operator = operator_for(banded_function, EVALUATION_POINT, mode=mode)
    expected = dense_jacobian(banded_function, EVALUATION_POINT)
    assert jnp.allclose(operator.as_matrix(), expected)
    assert JacobianColoring(operator.coloring).mode == mode


def test_pytree_structures_are_kept_while_the_matrix_stays_flat() -> None:
    """`in_structure`, `out_structure` and `mv` must speak in the pytrees the function
    actually has, while `as_bcoo` ravels to a plain matrix, which is what the solvers
    consume."""
    operator = operator_for(pytree_function, PYTREE_POINT)
    assert operator.in_structure() == jax.eval_shape(lambda: PYTREE_POINT)
    assert operator.as_bcoo().shape == (8, 8)

    tangent = jax.tree.map(jnp.ones_like, PYTREE_POINT)
    product = operator.mv(tangent)
    assert jax.tree.structure(product) == jax.tree.structure(PYTREE_POINT)


def test_jit_cache_is_stable_across_colorings_and_transposes() -> None:
    """Two operators over the same function and sparsity, and an operator transposed
    twice, must share a pytree structure, so a jitted function accepting them compiles
    exactly once. This is the property that makes carrying a coloring worthwhile.

    The operators are built through `from_function_operator`, which reuses the dense
    operator's already closure-converted function. Closure-converting the same function
    twice gives two jaxprs that compare unequal, and those are static, so going through
    the plain constructor twice would retrace for a reason that has nothing to do with
    the coloring.
    """
    linear_fn, structure = linearize(banded_function, EVALUATION_POINT)
    dense = lx.FunctionLinearOperator(linear_fn, structure)
    coloring = JacobianColoring.detect(banded_function, EVALUATION_POINT)
    trace_log: list[bool] = []

    @eqx.filter_jit
    def apply(operator, vector):
        trace_log.append(True)
        return operator.mv(vector)

    vector = jnp.arange(1.0, 7.0)
    operator = SparseFunctionLinearOperator.from_function_operator(
        dense, coloring=coloring
    )
    apply(operator, vector)

    # A coloring regenerated from scratch is a fresh object with fresh arrays, but the
    # same sparsity, so it must flatten to the same treedef.
    regenerated = JacobianColoring.detect(banded_function, EVALUATION_POINT)
    apply(
        SparseFunctionLinearOperator.from_function_operator(
            dense, coloring=regenerated
        ),
        vector,
    )
    assert len(trace_log) == 1, "a regenerated coloring retraced"

    apply(operator.transpose().transpose(), vector)
    assert len(trace_log) == 1, "a double transpose changed the pytree structure"


def test_materialise_returns_bcoo_operator() -> None:
    """`lineax.materialise` must produce a `BCOOLinearOperator` holding the correct
    matrix, for callers that want a concrete sparse operator."""
    operator = operator_for(banded_function, EVALUATION_POINT)
    materialised = lx.materialise(operator)
    assert isinstance(materialised, BCOOLinearOperator)
    assert jnp.allclose(
        materialised.as_matrix(), dense_jacobian(banded_function, EVALUATION_POINT)
    )


def test_materialise_refuses_to_drop_the_structure() -> None:
    """A `BCOOLinearOperator` carries only a matrix, so materialising a pytree operator
    would silently discard the structures the caller passed in. That must raise instead,
    pointing at `as_bcoo`."""
    operator = operator_for(pytree_function, PYTREE_POINT)
    with pytest.raises(
        ValueError, match="one-dimensional array in- and out-structures"
    ):
        lx.materialise(operator)


def test_linearise_is_the_identity() -> None:
    """The map is already linear, so there is no primal pass for `lineax.linearise` to
    cache. lineax's `linearise` is likewise the identity on a
    `lineax.FunctionLinearOperator`."""
    operator = operator_for(banded_function, EVALUATION_POINT)
    assert lx.linearise(operator) is operator


@pytest.mark.parametrize("transposed", [False, True], ids=["forward", "transposed"])
def test_linearising_a_jacobian_operator_gives_this_operator(transposed) -> None:
    """`lineax.linearise` on a `SparseJacobianLinearOperator` must hand back a
    `SparseFunctionLinearOperator` representing the same matrix, mirroring what lineax
    does with the dense pair. The transposed case is the one that can go wrong, since
    the linearised map takes the forward input whichever way the operator faces."""
    operator = SparseJacobianLinearOperator(banded_function, EVALUATION_POINT)
    if transposed:
        operator = operator.transpose()
    linearised = lx.linearise(operator)

    assert isinstance(linearised, SparseFunctionLinearOperator)
    assert jnp.allclose(linearised.as_matrix(), operator.as_matrix())
    assert linearised.in_structure() == operator.in_structure()
    assert linearised.out_structure() == operator.out_structure()
    vector = jax.tree.map(jnp.ones_like, operator.in_structure())
    assert jnp.allclose(linearised.mv(vector), operator.mv(vector))


def test_tags_drive_property_predicates() -> None:
    """Tags must flow through to the lineax structural predicates, including on the
    transpose, and an untagged operator must report no properties."""
    linear_fn, structure = linearize(elementwise_function, EVALUATION_POINT)
    plain = SparseFunctionLinearOperator(linear_fn, structure)
    assert lx.is_symmetric(plain) is False
    assert lx.is_diagonal(plain) is False
    assert lx.is_positive_semidefinite(plain) is False

    tagged = SparseFunctionLinearOperator(linear_fn, structure, tags=lx.symmetric_tag)
    assert lx.is_symmetric(tagged) is True
    # A symmetric operator transposes to itself.
    assert tagged.transpose() is tagged


def test_complex_structure_is_rejected() -> None:
    """A complex input structure must fail loudly, and at construction rather than at
    the first materialisation, since asdex colors real-valued matrices."""
    with pytest.raises(TypeError, match="real dtypes"):
        SparseFunctionLinearOperator(
            lambda vector: vector, jax.ShapeDtypeStruct((4,), jnp.complex64)
        )


def test_conflicting_precomputation_arguments_are_rejected() -> None:
    """`sparsity` and `coloring` are alternative shortcuts through the same
    precomputation, so passing both is a mistake worth reporting, as is passing
    something that is neither kind of coloring."""
    linear_fn, structure = linearize(banded_function, EVALUATION_POINT)
    coloring = JacobianColoring.detect(banded_function, EVALUATION_POINT)
    with pytest.raises(TypeError, match="at most one"):
        SparseFunctionLinearOperator(
            linear_fn, structure, sparsity=coloring.sparsity, coloring=coloring
        )
    with pytest.raises(TypeError, match="where `coloring` must be"):
        SparseFunctionLinearOperator(linear_fn, structure, coloring="not a coloring")  # ty: ignore[invalid-argument-type]


def test_from_function_operator_matches_the_dense_operator() -> None:
    """Converting a `lineax.FunctionLinearOperator` must keep its function, structure
    and tags, so the two operators represent the same matrix. Only how it is
    materialised differs."""
    linear_fn, structure = linearize(square_function, SQUARE_POINT)
    dense = lx.FunctionLinearOperator(linear_fn, structure, tags=lx.symmetric_tag)
    converted = SparseFunctionLinearOperator.from_function_operator(dense)

    assert jnp.allclose(converted.as_matrix(), dense.as_matrix())
    assert converted.in_structure() == dense.in_structure()
    assert converted.out_structure() == dense.out_structure()
    assert lx.is_symmetric(converted) is True


def test_from_function_operator_passes_precomputation_arguments_through() -> None:
    """The `sparsity`, `coloring` and `mode` arguments of the constructor must be
    available on the conversion too, so a caller who already knows the pattern does not
    pay for detection."""
    linear_fn, structure = linearize(banded_function, EVALUATION_POINT)
    dense = lx.FunctionLinearOperator(linear_fn, structure)
    coloring = JacobianColoring.detect(banded_function, EVALUATION_POINT)
    expected = dense_jacobian(banded_function, EVALUATION_POINT)

    from_coloring = SparseFunctionLinearOperator.from_function_operator(
        dense, coloring=coloring
    )
    assert jnp.allclose(from_coloring.as_matrix(), expected)

    from_sparsity = SparseFunctionLinearOperator.from_function_operator(
        dense, sparsity=coloring.sparsity, mode="bwd"
    )
    assert jnp.allclose(from_sparsity.as_matrix(), expected)
    assert JacobianColoring(from_sparsity.coloring).mode == "bwd"


@pytest.mark.parametrize(
    "solver", [KLU(), Spsolve(), AutoSparseLinearSolver()], ids=type
)
def test_linear_solve_matches_numpy(solver, enable_x64: None) -> None:
    """End-to-end integration proof: handing the function operator straight to each
    splineax solver must reproduce the dense solve. This exercises the `as_bcoo` branch
    inside every solver's `init`."""
    operator = operator_for(square_function, SQUARE_POINT)
    expected = np.linalg.solve(
        np.asarray(dense_jacobian(square_function, SQUARE_POINT), dtype=np.float64),
        np.asarray(RIGHT_HAND_SIDE, dtype=np.float64),
    )
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert np.allclose(np.asarray(solution), expected, atol=1e-5)


def test_transposed_solve_matches_numpy(enable_x64: None) -> None:
    """Solving with a transposed operator exercises the branch in
    `KLU.factorize_symbolic` that swaps the stored pattern's rows and columns. If that
    swap disagreed with the order asdex emits the values in, the solve would silently
    return the wrong answer rather than fail."""
    operator = operator_for(square_function, SQUARE_POINT).transpose()
    expected = np.linalg.solve(
        np.asarray(dense_jacobian(square_function, SQUARE_POINT), dtype=np.float64).T,
        np.asarray(RIGHT_HAND_SIDE, dtype=np.float64),
    )
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=KLU()).value
    assert np.allclose(np.asarray(solution), expected, atol=1e-5)
