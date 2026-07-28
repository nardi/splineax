"""Sparse function operator backed by `asdex` sparsity detection and coloring.

`lineax.FunctionLinearOperator` wraps an already-linear function into an operator and
materialises it densely, one column per input. The operator here is its sparse
analogue, and stands to it exactly as
[`splineax.SparseJacobianLinearOperator`][] stands to
`lineax.JacobianLinearOperator`: the matrix the linear map represents has its sparsity
pattern detected and colored once (by `asdex`), so materialising it costs one
evaluation per color rather than one per column or row.

The two operators are two views of the same thing, and they agree on it. Linearising a
function and wrapping the result here gives the same operator as taking its Jacobian
directly, so `SparseFunctionLinearOperator(jvp_of_fn, structure)` matches
`SparseJacobianLinearOperator(fn, x)` entry for entry. That is also what
`lineax.linearise` on a sparse Jacobian operator returns, mirroring what lineax's
`linearise` does with the dense pair.

A linear map has no distinguished evaluation point, since its Jacobian is the same
constant matrix everywhere. Detection and materialisation therefore trace at the zeros
of the input structure, and the operator carries a structure rather than a point.
"""

from collections.abc import Callable
from typing import cast

import asdex
import equinox as eqx
import equinox.internal as eqxi
import jax
import numpy as np
from asdex import ColoredPattern, SparsityPattern
from jax.experimental.sparse import BCOO
from jaxtyping import Array, Inexact, PyTree
from lineax import (
    AbstractLinearOperator,
    FunctionLinearOperator,
    is_symmetric,
    linearise,
    materialise,
)
from lineax._operator import _frozenset, _inexact_structure, strip_weak_dtype
from lineax._tags import transpose_tags

from ._bcoo import BCOOLinearOperator
from ._coloring import (
    JacobianColoring,
    JacobianMode,
    asdex_mode,
    check_real_dtypes,
    check_structures_survive_materialisation,
    flatten_map,
    register_ad_operator,
    zero_point,
)


class SparseFunctionLinearOperator(AbstractLinearOperator):
    """Wraps a *linear* function `fn: X -> Y` into a linear operator, kept sparse.

    The matrix the linear map represents has its sparsity pattern and a matching
    coloring determined once at construction (via `asdex`), so materialising it costs
    one evaluation of `fn` per color rather than one per column or row. Materialise it
    with `as_bcoo` or `lineax.materialise` (sparse) or `as_matrix` (dense), or hand the
    operator straight to a splineax sparse solver, which reads `as_bcoo`.

    This is the sparse analogue of `lineax.FunctionLinearOperator`, and the counterpart
    of [`splineax.SparseJacobianLinearOperator`][]. The two agree: linearising a
    function and wrapping the result here gives the same operator as taking the
    Jacobian directly.

    ```python
    import jax
    import jax.numpy as jnp

    import splineax as splx


    def residual(y, args):
        return 3.0 * y + y**2 + 0.5 * jnp.roll(y, 1) * y


    y0 = jnp.linspace(0.5, 1.5, 5)
    _, jvp_of_residual = jax.linearize(lambda y: residual(y, None), y0)

    function_operator = splx.SparseFunctionLinearOperator(
        jvp_of_residual, jax.eval_shape(lambda: y0)
    )
    jacobian_operator = splx.SparseJacobianLinearOperator(residual, y0)
    assert jnp.allclose(function_operator.as_matrix(), jacobian_operator.as_matrix())
    ```

    The coloring is stored as an `asdex.ColoredPattern`, which is a registered JAX
    pytree, so the operator carries it as an ordinary (dynamic) field and the whole
    operator can be passed as an argument into a jitted function. A precomputed
    coloring may be supplied through the `coloring` argument (either an
    `asdex.ColoredPattern` or a [`splineax.JacobianColoring`][]) to skip detection,
    which is what makes it cheap to build many operators for the same sparsity.

    The input and output of `fn` may be arrays of any shape, or pytrees of them, exactly
    as for `lineax.FunctionLinearOperator`. Only the dtypes are restricted, to real
    ones, which is checked leafwise at construction. Everything handed to asdex is
    raveled first, so the sparsity pattern and `as_bcoo` are over the flattened input
    and output and stay two-dimensional, while `in_structure`, `out_structure` and `mv`
    keep the structures `fn` actually has. `lineax.materialise` is the one operation
    that cannot preserve them, since a `BCOOLinearOperator` carries only a matrix, so
    it refuses anything but flat structures rather than dropping them silently.

    To convert an existing dense `lineax.FunctionLinearOperator`, use
    [`splineax.SparseFunctionLinearOperator.from_function_operator`][].
    """

    fn: Callable
    coloring: ColoredPattern
    transposed: bool = eqx.field(static=True)
    # Stored in the forward orientation, unlike `SparseJacobianLinearOperator`, which
    # stores its pair already swapped. `jax.linear_transpose` needs the structure of
    # what `fn` itself accepts, so keeping that one directly is simpler than recovering
    # it from a swapped pair. `in_structure` and `out_structure` do the swapping.
    _forward_in_structure: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    _forward_out_structure: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        fn: Callable,
        input_structure: PyTree[jax.ShapeDtypeStruct],
        *,
        sparsity: SparsityPattern | np.ndarray | BCOO | None = None,
        coloring: ColoredPattern | JacobianColoring | None = None,
        mode: JacobianMode | None = None,
        tags: object | frozenset[object] = (),
        transposed: bool = False,
        closure_convert: bool = True,
    ):
        """**Arguments:**

        - `fn`: a linear function `x -> y`, where `x` and `y` are arrays of real dtype,
            or pytrees of them. Linearity is unchecked, and a nonlinear function will
            give incorrect results rather than an error.
        - `input_structure`: a pytree of `jax.ShapeDtypeStruct`s describing the input to
            `fn`, as for `lineax.FunctionLinearOperator`. When later calling `self.mv(x)`
            this should match `jax.eval_shape(lambda: x)`.
        - `sparsity`: optional known sparsity pattern of the matrix, as an
            `asdex.SparsityPattern`, a dense boolean mask, or a `BCOO` matrix.
            Skips sparsity detection (the pattern is still colored here).
        - `coloring`: optional precomputed coloring, either an `asdex.ColoredPattern`
            or a [`splineax.JacobianColoring`][]. Skips both sparsity detection and
            coloring. At most one of `sparsity` and `coloring` may be given.
        - `mode`: optional coloring mode, either `"fwd"` (column coloring) or `"bwd"`
            (row coloring). If not given, asdex picks based on the pattern.
        - `tags`: any lineax tags indicating whether the operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if
            these tags are wrong.

        `transposed` and `closure_convert` are internal arguments, used by
        `transpose()` and `lineax.linearise`.
        """
        input_structure = _inexact_structure(input_structure)
        check_real_dtypes(input_structure, "SparseFunctionLinearOperator")
        if closure_convert:
            fn = eqx.filter_closure_convert(fn, input_structure)
        self.fn = fn
        self.tags = _frozenset(tags)
        self.transposed = transposed
        self._forward_in_structure = input_structure
        self._forward_out_structure = strip_weak_dtype(
            eqxi.cached_filter_eval_shape(fn, input_structure)
        )

        # Resolve the coloring down to a bare `asdex.ColoredPattern`, exactly as
        # `SparseJacobianLinearOperator` does, and for the same reason: the pattern is
        # a pytree, so storing it directly keeps jit caches warm, since any two
        # colorings of the same sparsity flatten to the same treedef.
        match (coloring, sparsity):
            case (JacobianColoring() as wrapper, None):
                self.coloring = wrapper.coloring
            case (ColoredPattern() as pattern, None):
                self.coloring = pattern
            case (None, None):
                self.coloring = asdex.jacobian_coloring(
                    *self._flat_map(), mode=asdex_mode(mode)
                )
            case (None, known_sparsity):
                assert known_sparsity is not None
                self.coloring = asdex.jacobian_coloring_from_sparsity(
                    known_sparsity, mode=asdex_mode(mode)
                )
            case _:
                raise TypeError(
                    "Pass at most one of `coloring` and `sparsity`, where `coloring` "
                    "must be an `asdex.ColoredPattern` or a `splineax.JacobianColoring`."
                )

    @classmethod
    def from_function_operator(
        cls,
        operator: FunctionLinearOperator,
        *,
        sparsity: SparsityPattern | np.ndarray | BCOO | None = None,
        coloring: ColoredPattern | JacobianColoring | None = None,
        mode: JacobianMode | None = None,
    ) -> "SparseFunctionLinearOperator":
        """Converts a `lineax.FunctionLinearOperator` into its sparse analogue.

        The function, the input structure and the tags are taken straight from
        `operator`, so the two operators represent the same matrix. The difference is
        how it is materialised: one evaluation per color instead of one per column.

        **Arguments:**

        - `operator`: the dense function operator to convert.
        - `sparsity`, `coloring`, `mode`: the precomputation arguments of the
            constructor, passed through unchanged. Giving neither `sparsity` nor
            `coloring` detects the sparsity here, which runs host-side, so either call
            this outside `jax.jit` or pass one of them in.
        """
        # `lineax.FunctionLinearOperator` closure-converts its function on
        # construction, so converting it again here would only repeat that trace.
        return cls(
            operator.fn,
            operator.in_structure(),
            sparsity=sparsity,
            coloring=coloring,
            mode=mode,
            tags=operator.tags,
            closure_convert=False,
        )

    def _flat_map(self) -> tuple[Callable[[Array], Array], Inexact[Array, " n"]]:
        """The raveled forward map and the flat point to evaluate it at.

        Everything asdex sees goes through here. The point is the zeros of the input
        structure: a linear map's Jacobian is the same matrix at every point, so any
        one of them will do.
        """
        return flatten_map(self.fn, zero_point(self._forward_in_structure))

    def mv(
        self, vector: PyTree[Inexact[Array, "..."]]
    ) -> PyTree[Inexact[Array, "..."]]:
        if self.transposed:
            transpose_fn = jax.linear_transpose(self.fn, self._forward_in_structure)
            (out,) = transpose_fn(vector)
            return out
        return self.fn(vector)

    def as_bcoo(self) -> Inexact[BCOO, "a b"]:
        """Materialises the operator as a `BCOO` matrix, using one evaluation of `fn`
        per color of the precomputed coloring.

        The matrix is always two-dimensional, over the raveled input and output, so an
        `fn` taking or returning a pytree still materialises to a plain matrix. The
        operator's own structures are the unraveled ones, so a caller that needs both
        must keep them itself, which is what the splineax solvers do.
        """
        flat_fn, flat_point = self._flat_map()
        matrix = cast(
            BCOO,
            asdex.jacobian_from_coloring(flat_fn, self.coloring, "bcoo")(flat_point),
        )
        if self.transposed:
            # `BCOO.T` is not well-typed, hence the cast.
            return cast(BCOO, matrix.T)
        return matrix

    def as_matrix(self) -> Inexact[Array, "a b"]:
        return self.as_bcoo().todense()

    def transpose(self) -> "SparseFunctionLinearOperator":
        if is_symmetric(self):
            return self
        # Stay sparse by flipping the transpose flag, reusing the function and the same
        # coloring pattern (so the jit cache identity is preserved). `mv` transposes
        # the linear map and `as_bcoo` transposes the materialised matrix. Building a
        # `jax.linear_transpose` operator instead would need the transposed pattern,
        # which means detecting the sparsity all over again.
        return SparseFunctionLinearOperator(
            self.fn,
            self._forward_in_structure,
            coloring=self.coloring,
            tags=transpose_tags(self.tags),
            transposed=not self.transposed,
            closure_convert=False,
        )

    def in_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        if self.transposed:
            return self._forward_out_structure
        return self._forward_in_structure

    def out_structure(self) -> PyTree[jax.ShapeDtypeStruct]:
        if self.transposed:
            return self._forward_in_structure
        return self._forward_out_structure


# Lineax `singledispatch` registrations. The tag-only ones are shared with
# `SparseJacobianLinearOperator` and installed by `register_ad_operator` at the bottom
# of this module. The two below are specific to this operator.


@materialise.register(SparseFunctionLinearOperator)
def _(operator: SparseFunctionLinearOperator) -> BCOOLinearOperator:
    # Convert to a concrete sparse operator, for callers that want one. The solvers do
    # not come through here: they read `as_bcoo()` and keep the operator's structures
    # themselves, precisely because a `BCOOLinearOperator` cannot carry them.
    check_structures_survive_materialisation(operator)
    return BCOOLinearOperator(operator.as_bcoo(), operator.tags)


@linearise.register(SparseFunctionLinearOperator)
def _(operator: SparseFunctionLinearOperator) -> SparseFunctionLinearOperator:
    # The map is already linear, so there is no primal pass to cache. lineax's
    # `linearise` is the identity on a `FunctionLinearOperator` for the same reason.
    return operator


register_ad_operator(SparseFunctionLinearOperator)
