"""Sparse Jacobian operator backed by `asdex` sparsity detection and coloring.

`lineax.JacobianLinearOperator` represents the Jacobian of a function densely.
The operator here is its sparse analogue: the Jacobian's sparsity pattern and a
matching row or column coloring are computed once (by `asdex`) and stored, so
materialising the Jacobian at a point costs one JVP or VJP per color rather than
one per column or row. The result is a `jax.experimental.sparse.BCOO` matrix
that the splineax sparse solvers consume directly. As in lineax, the function may
take and return pytrees, which are raveled to keep that result a plain matrix.

The coloring machinery itself lives in `_coloring.py`, shared with
`SparseFunctionLinearOperator`, along with the note on why a coloring computed once
stays valid at every evaluation point.
"""

from collections.abc import Callable
from typing import Any, cast

import asdex
import equinox as eqx
import jax
import jax.tree_util as jtu
import numpy as np
from asdex import ColoredPattern, SparsityPattern
from jax.experimental.sparse import BCOO
from jaxtyping import Array, ArrayLike, Inexact, PyTree
from lineax import (
    AbstractLinearOperator,
    JacobianLinearOperator,
    is_symmetric,
    materialise,
)
from lineax._operator import _frozenset, inexact_asarray, strip_weak_dtype
from lineax._tags import transpose_tags

from ._bcoo import BCOOLinearOperator
from ._coloring import (
    JacobianColoring,
    JacobianMode,
    asdex_mode,
    check_real_dtypes,
    check_structures_survive_materialisation,
    example_point,
    flatten_map,
    register_ad_operator,
)


class SparseJacobianLinearOperator(AbstractLinearOperator):
    """Given a function `fn: X -> Y` and a point `x`, the linear operator
    `(d(fn)/dx)(x)`, kept sparse.

    The Jacobian's sparsity pattern and a matching coloring are determined once
    at construction (via `asdex`), so materialising the Jacobian at `x` costs
    one JVP or VJP per color rather than one per column or row. Materialise it
    with `as_bcoo` or `lineax.materialise` (sparse) or `as_matrix` (dense), or
    hand the operator straight to a splineax sparse solver, which reads `as_bcoo`.

    The coloring is stored as an `asdex.ColoredPattern`, which is a registered JAX
    pytree, so the operator carries it as an ordinary (dynamic) field and the whole
    operator can be passed as an argument into a jitted function. A precomputed
    coloring may be supplied through the `coloring` argument (either an
    `asdex.ColoredPattern` or a [`splineax.JacobianColoring`][]) to skip detection,
    which is what makes it cheap to build many operators for the same sparsity.

    `x` and the output of `fn` may be arrays of any shape, or pytrees of them, exactly
    as for `lineax.JacobianLinearOperator`. Only the dtypes are restricted, to real
    ones, which is checked leafwise at construction. Everything handed to asdex is
    raveled first, so the sparsity pattern and `as_bcoo` are over the flattened input
    and output and stay two-dimensional, while `in_structure`, `out_structure` and `mv`
    keep the structures `fn` actually has. `lineax.materialise` is the one operation
    that cannot preserve them, since a `BCOOLinearOperator` carries only a matrix, so
    it refuses anything but flat structures rather than dropping them silently.

    To build many operators for the same function at different points without repeating
    sparsity detection, use
    [`splineax.SparseJacobianLinearOperatorColoring`][]. To convert an existing dense
    `lineax.JacobianLinearOperator`, use
    [`splineax.SparseJacobianLinearOperator.from_jacobian_operator`][].
    """

    fn: Callable
    x: PyTree[Inexact[Array, "..."]]
    args: PyTree[Any]
    coloring: ColoredPattern
    transposed: bool = eqx.field(static=True)
    _in_structure: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    _out_structure: PyTree[jax.ShapeDtypeStruct] = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        fn: Callable,
        x: PyTree[Inexact[ArrayLike, "..."]],
        args: PyTree[Any] = None,
        *,
        sparsity: SparsityPattern | np.ndarray | BCOO | None = None,
        coloring: ColoredPattern | JacobianColoring | None = None,
        mode: JacobianMode | None = None,
        tags: object | frozenset[object] = (),
        transposed: bool = False,
        closure_convert: bool = True,
    ):
        """**Arguments:**

        - `fn`: a function `(x, args) -> y`, where `x` and `y` are arrays of real
            dtype, or pytrees of them. Its Jacobian `d(fn)/dx` is the linear operator.
        - `x`: the point at which to evaluate `d(fn)/dx`.
        - `args`: extra arguments to `fn` that are not differentiated.
        - `sparsity`: optional known sparsity pattern of the Jacobian, as an
            `asdex.SparsityPattern`, a dense boolean mask, or a `BCOO` matrix.
            Skips sparsity detection (the pattern is still colored here).
        - `coloring`: optional precomputed coloring, either an `asdex.ColoredPattern`
            or a [`splineax.JacobianColoring`][]. Skips both sparsity detection and
            coloring. At most one of `sparsity` and `coloring` may be given.
        - `mode`: optional coloring mode, either `"fwd"` (column coloring,
            materialised with JVPs) or `"bwd"` (row coloring, materialised with
            VJPs). If not given, asdex picks based on the pattern.
        - `tags`: any lineax tags indicating whether the Jacobian has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if
            these tags are wrong.

        `transposed` and `closure_convert` are internal arguments, used by
        `transpose()` and [`splineax.SparseJacobianLinearOperatorColoring`][].
        """
        self.x = jtu.tree_map(inexact_asarray, x)
        check_real_dtypes(self.x, "SparseJacobianLinearOperator")
        if closure_convert:
            fn = eqx.filter_closure_convert(fn, self.x, args)
        self.fn = fn
        self.args = args
        self.tags = _frozenset(tags)
        self.transposed = transposed

        def function_of_point(point: PyTree[Array]) -> PyTree[Array]:
            return fn(point, args)

        forward_in_structure = strip_weak_dtype(jax.eval_shape(lambda: self.x))
        forward_out_structure = strip_weak_dtype(
            jax.eval_shape(function_of_point, self.x)
        )
        if transposed:
            self._in_structure = forward_out_structure
            self._out_structure = forward_in_structure
        else:
            self._in_structure = forward_in_structure
            self._out_structure = forward_out_structure

        # Resolve the coloring down to a bare `asdex.ColoredPattern`, which is what
        # is stored and later handed to `asdex.jacobian_from_coloring`. A
        # `JacobianColoring` (passed by `transpose()`, `operator_at()`, or the caller)
        # is unwrapped to its inner pattern. Because the pattern is a pytree, storing
        # it directly is enough for jit caches to stay warm: any two colorings of the
        # same sparsity flatten to the same treedef.
        match (coloring, sparsity):
            case (JacobianColoring() as wrapper, None):
                self.coloring = wrapper.coloring
            case (ColoredPattern() as pattern, None):
                self.coloring = pattern
            case (None, None):
                flat_function_of_point, flat_point = flatten_map(
                    function_of_point, self.x
                )
                self.coloring = asdex.jacobian_coloring(
                    flat_function_of_point, flat_point, mode=asdex_mode(mode)
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
    def from_jacobian_operator(
        cls,
        operator: JacobianLinearOperator,
        *,
        sparsity: SparsityPattern | np.ndarray | BCOO | None = None,
        coloring: ColoredPattern | JacobianColoring | None = None,
        mode: JacobianMode | None = None,
    ) -> "SparseJacobianLinearOperator":
        """Converts a `lineax.JacobianLinearOperator` into its sparse analogue.

        The function, the point, the extra arguments and the tags are taken straight
        from `operator`, so the two operators represent the same Jacobian. The
        difference is how it is materialised: one JVP or VJP per color instead of one
        per column or row.

        `x` may be any pytree, as for the operator being converted. Only the real-dtype
        restriction is narrower than what `lineax.JacobianLinearOperator` accepts.

        **Arguments:**

        - `operator`: the dense Jacobian operator to convert.
        - `sparsity`, `coloring`, `mode`: the precomputation arguments of the
            constructor, passed through unchanged. Giving neither `sparsity` nor
            `coloring` detects the sparsity here, which runs host-side, so either call
            this outside `jax.jit` or pass one of them in. `mode` defaults to the
            operator's own `jac`, which is spelled the same way.
        """
        # `lineax.JacobianLinearOperator` closure-converts its function on
        # construction, so converting it again here would only repeat that trace.
        return cls(
            operator.fn,
            operator.x,
            operator.args,
            sparsity=sparsity,
            coloring=coloring,
            mode=operator.jac if mode is None else mode,
            tags=operator.tags,
            closure_convert=False,
        )

    def _function_of_point(self) -> Callable[[PyTree[Array]], PyTree[Array]]:
        """Returns the forward map `x -> fn(x, args)` with `args` bound."""

        def function_of_point(point: PyTree[Array]) -> PyTree[Array]:
            return self.fn(point, self.args)

        return function_of_point

    def mv(
        self, vector: PyTree[Inexact[Array, "..."]]
    ) -> PyTree[Inexact[Array, "..."]]:
        if self.transposed:
            _, vjp_function = jax.vjp(self._function_of_point(), self.x)
            (out,) = vjp_function(vector)
            return out
        _, out = jax.jvp(self._function_of_point(), (self.x,), (vector,))
        return out

    def as_bcoo(self) -> Inexact[BCOO, "a b"]:
        """Materialises the Jacobian at `x` as a `BCOO` matrix, using one JVP or VJP
        per color of the precomputed coloring.

        The matrix is always two-dimensional, over the raveled input and output, so a
        `fn` taking or returning a pytree still materialises to a plain matrix. The
        operator's own structures are the unraveled ones, so a caller that needs both
        must keep them itself, which is what the splineax solvers do.
        """
        flat_function_of_point, flat_point = flatten_map(
            self._function_of_point(), self.x
        )
        jacobian = cast(
            BCOO,
            asdex.jacobian_from_coloring(
                flat_function_of_point,
                self.coloring,
                "bcoo",
            )(flat_point),
        )
        if self.transposed:
            # `BCOO.T` is not well-typed, hence the cast.
            return cast(BCOO, jacobian.T)
        return jacobian

    def as_matrix(self) -> Inexact[Array, "a b"]:
        return self.as_bcoo().todense()

    def transpose(self) -> "SparseJacobianLinearOperator":
        if is_symmetric(self):
            return self
        # Stay sparse by flipping the transpose flag, reusing the function, the
        # point and the same coloring pattern (so the jit cache identity is
        # preserved). `mv` switches between JVP and VJP and `as_bcoo` transposes
        # the materialised Jacobian.
        return SparseJacobianLinearOperator(
            self.fn,
            self.x,
            self.args,
            coloring=self.coloring,
            tags=transpose_tags(self.tags),
            transposed=not self.transposed,
            closure_convert=False,
        )

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return self._in_structure

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self._out_structure


class SparseJacobianLinearOperatorColoring(eqx.Module):
    """A [`splineax.JacobianColoring`][] bound to a specific function, reusable across
    evaluation points.

    Where a [`splineax.JacobianColoring`][] carries only the coloring, this class also
    holds the (closure-converted) function whose Jacobian was colored. That pairing is
    what a [`splineax.SparseJacobianLinearOperator`][] needs, so
    [`splineax.SparseJacobianLinearOperatorColoring.operator_at`][] can produce an
    operator at any point without repeating sparsity detection or coloring.

    Build one with [`splineax.SparseJacobianLinearOperatorColoring.detect`][] or
    [`splineax.SparseJacobianLinearOperatorColoring.from_sparsity`][], or from an
    existing [`splineax.JacobianColoring`][] with
    [`splineax.SparseJacobianLinearOperatorColoring.from_jacobian_coloring`][]. All
    operators built from one instance share the same closure-converted function and the
    same coloring pattern, so passing them through the same jitted computation compiles
    only once.

    The coloring is valid for any `x` and `args` of the same abstract structure
    (shapes and dtypes) as those it was computed with. asdex guarantees that the
    sparsity pattern depends only on the traced computation graph, not on numerical
    values, so reusing the coloring at other points is always sound.
    """

    fn: Callable
    """The closure-converted function whose Jacobian was colored. Shared by every
    operator built through `operator_at`, so their pytree structures compare
    equal."""

    coloring: JacobianColoring
    """The function-agnostic coloring. Carried as an ordinary (dynamic) pytree field,
    since `JacobianColoring` wraps a pytree `asdex.ColoredPattern`."""

    @classmethod
    def from_jacobian_coloring(
        cls,
        coloring: JacobianColoring,
        fn: Callable,
        x: PyTree[Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct],
        args: PyTree[Any] = None,
    ) -> "SparseJacobianLinearOperatorColoring":
        """Binds an existing [`splineax.JacobianColoring`][] to a function.

        This is the bridge from a bare coloring to an operator factory. The coloring
        may have come from [`splineax.JacobianColoring.detect`][] on this same
        function or from [`splineax.JacobianColoring.from_sparsity`][] on a pattern
        you know matches `fn`. The function is closure-converted here (once), and the
        result reused by every `operator_at` call.

        **Arguments:**

        - `coloring`: the coloring to bind, as a [`splineax.JacobianColoring`][].
        - `fn`: a function `(x, args) -> y`, where `x` and `y` are arrays of real
            dtype, or pytrees of them. Its Jacobian must have the sparsity the coloring
            describes.
        - `x`: a representative point, or `jax.ShapeDtypeStruct`s describing one. Only
            the shapes and dtypes matter here, used to closure-convert `fn`.
        - `args`: extra arguments to `fn` that are not differentiated.
        """
        point = example_point(x)
        converted_fn = eqx.filter_closure_convert(fn, point, args)
        return cls(converted_fn, coloring)

    @classmethod
    def detect(
        cls,
        fn: Callable,
        x: PyTree[Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct],
        args: PyTree[Any] = None,
        *,
        mode: JacobianMode | None = None,
    ) -> "SparseJacobianLinearOperatorColoring":
        """Detects the Jacobian sparsity of `fn`, colors it, and binds it to `fn`.

        A convenience wrapper equivalent to
        [`splineax.JacobianColoring.detect`][] followed by
        [`splineax.SparseJacobianLinearOperatorColoring.from_jacobian_coloring`][].

        **Arguments:**

        - `fn`: a function `(x, args) -> y`, where `x` and `y` are arrays of real
            dtype, or pytrees of them.
        - `x`: a representative point, or `jax.ShapeDtypeStruct`s describing one. Only
            the shapes and dtypes matter, since sparsity detection is structural.
        - `args`: extra arguments to `fn` that are not differentiated.
        - `mode`: optional coloring mode, `"fwd"` or `"bwd"`.
        """
        jacobian_coloring = JacobianColoring.detect(fn, x, args, mode=mode)
        return cls.from_jacobian_coloring(jacobian_coloring, fn, x, args)

    @classmethod
    def from_sparsity(
        cls,
        fn: Callable,
        x: PyTree[Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct],
        sparsity: SparsityPattern | np.ndarray | BCOO,
        args: PyTree[Any] = None,
        *,
        mode: JacobianMode | None = None,
    ) -> "SparseJacobianLinearOperatorColoring":
        """Colors a known Jacobian sparsity pattern of `fn`, skipping detection, and
        binds it to `fn`.

        A convenience wrapper equivalent to
        [`splineax.JacobianColoring.from_sparsity`][] followed by
        [`splineax.SparseJacobianLinearOperatorColoring.from_jacobian_coloring`][].

        **Arguments:**

        - `fn`: a function `(x, args) -> y`, where `x` and `y` are arrays of real
            dtype, or pytrees of them.
        - `x`: a representative point, or `jax.ShapeDtypeStruct`s describing one. Not
            used for detection, but required to closure-convert `fn`.
        - `sparsity`: the known sparsity pattern of the Jacobian, as an
            `asdex.SparsityPattern`, a dense boolean mask, or a `BCOO` matrix.
        - `args`: extra arguments to `fn` that are not differentiated.
        - `mode`: optional coloring mode, `"fwd"` or `"bwd"`.
        """
        jacobian_coloring = JacobianColoring.from_sparsity(sparsity, mode=mode)
        return cls.from_jacobian_coloring(jacobian_coloring, fn, x, args)

    def operator_at(
        self,
        x: PyTree[Inexact[ArrayLike, "..."]],
        args: PyTree[Any] = None,
        tags: object | frozenset[object] = (),
    ) -> SparseJacobianLinearOperator:
        """Builds a [`splineax.SparseJacobianLinearOperator`][] at the point `x`,
        reusing the precomputed coloring.

        **Arguments:**

        - `x`: the point at which to evaluate the Jacobian. Must have the same
            structure, shapes and dtypes as the point the coloring was computed for.
        - `args`: extra arguments to `fn` that are not differentiated. Must have
            the same abstract structure as the `args` the coloring was computed
            for.
        - `tags`: any lineax tags for the resulting operator.
        """
        return SparseJacobianLinearOperator(
            self.fn,
            x,
            args,
            coloring=self.coloring,
            tags=tags,
            closure_convert=False,
        )


# Lineax `singledispatch` registrations. The tag-only ones are shared with
# `SparseFunctionLinearOperator` and installed by `register_ad_operator` at the bottom
# of this module. `linearise` is registered in `_function.py`, since what it returns is
# a function operator.


@materialise.register(SparseJacobianLinearOperator)
def _(operator: SparseJacobianLinearOperator) -> BCOOLinearOperator:
    # Convert to a concrete sparse operator, for callers that want one. The solvers do
    # not come through here: they read `as_bcoo()` and keep the operator's structures
    # themselves, precisely because a `BCOOLinearOperator` cannot carry them.
    check_structures_survive_materialisation(operator)
    return BCOOLinearOperator(operator.as_bcoo(), operator.tags)


register_ad_operator(SparseJacobianLinearOperator)
