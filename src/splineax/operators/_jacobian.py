"""Sparse Jacobian operator backed by `asdex` sparsity detection and coloring.

`lineax.JacobianLinearOperator` represents the Jacobian of a function densely.
The operator here is its sparse analogue: the Jacobian's sparsity pattern and a
matching row or column coloring are computed once (by `asdex`) and stored, so
materialising the Jacobian at a point costs one JVP or VJP per color rather than
one per column or row. The result is a `jax.experimental.sparse.BCOO` matrix
that the splineax sparse solvers consume directly. As in lineax, the function may
take and return pytrees, which are raveled to keep that result a plain matrix.

The sparsity pattern and coloring depend only on the traced computation graph of
the function, not on the numerical values of its inputs. They are therefore
computed at construction time and reused for every evaluation point. Two
colorings of the same sparsity pattern flatten to identical treedefs, so a
jitted function that accepts one compiles exactly once. The public entry point
for creating and carrying colorings is [`splineax.JacobianColoring`][].
"""

from collections.abc import Callable
from typing import Any, Literal, cast

import asdex
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from asdex import ColoredPattern, SparsityPattern
from jax.experimental.sparse import BCOO
from jaxtyping import Array, ArrayLike, Inexact, PyTree
from lineax import (
    AbstractLinearOperator,
    JacobianLinearOperator,
    has_unit_diagonal,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_tridiagonal,
    is_upper_triangular,
    linearise,
    materialise,
)
from lineax._operator import _frozenset, inexact_asarray, strip_weak_dtype
from lineax._solver.misc import PackedStructures, ravel_vector, unravel_solution
from lineax._tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    transpose_tags,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
)

from ._bcoo import BCOOLinearOperator

JacobianMode = Literal["fwd", "bwd"]
"""The AD mode a Jacobian coloring is built for: `"fwd"` (column coloring, JVPs) or
`"bwd"` (row coloring, VJPs). Spelled as in lineax and JAX, so a
`lineax.JacobianLinearOperator`'s `jac` argument carries over unchanged. asdex spells
reverse mode `"rev"`, which `_asdex_mode` translates to."""


class JacobianColoring(eqx.Module):
    """A function-agnostic Jacobian sparsity coloring, backed by `asdex`.

    A `JacobianColoring` wraps an `asdex.ColoredPattern`: a Jacobian sparsity pattern
    together with a matching row or column coloring. The coloring is what lets the
    Jacobian be materialised with one JVP or VJP per color rather than one per column
    or row.

    This wrapper carries only the coloring, not any particular function or evaluation
    point. Create one with either [`splineax.JacobianColoring.detect`][], which
    detects the sparsity of a function and colors it, or
    [`splineax.JacobianColoring.from_sparsity`][], which colors a sparsity pattern you
    already know. Both run host-side (they use numpy, scipy, and graph coloring under
    the hood), so build the coloring outside `jax.jit` and pass the finished object
    in as an argument.

    Because `asdex.ColoredPattern` is a registered JAX pytree, a `JacobianColoring`
    is itself a pytree and can be threaded through jitted computations. Any two
    colorings of the same sparsity pattern flatten to the same treedef, so a jitted
    function that receives a `JacobianColoring` compiles once and stays cached even
    when the coloring is regenerated from scratch.

    To turn a coloring into a linear operator, either pass it as the `coloring`
    argument of [`splineax.SparseJacobianLinearOperator`][] together with a function
    and a point, or bind it to a specific function once with
    [`splineax.SparseJacobianLinearOperatorColoring.from_jacobian_coloring`][] and
    then call `operator_at` at many points.
    """

    coloring: ColoredPattern
    """The wrapped asdex coloring, holding both the sparsity pattern and the row or
    column coloring of it. Stored as an ordinary (dynamic) pytree field, so it can be
    carried through jitted functions."""

    @classmethod
    def detect(
        cls,
        fn: Callable,
        x: PyTree[Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct],
        args: PyTree[Any] = None,
        *,
        mode: JacobianMode | None = None,
    ) -> "JacobianColoring":
        """Detects the Jacobian sparsity of `fn` and colors it.

        Detection is structural: asdex traces the computation graph of `fn` and reads
        off which outputs depend on which inputs, without evaluating any derivatives.
        Only the shape and dtype of `x` matter, so a `jax.ShapeDtypeStruct` may be
        passed in place of a concrete point. Detection and coloring run host-side, so
        call this outside `jax.jit` and pass the resulting coloring in.

        **Arguments:**

        - `fn`: a function `(x, args) -> y`, where `x` and `y` are arrays of real
            dtype, or pytrees of them. Its Jacobian's sparsity is detected.
        - `x`: a representative point, or `jax.ShapeDtypeStruct`s describing one. Only
            the shapes and dtypes are used.
        - `args`: extra arguments to `fn` that are not differentiated.
        - `mode`: optional coloring mode, either `"fwd"` (column coloring,
            materialised with JVPs) or `"bwd"` (row coloring, materialised with VJPs).
            If not given, asdex picks based on the pattern.
        """
        example_point = _example_point(x)

        def function_of_point(point: PyTree[Array]) -> PyTree[Array]:
            return fn(point, args)

        # Detect against the raveled map, so that the pattern is over the same flat
        # index order the operator materialises in, whatever structure `x` has.
        flat_function_of_point, flat_point = _flatten(function_of_point, example_point)
        detected = asdex.jacobian_coloring(
            flat_function_of_point, flat_point, mode=_asdex_mode(mode)
        )
        return cls(detected)

    @classmethod
    def from_sparsity(
        cls,
        sparsity: SparsityPattern | np.ndarray | BCOO,
        *,
        mode: JacobianMode | None = None,
    ) -> "JacobianColoring":
        """Colors a known Jacobian sparsity pattern, skipping detection.

        No function is needed, since the sparsity pattern already describes which
        Jacobian entries are nonzero. Coloring runs host-side, so call this outside
        `jax.jit` and pass the resulting coloring in.

        **Arguments:**

        - `sparsity`: the known sparsity pattern of the Jacobian, as an
            `asdex.SparsityPattern`, a dense boolean mask, or a `BCOO` matrix.
        - `mode`: optional coloring mode, either `"fwd"` (column coloring,
            materialised with JVPs) or `"bwd"` (row coloring, materialised with VJPs).
            If not given, asdex picks based on the pattern.
        """
        colored = asdex.jacobian_coloring_from_sparsity(
            sparsity, mode=_asdex_mode(mode)
        )
        return cls(colored)

    @property
    def sparsity(self) -> SparsityPattern:
        """The `asdex.SparsityPattern` that was colored. The splineax solvers read the
        row and column indices from here to pre-analyze the sparsity host-side."""
        return self.coloring.sparsity

    @property
    def mode(self) -> JacobianMode:
        """The resolved coloring mode, either `"fwd"` or `"bwd"`. This is the mode
        asdex chose, never the unresolved `None` the caller may have passed, and it is
        reported in the lineax spelling rather than asdex's `"fwd"`/`"rev"`."""
        return "bwd" if self.coloring.mode == "rev" else "fwd"

    @property
    def num_colors(self) -> int:
        """The number of colors, and so the number of JVPs or VJPs one Jacobian
        materialisation costs."""
        return self.coloring.num_colors


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
        for leaf in jtu.tree_leaves(self.x):
            if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                raise TypeError(
                    "`SparseJacobianLinearOperator` only supports real dtypes, but `x` "
                    f"has a leaf of dtype {leaf.dtype}."
                )
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
                flat_function_of_point, flat_point = _flatten(function_of_point, self.x)
                self.coloring = asdex.jacobian_coloring(
                    flat_function_of_point, flat_point, mode=_asdex_mode(mode)
                )
            case (None, known_sparsity):
                assert known_sparsity is not None
                self.coloring = asdex.jacobian_coloring_from_sparsity(
                    known_sparsity, mode=_asdex_mode(mode)
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
        flat_function_of_point, flat_point = _flatten(self._function_of_point(), self.x)
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
        example_point = _example_point(x)
        converted_fn = eqx.filter_closure_convert(fn, example_point, args)
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


def _asdex_mode(mode: JacobianMode | None) -> Literal["fwd", "rev"] | None:
    """Translates a coloring mode into the asdex spelling. Both call forward mode
    `"fwd"`, but reverse mode is `"bwd"` here, as in lineax and JAX, where asdex calls
    it `"rev"`. `None` is passed through, leaving the choice to asdex."""
    return "rev" if mode == "bwd" else mode


def _example_point(
    x: PyTree[Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct],
) -> PyTree[Inexact[Array, "..."]]:
    """Turns a concrete point or a `jax.ShapeDtypeStruct` into a concrete array
    usable for tracing, leafwise. Only the shape and dtype are meaningful to the
    callers."""

    def concrete_leaf(leaf: ArrayLike | jax.ShapeDtypeStruct) -> Array:
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jnp.empty(leaf.shape, leaf.dtype)
        return inexact_asarray(leaf)

    return jtu.tree_map(
        concrete_leaf, x, is_leaf=lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct)
    )


def _packed_structures(
    out_structure: PyTree[jax.ShapeDtypeStruct],
    in_structure: PyTree[jax.ShapeDtypeStruct],
) -> PackedStructures:
    """The ravel bookkeeping `lineax.ravel_vector` and `lineax.unravel_solution` read,
    built from a pair of structures.

    `lineax._solver.misc.pack_structures` builds the same thing from an operator, which
    is not what is wanted here: the flattening below describes `fn` itself, whose input
    and output do not swap places when the operator is transposed.
    """
    leaves, treedef = jtu.tree_flatten(
        (strip_weak_dtype(out_structure), strip_weak_dtype(in_structure))
    )
    return PackedStructures(eqxi.Static((leaves, treedef)))


def _flatten(
    function_of_point: Callable[[PyTree[Array]], PyTree[Array]],
    x: PyTree[Inexact[Array, "..."]],
) -> tuple[Callable[[Array], Array], Inexact[Array, " n"]]:
    """Rewrites a forward map as a map between raveled arrays, and ravels `x` to match.

    asdex detects, colors and materialises flat Jacobians, so everything that talks to
    it goes through here. That is what lets `x` and the output of `fn` be arrays of any
    shape, or pytrees of them, while the stored sparsity pattern and the materialised
    matrix stay two-dimensional.

    Ravelling goes through lineax's own `ravel_vector` and `unravel_solution`, the same
    helpers the sparse solvers use on a right-hand side and its solution. Both sides
    therefore agree on the ordering by construction, which `jax.flatten_util.ravel_pytree`
    would not guarantee.
    """
    in_structure = strip_weak_dtype(jax.eval_shape(lambda: x))
    out_structure = strip_weak_dtype(jax.eval_shape(function_of_point, x))
    packed = _packed_structures(out_structure, in_structure)
    # Ravelling `x` reads the pair's output side, so it needs one of its own.
    point_packed = _packed_structures(in_structure, in_structure)

    def flat_function_of_point(flat_point: Inexact[Array, " n"]) -> Array:
        point = unravel_solution(flat_point, packed)
        return ravel_vector(function_of_point(point), packed)

    return flat_function_of_point, ravel_vector(x, point_packed)


# Lineax `singledispatch` registrations. `register_sparse_operator` cannot be reused
# here because its implementations read `operator.matrix`, which this operator does
# not have. `conj`, `diagonal` and `tridiagonal` are intentionally left unregistered:
# the operator is real-valued and materialise-first, so lineax's informative default
# errors apply.


@materialise.register(SparseJacobianLinearOperator)
def _(operator: SparseJacobianLinearOperator) -> BCOOLinearOperator:
    # Convert to a concrete sparse operator, for callers that want one. The solvers do
    # not come through here: they read `as_bcoo()` and keep the operator's structures
    # themselves, precisely because a `BCOOLinearOperator` cannot carry them.
    _check_structures_survive_materialisation(operator)
    return BCOOLinearOperator(operator.as_bcoo(), operator.tags)


def _check_structures_survive_materialisation(
    operator: SparseJacobianLinearOperator,
) -> None:
    """Raises unless both of the operator's structures are a single one-dimensional
    array, which is all a `BCOOLinearOperator` can represent.

    `as_bcoo` ravels, so a richer structure materialises to a perfectly good matrix
    whose operator would then report the flat structures the matrix shape implies. That
    would drop the structure the caller passed in without a word, so it is refused here
    instead.
    """
    structures = (
        ("in", operator.in_structure()),
        ("out", operator.out_structure()),
    )
    for name, structure in structures:
        if not isinstance(structure, jax.ShapeDtypeStruct) or len(structure.shape) != 1:
            raise ValueError(
                "`lineax.materialise` on a `SparseJacobianLinearOperator` requires "
                "one-dimensional array in- and out-structures, since the resulting "
                "`BCOOLinearOperator` carries only a matrix and would drop anything "
                f"richer. The {name}-structure is {structure}. Use `as_bcoo()` "
                "instead, keeping the structures yourself, as the splineax solvers do."
            )


@linearise.register(SparseJacobianLinearOperator)
def _(operator: SparseJacobianLinearOperator) -> SparseJacobianLinearOperator:
    # Cache the primal pass with `jax.linearize`, then wrap the resulting linear
    # map in a new operator so it can still be sparsely materialised (the Jacobian
    # of the linearised map is the same constant matrix with the same sparsity).
    # The original coloring wrapper is passed through, so the static key stays
    # stable even though the linearised function is a fresh object per call.
    _, jvp_function = jax.linearize(operator._function_of_point(), operator.x)

    def linearised_fn(point: Array, args: PyTree[Any]) -> Array:
        del args
        return jvp_function(point)

    return SparseJacobianLinearOperator(
        linearised_fn,
        operator.x,
        coloring=operator.coloring,
        tags=operator.tags,
        transposed=operator.transposed,
    )


# Tag-only structural predicates, mirroring `_operations.py` but reading only
# `operator.tags`. The positive/negative-semidefinite-implies-symmetric rule needs no
# dtype check here: for complex dtypes positive semidefinite means Hermitian, which
# differs from symmetric, so lineax restricts the implication to real dtypes. This
# operator rejects complex `x` at construction, so the implication always applies.


def _is_symmetric(operator: SparseJacobianLinearOperator) -> bool:
    return (
        symmetric_tag in operator.tags
        or diagonal_tag in operator.tags
        or positive_semidefinite_tag in operator.tags
        or negative_semidefinite_tag in operator.tags
    )


def _is_diagonal(operator: SparseJacobianLinearOperator) -> bool:
    return diagonal_tag in operator.tags or (
        operator.in_size() == 1 and operator.out_size() == 1
    )


def _is_tridiagonal(operator: SparseJacobianLinearOperator) -> bool:
    return tridiagonal_tag in operator.tags or diagonal_tag in operator.tags


def _has_unit_diagonal(operator: SparseJacobianLinearOperator) -> bool:
    return unit_diagonal_tag in operator.tags


def _is_lower_triangular(operator: SparseJacobianLinearOperator) -> bool:
    return lower_triangular_tag in operator.tags


def _is_upper_triangular(operator: SparseJacobianLinearOperator) -> bool:
    return upper_triangular_tag in operator.tags


def _is_positive_semidefinite(operator: SparseJacobianLinearOperator) -> bool:
    return positive_semidefinite_tag in operator.tags


def _is_negative_semidefinite(operator: SparseJacobianLinearOperator) -> bool:
    return negative_semidefinite_tag in operator.tags


is_symmetric.register(SparseJacobianLinearOperator, _is_symmetric)
is_diagonal.register(SparseJacobianLinearOperator, _is_diagonal)
is_tridiagonal.register(SparseJacobianLinearOperator, _is_tridiagonal)
has_unit_diagonal.register(SparseJacobianLinearOperator, _has_unit_diagonal)
is_lower_triangular.register(SparseJacobianLinearOperator, _is_lower_triangular)
is_upper_triangular.register(SparseJacobianLinearOperator, _is_upper_triangular)
is_positive_semidefinite.register(
    SparseJacobianLinearOperator, _is_positive_semidefinite
)
is_negative_semidefinite.register(
    SparseJacobianLinearOperator, _is_negative_semidefinite
)
