"""Sparse Jacobian operator backed by `asdex` sparsity detection and coloring.

`lineax.JacobianLinearOperator` represents the Jacobian of a function densely. The
operator here is its sparse analogue: the Jacobian's sparsity pattern and a matching
row or column coloring are computed once (by `asdex`) and stored, so materialising the
Jacobian at a point costs one JVP or VJP per color rather than one per column or row.
The result is a `jax.experimental.sparse.BCOO` matrix that the splineax sparse solvers
consume directly.

The sparsity pattern and coloring depend only on the traced computation graph of the
function, not on the numerical values of its inputs. They are therefore computed at
construction time and reused for every evaluation point.
"""

import dataclasses
from collections.abc import Callable, Hashable
from typing import Any, Literal, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from asdex import ColoredPattern, SparsityPattern
from asdex import (
    jacobian_coloring as asdex_jacobian_coloring,
)
from asdex import (
    jacobian_coloring_from_sparsity as asdex_jacobian_coloring_from_sparsity,
)
from asdex import (
    jacobian_from_coloring as asdex_jacobian_from_coloring,
)
from jax.experimental.sparse import BCOO
from jaxtyping import Array, ArrayLike, Inexact, PyTree
from lineax import (
    AbstractLinearOperator,
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

JacobianMode = Literal["fwd", "rev"]


def wrap_in_jax_partial(fn: Callable | jtu.Partial) -> jtu.Partial:
    return fn if isinstance(fn, jtu.Partial) else jtu.Partial(fn)


@dataclasses.dataclass(frozen=True, eq=False)
class _StaticColoring:
    """Hashable holder for an `asdex.ColoredPattern`, used as an equinox static field.

    A `ColoredPattern` is a frozen dataclass whose fields are numpy arrays. Numpy
    arrays do not define `__hash__`, and comparing them with `==` returns an array
    rather than a bool, so the pattern can neither be hashed nor safely
    equality-compared. It therefore must never appear bare in a pytree treedef, which
    jax hashes and compares when looking up jit caches.

    Instead, this wrapper is keyed by what uniquely determines the coloring: the
    identity of the function that was colored and the abstract structure (shapes and
    dtypes) of the arguments it was traced with. Two wrappers with equal keys hold
    the same coloring, so jit caches stay warm across operators rebuilt from the same
    function at same-shaped points. The pattern itself takes no part in hashing or
    equality.
    """

    internal_coloring: ColoredPattern
    """The wrapped asdex coloring. This is the payload carried along for sparse
    materialisation and it takes no part in hashing or equality comparison."""

    fn: jtu.Partial
    """The function whose Jacobian was colored, as passed by the caller before any
    closure conversion. Compared by object identity. The wrapper holds a strong
    reference to it, so the identity stays valid (and cannot be recycled by the
    garbage collector) for the wrapper's lifetime."""

    abstract_arguments: Hashable
    """A hashable description of the abstract arguments the coloring was computed
    for. It consists of the pytree structure of `(x, args)` together with the tuple
    of `jax.ShapeDtypeStruct` leaves, with weak dtypes stripped."""

    @classmethod
    def create(
        cls, internal_coloring: ColoredPattern, fn: Callable, x: Any, args: Any
    ) -> "_StaticColoring":
        """Builds the wrapper, deriving the hashable key from the arguments.

        Uses `jax.eval_shape` over `(x, args)`, so both concrete arrays and
        `jax.ShapeDtypeStruct`s are accepted. The resulting pytree of structs is
        flattened into a `(treedef, leaves)` tuple and the leaves are passed through
        lineax's `strip_weak_dtype`, so that a weakly typed point and a strongly
        typed point of the same shape and dtype produce equal keys.
        """
        abstract = strip_weak_dtype(
            jax.eval_shape(lambda point, extra: (point, extra), x, args)
        )
        leaves, structure = jtu.tree_flatten(abstract)
        return cls(
            internal_coloring, wrap_in_jax_partial(fn), (structure, tuple(leaves))
        )

    def __hash__(self) -> int:
        return hash((id(self.fn), self.abstract_arguments))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _StaticColoring)
            and self.fn is other.fn
            and self.abstract_arguments == other.abstract_arguments
        )


class SparseJacobianLinearOperator(AbstractLinearOperator):
    """Given a function `fn: X -> Y` and a point `x`, the linear operator
    `(d(fn)/dx)(x)`, kept sparse.

    The Jacobian's sparsity pattern and a matching coloring are determined once
    at construction (via `asdex`), so materialising the Jacobian at `x` costs
    one JVP or VJP per color rather than one per column or row. Materialise it
    with `as_bcoo` or `lineax.materialise` (sparse) or `as_matrix` (dense), or
    hand the operator straight to a splineax sparse solver, which materialises
    it through `lineax.materialise`.

    Only real dtypes are supported. To build many operators for the same
    function at different points without repeating sparsity detection, use
    [`splineax.SparseJacobianColoring`][].
    """

    fn: jtu.Partial
    x: Inexact[Array, " n"]
    args: PyTree[Any]
    coloring: _StaticColoring = eqx.field(static=True)
    transposed: bool = eqx.field(static=True)
    _in_structure: jax.ShapeDtypeStruct = eqx.field(static=True)
    _out_structure: jax.ShapeDtypeStruct = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)

    def __init__(
        self,
        fn: Callable,
        x: Inexact[ArrayLike, " n"],
        args: PyTree[Any] = None,
        *,
        sparsity: SparsityPattern | np.ndarray | BCOO | None = None,
        coloring: ColoredPattern | _StaticColoring | None = None,
        mode: JacobianMode | None = None,
        tags: object | frozenset[object] = (),
        transposed: bool = False,
        closure_convert: bool = True,
    ):
        """**Arguments:**

        - `fn`: a function `(x, args) -> y`, where both `x` and `y` are
            one-dimensional arrays of real dtype. Its Jacobian `d(fn)/dx` is the
            linear operator.
        - `x`: the point at which to evaluate `d(fn)/dx`.
        - `args`: extra arguments to `fn` that are not differentiated.
        - `sparsity`: optional known sparsity pattern of the Jacobian, as an
            `asdex.SparsityPattern`, a dense boolean mask, or a `BCOO` matrix.
            Skips sparsity detection (the pattern is still colored here).
        - `coloring`: optional precomputed `asdex.ColoredPattern`. Skips both
            sparsity detection and coloring. At most one of `sparsity` and
            `coloring` may be given.
        - `mode`: optional asdex coloring mode, either `"fwd"` (column coloring,
            materialised with JVPs) or `"rev"` (row coloring, materialised with
            VJPs). If not given, asdex picks based on the pattern.
        - `tags`: any lineax tags indicating whether the Jacobian has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if
            these tags are wrong.

        `transposed` and `closure_convert` are internal arguments, used by
        `transpose()` and [`splineax.SparseJacobianColoring`][].
        """
        # Keep the caller's function around for keying the coloring. Closure
        # conversion below produces a fresh object per call, so the original
        # identity is the stable one.
        unconverted_fn = wrap_in_jax_partial(fn)

        self.x = inexact_asarray(x)
        if jnp.issubdtype(self.x.dtype, jnp.complexfloating):
            raise TypeError(
                "`SparseJacobianLinearOperator` only supports real dtypes, but `x` "
                f"has dtype {self.x.dtype}."
            )
        if self.x.ndim != 1:
            raise ValueError(
                "`SparseJacobianLinearOperator` requires `x` to be one-dimensional, "
                f"but it has shape {self.x.shape}."
            )
        if closure_convert:
            fn = eqx.filter_closure_convert(fn, self.x, args)
        self.fn = wrap_in_jax_partial(fn)
        self.args = args
        self.tags = _frozenset(tags)
        self.transposed = transposed

        def function_of_point(point: Array) -> Array:
            return fn(point, args)

        match (coloring, sparsity):
            case (_StaticColoring() as wrapped, None):
                # `transpose()` and `SparseJacobianColoring.operator_at()` pass an existing
                # wrapper through, preserving its key and thereby the jit cache
                # identity of the operator.
                resolved = wrapped
            case (ColoredPattern() as pattern, None):
                resolved = _StaticColoring.create(pattern, unconverted_fn, self.x, args)
            case (None, None):
                detected = asdex_jacobian_coloring(function_of_point, self.x, mode=mode)
                resolved = _StaticColoring.create(
                    detected, unconverted_fn, self.x, args
                )
            case (None, known_sparsity):
                colored = asdex_jacobian_coloring_from_sparsity(
                    known_sparsity, mode=mode
                )
                resolved = _StaticColoring.create(colored, unconverted_fn, self.x, args)
            case _:
                raise TypeError(
                    "Pass at most one of `coloring` and `sparsity`, where "
                    "`coloring` must be an `asdex.ColoredPattern`."
                )
        self.coloring = resolved

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

    def _function_of_point(self) -> Callable[[Array], Array]:
        """Returns the forward map `x -> fn(x, args)` with `args` bound."""

        def function_of_point(point: Array) -> Array:
            return self.fn(point, self.args)

        return function_of_point

    def mv(self, vector: Inexact[Array, " b"]) -> Inexact[Array, " a"]:
        if self.transposed:
            _, vjp_function = jax.vjp(self._function_of_point(), self.x)
            (out,) = vjp_function(vector)
            return out
        _, out = jax.jvp(self._function_of_point(), (self.x,), (vector,))
        return out

    def as_bcoo(self) -> Inexact[BCOO, "a b"]:
        """Materialises the Jacobian at `x` as a `BCOO` matrix, using one JVP or VJP
        per color of the precomputed coloring."""
        jacobian = cast(
            BCOO,
            asdex_jacobian_from_coloring(
                self._function_of_point(),
                self.coloring.internal_coloring,
                "bcoo",
            )(self.x),
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
        # point and the same coloring wrapper (so the jit cache identity is
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


class SparseJacobianColoring(eqx.Module):
    """A precomputed Jacobian sparsity coloring for a function, reusable across
    evaluation points.

    Build one with [`splineax.SparseJacobianColoring.detect`][] or
    [`splineax.SparseJacobianColoring.from_sparsity`][], then call
    [`splineax.SparseJacobianColoring.operator_at`][] to obtain a
    [`splineax.SparseJacobianLinearOperator`][] at any point without repeating
    sparsity detection or coloring. All operators built from one instance share the
    same closure-converted function and coloring key, so passing them through the
    same jitted computation compiles only once.

    The coloring is valid for any `x` and `args` of the same abstract structure
    (shapes and dtypes) as those it was computed with. asdex guarantees that the
    sparsity pattern depends only on the traced computation graph, not on numerical
    values, so reusing the coloring at other points is always sound.
    """

    fn: jtu.Partial
    """The closure-converted function whose Jacobian was colored. Shared by every
    operator built through `operator_at`, so their pytree structures compare
    equal."""

    coloring: _StaticColoring = eqx.field(static=True)
    """The detected and colored sparsity pattern, wrapped so it can be carried as an
    equinox static field."""

    @classmethod
    def detect(
        cls,
        fn: Callable,
        x: Inexact[ArrayLike, " n"] | jax.ShapeDtypeStruct,
        args: PyTree[Any] = None,
        *,
        mode: JacobianMode | None = None,
    ) -> "SparseJacobianColoring":
        """Detects the Jacobian sparsity of `fn` and colors it.

        **Arguments:**

        - `fn`: a function `(x, args) -> y`, where both `x` and `y` are
            one-dimensional arrays of real dtype.
        - `x`: a representative point, or a `jax.ShapeDtypeStruct` describing one.
            Only its shape and dtype matter, since sparsity detection is structural.
        - `args`: extra arguments to `fn` that are not differentiated.
        - `mode`: optional asdex coloring mode, `"fwd"` or `"rev"`.
        """
        example_point = _example_point(x)
        converted_fn = wrap_in_jax_partial(
            eqx.filter_closure_convert(fn, example_point, args)
        )

        def function_of_point(point: Array) -> Array:
            return converted_fn(point, args)

        detected = asdex_jacobian_coloring(function_of_point, example_point, mode=mode)
        return cls(converted_fn, _StaticColoring.create(detected, fn, x, args))

    @classmethod
    def from_sparsity(
        cls,
        fn: Callable,
        x: Inexact[ArrayLike, " n"] | jax.ShapeDtypeStruct,
        sparsity: SparsityPattern | np.ndarray | BCOO,
        args: PyTree[Any] = None,
        *,
        mode: JacobianMode | None = None,
    ) -> "SparseJacobianColoring":
        """Colors a known Jacobian sparsity pattern of `fn`, skipping detection.

        **Arguments:**

        - `fn`: a function `(x, args) -> y`, where both `x` and `y` are
            one-dimensional arrays of real dtype.
        - `x`: a representative point, or a `jax.ShapeDtypeStruct` describing one.
            Not used for detection, but required to key the coloring for jit
            caching and to closure-convert `fn`.
        - `sparsity`: the known sparsity pattern of the Jacobian, as an
            `asdex.SparsityPattern`, a dense boolean mask, or a `BCOO` matrix.
        - `args`: extra arguments to `fn` that are not differentiated.
        - `mode`: optional asdex coloring mode, `"fwd"` or `"rev"`.
        """
        example_point = _example_point(x)
        converted_fn = wrap_in_jax_partial(
            eqx.filter_closure_convert(fn, example_point, args)
        )
        colored = asdex_jacobian_coloring_from_sparsity(sparsity, mode=mode)
        return cls(converted_fn, _StaticColoring.create(colored, fn, x, args))

    def operator_at(
        self,
        x: Inexact[ArrayLike, " n"],
        args: PyTree[Any] = None,
        tags: object | frozenset[object] = (),
    ) -> SparseJacobianLinearOperator:
        """Builds a [`splineax.SparseJacobianLinearOperator`][] at the point `x`,
        reusing the precomputed coloring.

        **Arguments:**

        - `x`: the point at which to evaluate the Jacobian. Must have the same
            shape and dtype as the point the coloring was computed for.
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


def _example_point(
    x: Inexact[ArrayLike, " n"] | jax.ShapeDtypeStruct,
) -> Inexact[Array, " n"]:
    """Turns a concrete point or a `jax.ShapeDtypeStruct` into a concrete array
    usable for tracing. Only the shape and dtype are meaningful to the callers."""
    if isinstance(x, jax.ShapeDtypeStruct):
        return jnp.empty(x.shape, x.dtype)
    return inexact_asarray(x)


# Lineax `singledispatch` registrations. `register_sparse_operator` cannot be reused
# here because its implementations read `operator.matrix`, which this operator does
# not have. `conj`, `diagonal` and `tridiagonal` are intentionally left unregistered:
# the operator is real-valued and materialise-first, so lineax's informative default
# errors apply.


@materialise.register(SparseJacobianLinearOperator)
def _(operator: SparseJacobianLinearOperator) -> BCOOLinearOperator:
    # Convert to a concrete sparse operator. This is how the solvers consume it.
    return BCOOLinearOperator(operator.as_bcoo(), operator.tags)


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
