"""Shared coloring machinery for the automatic-differentiation operators.

`SparseJacobianLinearOperator` and `SparseFunctionLinearOperator` are the sparse
analogues of the two lineax operators that are defined by a function rather than by
a matrix. Neither stores a matrix, so they cannot compose their behavior from
`_operations.py` the way `BCOOLinearOperator` and `BCSRLinearOperator` do. What they
share instead lives here: the `asdex` coloring wrapper, the raveling that lets a
function take and return pytrees while the sparsity pattern stays two-dimensional,
and the lineax `singledispatch` registrations that read only an operator's tags.

The sparsity pattern and coloring depend only on the traced computation graph of the
function, not on the numerical values of its inputs. They are therefore computed once,
at construction time, and reused for every evaluation afterwards. Two colorings of the
same sparsity pattern flatten to identical treedefs, so a jitted function that accepts
one compiles exactly once. The public entry point for creating and carrying colorings
is [`splineax.JacobianColoring`][].
"""

from collections.abc import Callable
from typing import Any, Literal

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
    has_unit_diagonal,
    is_diagonal,
    is_lower_triangular,
    is_negative_semidefinite,
    is_positive_semidefinite,
    is_symmetric,
    is_tridiagonal,
    is_upper_triangular,
)
from lineax._operator import inexact_asarray, strip_weak_dtype
from lineax._solver.misc import PackedStructures, ravel_vector, unravel_solution
from lineax._tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
)

JacobianMode = Literal["fwd", "bwd"]
"""The AD mode a Jacobian coloring is built for: `"fwd"` (column coloring, JVPs) or
`"bwd"` (row coloring, VJPs). Spelled as in lineax and JAX, so a
`lineax.JacobianLinearOperator`'s `jac` argument carries over unchanged. asdex spells
reverse mode `"rev"`, which `asdex_mode` translates to."""


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
    argument of [`splineax.SparseJacobianLinearOperator`][] or
    [`splineax.SparseFunctionLinearOperator`][] together with a function, or bind it to
    a specific function once with
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
        point = example_point(x)

        def function_of_point(point: PyTree[Array]) -> PyTree[Array]:
            return fn(point, args)

        # Detect against the raveled map, so that the pattern is over the same flat
        # index order the operator materialises in, whatever structure `x` has.
        flat_function_of_point, flat_point = flatten_map(function_of_point, point)
        detected = asdex.jacobian_coloring(
            flat_function_of_point, flat_point, mode=asdex_mode(mode)
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
        colored = asdex.jacobian_coloring_from_sparsity(sparsity, mode=asdex_mode(mode))
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


def asdex_mode(mode: JacobianMode | None) -> Literal["fwd", "rev"] | None:
    """Translates a coloring mode into the asdex spelling. Both call forward mode
    `"fwd"`, but reverse mode is `"bwd"` here, as in lineax and JAX, where asdex calls
    it `"rev"`. `None` is passed through, leaving the choice to asdex."""
    return "rev" if mode == "bwd" else mode


def example_point(
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


def zero_point(
    structure: PyTree[jax.ShapeDtypeStruct],
) -> PyTree[Inexact[Array, "..."]]:
    """Builds a pytree of zeros with the given structure.

    A linear map has the same Jacobian everywhere, so
    [`splineax.SparseFunctionLinearOperator`][] is free to trace and materialise at any
    point it likes. It uses this one. Zeros rather than the uninitialised memory
    `example_point` hands out, since that memory can hold NaNs, which would survive a
    multiplication by zero and end up in the materialised matrix.
    """
    return jtu.tree_map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), structure)


def check_real_dtypes(x: PyTree[Inexact[Array, "..."]], operator_name: str) -> None:
    """Raises unless every leaf of `x` has a real dtype.

    Both AD operators hand their function to asdex, which colors real-valued Jacobians.
    Rejecting complex leaves up front also justifies the unconditional
    positive-semidefinite-implies-symmetric rule in `register_ad_operator`.
    """
    for leaf in jtu.tree_leaves(x):
        if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
            raise TypeError(
                f"`{operator_name}` only supports real dtypes, but `x` has a leaf of "
                f"dtype {leaf.dtype}."
            )


def packed_structures(
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


def flatten_map(
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
    packed = packed_structures(out_structure, in_structure)
    # Ravelling `x` reads the pair's output side, so it needs one of its own.
    point_packed = packed_structures(in_structure, in_structure)

    def flat_function_of_point(flat_point: Inexact[Array, " n"]) -> Array:
        point = unravel_solution(flat_point, packed)
        return ravel_vector(function_of_point(point), packed)

    return flat_function_of_point, ravel_vector(x, point_packed)


def check_structures_survive_materialisation(operator: AbstractLinearOperator) -> None:
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
                "`lineax.materialise` on a "
                f"`{type(operator).__name__}` requires "
                "one-dimensional array in- and out-structures, since the resulting "
                "`BCOOLinearOperator` carries only a matrix and would drop anything "
                f"richer. The {name}-structure is {structure}. Use `as_bcoo()` "
                "instead, keeping the structures yourself, as the splineax solvers do."
            )


# Tag-only structural predicates, mirroring `_operations.py` but reading only
# `operator.tags`. The positive/negative-semidefinite-implies-symmetric rule needs no
# dtype check here: for complex dtypes positive semidefinite means Hermitian, which
# differs from symmetric, so lineax restricts the implication to real dtypes. The AD
# operators reject complex inputs at construction, so the implication always applies.


def _is_symmetric(operator: Any) -> bool:
    return (
        symmetric_tag in operator.tags
        or diagonal_tag in operator.tags
        or positive_semidefinite_tag in operator.tags
        or negative_semidefinite_tag in operator.tags
    )


def _is_diagonal(operator: Any) -> bool:
    return diagonal_tag in operator.tags or (
        operator.in_size() == 1 and operator.out_size() == 1
    )


def _is_tridiagonal(operator: Any) -> bool:
    return tridiagonal_tag in operator.tags or diagonal_tag in operator.tags


def _has_unit_diagonal(operator: Any) -> bool:
    return unit_diagonal_tag in operator.tags


def _is_lower_triangular(operator: Any) -> bool:
    return lower_triangular_tag in operator.tags


def _is_upper_triangular(operator: Any) -> bool:
    return upper_triangular_tag in operator.tags


def _is_positive_semidefinite(operator: Any) -> bool:
    return positive_semidefinite_tag in operator.tags


def _is_negative_semidefinite(operator: Any) -> bool:
    return negative_semidefinite_tag in operator.tags


def register_ad_operator(cls: type[AbstractLinearOperator]) -> None:
    """Installs the tag-only lineax `singledispatch` registrations on an AD operator.

    The counterpart of `register_sparse_operator` in `_operations.py`, which cannot be
    reused here because its implementations read `operator.matrix`, which neither AD
    operator has. `conj`, `diagonal` and `tridiagonal` are intentionally left
    unregistered: these operators are real-valued and materialise-first, so lineax's
    informative default errors apply. `linearise` and `materialise` differ between the
    two operators and are registered by each of them.
    """
    is_symmetric.register(cls, _is_symmetric)
    is_diagonal.register(cls, _is_diagonal)
    is_tridiagonal.register(cls, _is_tridiagonal)
    has_unit_diagonal.register(cls, _has_unit_diagonal)
    is_lower_triangular.register(cls, _is_lower_triangular)
    is_upper_triangular.register(cls, _is_upper_triangular)
    is_positive_semidefinite.register(cls, _is_positive_semidefinite)
    is_negative_semidefinite.register(cls, _is_negative_semidefinite)
