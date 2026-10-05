"""Tag objects for sparse operators.

These mark a property a solver may rely on, matching lineax's own repr-identified tags.
"""

import secrets
from collections.abc import Callable
from typing import Any, Literal, cast, overload

import asdex
import jax
import jax.core
import jax.numpy as jnp
import lineax as lx
import numpy as np
from asdex import ColoredPattern, SparsityPattern
from jax.experimental.sparse import BCOO, BCSR
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, Inexact, PyTree
from lineax._operator import inexact_asarray

from ._sparse import SparseLinearOperator

JacobianDirection = Literal["fwd", "bwd"]
"""The direction a Jacobian is computed in, named as in `lineax.JacobianLinearOperator`.

`"fwd"` colors columns and uses one JVP per color. `"bwd"` colors rows and uses one VJP
per color.
"""

_AsdexMode = Literal["fwd", "rev"]
"""The mode names asdex uses for a Jacobian coloring, where `"rev"` means `"bwd"`."""


def _asdex_mode(jac: JacobianDirection | None) -> _AsdexMode | None:
    """Translate a lineax direction into the mode name asdex uses."""
    match jac:
        case "bwd":
            return "rev"
        case "fwd" | None:
            return jac


def _jac_from_asdex_mode(mode: str) -> JacobianDirection:
    """Translate an asdex mode name into the lineax direction."""
    match mode:
        case "fwd":
            return "fwd"
        case "rev":
            return "bwd"
        case _:
            raise ValueError(
                f"A Jacobian coloring must have mode `fwd` or `rev`, got `{mode}`."
            )


def _transposed_direction(jac: JacobianDirection | None) -> JacobianDirection | None:
    """Swap `"fwd"` and `"bwd"`, keeping None."""
    match jac:
        case "fwd":
            return "bwd"
        case "bwd":
            return "fwd"
        case None:
            return None


def _coloring_on_entries(
    coloring: ColoredPattern,
    indices: np.ndarray,
    shape: tuple[int, ...],
    mode: _AsdexMode,
) -> ColoredPattern:
    """Build a coloring with the colors of `coloring` on a new list of entries.

    The colors belong to rows or columns, not to entries. So the same colors stay valid
    when the entries change order, and also when the pattern is transposed together with
    a swap of `mode`.
    """
    sparsity = SparsityPattern.from_coo(
        indices[:, 0], indices[:, 1], (shape[0], shape[1])
    )
    return ColoredPattern(
        sparsity=sparsity,
        colors=coloring.colors,
        num_colors=coloring.num_colors,
        symmetric=False,
        mode=mode,
    )


def _reusable_mode(coloring: ColoredPattern) -> _AsdexMode | None:
    """Return the mode of a plain row or column coloring, or None for any other kind.

    Only a plain coloring can move to new entries with `_coloring_on_entries`. A
    symmetric (star) coloring has extra structure tied to its entries.
    """
    if coloring.symmetric:
        return None
    match coloring.mode:
        case "fwd":
            return "fwd"
        case "rev":
            return "rev"
        case _:
            return None


def _transpose_coloring(coloring: ColoredPattern) -> ColoredPattern | None:
    """Return a coloring of the transposed pattern, or None if it cannot be reused.

    A column coloring of a matrix is a row coloring of its transpose, so the colors carry
    over with the mode swapped. A symmetric (star) coloring has extra structure that does
    not carry over this way, so it returns None and the transposed tag will recolor.
    """
    mode = _reusable_mode(coloring)
    if mode is None:
        return None
    indices, shape = coloring_index_array(coloring)
    transposed_mode: _AsdexMode = "rev" if mode == "fwd" else "fwd"
    return _coloring_on_entries(
        coloring, indices[:, ::-1], shape[::-1], transposed_mode
    )


class _HasRepr:
    """A tag object whose only content is its repr, matching lineax's own tags."""

    def __init__(self, string: str) -> None:
        self.string = string

    def __repr__(self) -> str:
        return self.string


sparse_indices_sorted = _HasRepr("sparse_indices_sorted")
"""One global assertion that an operator's indices are already row-major sorted, so
`Pardiso` and `Spsolve` may skip the sort they would otherwise do in `init`.

`BCOOLinearOperator` and `BCSRLinearOperator` add this automatically when the matrix they
wrap already carries `indices_sorted`.
"""


class _ContentPatternTag:
    """A sparsity-pattern tag identified by the content of its index arrays.

    Two tags are equal when their indices match exactly, so operators built with the same
    pattern reuse each other's factorization even when tagged separately. This follows
    asdex's `_HashableEntries`, which carries concrete index arrays as hashable static aux
    data.

    The tag also carries the Jacobian coloring that a solver needs to turn a tagged
    `lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator` into a `BCOO`.
    The coloring is either given up front or computed on first use and cached. It is
    not part of equality, so a colored tag and an uncolored tag of one pattern compare
    equal.
    """

    __slots__ = ("_colorings", "_hash", "coloring", "indices", "shape")

    def __init__(
        self,
        indices: np.ndarray,
        shape: tuple[int, ...],
        coloring: ColoredPattern | None = None,
    ) -> None:
        # One index dtype for every source, so equal patterns compare and hash equal.
        self.indices = np.asarray(indices, dtype=np.int64)
        """The `(nse, 2)` array of row and column indices, in entry order."""
        self.shape = shape
        """The shape of the matrix the pattern belongs to."""
        self.coloring = coloring
        """A coloring fixed up front, which every conversion will use. Its entries are in
        the same order as `indices`."""
        self._hash: int | None = None
        self._colorings: dict[JacobianDirection | None, ColoredPattern] = {}

    def sparsity_pattern(self) -> SparsityPattern:
        """Return the pattern as an `asdex.SparsityPattern`, keeping the entry order."""
        return SparsityPattern.from_coo(
            self.indices[:, 0], self.indices[:, 1], (self.shape[0], self.shape[1])
        )

    def jacobian_coloring(self, jac: JacobianDirection | None = None) -> ColoredPattern:
        """Return a Jacobian coloring of the pattern.

        A coloring fixed up front wins, and then `jac` is ignored. Otherwise the pattern
        is colored for `jac` on first use and the result is cached on the tag. With `jac`
        set to None, asdex picks the direction. The coloring keeps the entry order of
        `indices`, so a `BCOO` built from it has its entries in that order too.
        """
        if self.coloring is not None:
            return self.coloring
        cached = self._colorings.get(jac)
        if cached is None:
            cached = asdex.jacobian_coloring_from_sparsity(
                self.sparsity_pattern(), mode=_asdex_mode(jac)
            )
            self._colorings[jac] = cached
        return cached

    def __hash__(self) -> int:
        # Content hashing is O(nnz), so compute it lazily and cache it. A frozenset
        # hashes the tag every time it is built.
        if self._hash is None:
            self._hash = hash((self.indices.tobytes(), self.shape))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, _ContentPatternTag):
            return NotImplemented
        return self.shape == other.shape and np.array_equal(self.indices, other.indices)

    def transpose(self) -> "_ContentPatternTag":
        """Return the tag of the transposed pattern.

        The two index columns swap places and the entries keep their order. `BCOO.T`
        transposes the same way, so the new tag still lines up with the transposed
        matrix's values.
        """
        transposed_indices = np.ascontiguousarray(self.indices[:, ::-1])
        transposed_coloring = (
            None if self.coloring is None else _transpose_coloring(self.coloring)
        )
        transposed = _ContentPatternTag(
            transposed_indices, self.shape[::-1], transposed_coloring
        )
        # Lazily computed colorings carry over as well, with their direction swapped.
        for jac, coloring in self._colorings.items():
            transposed_cached = _transpose_coloring(coloring)
            if transposed_cached is not None:
                transposed._colorings[_transposed_direction(jac)] = transposed_cached
        return transposed

    def row_major_sorted(self) -> "_ContentPatternTag":
        """Return the tag with its entries sorted by row, then by column.

        A `BCSR` matrix always stores its entries in this order, so a `BCSR` operator
        uses this to keep its tag in line with its own entries.
        """
        order = np.lexsort((self.indices[:, 1], self.indices[:, 0]))
        sorted_indices = self.indices[order]
        sorted_coloring = None
        if self.coloring is not None:
            mode = _reusable_mode(self.coloring)
            if mode is not None:
                sorted_coloring = _coloring_on_entries(
                    self.coloring, sorted_indices, self.shape, mode
                )
        return _ContentPatternTag(sorted_indices, self.shape, sorted_coloring)


class _IdentityPatternTag:
    """A sparsity-pattern tag identified by a random id, for use under jit.

    A traced index array cannot be hashed, so this stands in for the content tag. Two
    instances differ, and the same instance threaded onto several operators marks them as
    sharing a pattern. The random id keeps the tag hashable and stable across pytree
    flatten and unflatten, where a bare `object()` identity would not survive.
    """

    __slots__ = ("_id", "transposed")

    def __init__(self, identifier: int | None = None, transposed: bool = False) -> None:
        self._id = secrets.randbits(128) if identifier is None else identifier
        self.transposed = transposed

    def __hash__(self) -> int:
        return hash((self._id, self.transposed))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _IdentityPatternTag):
            return NotImplemented
        return self._id == other._id and self.transposed == other.transposed

    def transpose(self) -> "_IdentityPatternTag":
        """Return the tag of the transposed pattern.

        It keeps the random id and flips `transposed`. Transposing twice will return a
        tag equal to the original, and every transpose of one tag compares equal.
        """
        return _IdentityPatternTag(self._id, not self.transposed)


PatternTag = _ContentPatternTag | _IdentityPatternTag
"""Either kind of sparsity-pattern tag."""


def find_pattern_tag(tags: frozenset[object]) -> PatternTag | None:
    """Return the sparsity-pattern tag among `tags`, or None if there is none.

    An operator carries at most one pattern tag, so the first one found is the only one.
    """
    for tag in tags:
        if isinstance(tag, (_ContentPatternTag, _IdentityPatternTag)):
            return tag
    return None


def _transpose_pattern_tag_rule(tags: frozenset[object]) -> PatternTag | None:
    """Transpose rule for `lineax.transpose_tags`, which drops tags no rule matches.

    With this rule a transposed operator carries the tag of the transposed pattern. That
    holds for `lineax.JacobianLinearOperator`, `lineax.FunctionLinearOperator`,
    `lineax.TaggedLinearOperator` and the sparse operators in this package.
    """
    tag = find_pattern_tag(tags)
    if tag is None:
        return None
    return tag.transpose()


lx.transpose_tags_rules.append(_transpose_pattern_tag_rule)


def coloring_index_array(
    coloring: ColoredPattern,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Read a coloring's COO index array and shape as concrete numpy data.

    The pattern comes from the precomputed asdex coloring, whose row and column indices are
    always concrete, so this never sees a traced array.
    """
    sparsity = coloring.sparsity
    indices = np.stack([np.asarray(sparsity.rows), np.asarray(sparsity.cols)], axis=1)
    return indices, tuple(sparsity.shape)


def sparsity_tag_from_coloring(coloring: ColoredPattern) -> _ContentPatternTag:
    """Build a content pattern tag that holds `coloring` and takes its entry order."""
    indices, shape = coloring_index_array(coloring)
    return _ContentPatternTag(indices, shape, coloring)


PatternSource = (
    BCOO | BCSR | SparseLinearOperator | SparsityPattern | ColoredPattern | np.ndarray
)
"""Everything a sparsity pattern can be read from without a function."""

TaggedOperator = (
    lx.JacobianLinearOperator | lx.FunctionLinearOperator | lx.TaggedLinearOperator
)
"""The lineax operators that a solver accepts when they carry a sparsity-pattern tag."""


def operator_pattern_tag(operator: lx.AbstractLinearOperator) -> PatternTag | None:
    """Return the operator's sparsity-pattern tag, or None if it carries none.

    Solvers read this in `update` to decide whether an operator shares a state's pattern.
    """
    return find_pattern_tag(getattr(operator, "tags", frozenset()))


def pattern_indices(
    pattern: PatternSource | _ContentPatternTag | TaggedOperator,
) -> tuple[np.ndarray | None, tuple[int, ...] | None]:
    """Read a pattern's COO index array and shape as concrete numpy data.

    Returns `(None, None)` when the indices are traced, or for a tagged lineax operator
    whose tag carries no indices. A dense boolean mask lists its entries in row-major
    order, the same order asdex uses for it.
    """
    match pattern:
        case _ContentPatternTag():
            return pattern.indices, tuple(pattern.shape)
        case (
            lx.JacobianLinearOperator()
            | lx.FunctionLinearOperator()
            | lx.TaggedLinearOperator()
        ):
            tag = operator_pattern_tag(pattern)
            if isinstance(tag, _ContentPatternTag):
                return tag.indices, tuple(tag.shape)
            return None, None
        case ColoredPattern():
            return coloring_index_array(pattern)
        case SparsityPattern(rows=rows, cols=cols, shape=shape):
            indices = np.stack([np.asarray(rows), np.asarray(cols)], axis=1)
            return indices, tuple(shape)
        case np.ndarray():
            return np.argwhere(pattern), tuple(pattern.shape)
        case BCOO():
            indices, shape = pattern.indices, pattern.shape
        case BCSR():
            bcoo = pattern.to_bcoo()
            indices, shape = bcoo.indices, bcoo.shape
        case _ if isinstance(pattern, SparseLinearOperator):
            return pattern_indices(pattern.matrix)
        case _:
            return None, None
    if isinstance(indices, jax.core.Tracer):
        return None, None
    return np.asarray(indices), tuple(shape)


def sparsity_pattern_tag(
    pattern: PatternSource | _ContentPatternTag | TaggedOperator | None = None,
) -> PatternTag:
    """Create a tag marking an operator's structural sparsity pattern.

    Attach the tag to operators through their `tags` argument. Two operators carrying
    equal tags are asserted to have exactly the same index arrays, in the same order, so
    a solver may reuse one operator's factorization for the other.

    Given a concrete `pattern`, the tag is content-hashed, so independently tagged
    operators with the same indices get equal tags. With no argument, or a pattern whose
    indices are traced under jit, the tag instead carries a random id. Thread that one
    tag object onto every operator sharing the pattern to mark them as equal.

    A content-hashed tag also lets a solver turn a tagged `lineax.JacobianLinearOperator`
    or `lineax.FunctionLinearOperator` into a `BCOO`. The Jacobian coloring this needs is
    computed on first use and cached on the tag. A pattern that already holds a coloring
    (an `asdex.ColoredPattern`) keeps it. Use [`splineax.sparsity_coloring_tag`][] to
    compute the coloring up front.

    Given a tag, or a lineax operator that carries one, that tag is returned.
    """
    match pattern:
        case None:
            return _IdentityPatternTag()
        case ColoredPattern():
            return sparsity_tag_from_coloring(pattern)
        case _ContentPatternTag():
            return pattern
        case (
            lx.JacobianLinearOperator()
            | lx.FunctionLinearOperator()
            | lx.TaggedLinearOperator()
        ):
            operator_tag = operator_pattern_tag(pattern)
            if operator_tag is not None:
                return operator_tag
    indices, shape = pattern_indices(pattern)
    if indices is None or shape is None:
        return _IdentityPatternTag()
    return _ContentPatternTag(indices, shape)


def example_point(
    point: PyTree[Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct],
) -> PyTree[Inexact[Array, "..."]]:
    """Turn a pytree of arrays or `jax.ShapeDtypeStruct`s into concrete arrays.

    Only the shapes and dtypes are meaningful to the callers, which use the result to
    trace a function.
    """

    def example_leaf(
        leaf: Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct,
    ) -> Inexact[Array, "..."]:
        if isinstance(leaf, jax.ShapeDtypeStruct):
            return jnp.zeros(leaf.shape, leaf.dtype)
        return inexact_asarray(leaf)

    return jax.tree.map(example_leaf, point)


def flat_function(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    point: PyTree[Array],
) -> tuple[Callable[[Array], Array], Array]:
    """Wrap `function` so it maps a flat vector to a flat vector.

    Returns the wrapped function and `point` raveled. Inputs and outputs are raveled in
    pytree leaf order, which is the order the splineax solvers ravel vectors in.
    """
    flat_point, unravel_point = ravel_pytree(point)

    def function_of_flat_point(flat_input: Array) -> Array:
        flat_output, _ = ravel_pytree(function(unravel_point(flat_input)))
        return flat_output

    return function_of_flat_point, flat_point


@overload
def sparsity_coloring_tag(
    pattern: PatternSource,
    *,
    jac: JacobianDirection | None = None,
) -> _ContentPatternTag: ...


@overload
def sparsity_coloring_tag(
    pattern: Callable[..., Any],
    point: PyTree[Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct],
    args: PyTree[Any] = None,
    *,
    jac: JacobianDirection | None = None,
) -> _ContentPatternTag: ...


def sparsity_coloring_tag(
    pattern: PatternSource | Callable[..., Any],
    point: PyTree[Inexact[ArrayLike, "..."] | jax.ShapeDtypeStruct] = None,
    args: PyTree[Any] = None,
    *,
    jac: JacobianDirection | None = None,
) -> _ContentPatternTag:
    """Create a sparsity-pattern tag that holds a Jacobian coloring.

    A tagged `lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator` is turned
    into a `BCOO` with one JVP or VJP per color. This tag computes the coloring now, so
    no solver has to compute it later. The tag compares equal to a
    [`splineax.sparsity_pattern_tag`][] of the same pattern, so operators with either
    tag will share a factorization.

    Coloring runs host-side on numpy data, so call this outside `jax.jit` and close over
    the tag or pass it in as a static value.

    **Arguments:**

    - `pattern`: either a known pattern, or a function `fn(x, args) -> y` whose Jacobian
        pattern is detected with asdex. A known pattern is a `BCOO`, a `BCSR`, a sparse
        operator from this package, an `asdex.SparsityPattern`, a dense boolean mask, or
        an `asdex.ColoredPattern`. A colored pattern is used as it is.
    - `point`: with a function, a point of the right structure, as arrays or
        `jax.ShapeDtypeStruct`s. Only the shapes and dtypes matter. Inputs and outputs
        may be pytrees, which are raveled in leaf order.
    - `args`: with a function, extra arguments to it that are not differentiated.
    - `jac`: `"fwd"` colors columns for JVPs, `"bwd"` colors rows for VJPs. With None,
        asdex picks whichever needs fewer colors.
    """
    if point is not None:
        # The overloads only allow a `point` together with a function.
        function = cast(Callable[[PyTree[Array], PyTree[Any]], PyTree[Array]], pattern)
        function_of_flat_point, flat_point = flat_function(
            lambda input_point: function(input_point, args), example_point(point)
        )
        coloring = asdex.jacobian_coloring(
            function_of_flat_point, flat_point, mode=_asdex_mode(jac)
        )
        return sparsity_tag_from_coloring(coloring)
    match pattern:
        case ColoredPattern():
            coloring_jac = _jac_from_asdex_mode(pattern.mode)
            if jac is not None and coloring_jac != jac:
                raise ValueError(
                    f"The coloring was computed for `jac={coloring_jac!r}`, which does "
                    f"not match `jac={jac!r}`."
                )
            return sparsity_tag_from_coloring(pattern)
        case _ if callable(pattern) and not isinstance(pattern, SparseLinearOperator):
            raise TypeError(
                "Detecting the pattern of a function needs a `point` of the right "
                "structure."
            )
        case _:
            # Without a `point`, the overloads only allow a known pattern.
            indices, shape = pattern_indices(cast(PatternSource, pattern))
            if indices is None or shape is None:
                raise TypeError(
                    "`sparsity_coloring_tag` needs concrete indices, so it cannot be "
                    "called on a traced pattern or on an unsupported type, got "
                    f"`{type(pattern).__name__}`."
                )
            tag = _ContentPatternTag(indices, shape)
            tag.coloring = tag.jacobian_coloring(jac)
            return tag
