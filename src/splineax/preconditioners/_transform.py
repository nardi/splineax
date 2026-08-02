"""Transformations applied to the whole system before it is preconditioned.

A reordering such as reverse Cuthill-McKee is not a parameter of any one
preconditioner. It applies to the entire system or not at all, and nesting it inside
(say) a block partitioner would make that object responsible for something well beyond
deriving its own parameter. So transforms live on the solver, as an ordered sequence,
and the preconditioner only ever sees the pattern they produced.

Every transform is an invertible pair `(L, R)`: the solver solves `(L A R) y = L b` and
recovers `x = R y`. That one shape covers a symmetric permutation `(P, P^T)`, the
unsymmetric row permutation a bipartite matching produces `(P, I)`, and row/column
equilibration `(D_row, D_col)`, so the union below can grow without the solver changing.

`SystemTransform` is deliberately a *closed* union: it is the solver's internal
currency, and every function here must handle it exhaustively. Users extend this by
writing new *stages* -- objects satisfying `SymbolicTransform` or `NumericTransform`,
which produce transforms -- not new members of the union.
"""

from typing import Any, Protocol, TypeAlias, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from jaxtyping import Array, Inexact, Integer, PyTree
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
    materialise,
)
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

from splineax._pattern import ConcreteCooPattern, CooPattern
from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import SparseJacobianLinearOperator


class IdentityTransform(eqx.Module):
    """The transform that does nothing.

    Every function below short-circuits on it, so a solver with no transforms costs
    exactly what it would have without the machinery.
    """


class SymmetricPermutation(eqx.Module):
    """Reorders rows and columns alike: `A -> P A P^T`.

    Entry `(r, c)` of the original matrix becomes entry `(inverse[r], inverse[c])` of
    the transformed one, and `permutation[i]` is the original index that ends up at
    position `i`. Symmetry and definiteness both survive this, which is what lets a
    reordering be used with `lineax.CG`.
    """

    permutation: Integer[Array, " size"]
    """`permutation[i]` is the original index placed at position `i`."""
    inverse: Integer[Array, " size"]
    """`inverse[permutation[i]] == i`: where each original index ends up."""

    @staticmethod
    def from_order(order: np.ndarray) -> "SymmetricPermutation":
        """Build from an ordering, computing the inverse."""
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size, dtype=order.dtype)
        return SymmetricPermutation(jnp.asarray(order), jnp.asarray(inverse))


class ComposedTransform(eqx.Module):
    """Several transforms applied in order; the inverse folds in reverse."""

    transforms: tuple["SystemTransform", ...]


SystemTransform: TypeAlias = (
    IdentityTransform | SymmetricPermutation | ComposedTransform
)


@runtime_checkable
class SymbolicTransform(Protocol):
    """A transform stage derivable from the sparsity pattern alone.

    Reorderings are all of this kind: reverse Cuthill-McKee, minimum degree, a maximum
    bipartite matching.
    """

    def symbolic(self, pattern: CooPattern) -> SystemTransform:
        """Derive the transform this stage applies to a pattern."""
        ...


@runtime_checkable
class NumericTransform(Protocol):
    """A transform stage that needs the matrix values, such as equilibration."""

    def numeric(self, operator: AbstractLinearOperator) -> SystemTransform:
        """Derive the transform this stage applies to an operator."""
        ...


# Structural tags describe where the nonzeros are, so a reordering invalidates all of
# them. Definiteness and symmetry are properties of the quadratic form and survive a
# *symmetric* permutation.
_STRUCTURAL_TAGS = frozenset(
    {
        diagonal_tag,
        tridiagonal_tag,
        unit_diagonal_tag,
        lower_triangular_tag,
        upper_triangular_tag,
    }
)


def compose(*transforms: SystemTransform) -> SystemTransform:
    """Combine transforms into one applied left to right, flattening and dropping
    identities so that the common no-transform case stays free."""
    flat: list[SystemTransform] = []
    for transform in transforms:
        match transform:
            case IdentityTransform():
                continue
            case ComposedTransform(inner):
                flat.extend(inner)
            case _:
                flat.append(transform)
    match flat:
        case []:
            return IdentityTransform()
        case [only]:
            return only
        case _:
            return ComposedTransform(tuple(flat))


def preserves_symmetry(transform: SystemTransform) -> bool:
    """Whether `A` symmetric implies the transformed matrix is symmetric.

    Load-bearing: it decides whether definiteness tags survive, and `lineax.CG` refuses
    an operator that is not tagged positive- or negative-semidefinite.
    """
    match transform:
        case IdentityTransform() | SymmetricPermutation():
            return True
        case ComposedTransform(inner):
            return all(preserves_symmetry(t) for t in inner)


def transformed_tags(
    transform: SystemTransform, tags: frozenset[object]
) -> frozenset[object]:
    """The tags that still hold after `transform`.

    Structural tags never survive a reordering. Symmetry and definiteness do, but only
    for a transform that preserves symmetry -- an unsymmetric row permutation, as a
    matching produces, destroys both.
    """
    match transform:
        case IdentityTransform():
            return tags
        case ComposedTransform(inner):
            for t in inner:
                tags = transformed_tags(t, tags)
            return tags
        case SymmetricPermutation():
            kept = tags - _STRUCTURAL_TAGS
            if preserves_symmetry(transform):
                return kept
            return kept - {
                symmetric_tag,
                positive_semidefinite_tag,
                negative_semidefinite_tag,
            }


def transpose_transform(transform: SystemTransform) -> SystemTransform:
    """The transform for the transposed system.

    `(P A P^T)^T = P A^T P^T`, so a symmetric permutation is its own transpose.
    """
    match transform:
        case IdentityTransform() | SymmetricPermutation():
            return transform
        case ComposedTransform(inner):
            return ComposedTransform(tuple(transpose_transform(t) for t in inner))


def conj_transform(transform: SystemTransform) -> SystemTransform:
    """The transform for the conjugated system. Permutations are real, so unchanged."""
    match transform:
        case IdentityTransform() | SymmetricPermutation():
            return transform
        case ComposedTransform(inner):
            return ComposedTransform(tuple(conj_transform(t) for t in inner))


def transform_pattern(transform: SystemTransform, pattern: CooPattern) -> CooPattern:
    """Apply a transform to a sparsity pattern, so the next stage sees its output."""
    match transform:
        case IdentityTransform():
            return pattern
        case ComposedTransform(inner):
            for t in inner:
                pattern = transform_pattern(t, pattern)
            return pattern
        case SymmetricPermutation(_, inverse):
            return CooPattern(
                inverse[pattern.rows], inverse[pattern.cols], pattern.shape
            )


def transform_operator(
    transform: SystemTransform, operator: AbstractLinearOperator
) -> AbstractLinearOperator:
    """Apply a transform to an operator, keeping it sparse.

    A permutation is pure index arithmetic on the stored coordinates: no values move,
    and nothing is materialised. The result is always a `BCOOLinearOperator`, whatever
    sparse operator went in.
    """
    match transform:
        case IdentityTransform():
            return operator
        case ComposedTransform(inner):
            for t in inner:
                operator = transform_operator(t, operator)
            return operator
        case SymmetricPermutation(_, inverse):
            matrix = _as_bcoo_matrix(operator)
            rows, cols = matrix.indices[:, 0], matrix.indices[:, 1]
            indices = jnp.stack([inverse[rows], inverse[cols]], axis=1)
            permuted = BCOO(
                (matrix.data, indices), shape=matrix.shape, indices_sorted=False
            )
            return BCOOLinearOperator(
                permuted, transformed_tags(transform, operator_tags(operator))
            )


def transform_vector(
    transform: SystemTransform, vector: PyTree[Inexact[Array, " size"]]
) -> PyTree[Inexact[Array, " size"]]:
    """Move a right-hand side into the transformed system's coordinates."""
    match transform:
        case IdentityTransform():
            return vector
        case ComposedTransform(inner):
            for t in inner:
                vector = transform_vector(t, vector)
            return vector
        case SymmetricPermutation(permutation, _):
            return _permute_leaf(vector, permutation, "vector")


def transform_solution(
    transform: SystemTransform, solution: PyTree[Inexact[Array, " size"]]
) -> PyTree[Inexact[Array, " size"]]:
    """Move a solution-space vector (such as `options["y0"]`) into transformed
    coordinates."""
    match transform:
        case IdentityTransform():
            return solution
        case ComposedTransform(inner):
            for t in inner:
                solution = transform_solution(t, solution)
            return solution
        case SymmetricPermutation(permutation, _):
            return _permute_leaf(solution, permutation, "y0")


def untransform_solution(
    transform: SystemTransform, solution: PyTree[Inexact[Array, " size"]]
) -> PyTree[Inexact[Array, " size"]]:
    """Move a solution back out of the transformed system's coordinates."""
    match transform:
        case IdentityTransform():
            return solution
        case ComposedTransform(inner):
            for t in reversed(inner):
                solution = untransform_solution(t, solution)
            return solution
        case SymmetricPermutation(_, inverse):
            return _permute_leaf(solution, inverse, "solution")


def operator_tags(operator: AbstractLinearOperator) -> frozenset[object]:
    """Read an operator's properties back out as a tag set.

    Goes through lineax's predicates rather than reading a `tags` attribute, so this
    works for any operator, including ones that derive their properties rather than
    storing them.
    """
    predicates = (
        (is_symmetric, symmetric_tag),
        (is_diagonal, diagonal_tag),
        (is_tridiagonal, tridiagonal_tag),
        (has_unit_diagonal, unit_diagonal_tag),
        (is_lower_triangular, lower_triangular_tag),
        (is_upper_triangular, upper_triangular_tag),
        (is_positive_semidefinite, positive_semidefinite_tag),
        (is_negative_semidefinite, negative_semidefinite_tag),
    )
    return frozenset(tag for predicate, tag in predicates if predicate(operator))


def _permute_leaf(
    vector: PyTree[Any], permutation: Integer[Array, " size"], what: str
) -> Any:
    if not eqx.is_array_like(vector) or jnp.ndim(vector) != 1:
        raise NotImplementedError(
            f"Applying a `SymmetricPermutation` to a PyTree-structured {what} is not "
            "supported yet; it must be a single one-dimensional array. Ravel the "
            "system yourself, or drop the solver's `transforms`."
        )
    return jnp.asarray(vector)[permutation]


def _as_bcoo_matrix(operator: AbstractLinearOperator) -> BCOO:
    match operator:
        case BCOOLinearOperator(matrix):
            return matrix
        case BCSRLinearOperator(matrix):
            return matrix.to_bcoo()
        case SparseJacobianLinearOperator():
            return _as_bcoo_matrix(materialise(operator))
        case _:
            raise TypeError(
                "Transforming an operator requires a sparse operator backed by a "
                "`BCOO` or `BCSR` matrix (e.g. `splineax.BCOOLinearOperator`), or a "
                f"`splineax.SparseJacobianLinearOperator`; got "
                f"{type(operator).__name__}."
            )


class ReverseCuthillMcKee(eqx.Module):
    """Reorders the system to reduce its bandwidth, symmetrically.

    Cuthill-McKee grows a breadth-first ordering from a low-degree start node, visiting
    each level's neighbours in increasing order of degree; reversing the result is a
    standard improvement that reduces fill without changing the bandwidth. The nonzeros
    end up clustered near the diagonal, which is what makes contiguous diagonal blocks
    worth taking: [`splineax.BlockJacobi`][] drops whatever falls outside its blocks, so
    a narrower band means less of the matrix is thrown away.

    The permutation is symmetric (`A -> P A P^T`), so symmetry and definiteness survive
    it and this can be used with `lineax.CG`.

    Satisfies the `SymbolicTransform` protocol, so it goes in the solver's `transforms`:

    ```python
    import lineax as lx
    import splineax as splx

    solver = splx.PreconditionedIterativeLinearSolver(
        lx.GMRES(rtol=1e-8, atol=1e-8),
        splx.BlockJacobi(blocks=splx.MaximalCaptureBlockPartitioner()),
        transforms=(splx.ReverseCuthillMcKee(),),
    )
    assert solver.transforms[0] == splx.ReverseCuthillMcKee()
    ```
    """

    def symbolic(self, pattern: CooPattern) -> SystemTransform:
        """Derive the reordering from the pattern."""
        concrete = pattern.concrete("`ReverseCuthillMcKee`")
        if concrete.shape[0] != concrete.shape[1]:
            raise ValueError(
                "`ReverseCuthillMcKee` requires a square pattern; got shape "
                f"{concrete.shape}."
            )
        return SymmetricPermutation.from_order(_reverse_cuthill_mckee(concrete))


def _symmetric_adjacency(pattern: ConcreteCooPattern) -> tuple[np.ndarray, np.ndarray]:
    """The pattern's undirected adjacency, as a CSR-style `(indptr, neighbours)` pair.

    Symmetrised by including both `(r, c)` and `(c, r)`, since a reordering is about
    the *undirected* coupling between indices, and self-loops dropped so that a
    vertex's degree counts its neighbours.
    """
    size = pattern.shape[0]
    rows = np.concatenate([pattern.rows, pattern.cols]).astype(np.int64)
    cols = np.concatenate([pattern.cols, pattern.rows]).astype(np.int64)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    # Sorting by row and de-duplicating gives each vertex a contiguous neighbour run.
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    if rows.size:
        unique = np.empty(rows.size, dtype=bool)
        unique[0] = True
        unique[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
        rows, cols = rows[unique], cols[unique]
    indptr = np.zeros(size + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=size), out=indptr[1:])
    return indptr, cols


def _reverse_cuthill_mckee(pattern: ConcreteCooPattern) -> np.ndarray:
    """The reverse Cuthill-McKee ordering of a pattern, as `order[i] = original index`."""
    size = pattern.shape[0]
    indptr, neighbours = _symmetric_adjacency(pattern)
    degrees = np.diff(indptr)
    visited = np.zeros(size, dtype=bool)
    order = np.empty(size, dtype=np.int64)
    filled = 0
    # Visit components in order of their lowest-degree vertex, so a disconnected
    # pattern is handled without a separate pass.
    for start in np.argsort(degrees, kind="stable"):
        if visited[start]:
            continue
        visited[start] = True
        order[filled] = start
        filled += 1
        head = filled - 1
        # Breadth-first, taking each frontier's unvisited neighbours by rising degree.
        while head < filled:
            vertex = order[head]
            head += 1
            candidates = neighbours[indptr[vertex] : indptr[vertex + 1]]
            candidates = candidates[~visited[candidates]]
            if candidates.size == 0:
                continue
            candidates = candidates[np.argsort(degrees[candidates], kind="stable")]
            visited[candidates] = True
            order[filled : filled + candidates.size] = candidates
            filled += candidates.size
    return order[::-1].copy()
