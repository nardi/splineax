from typing import Protocol, TypeVar, runtime_checkable

import jax
import numpy as np
from lineax import AbstractLinearOperator


class SparseMatrix(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...


SparseMatrixT = TypeVar("SparseMatrixT", bound=SparseMatrix, covariant=True)


@runtime_checkable
class TaggedLinearOperator(Protocol):
    """Structural type implemented by every operator in this package.

    Deliberately says nothing about *storage*: it is the interface the tag-based
    `singledispatch` implementations need (`is_symmetric`, `is_diagonal`, and the rest),
    all of which are answered from `tags` plus the operator's shape. Operators that also
    hold a `matrix` implement the narrower `SparseLinearOperator` below; those that do
    not -- `BlockDiagonalLinearOperator`, which stores a stack of dense blocks -- still
    get the whole tag layer from here.
    """

    @property
    def tags(self) -> frozenset[object]: ...

    def in_structure(self) -> jax.ShapeDtypeStruct: ...
    def out_structure(self) -> jax.ShapeDtypeStruct: ...
    def in_size(self) -> int: ...
    def out_size(self) -> int: ...
    def _conj(self) -> AbstractLinearOperator: ...


@runtime_checkable
class SparseLinearOperator(TaggedLinearOperator, Protocol[SparseMatrixT]):
    """Structural type implemented by the sparse *matrix-backed* operators.

    Used to type the shared helpers without giving the operators a common base class:
    anything providing these members can be passed to the functions below.
    """

    @property
    def matrix(self) -> SparseMatrixT: ...
