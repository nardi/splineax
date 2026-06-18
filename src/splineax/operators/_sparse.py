from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from lineax import AbstractLinearOperator


class SparseMatrix(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...


SparseMatrixT = TypeVar("SparseMatrixT", bound=SparseMatrix, covariant=True)


@runtime_checkable
class SparseLinearOperator(Protocol[SparseMatrixT]):
    """Structural type implemented by all sparse operators.

    Used to type the shared helpers without giving the operators a common base class:
    anything providing these members can be passed to the functions below.
    """

    @property
    def matrix(self) -> SparseMatrixT: ...
    @property
    def tags(self) -> frozenset[object]: ...

    def in_size(self) -> int: ...
    def out_size(self) -> int: ...
    def _conj(self) -> AbstractLinearOperator: ...
