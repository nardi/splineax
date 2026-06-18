from enum import IntEnum
from typing import Any, TypeAlias

from jax import custom_batching
from jax.experimental.sparse import BCSR
from jax.experimental.sparse.linalg import _csr_transpose, spsolve
from jaxtyping import Array, Inexact, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver
from lineax._solver.misc import (
    PackedStructures,
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator

_SpsolveState: TypeAlias = tuple[BCSR, PackedStructures]


class ReorderingScheme(IntEnum):
    NO_REORDERING = 0
    SYMRCM = 1
    SYMAMD = 2
    CSRMETISND = 3


def _spsolve(
    data: Inexact[Array, " nse"],
    indices: Array,
    indptr: Array,
    b: Inexact[Array, " size"],
    tol: float,
    reorder: "ReorderingScheme",
) -> Inexact[Array, " size"]:
    """`spsolve` augmented with the sequential `vmap` rule it does not provide natively.

    `jax.experimental.sparse.linalg.spsolve` has no batching rule, so `jax.vmap` over it
    (and hence `jax.jacfwd`/`jax.jacrev`) would otherwise raise. `sequential_vmap` adds a
    rule that loops over the batch via `lax.map`. `tol`/`reorder` are closed over rather
    than passed through the `custom_vmap` boundary, where they would become tracers that
    `spsolve` rejects as non-static parameters.
    """

    @custom_batching.sequential_vmap
    def spsolve_with_sequential_vmap(
        data: Inexact[Array, " nse"],
        indices: Array,
        indptr: Array,
        b: Inexact[Array, " size"],
    ) -> Inexact[Array, " size"]:
        return spsolve(data, indices, indptr, b, tol=tol, reorder=reorder)

    return spsolve_with_sequential_vmap(data, indices, indptr, b)


class Spsolve(AbstractLinearSolver[_SpsolveState]):
    """Sparse direct solver wrapping `jax.experimental.sparse.linalg.spsolve`.

    This solver keeps the operator in its native sparse (CSR) storage rather than
    densifying it, and so is intended for use with the sparse operators in this package
    (`BCOOLinearOperator` and `BCSRLinearOperator`). Internally `spsolve` performs a
    sparse QR factorization (CUDA native; on CPU it falls back to
    `scipy.sparse.linalg.spsolve`).

    This solver can only handle square nonsingular operators.
    """

    tol: float = 1e-6
    reorder: ReorderingScheme = ReorderingScheme.SYMRCM

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _SpsolveState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Spsolve` may only be used for linear solves with square matrices"
            )

        # `spsolve` consumes a CSR triple whose column indices are sorted within
        # each row. We assume the matrix is coalesced (no duplicate indices) and
        # only ensure the sorting here.
        match operator:
            case BCSRLinearOperator(matrix):
                # Round-trip an unsorted `BCSR` through `BCOO`, since
                # `BCSR.from_bcoo` sorts.
                matrix_bcsr = (
                    matrix
                    if matrix.indices_sorted
                    else BCSR.from_bcoo(matrix.to_bcoo())
                )
            case BCOOLinearOperator(matrix):
                # `BCSR.from_bcoo` sorts the indices itself when they are not
                # already sorted.
                matrix_bcsr = BCSR.from_bcoo(matrix)
            case _:
                raise TypeError(
                    "`Spsolve` requires a sparse operator backed by a `BCOO` or `BCSR` "
                    "matrix (e.g. `splineax.BCOOLinearOperator` or "
                    f"`splineax.BCSRLinearOperator`); got {type(operator).__name__}."
                )

        return matrix_bcsr, pack_structures(operator)

    def compute(
        self, state: _SpsolveState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        matrix, packed_structures = state
        vector = ravel_vector(vector, packed_structures)
        # `spsolve` requires the right-hand side to share the matrix dtype.
        vector = vector.astype(matrix.dtype)
        solution = _spsolve(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            vector,
            tol=self.tol,
            reorder=self.reorder,
        )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(
        self, state: _SpsolveState, options: dict[str, Any]
    ) -> tuple[_SpsolveState, dict[str, Any]]:
        del options
        matrix, packed_structures = state
        matrix_T = BCSR(
            _csr_transpose(matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape[::-1],
        )
        transpose_state = (matrix_T, transpose_packed_structures(packed_structures))
        return transpose_state, {}

    def conj(
        self, state: _SpsolveState, options: dict[str, Any]
    ) -> tuple[_SpsolveState, dict[str, Any]]:
        del options
        matrix, packed_structures = state
        matrix_conj = BCSR(
            (matrix.data.conj(), matrix.indices, matrix.indptr), shape=matrix.shape
        )
        return (matrix_conj, packed_structures), {}

    def assume_full_rank(self) -> bool:
        return True


Spsolve.__init__.__doc__ = """**Arguments:**

- `tol`: tolerance passed to `spsolve` for deciding whether the system is singular.
    Defaults to `1e-6`.
- `reorder`: the fill-reducing reordering scheme passed to `spsolve`. `0` for no
    reordering, otherwise `1`, `2`, or `3` for symrcm, symamd, or csrmetisnd
    respectively. Defaults to `1` (symrcm).
"""
