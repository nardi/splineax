from contextlib import AbstractContextManager, contextmanager
from enum import IntEnum
from typing import Any, Iterator, Literal, NamedTuple, overload

import equinox as eqx
from jax import custom_batching
from jax.experimental.sparse import BCSR
from jax.experimental.sparse.linalg import _csr_transpose, spsolve
from jaxtyping import Array, Inexact, PyTree
from lineax import (
    AbstractLinearOperator,
    FunctionLinearOperator,
    JacobianLinearOperator,
)
from lineax._solution import RESULTS
from lineax._solver.misc import (
    PackedStructures,
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)

from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._function import SparseFunctionLinearOperator
from splineax.operators._jacobian import (
    SparseJacobianLinearOperator,
)
from splineax.solvers._sparse import (
    AbstractSparseLinearSolver,
    SparseNumericState,
    SymbolicScopedSparseLinearSolver,
    _Sparsity,
    as_scoped_solver,
    factorize_through_init,
    sparse_function_operator,
    sparse_jacobian_operator,
)


class _SpsolveState(NamedTuple):
    matrix: BCSR
    packed_structures: PackedStructures

    @contextmanager
    def factorize(self) -> Iterator["_SpsolveState"]:
        # No-op: Spsolve has no separate numeric factorization phase.
        yield self


class _SpsolveSymbolicScope(NamedTuple):
    solver: "Spsolve"
    """The originating solver, so built states keep its tol/reorder config."""
    sparsity: _Sparsity
    """The object the scope was opened from, kept to sparsely materialise a dense
    `lineax.JacobianLinearOperator` or `lineax.FunctionLinearOperator` handed to
    `init`."""

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _SpsolveState:
        # No-op symbolic reuse: Spsolve cannot pre-analyze, so this is a normal init.
        # The one thing the scope does add is its sparsity, which turns a dense lineax
        # operator defined by a function into its sparse analogue, materialised with one
        # evaluation per color rather than one per column or row.
        if isinstance(operator, JacobianLinearOperator):
            operator = sparse_jacobian_operator(operator, self.sparsity)
        elif isinstance(operator, FunctionLinearOperator):
            operator = sparse_function_operator(operator, self.sparsity)
        return self.solver.init(operator, options)

    @contextmanager
    def factorize(self, operator: AbstractLinearOperator) -> Iterator[_SpsolveState]:
        with self.init(operator).factorize() as state:
            yield state


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


class Spsolve(AbstractSparseLinearSolver[_SpsolveState]):
    """Sparse direct solver wrapping `jax.experimental.sparse.linalg.spsolve`.

    This solver keeps the operator in its native sparse (CSR) storage rather than
    densifying it, and so is intended for use with the sparse operators in this package
    (`BCOOLinearOperator` and `BCSRLinearOperator`). Internally `spsolve` performs a
    sparse QR factorization (CUDA native; on CPU it falls back to
    `scipy.sparse.linalg.spsolve`).

    This solver can only handle square nonsingular operators.
    """

    tol: float = eqx.field(default=1e-6, static=True)
    reorder: ReorderingScheme = eqx.field(default=ReorderingScheme.SYMRCM, static=True)

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _SpsolveState:
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Spsolve` may only be used for linear solves with square matrices"
            )

        # `spsolve` consumes a CSR triple whose column indices are sorted within
        # each row. We assume the matrix is coalesced (no duplicate indices) and
        # only ensure the sorting here.
        match operator:
            case SparseJacobianLinearOperator() | SparseFunctionLinearOperator():
                # Materialise the Jacobian and sort it, as the `BCOO` case below does.
                # `operator` stays bound to it, so `pack_structures` sees the caller's
                # structures rather than the flat pair a materialised operator reports.
                matrix_bcsr = BCSR.from_bcoo(operator.as_bcoo())
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
                    "`splineax.BCSRLinearOperator`), or a "
                    "`splineax.SparseJacobianLinearOperator` or "
                    f"`splineax.SparseFunctionLinearOperator`; "
                    f"got {type(operator).__name__}."
                )

        return _SpsolveState(matrix_bcsr, pack_structures(operator))

    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        # No-op factorization for parity with KLU: yields the ordinary solver state.
        return factorize_through_init(self, operator, options)

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[False] = False
    ) -> AbstractContextManager[_SpsolveSymbolicScope]: ...

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[True]
    ) -> AbstractContextManager[SymbolicScopedSparseLinearSolver]: ...

    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: bool = False
    ) -> AbstractContextManager[
        _SpsolveSymbolicScope | SymbolicScopedSparseLinearSolver
    ]:
        """Open a no-op symbolic-factorization scope, for parity with `KLU`.

        Args:
            sparsity: accepted for parity with `KLU`, since `Spsolve` cannot
                      pre-analyze a sparsity pattern. It is only kept so that the
                      scope's `init` can sparsely materialise a dense
                      `lineax.JacobianLinearOperator` or
                      `lineax.FunctionLinearOperator` against it.
            as_solver: Yield a `SymbolicScopedSparseLinearSolver` pairing the scope
                       with this solver, instead of the bare scope, so that the two
                       need not be passed around together.
        """
        scope = self._factorize_symbolic(sparsity)
        return as_scoped_solver(self, scope) if as_solver else scope

    @contextmanager
    def _factorize_symbolic(
        self, sparsity: _Sparsity
    ) -> Iterator[_SpsolveSymbolicScope]:
        # No-op symbolic factorization: no pattern is analyzed, since Spsolve cannot
        # pre-analyze one. The sparsity is only kept for the same `init` handling of
        # Jacobian operators the other solvers offer. Kept separate from
        # `factorize_symbolic` above so that the public method can be overloaded on
        # `as_solver` (`@contextmanager` and `@overload` do not compose).
        yield _SpsolveSymbolicScope(self, sparsity)

    def compute(
        self, state: _SpsolveState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        matrix = state.matrix
        packed_structures = state.packed_structures
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
        matrix = state.matrix
        matrix_T = BCSR(
            _csr_transpose(matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape[::-1],
        )
        transpose_state = _SpsolveState(
            matrix_T, transpose_packed_structures(state.packed_structures)
        )
        return transpose_state, {}

    def conj(
        self, state: _SpsolveState, options: dict[str, Any]
    ) -> tuple[_SpsolveState, dict[str, Any]]:
        del options
        matrix = state.matrix
        matrix_conj = BCSR(
            (matrix.data.conj(), matrix.indices, matrix.indptr), shape=matrix.shape
        )
        return _SpsolveState(matrix_conj, state.packed_structures), {}

    def assume_full_rank(self) -> bool:
        return True


Spsolve.__init__.__doc__ = """**Arguments:**

- `tol`: tolerance passed to `spsolve` for deciding whether the system is singular.
    Defaults to `1e-6`.
- `reorder`: the fill-reducing reordering scheme passed to `spsolve`. `0` for no
    reordering, otherwise `1`, `2`, or `3` for symrcm, symamd, or csrmetisnd
    respectively. Defaults to `1` (symrcm).
"""
