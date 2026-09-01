from enum import IntEnum
from typing import Any

import equinox as eqx
from jax import custom_batching
from jax.experimental.sparse import BCSR
from jax.experimental.sparse.linalg import _csr_transpose, spsolve
from jaxtyping import Array, Inexact, PyTree
from lineax import AbstractLinearOperator, materialise
from lineax._solution import RESULTS
from lineax._solve import AbstractLinearSolver
from lineax._solver.misc import (
    PackedStructures,
    pack_structures,
    ravel_vector,
    transpose_packed_structures,
    unravel_solution,
)

from splineax._trace import record_event
from splineax.operators._bcoo import BCOOLinearOperator
from splineax.operators._bcsr import BCSRLinearOperator
from splineax.operators._jacobian import (
    SparseJacobianLinearOperator,
)
from splineax.solvers._sparse import (
    _Sparsity,
    sparse_indices_sorted,
    warn_if_unsorted,
)


class _SpsolveState(eqx.Module):
    """A Spsolve state.

    `spsolve` has no separate factorization phase, so the stateful API is a set of
    no-ops. A state straight from `init_symbolic` carries no matrix and is not solvable
    until `update` gives it an operator.
    """

    operator: AbstractLinearOperator | None
    """The operator this state was built on. Compared by identity in `update`."""
    matrix: BCSR | None
    """The sorted CSR matrix to solve, or None for a symbolic-only state."""
    packed_structures: PackedStructures | None
    """The lineax structure for ravel and unravel, None for a symbolic-only state."""

    def track(self, solution: Any) -> "_SpsolveState":
        """No-op, since a Spsolve state owns no memory to order a release after."""
        del solution
        return self

    def release(self) -> None:
        """No-op, since a Spsolve state owns nothing to free."""


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

    It has no separate factorization phase, so the stateful reuse API (`init_symbolic`,
    `update`, and the state's `release`) is a set of no-ops here, for parity with `KLU` and
    `Pardiso`.

    This solver can only handle square nonsingular operators.
    """

    tol: float = eqx.field(default=1e-6, static=True)
    reorder: ReorderingScheme = eqx.field(default=ReorderingScheme.SYMRCM, static=True)

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _SpsolveState:
        record_event("init", "spsolve", shape=(operator.out_size(), operator.in_size()))
        return self._build(operator, options)

    def _build(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _SpsolveState:
        """Sort `operator` into a solvable CSR state, no factorization.

        Shared by `init` and `update`'s rebuild path, so a rebuild does not re-emit an `init`
        boundary. `Spsolve` factors and solves in one fused call, so there is no analyze or
        factor to record here.
        """
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`Spsolve` may only be used for linear solves with square matrices"
            )

        # `spsolve` consumes a CSR triple whose column indices are sorted within each
        # row. We assume the matrix is coalesced (no duplicate indices) and only ensure
        # the sorting here. An operator tagged `sparse_indices_sorted` asserts it needs
        # no sort.
        sorted_asserted = sparse_indices_sorted in getattr(operator, "tags", ())
        match operator:
            case SparseJacobianLinearOperator():
                return self._build(materialise(operator), options)
            case BCSRLinearOperator(matrix):
                # Round-trip an unsorted `BCSR` through `BCOO`, since `BCSR.from_bcoo`
                # sorts.
                if matrix.indices_sorted or sorted_asserted:
                    matrix_bcsr = matrix
                else:
                    warn_if_unsorted(matrix, "Spsolve")
                    matrix_bcsr = BCSR.from_bcoo(matrix.to_bcoo())
            case BCOOLinearOperator(matrix):
                if sorted_asserted:
                    matrix_bcsr = BCSR.from_bcoo(matrix)
                else:
                    warn_if_unsorted(matrix, "Spsolve")
                    matrix_bcsr = BCSR.from_bcoo(matrix)
            case _:
                raise TypeError(
                    "`Spsolve` requires a sparse operator backed by a `BCOO` or `BCSR` "
                    "matrix (e.g. `splineax.BCOOLinearOperator` or "
                    "`splineax.BCSRLinearOperator`), or a "
                    f"`splineax.SparseJacobianLinearOperator`; "
                    f"got {type(operator).__name__}."
                )

        return _SpsolveState(operator, matrix_bcsr, pack_structures(operator))

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> _SpsolveState:
        """No-op symbolic init, for parity with `KLU`.

        `Spsolve` cannot pre-analyze a sparsity pattern, so this returns an empty state
        that `update` fills with the first real operator.
        """
        del sparsity, options
        record_event("init_symbolic", "spsolve", symbolic=True, note="no-op")
        return _SpsolveState(None, None, None)

    def update(
        self,
        state: _SpsolveState,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _SpsolveState:
        """Rebuild the state from `operator`, since `Spsolve` reuses no factorization.

        Repeated calls with the same operator object are a no-op.
        """
        if operator is state.operator:
            record_event(
                "update",
                "spsolve",
                outcome="noop",
                shape=state.matrix.shape if state.matrix is not None else None,
                reason="same operator object",
            )
            return state
        # `Spsolve` reuses nothing, so every changed operator is a full rebuild.
        record_event(
            "update",
            "spsolve",
            outcome="rebuilt",
            reused=False,
            reason="Spsolve keeps no factorization to reuse",
        )
        return self._build(operator, options)

    def compute(
        self, state: _SpsolveState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        if state.matrix is None or state.packed_structures is None:
            raise ValueError(
                "`Spsolve` cannot solve with a symbolic-only state; call `update` with "
                "an operator first."
            )
        matrix = state.matrix
        packed_structures = state.packed_structures
        vector = ravel_vector(vector, packed_structures)
        # `spsolve` requires the right-hand side to share the matrix dtype.
        vector = vector.astype(matrix.dtype)
        # Fused analyze+factor+solve, and unordered because `compute` runs inside lineax's
        # solve primitive (see `record_event`).
        record_event(
            "solve", "spsolve", shape=matrix.shape, note="fused", ordered=False
        )
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
        assert matrix is not None and state.packed_structures is not None
        matrix_T = BCSR(
            _csr_transpose(matrix.data, matrix.indices, matrix.indptr),
            shape=matrix.shape[::-1],
        )
        transpose_state = _SpsolveState(
            state.operator,
            matrix_T,
            transpose_packed_structures(state.packed_structures),
        )
        return transpose_state, {}

    def conj(
        self, state: _SpsolveState, options: dict[str, Any]
    ) -> tuple[_SpsolveState, dict[str, Any]]:
        del options
        matrix = state.matrix
        assert matrix is not None
        matrix_conj = BCSR(
            (matrix.data.conj(), matrix.indices, matrix.indptr), shape=matrix.shape
        )
        return _SpsolveState(state.operator, matrix_conj, state.packed_structures), {}

    def assume_full_rank(self) -> bool:
        return True


Spsolve.__init__.__doc__ = """**Arguments:**

- `tol`: tolerance passed to `spsolve` for deciding whether the system is singular.
    Defaults to `1e-6`.
- `reorder`: the fill-reducing reordering scheme passed to `spsolve`. `0` for no
    reordering, otherwise `1`, `2`, or `3` for symrcm, symamd, or csrmetisnd
    respectively. Defaults to `1` (symrcm).
"""
