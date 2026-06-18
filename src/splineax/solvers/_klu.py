from typing import Any, TypeAlias, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Inexact, Integer, PyTree
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

# `Ai` (row indices), `Aj` (column indices), `Ax` (values): the matrix in COO form.
_COO = tuple[Integer[Array, " a"], Integer[Array, " b"], Inexact[Array, " nse"]]
_KLUState: TypeAlias = tuple[_COO, PackedStructures]


def _klujax():
    # Lazy import: importing klujax enables jax x64 and forces the CPU platform globally,
    # so we defer it until a KLU solve actually runs (keeping it out of `import splineax`).
    import klujax

    return klujax


T = TypeVar("T")


def _ensure_cpu(args: T) -> T:
    """Return `x` unchanged, raising if the current platform is not CPU.

    Uses `jax.lax.platform_dependent` to produce a traced boolean and
    `equinox.error_if` to raise.
    """
    on_cpu = jax.lax.platform_dependent(
        args,
        default=lambda _: jnp.bool_(False),
        cpu=lambda _: jnp.bool_(True),
    )
    return eqx.error_if(
        args,
        ~on_cpu,
        "`KLU` can only solve on CPU; klujax wraps the CPU-only SuiteSparse KLU library.",
    )


class KLU(AbstractLinearSolver[_KLUState]):
    """Sparse direct solver wrapping the `klujax` (SuiteSparse KLU) library.

    This solver keeps the operator in its native sparse (COO) storage rather than
    densifying it, and so is intended for use with the sparse operators in this package
    (`BCOOLinearOperator` and `BCSRLinearOperator`).

    `klujax` is **CPU and double-precision only**: `float32`/`complex64` inputs are
    upcast to `float64`/`complex128`, and importing it enables JAX's x64 mode and forces
    the CPU platform globally (this happens lazily, on the first solve).

    This solver can only handle square nonsingular operators.
    """

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _KLUState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`KLU` may only be used for linear solves with square matrices"
            )

        # `klujax.solve` consumes a COO triple. We assume the matrix is coalesced (no
        # duplicate indices); KLU builds CSC internally, so the index order is irrelevant.
        match operator:
            case BCSRLinearOperator(matrix):
                matrix = matrix.to_bcoo()
            case BCOOLinearOperator(matrix):
                pass
            case _:
                raise TypeError(
                    "`KLU` requires a sparse operator backed by a `BCOO` or `BCSR` "
                    "matrix (e.g. `splineax.BCOOLinearOperator` or "
                    f"`splineax.BCSRLinearOperator`); got {type(operator).__name__}."
                )

        Ai = matrix.indices[:, 0].astype(jnp.int32)
        Aj = matrix.indices[:, 1].astype(jnp.int32)
        Ax = matrix.data
        return (Ai, Aj, Ax), pack_structures(operator)

    def compute(
        self, state: _KLUState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        (Ai, Aj, Ax), packed_structures = state
        b = ravel_vector(vector, packed_structures)
        Ai, Aj, Ax, b = _ensure_cpu((Ai, Aj, Ax, b))
        x = _klujax().solve(Ai, Aj, Ax, b)
        solution = unravel_solution(x, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(
        self, state: _KLUState, options: dict[str, Any]
    ) -> tuple[_KLUState, dict[str, Any]]:
        del options
        (Ai, Aj, Ax), packed_structures = state
        # The transpose of a COO matrix swaps its row and column indices.
        transpose_state = ((Aj, Ai, Ax), transpose_packed_structures(packed_structures))
        return transpose_state, {}

    def conj(
        self, state: _KLUState, options: dict[str, Any]
    ) -> tuple[_KLUState, dict[str, Any]]:
        del options
        (Ai, Aj, Ax), packed_structures = state
        return ((Ai, Aj, Ax.conj()), packed_structures), {}

    def assume_full_rank(self) -> bool:
        return True


KLU.__init__.__doc__ = """**Arguments:**

Nothing.
"""
