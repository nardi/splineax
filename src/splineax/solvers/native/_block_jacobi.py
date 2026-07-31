"""Block-Jacobi preconditioned GMRES, built only from array operations.

Unlike the other solvers in this package this one calls no external library, so the whole
solve traces into a single compiled computation and runs on any backend. It works in three
stages, which map onto the symbolic, numeric and solve tiers of the factorization-reuse API:
reorder the pattern to narrow its band, cut the reordered range into overlapping blocks, then
invert those blocks and hand them to GMRES as a preconditioner.

The reordering and the blocking see only the sparsity pattern, so a pattern analysed once
serves every matrix sharing it. That is what `factorize_symbolic` reuses here, and it is a
real saving rather than the no-op it is for `Spsolve`.

See the theory page in `docs/theory/block-jacobi-gmres.md` for the algorithm and the
reasoning behind it.
"""

from contextlib import AbstractContextManager, contextmanager
from enum import IntEnum
from typing import Any, Iterator, Literal, overload

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Bool, Inexact, Integer, PyTree
from lineax import AbstractLinearOperator, materialise
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
from splineax.operators._jacobian import SparseJacobianLinearOperator
from splineax.solvers._sparse import (
    AbstractSparseLinearSolver,
    SparseNumericState,
    SymbolicScopedSparseLinearSolver,
    _Sparsity,
    as_scoped_solver,
    factorize_through_init,
)
from splineax.solvers.native._blocks import (
    block_destinations,
    capture_fraction,
    choose_block_size,
    geometry,
    partition,
)
from splineax.solvers.native._matching import matching
from splineax.solvers.native._ordering import (
    Ordering,
    inverse_permutation,
    order,
)


class BlockInverse(IntEnum):
    """How to invert each diagonal block.

    Both routes are rank-revealing, and both leave a block's numerically dependent directions
    *unpreconditioned* rather than inverting them. Leaving them alone keeps the preconditioner
    invertible, which matters more than it might seem: GMRES is preconditioned on the left, so
    it solves `M A x = M b`, and that has the same solutions as `A x = b` only when `M` has no
    null space. A pseudo-inverse would send those directions to zero and quietly change the
    problem, letting the iteration converge to something that does not solve the original
    system.
    """

    SVD = 0
    """Invert via a singular value decomposition. Available on every backend."""
    QR = 1
    """Invert via a column-pivoted QR factorization. Measurably cheaper than the singular value
    decomposition, by a factor between about two and seven depending on the block size, but
    column pivoting is not implemented on TPU."""


class _BlockJacobiAnalysis(eqx.Module):
    """The symbolic factorization: everything that follows from the pattern alone.

    Held by a `factorize_symbolic` scope and reused for every matrix sharing the pattern, so
    that only the block assembly and inversion repeat per set of values.
    """

    perm: Integer[Array, " n"]
    """Original index at each new position, for reordering a vector."""
    inv_perm: Integer[Array, " n"]
    """New position of each original index, for restoring a solution."""
    constraint: Bool[Array, " n"]
    """Which reordered rows are constraint rows of a detected saddle point, meaning they have
    no diagonal entry and the pattern passed the guard in `_analyse`. All `False` otherwise.
    Diagnostic: nothing downstream reads it, but it is what a caller checks to see whether the
    matching-informed grouping engaged."""
    gather_index: Integer[Array, "num_blocks b"]
    """Rows covered by each block."""
    core_mask: Bool[Array, "num_blocks b"]
    """Rows each block owns, so every row is written back exactly once."""
    destinations: Integer[Array, "num_dest nse"]
    """Where each stored entry lands in the flattened block array, or one past its end."""
    sort_order: Integer[Array, " nse"]
    """Reads the values into CSR order, so the numeric stage never sorts."""
    sorted_rows: Integer[Array, " nse"]
    """Reordered row index of each value in CSR order, used to build the transpose."""
    col_indices: Integer[Array, " nse"]
    """Reordered column index of each value in CSR order."""
    indptr: Integer[Array, " n+1"]
    """Where each reordered row begins."""
    block_size: int = eqx.field(static=True)
    num_blocks: int = eqx.field(static=True)
    size: int = eqx.field(static=True)
    captured: float = eqx.field(static=True)
    """Fraction of the entries the blocks cover, as a diagnostic. `nan` when the pattern was
    traced and so could not be measured."""
    structural_rank: int = eqx.field(static=True)
    """The size of a maximum matching between the pattern's rows and columns, as a diagnostic.
    Equal to `size` unless the pattern is structurally singular, in which case `_analyse` has
    already raised rather than returning. `-1` when the pattern was traced and so could not be
    measured, mirroring `captured`."""


class _BlockJacobiState(eqx.Module):
    """A fully factorized state. One class serves all three tiers, as for `Spsolve`: there is
    no native handle to release, so a numeric factorization is just arrays."""

    operator: AbstractLinearOperator
    """The reordered matrix. CSR going forwards, coordinate form once transposed, since
    transposing coordinate form is free."""
    inv_blocks: Inexact[Array, "num_blocks b b"]
    analysis: _BlockJacobiAnalysis
    packed_structures: PackedStructures
    solver: "BlockJacobiGMRES"
    """The originating solver, so a state carries the tolerances it will be solved with."""

    @contextmanager
    def factorize(self) -> Iterator["_BlockJacobiState"]:
        # The blocks are already inverted, so there is no further numeric stage.
        yield self


class _BlockJacobiSymbolicScope(eqx.Module):
    """An open symbolic-factorization scope holding one analysed pattern."""

    solver: "BlockJacobiGMRES"
    analysis: _BlockJacobiAnalysis

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _BlockJacobiState:
        del options
        return self.solver._factorize(operator, self.analysis)

    @contextmanager
    def factorize(
        self, operator: AbstractLinearOperator
    ) -> Iterator[_BlockJacobiState]:
        with self.init(operator).factorize() as state:
            yield state


def _as_bcoo(operator: AbstractLinearOperator) -> BCOO:
    """The operator's matrix in coordinate form, densifying nothing."""
    match operator:
        case SparseJacobianLinearOperator():
            return _as_bcoo(materialise(operator))
        case BCSRLinearOperator(matrix):
            return matrix.to_bcoo()
        case BCOOLinearOperator(matrix):
            return matrix
        case _:
            raise TypeError(
                "`BlockJacobiGMRES` requires a sparse operator backed by a `BCOO` or "
                "`BCSR` matrix (e.g. `splineax.BCOOLinearOperator` or "
                "`splineax.BCSRLinearOperator`), or a "
                "`splineax.SparseJacobianLinearOperator`; got "
                f"{type(operator).__name__}."
            )


def _sparsity_pattern(sparsity: _Sparsity) -> BCOO:
    """The coordinate pattern of anything `factorize_symbolic` accepts."""
    if isinstance(sparsity, (BCOO, BCSR)):
        return sparsity.to_bcoo() if isinstance(sparsity, BCSR) else sparsity
    if isinstance(sparsity, AbstractLinearOperator):
        return _as_bcoo(sparsity)
    # A `JacobianColoring` or a coloring-carrying operator exposes the pattern directly.
    pattern = getattr(sparsity, "sparsity", None)
    if pattern is None:
        raise TypeError(
            "`BlockJacobiGMRES.factorize_symbolic` requires a `BCOO`, a `BCSR`, one of "
            "this package's sparse operators, or a `splineax.JacobianColoring`; got "
            f"{type(sparsity).__name__}."
        )
    return BCOO.fromdense(pattern) if not isinstance(pattern, BCOO) else pattern


def _invert_blocks(
    blocks: Inexact[Array, "num_blocks b b"],
    block_inverse: BlockInverse,
    rcond: float | None,
) -> Inexact[Array, "num_blocks b b"]:
    """Invert every block at once.

    The blocks are inverted rather than factored so that applying the preconditioner is a
    dense product per block with nothing waiting on anything else. Keeping factors instead
    would make it a triangular solve, whose sequential structure is exactly what splitting the
    matrix into independent blocks exists to avoid.

    A diagonal block can be singular even when the matrix is not, so both routes are
    rank-revealing and leave the directions they cannot invert alone. See `BlockInverse` for
    why leaving them alone rather than zeroing them is what keeps the result usable.
    """
    match block_inverse:
        case BlockInverse.SVD:
            return jax.vmap(lambda block: _svd_inverse(block, rcond))(blocks)
        case BlockInverse.QR:
            return jax.vmap(lambda block: _pivoted_qr_inverse(block, rcond))(blocks)
        case _:
            raise ValueError(
                "`block_inverse` must be `BlockInverse.SVD` or `BlockInverse.QR`; got "
                f"{block_inverse!r}."
            )


def _relative_threshold(block: Inexact[Array, "b b"], rcond: float | None) -> Array:
    """Relative size below which a direction of a block counts as numerically absent.

    The default is the square root of the working precision, which is far more conservative
    than the usual choice for a pseudo-inverse. A preconditioner does not need marginal
    directions inverted, and inverting them is actively harmful: it bounds the condition
    number of the preconditioner by the reciprocal of this threshold, and left preconditioning
    compares `M (b - A x)` against the tolerance, so a preconditioner conditioned near the
    reciprocal of the working precision can make that quantity vanish while the real residual
    is still large. The iteration would then report success on a wrong answer.
    """
    if rcond is not None:
        return jnp.asarray(rcond, dtype=jnp.finfo(block.dtype).dtype)
    return jnp.sqrt(jnp.finfo(block.dtype).eps)


def _svd_inverse(
    block: Inexact[Array, "b b"], rcond: float | None
) -> Inexact[Array, "b b"]:
    """Invert one block through its singular value decomposition.

    Singular values too small to invert are left at one rather than inverted or dropped, so
    the result is bounded and still invertible. For a block of full rank nothing is altered
    and this is the ordinary inverse.
    """
    left, values, right = jnp.linalg.svd(block)
    keep = values > _relative_threshold(block, rcond) * jnp.max(values)
    scaled = jnp.where(keep, 1.0 / jnp.where(keep, values, 1.0), 1.0)
    return right.conj().T @ (scaled[:, None] * left.conj().T)


def _pivoted_qr_inverse(
    block: Inexact[Array, "b b"], rcond: float | None
) -> Inexact[Array, "b b"]:
    """Invert one block through its column-pivoted QR factorization.

    Pivoting orders the diagonal of `R` by decreasing magnitude, so a small trailing entry
    marks a numerically dependent direction. Replacing those rows of `R` with rows of the
    identity leaves the triangular solve well conditioned while keeping it invertible, which
    is what leaves those directions unpreconditioned instead of dropping them.

    For a block of full rank nothing is replaced and this is the ordinary inverse.
    """
    width = block.shape[0]
    factor_q, factor_r, columns = jax.lax.linalg.qr(
        block, full_matrices=False, pivoting=True
    )
    diagonal = jnp.abs(jnp.diag(factor_r))
    keep = diagonal > _relative_threshold(block, rcond) * jnp.max(diagonal)

    identity = jnp.eye(width, dtype=factor_r.dtype)
    safe_r = jnp.where(keep[:, None], factor_r, identity)
    inverse = jax.scipy.linalg.solve_triangular(safe_r, factor_q.conj().T, lower=False)
    # Undo the column pivoting: row `i` of the solve belongs to original column `columns[i]`.
    return jnp.zeros_like(inverse).at[columns].set(inverse)


class BlockJacobiGMRES(AbstractSparseLinearSolver[_BlockJacobiState]):
    """Restarted GMRES preconditioned by the inverses of overlapping diagonal blocks.

    Written only from array operations, so unlike the other solvers here it needs no external
    library, traces into a single compiled computation and runs on any backend. It is an
    *iterative* solver: the result is accurate to the requested tolerance rather than to the
    precision of a factorization, and a hard problem may fail to converge, which is reported
    as an unsuccessful `RESULTS` rather than raised.

    The preconditioner keeps only the entries that fall inside some block. Those left out are
    dropped from the preconditioner alone, never from the matrix the iteration multiplies by,
    so discarding them costs iterations rather than accuracy. When the operator is small
    enough to fit inside one block the preconditioner is an exact inverse and the solve
    converges immediately.

    This solver can only handle square operators.
    """

    rtol: float = eqx.field(default=1e-6, static=True)
    atol: float = eqx.field(default=1e-6, static=True)
    restart: int = eqx.field(default=20, static=True)
    max_steps: int | None = eqx.field(default=None, static=True)
    stagnation_iters: int = eqx.field(default=20, static=True)
    ordering: Ordering = eqx.field(default=Ordering.RCM, static=True)
    block_size: int | None = eqx.field(default=None, static=True)
    max_block_size: int = eqx.field(default=128, static=True)
    overlap_fraction: float = eqx.field(default=0.25, static=True)
    capture_target: float = eqx.field(default=0.8, static=True)
    block_inverse: BlockInverse = eqx.field(default=BlockInverse.SVD, static=True)
    rcond: float | None = eqx.field(default=None, static=True)

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _BlockJacobiState:
        del options
        if operator.in_size() != operator.out_size():
            raise ValueError(
                "`BlockJacobiGMRES` may only be used for linear solves with square "
                "matrices"
            )
        matrix = _as_bcoo(operator)
        analysis = self._analyse(matrix, operator.in_size())
        return self._factorize(operator, analysis)

    def factorize(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> AbstractContextManager[SparseNumericState]:
        return factorize_through_init(self, operator, options)

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[False] = False
    ) -> AbstractContextManager[_BlockJacobiSymbolicScope]: ...

    @overload
    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: Literal[True]
    ) -> AbstractContextManager[SymbolicScopedSparseLinearSolver]: ...

    def factorize_symbolic(
        self, sparsity: _Sparsity, *, as_solver: bool = False
    ) -> AbstractContextManager[
        _BlockJacobiSymbolicScope | SymbolicScopedSparseLinearSolver
    ]:
        """Analyse a sparsity pattern once, for reuse across matrices sharing it.

        The reordering and the block partition depend only on the pattern, so this is where
        almost all of the analysis happens. Every state derived from the scope skips straight
        to assembling and inverting the blocks.

        Args:
            sparsity: the pattern to analyse. Only its indices are read.
            as_solver: yield a `SymbolicScopedSparseLinearSolver` pairing the scope with this
                       solver, instead of the bare scope, so the two need not be passed
                       around together.
        """
        scope = self._factorize_symbolic(sparsity)
        return as_scoped_solver(self, scope) if as_solver else scope

    @contextmanager
    def _factorize_symbolic(
        self, sparsity: _Sparsity
    ) -> Iterator[_BlockJacobiSymbolicScope]:
        # Kept separate from the public method so that one can be overloaded on `as_solver`,
        # since `@contextmanager` and `@overload` do not compose.
        matrix = _sparsity_pattern(sparsity)
        rows, _ = matrix.shape
        yield _BlockJacobiSymbolicScope(self, self._analyse(matrix, rows))

    def _analyse(self, matrix: BCOO, size: int) -> _BlockJacobiAnalysis:
        """Reorder the pattern and lay out the blocks. Reads no values."""
        if self.block_inverse is BlockInverse.QR and jax.default_backend() == "tpu":
            raise ValueError(
                "`BlockJacobiGMRES(block_inverse=BlockInverse.QR)` needs column-pivoted "
                "QR, which is not implemented on TPU. Use `BlockInverse.SVD` instead."
            )

        # Everything below is index arithmetic on a pattern that is normally a compile-time
        # constant, so under `jit` it folds away rather than becoming part of the program.
        with jax.ensure_compile_time_eval():
            rows = matrix.indices[:, 0]
            cols = matrix.indices[:, 1]
            measurable = not isinstance(rows, jax.core.Tracer)

            partner, matched = matching(rows, cols, size)
            if measurable:
                structural_rank = int(matched)
                if structural_rank < size:
                    raise ValueError(
                        "`BlockJacobiGMRES` was given an operator whose sparsity pattern is "
                        "structurally singular: a maximum matching between its rows and "
                        f"columns covers only {structural_rank} of its {size} rows, so by "
                        "the Frobenius-König theorem its determinant is identically zero and "
                        "no assignment of values can make it invertible."
                    )
            else:
                structural_rank = -1

            # A constraint row has no stored diagonal entry. Not every such row means a
            # saddle point, so the guard requires both kinds of row to be present and no
            # stored entry inside the constraint-by-constraint block, which in a genuine
            # saddle point is exactly the zero block that makes it one.
            has_diagonal = (
                jax.ops.segment_max(
                    (rows == cols).astype(jnp.int32), rows, num_segments=size
                )
                > 0
            )
            candidate = ~has_diagonal
            is_saddle_point = (
                jnp.any(candidate)
                & jnp.any(~candidate)
                & ~jnp.any(candidate[rows] & candidate[cols])
            )
            constraint = candidate & is_saddle_point

            base_perm = order(rows, cols, size, self.ordering)
            rank = inverse_permutation(base_perm)
            positions = jnp.arange(size, dtype=jnp.int32)
            safe_partner = jnp.where(partner >= 0, partner, positions)
            # Ordinary unknowns keep their bandwidth-reduced order. Each constraint unknown
            # moves to sit immediately after the unknown its matching pairs it with. See the
            # theory page's "Saddle-point systems" section for why that makes every block
            # holding a constraint and its partner invertible.
            group_key = jnp.where(constraint, 2 * rank[safe_partner] + 1, 2 * rank)
            group_perm = jnp.argsort(group_key).astype(jnp.int32)

            # A pattern with a hidden but otherwise ordinary diagonal, rather than a genuine
            # saddle point, wants a different repair: permute rows by the matching so the
            # diagonal becomes populated, then reorder as usual. Doing that needs the matching
            # to be known complete, which is only checked above when the pattern is not
            # traced, so it is only attempted then. A traced pattern with a hidden diagonal is
            # ordered as if it had none, which is safe, only less well preconditioned. See the
            # theory page's "Repairing an accidental diagonal" for why this and the grouping
            # above are alternatives rather than something to combine, and why `perm` and
            # `inv_perm` deliberately stop being mutual inverses here. Reordering equations
            # permutes the right-hand side but never the solution.
            if measurable and bool(jnp.any(candidate)) and not bool(is_saddle_point):
                mcol = inverse_permutation(partner)
                perm = mcol[base_perm]
                inv_perm = rank
                effective_rows = partner[rows]
            else:
                perm = jnp.where(is_saddle_point, group_perm, base_perm)
                inv_perm = inverse_permutation(perm)
                effective_rows = rows

            reordered_rows = inv_perm[effective_rows]
            reordered_cols = inv_perm[cols]
            reordered_constraint = constraint[perm]

            block_size, captured = self._resolve_block_size(
                reordered_rows, reordered_cols, size, matrix.nse
            )
            resolved, _, num_blocks = geometry(size, block_size, self.overlap_fraction)
            gather_index, core_mask = partition(size, block_size, self.overlap_fraction)
            destinations = block_destinations(
                reordered_rows, reordered_cols, size, block_size, self.overlap_fraction
            )

            # CSR order depends only on the pattern, so sorting happens here rather than
            # once per set of values.
            sort_order = jnp.lexsort((reordered_cols, reordered_rows))
            sorted_rows = reordered_rows[sort_order]
            per_row = jnp.bincount(reordered_rows, length=size)
            indptr = jnp.concatenate(
                [jnp.zeros(1, dtype=jnp.int32), jnp.cumsum(per_row).astype(jnp.int32)]
            )

        return _BlockJacobiAnalysis(
            perm=perm,
            inv_perm=inv_perm,
            constraint=reordered_constraint,
            gather_index=gather_index,
            core_mask=core_mask,
            destinations=destinations,
            sort_order=sort_order.astype(jnp.int32),
            sorted_rows=sorted_rows.astype(jnp.int32),
            col_indices=reordered_cols[sort_order].astype(jnp.int32),
            indptr=indptr,
            block_size=resolved,
            num_blocks=num_blocks,
            size=size,
            captured=captured,
            structural_rank=structural_rank,
        )

    def _resolve_block_size(
        self,
        reordered_rows: Integer[Array, " nse"],
        reordered_cols: Integer[Array, " nse"],
        size: int,
        stored: int,
    ) -> tuple[int, float]:
        """The block size to use, and how much of the pattern it captures.

        Measuring capture needs the pattern's indices as values, not placeholders, because
        the block size it chooses sets array shapes. Indices are values when the pattern is
        analysed eagerly, which is what `factorize_symbolic` is for, but not when
        `lineax.linear_solve` stages `init` into its own trace. The estimate below covers
        that case using only the shape.
        """
        measurable = not isinstance(reordered_rows, jax.core.Tracer)
        if self.block_size is not None:
            chosen = min(self.block_size, size)
        elif measurable:
            return choose_block_size(
                reordered_rows,
                reordered_cols,
                size,
                self.max_block_size,
                self.overlap_fraction,
                self.capture_target,
            )
        else:
            chosen = self._estimated_block_size(size, stored)

        if not measurable:
            return chosen, float("nan")
        measured = capture_fraction(
            reordered_rows, reordered_cols, size, chosen, self.overlap_fraction
        )
        return chosen, float(measured)

    def _estimated_block_size(self, size: int, stored: int) -> int:
        """A block size guessed from the pattern's shape alone.

        An operator small enough to fit inside the largest permitted block gets exactly that,
        since one block is then an exact inverse and costs no more than the caller has already
        agreed to pay. This mirrors what measuring would have chosen, and it matters: a
        needlessly partitioned small operator gives a weak preconditioner for no saving.

        Otherwise entries per row estimates the width of the band, since a band of half-width
        `h` holds about `2h + 1` entries in each row, and blocks of about `2.5h` are what it
        takes to cover most of such a band.
        """
        if size <= self.max_block_size:
            return size
        per_row = stored / max(size, 1)
        return int(min(max(round(1.25 * per_row), 4), self.max_block_size))

    def _factorize(
        self, operator: AbstractLinearOperator, analysis: _BlockJacobiAnalysis
    ) -> _BlockJacobiState:
        """Assemble the blocks from the operator's values and invert them."""
        if operator.in_size() != analysis.size:
            raise ValueError(
                "`BlockJacobiGMRES` was given an operator of size "
                f"{operator.in_size()}, but its symbolic factorization was computed for "
                f"size {analysis.size}."
            )
        matrix = _as_bcoo(operator)
        values = matrix.data[analysis.sort_order]
        width = analysis.block_size

        # One slot past the end of the block array absorbs every entry no block covers, and
        # is then dropped. This is the same trick `_bcoo_band` uses to discard off-band
        # entries, and it means nothing has to be tested for per entry.
        flat = jnp.zeros(
            analysis.num_blocks * width * width + 1, dtype=matrix.data.dtype
        )
        flat = flat.at[analysis.destinations].add(matrix.data[None, :])
        blocks = flat[:-1].reshape(analysis.num_blocks, width, width)

        reordered = BCSRLinearOperator(
            BCSR(
                (values, analysis.col_indices, analysis.indptr),
                shape=(analysis.size, analysis.size),
            )
        )
        return _BlockJacobiState(
            operator=reordered,
            inv_blocks=_invert_blocks(blocks, self.block_inverse, self.rcond),
            analysis=analysis,
            packed_structures=pack_structures(operator),
            solver=self,
        )

    def compute(
        self,
        state: _BlockJacobiState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        analysis = state.analysis
        packed_structures = state.packed_structures
        raveled = ravel_vector(vector, packed_structures)
        reordered = raveled.astype(state.inv_blocks.dtype)[analysis.perm]

        gmres = lx.GMRES(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            restart=self.restart,
            stagnation_iters=self.stagnation_iters,
        )
        preconditioner = lx.FunctionLinearOperator(
            _schwarz_apply(analysis, state.inv_blocks),
            state.operator.in_structure(),
        )
        solution, result, _ = gmres.compute(
            gmres.init(state.operator, {}),
            reordered,
            {"preconditioner": preconditioner},
        )
        result = self._certify(state, reordered, solution, result)

        restored = solution[analysis.inv_perm]
        return unravel_solution(restored, packed_structures), result, {}

    def _certify(
        self,
        state: _BlockJacobiState,
        vector: Inexact[Array, " n"],
        solution: Inexact[Array, " n"],
        result: RESULTS,
    ) -> RESULTS:
        """Downgrade a reported success that the true residual does not support.

        GMRES here is preconditioned on the left, and it measures convergence on `M (b - A x)`
        rather than on `b - A x`. Those agree only while `M` is well conditioned. A block
        structure that captures little of a hard matrix can be far from that, and then the
        preconditioned residual can fall below the tolerance while the real one has not, so the
        iteration would report success on an answer that does not solve the system.

        One extra product with the matrix settles it, using the same criterion GMRES applies
        but to the unpreconditioned residual. The cost is one matvec per solve, against the
        alternative of a success flag that cannot be relied on.
        """
        residual = vector - state.operator.mv(solution)
        scale = self.atol + self.rtol * jnp.abs(vector)
        converged = jnp.max(jnp.abs(residual) / scale) <= 1
        claimed = result == RESULTS.successful
        return RESULTS.where(claimed & jnp.invert(converged), RESULTS.breakdown, result)

    def transpose(
        self, state: _BlockJacobiState, options: dict[str, Any]
    ) -> tuple[_BlockJacobiState, dict[str, Any]]:
        del options
        analysis = state.analysis
        # The block partition sits at fixed index ranges, so a diagonal block of the
        # transpose occupies the same range as the matching block of the original and equals
        # its transpose, whatever produced that block. The transposed state is therefore
        # exact rather than an approximation, and every field but `perm` and `inv_perm`
        # carries over unchanged.
        values = _reordered_values(state)
        indices = jnp.stack([analysis.col_indices, analysis.sorted_rows], axis=-1)
        transposed = BCOOLinearOperator(
            BCOO((values, indices), shape=(analysis.size, analysis.size))
        )
        # `perm` and `inv_perm` are mutual inverses except when the second stage's row
        # permutation is active, and swapping the roles of a right-hand side and a solution
        # is exactly what transposing the system does: what reordered the input now restores
        # the output, and what restored the output now reorders the input. Inverting each
        # after the swap is what turns "new position of an original index" back into
        # "original index at a new position" and back, which is the only other difference
        # transposing makes. This reduces to reusing `analysis` unchanged whenever `perm` and
        # `inv_perm` were already mutual inverses, which is every case but that one.
        transposed_analysis = _BlockJacobiAnalysis(
            perm=inverse_permutation(analysis.inv_perm),
            inv_perm=inverse_permutation(analysis.perm),
            constraint=analysis.constraint,
            gather_index=analysis.gather_index,
            core_mask=analysis.core_mask,
            destinations=analysis.destinations,
            sort_order=analysis.sort_order,
            sorted_rows=analysis.sorted_rows,
            col_indices=analysis.col_indices,
            indptr=analysis.indptr,
            block_size=analysis.block_size,
            num_blocks=analysis.num_blocks,
            size=analysis.size,
            captured=analysis.captured,
            structural_rank=analysis.structural_rank,
        )
        return (
            _BlockJacobiState(
                operator=transposed,
                inv_blocks=jnp.matrix_transpose(state.inv_blocks),
                analysis=transposed_analysis,
                packed_structures=transpose_packed_structures(state.packed_structures),
                solver=state.solver,
            ),
            {},
        )

    def conj(
        self, state: _BlockJacobiState, options: dict[str, Any]
    ) -> tuple[_BlockJacobiState, dict[str, Any]]:
        del options
        return (
            _BlockJacobiState(
                operator=lx.conj(state.operator),
                inv_blocks=state.inv_blocks.conj(),
                analysis=state.analysis,
                packed_structures=state.packed_structures,
                solver=state.solver,
            ),
            {},
        )

    def assume_full_rank(self) -> bool:
        return True


def _reordered_values(state: _BlockJacobiState) -> Inexact[Array, " nse"]:
    """The reordered matrix's values, in the CSR order the analysis fixed."""
    return _as_bcoo(state.operator).data


def _schwarz_apply(
    analysis: _BlockJacobiAnalysis, inv_blocks: Inexact[Array, "num_blocks b b"]
):
    """Build the preconditioner's action on a vector.

    Each block reads the entries it covers, multiplies by its inverse, and writes back only
    the rows it owns. Restricting the write is what keeps the overlaps from being counted
    twice, and it is why the result is a well-defined operator rather than a sum of
    overlapping contributions.
    """

    def apply(vector: Inexact[Array, " n"]) -> Inexact[Array, " n"]:
        gathered = vector[analysis.gather_index]
        products = jax.vmap(jnp.matmul)(inv_blocks, gathered)
        owned = jnp.where(analysis.core_mask, products, 0)
        return jnp.zeros_like(vector).at[analysis.gather_index].add(owned)

    return apply


BlockJacobiGMRES.__init__.__doc__ = """**Arguments:**

- `rtol`: relative tolerance for terminating the iteration. Defaults to `1e-6`.
- `atol`: absolute tolerance for terminating the iteration. Defaults to `1e-6`.
- `restart`: size of the Krylov subspace built between restarts. This is the only part of
    the solve whose memory grows with the iteration count. Defaults to `20`.
- `max_steps`: maximum number of restarts before the solve is reported as failed. Defaults
    to `None`, meaning no limit.
- `stagnation_iters`: how many restarts may pass without the residual decreasing before the
    solve is halted. Defaults to `20`.
- `ordering`: which bandwidth-reducing reordering to apply. Defaults to
    `splineax.Ordering.RCM`. Use `splineax.Ordering.NONE` for an operator that is already
    narrowly banded, and `splineax.Ordering.SPECTRAL` for a pattern whose graph has very
    many breadth-first levels, where `RCM` becomes expensive.
- `block_size`: fix the block size instead of choosing one. Defaults to `None`, meaning
    choose. Choosing needs a pattern whose indices are known values, since the block size
    sets array shapes; setting this explicitly makes the solver traceable even when they are
    not.
- `max_block_size`: largest block size that may be chosen. Since inverting the blocks costs
    on the order of `n * max_block_size^2`, this is the main control on the cost of a
    numeric factorization. Defaults to `128`.
- `overlap_fraction`: fraction of each block that overlaps its neighbour. Raising it is
    usually the cheapest way to improve convergence, since it increases the part of the
    matrix the preconditioner captures at a cost of only `1 / (1 - overlap_fraction)`.
    Defaults to `0.25`.
- `capture_target`: fraction of the entries the blocks should cover, which is what the block
    size is chosen to reach. Defaults to `0.8`.
- `block_inverse`: how to invert each block. Defaults to `splineax.BlockInverse.SVD`, which
    works on every backend. `splineax.BlockInverse.QR` is cheaper, by a factor between about
    two and seven depending on the block size, but relies on column-pivoted QR, which is
    unavailable on TPU.
- `rcond`: relative threshold below which a direction of a block is treated as absent, either
    as a singular value or as a diagonal entry of its QR factor. Defaults to `None`, meaning
    the square root of the working precision. Deliberately looser than the usual choice for a
    pseudo-inverse, since it bounds how ill-conditioned the preconditioner may become.
"""
