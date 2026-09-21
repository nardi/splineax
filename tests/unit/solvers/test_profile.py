"""Tests for `splineax.SolveProfile`, the opt-in debugging log of solver operations.

A `SolveProfile` (created with `create_solve_profile`, entered with `with`) records the generic
stateful-API operations (`init`, `update`, `compute`, `track`, `release`) and the
solver-specific operations nested under them (`KLU.analyze`, `KLU.refactor`, ...), grouped
into state-sequences. These tests check that the right operations are recorded in order, that
reuse and rebuild are distinguished, that iterative refinement records its steps, that tracing
adds nothing outside any `with` block, that solves stay in program order under `jit`, `vmap`,
and both autodiff modes, that independent profiles stay independent, and that
`profile_solves` profiles a jitted function correctly across repeated calls. They lean on the
`solver`/`make_operator`/`enable_x64` fixtures from [conftest.py](conftest.py).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS

import splineax as splx
from splineax import IterativeRefinement, ProfileRecord
from splineax._profile import _active
from splineax.solvers._sparse import _Sparsity

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


def _ordered(profile: splx.SolveProfile) -> list[ProfileRecord]:
    return sorted(profile.records, key=lambda record: record.order)


def _ops(profile: splx.SolveProfile) -> list[str]:
    return [record.operation for record in _ordered(profile)]


class _NoFilter:
    """Sentinel distinguishing "no `solver` filter given" from the meaningful `solver=None`."""


_NO_FILTER = _NoFilter()


def _by_op(
    profile: splx.SolveProfile,
    operation: str,
    solver: str | None | _NoFilter = _NO_FILTER,
) -> list[ProfileRecord]:
    records = [r for r in _ordered(profile) if r.operation == operation]
    if not isinstance(solver, _NoFilter):
        records = [r for r in records if r.solver == solver]
    return records


def test_no_active_profile_outside_context() -> None:
    """With no `SolveProfile` entered there is no active profile, so nothing is ever recorded."""
    assert _active() is None
    with splx.create_solve_profile():
        assert _active() is not None
    assert _active() is None


def test_empty_block_records_nothing() -> None:
    """A profile with no solves in it collects no records and renders as empty."""
    profile = splx.create_solve_profile()
    with profile:
        pass
    assert profile.records == []
    assert profile.sequences == []
    assert "empty" in profile.render(color=False)


def test_records_generic_operations(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """A solve inside the context opens with a generic `init`, runs a `compute`, and lands in
    a single state-sequence, for every backend."""
    operator = make_operator(SQUARE_MATRIX)
    profile = splx.create_solve_profile()
    with profile:
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
        state.release()
    ops = _ops(profile)
    assert ops[0] == "init"
    assert "compute" in ops
    assert len(profile.sequences) == 1
    # A generic `init` carries no solver. The work under it does.
    assert _by_op(profile, "init")[0].solver is None
    assert any(record.solver is not None for record in profile.records)


def test_klu_nests_native_operations_in_order(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`KLU`'s analyze, factor, and triangular solve are recorded as `KLU.*` operations, in
    that order."""
    operator = make_operator(SQUARE_MATRIX)
    profile = splx.create_solve_profile()
    with profile:
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
    ops = _ops(profile)
    assert ops.index("analyze") < ops.index("factor") < ops.index("solve_with_numeric")
    assert _by_op(profile, "analyze")[0].solver == "KLU"
    # The solve reports the rebuild reason: the resident factorization was a cache hit.
    solves = _by_op(profile, "solve_with_numeric", "KLU")
    assert solves[0].outputs["rebuild"] == 0
    # The native free operations nest under `release`.
    assert _by_op(profile, "free_numeric", "KLU") and _by_op(
        profile, "free_symbolic", "KLU"
    )


def test_update_reuses_analysis_on_shared_pattern(enable_x64: None) -> None:
    """`update` on a shared `sparsity_pattern_tag` records a `reused` outcome and a
    `KLU.refactor` carrying a finite `rcond`, rather than a fresh analyze."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    first = splx.BCOOLinearOperator(sparsity, tags=tag)
    second = splx.BCOOLinearOperator(BCOO.fromdense(2.0 * SQUARE_MATRIX), tags=tag)
    profile = splx.create_solve_profile()
    with profile:
        _, state = splx.linear_solve(first, RIGHT_HAND_SIDE, splx.KLU())
        _, state = splx.linear_solve(second, RIGHT_HAND_SIDE, splx.KLU(), state=state)
        state.release()
    updates = _by_op(profile, "update")
    assert [update.outputs["outcome"] for update in updates] == ["reused"]
    refactors = _by_op(profile, "refactor", "KLU")
    assert len(refactors) == 1
    assert refactors[0].outputs["reused"] is True
    assert refactors[0].outputs["rcond"] > 0.0
    assert "stable" in refactors[0].outputs["reason"]
    # The refactor reports the rebuild reason of the numeric handle it refreshed.
    assert refactors[0].outputs["rebuild"] == 0
    # Reusing the analysis means no second analyze was recorded.
    assert _ops(profile).count("analyze") == 1


def test_symbolic_state_records_factor_reason(enable_x64: None) -> None:
    """A first `update` on a symbolic-only state factors (there is no numeric to refactor),
    and the profile says so."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    operator = splx.BCOOLinearOperator(sparsity, tags=tag)
    solver = splx.KLU()
    profile = splx.create_solve_profile()
    with profile:
        state = solver.init_symbolic(sparsity)
        state = solver.update(state, operator)
        state.release()
    factors = _by_op(profile, "factor", "KLU")
    assert len(factors) == 1
    assert factors[0].outputs["reason"] == "No prior factorization"


def test_update_rebuilds_on_changed_pattern(enable_x64: None) -> None:
    """`update` with a different sparsity pattern records a `rebuilt` outcome and re-analyzes,
    so a lost reuse is explicit in the log."""
    first = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    # A different sparsity pattern (reversed rows), so the tags cannot match.
    second = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX[::-1]))
    profile = splx.create_solve_profile()
    with profile:
        _, state = splx.linear_solve(first, RIGHT_HAND_SIDE, splx.KLU())
        _, state = splx.linear_solve(second, RIGHT_HAND_SIDE, splx.KLU(), state=state)
        state.release()
    updates = _by_op(profile, "update")
    assert [update.outputs["outcome"] for update in updates] == ["rebuilt"]
    # The rebuild is motivated: neither operator carries a sparsity tag to match on.
    assert "tag" in updates[0].outputs["reason"]
    # A rebuild re-analyzes, so there are two analyze operations and still one sequence.
    assert _ops(profile).count("analyze") == 2
    assert len(profile.sequences) == 1


def test_two_lineages_are_separate_sequences(enable_x64: None) -> None:
    """Two independent `init` ... `release` lineages in one block are two sequences."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    profile = splx.create_solve_profile()
    with profile:
        _, first = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        first.release()
        _, second = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        second.release()
    assert len(profile.sequences) == 2
    assert all(sequence[0].operation == "init" for sequence in profile.sequences)


def test_independent_profiles_do_not_cross_talk(enable_x64: None) -> None:
    """Entering a second profile after the first has exited only ever adds to the second."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))

    profile1 = splx.create_solve_profile()
    with profile1:
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
    first_count = len(profile1.records)
    assert first_count > 0

    profile2 = splx.create_solve_profile()
    with profile2:
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
    assert len(profile1.records) == first_count
    assert len(profile2.records) > 0


def test_reentering_same_profile_accumulates(enable_x64: None) -> None:
    """Entering the same `SolveProfile` object a second time adds further records to it."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    profile = splx.create_solve_profile()
    with profile:
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
    first_count = len(profile.records)
    with profile:
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
    assert len(profile.records) > first_count


def test_call_outside_block_after_entering_records_nothing_further(
    enable_x64: None,
) -> None:
    """A call to an already-compiled jitted function, made after the profile that compiled it
    has exited, records nothing further. The callback baked into that compiled executable
    looks up the active profile fresh each time it fires, so it sees none active and no-ops,
    rather than keep appending into a profile whose `with` block has already closed."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape

    @eqx.filter_jit
    def run(data: Array) -> Array:
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=shape, indices_sorted=True)
        )
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
        return state.shape[0]  # type: ignore[return-value]

    profile = splx.create_solve_profile()
    with profile:
        run(sparsity.data)
    first_count = len(profile.records)
    assert first_count > 0

    # Same shape, so this is a cache hit against the executable built above, and it runs
    # outside any profile.
    run(sparsity.data)
    assert len(profile.records) == first_count


class _JacobiState(eqx.Module):
    """State of `_JacobiSolver`: the operator's diagonal and the operator itself."""

    diagonal: Array
    operator: AbstractLinearOperator

    def release(self) -> None:
        """No-op, since a Jacobi state owns nothing to free."""


class _JacobiSolver(lx.AbstractLinearSolver[_JacobiState]):
    """A weak stateful solver (one Jacobi sweep, `x = b / diag(A)`), only for these tests.

    Wrapped in `IterativeRefinement` the correction loop becomes a Jacobi iteration that
    needs several steps to converge, so the profile records more than one `refine_step`.
    """

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any] = {}
    ) -> _JacobiState:
        del options
        return _JacobiState(jnp.diag(operator.as_matrix()), operator)

    def update(
        self,
        state: _JacobiState,
        operator: AbstractLinearOperator,
        options: dict[str, Any] = {},
    ) -> _JacobiState:
        return self.init(operator, options)

    def compute(
        self, state: _JacobiState, vector: PyTree[Array], options: dict[str, Any]
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options
        return vector / state.diagonal, RESULTS.successful, {}

    def transpose(
        self, state: _JacobiState, options: dict[str, Any]
    ) -> tuple[_JacobiState, dict[str, Any]]:
        del options
        transposed = state.operator.transpose()
        return _JacobiState(jnp.diag(transposed.as_matrix()), transposed), {}

    def conj(
        self, state: _JacobiState, options: dict[str, Any]
    ) -> tuple[_JacobiState, dict[str, Any]]:
        del options
        return state, {}

    def assume_full_rank(self) -> bool:
        return True

    def init_symbolic(
        self, sparsity: _Sparsity, options: dict[str, Any] = {}
    ) -> _JacobiState:
        """Not supported: a dense Jacobi sweep has no sparsity pattern to analyze."""
        raise NotImplementedError(
            "_JacobiSolver has no symbolic phase; it only satisfies "
            "`SparseLinearSolver` so `IterativeRefinement` can wrap it in these tests."
        )


def test_iterative_refinement_records_steps(enable_x64: None) -> None:
    """Iterative refinement records one `refine_start`, one `refine_step` per correction with
    an increasing step and a non-increasing residual, and a converged `refine_result`, all as
    `IterativeRefinement.*` operations nested under a single `compute`."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = IterativeRefinement(_JacobiSolver(), tol=1e-8, max_steps=50)
    profile = splx.create_solve_profile()
    with profile:
        solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert jnp.allclose(
        solution,
        jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE)),
        atol=1e-6,
    )
    ops = _ops(profile)
    assert ops.count("refine_start") == 1
    assert ops.count("refine_result") == 1
    assert ops.count("compute") == 1
    steps = _by_op(profile, "refine_step", "IterativeRefinement")
    assert len(steps) >= 2
    assert [record.outputs["step"] for record in steps] == list(
        range(1, len(steps) + 1)
    )
    norms = [record.outputs["residual_norm"] for record in steps]
    assert all(later <= earlier for earlier, later in zip(norms, norms[1:]))
    result = _by_op(profile, "refine_result")[0]
    assert result.outputs["converged"] is True


def test_refinement_needing_no_correction_records_nothing(enable_x64: None) -> None:
    """A refinement whose initial solve already meets the tolerance records no
    `IterativeRefinement` operations at all, so the profile looks as if refinement
    was not used."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = IterativeRefinement(splx.KLU(), tol=1e-2, max_steps=5)
    profile = splx.create_solve_profile()
    with profile:
        solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
        state.release()
    assert solution.result == lx.RESULTS.successful
    # Only the plain `KLU` operations ran: no refine_start, refine_step, or refine_result.
    assert not any(record.operation.startswith("refine") for record in profile.records)
    assert _ops(profile).count("compute") == 1


def test_refinement_exhausting_steps_still_records(enable_x64: None) -> None:
    """A refinement that exhausts `max_steps` without converging still records its
    `refine_start`, every `refine_step`, and a `refine_result` with
    `converged=False`."""
    solver = IterativeRefinement(_JacobiSolver(), tol=1e-14, max_steps=3)
    operator = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    profile = splx.create_solve_profile()
    with profile:
        solution, state = splx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver, throw=False
        )
    assert solution.result == lx.RESULTS.max_steps_reached
    assert jnp.isnan(solution.value).all()
    ops = _ops(profile)
    assert ops.count("refine_start") == 1
    assert ops.count("refine_result") == 1
    steps = _by_op(profile, "refine_step", "IterativeRefinement")
    assert [record.outputs["step"] for record in steps] == [1, 2, 3]
    assert _by_op(profile, "refine_result")[0].outputs["converged"] is False


def test_profile_of_vmap_and_grad_plain_solve(enable_x64: None) -> None:
    """A plain solve under `jax.vmap`, `jax.grad`, `jax.jacfwd`, and `jax.jacrev` runs
    and records its operations, so the callbacks compose with batching and both
    autodiff modes. (`vmap` over a refinement loop is the one combination that does
    not compose, from the batched `while_loop` predicate rejecting unordered IO.)"""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape

    def solve(data: Array, rhs: Array) -> Array:
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=shape, indices_sorted=True)
        )
        solution, state = splx.linear_solve(operator, rhs, splx.KLU())
        return solution.value

    data = sparsity.data
    batched = jnp.stack([RIGHT_HAND_SIDE, 2 * RIGHT_HAND_SIDE])
    for name, fn in (
        ("vmap", jax.vmap(solve, in_axes=(None, 0))),
        ("grad", jax.grad(lambda d, b: solve(d, b).sum())),
        ("jacfwd", jax.jacfwd(solve)),
        ("jacrev", jax.jacrev(solve)),
    ):
        profile = splx.create_solve_profile()
        with profile:
            if name == "vmap":
                fn(data, batched)
            else:
                fn(data, RIGHT_HAND_SIDE)
        assert len(profile.records) > 0, f"{name} recorded nothing"


def test_records_under_jit(enable_x64: None) -> None:
    """Profiling works through `jax.jit`: every expected operation is recorded, and the
    trace-time order key keeps the printed tree in program order."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape

    @eqx.filter_jit
    def run(data: Array) -> Array:
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=shape, indices_sorted=True)
        )
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
        return state.shape[0]  # type: ignore[return-value]

    profile = splx.create_solve_profile()
    with profile:
        run(sparsity.data)
    ops = _ops(profile)
    for expected in (
        "init",
        "analyze",
        "factor",
        "solve_with_numeric",
        "track",
        "release",
    ):
        assert expected in ops
    # Sorting by the order key recovers program order, even under jit.
    assert (
        ops.index("init")
        < ops.index("analyze")
        < ops.index("factor")
        < ops.index("solve_with_numeric")
        < ops.index("release")
    )


def _make_chained_solves(sparsity: BCOO) -> Callable[[Array, Array], Array]:
    """Build a function that solves twice through one threaded state, then releases it.

    Both operators carry `sparsity`'s indices and one shared tag, so the second solve runs
    an `update` against the state the first solve returned.
    """
    tag = splx.sparsity_pattern_tag(sparsity)

    def operator_of(values: Array) -> splx.BCOOLinearOperator:
        return splx.BCOOLinearOperator(
            BCOO(
                (values, sparsity.indices),
                shape=sparsity.shape,
                indices_sorted=True,
            ),
            tags=tag,
        )

    def chained_solves(data: Array, rhs: Array) -> Array:
        first, state = splx.linear_solve(operator_of(data), rhs, splx.KLU())
        second, state = splx.linear_solve(
            operator_of(2.0 * data), first.value, splx.KLU(), state=state
        )
        state.release()
        return second.value.sum()

    return chained_solves


@pytest.mark.parametrize(
    ("transform", "batch_rhs", "expected_ops"),
    [
        (
            eqx.filter_jit,
            False,
            ["init", "compute", "track", "update", "compute", "track", "release"],
        ),
        (
            lambda fn: eqx.filter_jit(jax.vmap(fn, in_axes=(None, 0))),
            True,
            ["init", "compute", "track", "update", "compute", "track", "release"],
        ),
        # The tangent solve shares the primal's factorization, so it runs before the
        # `track` that closes the primal's factorization window.
        (
            lambda fn: eqx.filter_jit(jax.jacfwd(fn, argnums=1)),
            False,
            [
                "init",
                "compute",
                "compute",
                "track",
                "update",
                "compute",
                "compute",
                "track",
                "release",
            ],
        ),
        # The backward solves run after the whole forward pass, second solve first.
        (
            lambda fn: eqx.filter_jit(jax.grad(fn, argnums=1)),
            False,
            [
                "init",
                "compute",
                "track",
                "update",
                "compute",
                "track",
                "release",
                "compute",
                "compute",
            ],
        ),
    ],
    ids=["jit", "vmap", "jacfwd", "grad"],
)
def test_generic_operations_in_program_order_under_transforms(
    enable_x64: None,
    transform: Callable[[Callable[[Array, Array], Array]], Callable[..., Any]],
    batch_rhs: bool,
    expected_ops: list[str],
) -> None:
    """Under `jit`, `vmap`, `jacfwd`, and `grad`, the generic operations of two chained
    solves sort in program order. lineax traces each solve's `compute` late, so this
    checks that the order slot reserved at each call site places the `compute` there."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    chained_solves = _make_chained_solves(sparsity)
    rhs = (
        jnp.stack([RIGHT_HAND_SIDE, 2.0 * RIGHT_HAND_SIDE])
        if batch_rhs
        else RIGHT_HAND_SIDE
    )
    profile = splx.create_solve_profile()
    with profile:
        jax.block_until_ready(transform(chained_solves)(sparsity.data, rhs))
    generic_ops = [
        record.operation for record in _ordered(profile) if record.solver is None
    ]
    assert generic_ops == expected_ops


def test_function_compiled_outside_context_records_nothing(enable_x64: None) -> None:
    """Tracing is applied at profile time, so a function compiled before the block emits
    nothing when later called inside it (a documented limitation)."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    indices, shape = sparsity.indices, sparsity.shape

    @eqx.filter_jit
    def run(data: Array) -> Array:
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=shape, indices_sorted=True)
        )
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
        return state.shape[0]  # type: ignore[return-value]

    # Compile (and run) once outside any profile, so the cached computation has no callbacks.
    run(sparsity.data)
    profile = splx.create_solve_profile()
    with profile:
        run(sparsity.data)
    assert profile.records == []


def _make_solve(
    indices: Array, shape: tuple[int, ...]
) -> Callable[..., tuple[int, splx.SolveProfile | None]]:
    @splx.profile_solves
    @eqx.filter_jit
    def solve(data: Array) -> Array:
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=shape, indices_sorted=True)
        )
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
        return state.shape[0]  # type: ignore[return-value]

    return solve


def test_profile_solves_returns_result_and_profile(enable_x64: None) -> None:
    """A default call returns `(result, profile)`, with the profile populated as usual."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    solve = _make_solve(sparsity.indices, sparsity.shape)
    result, profile = solve(sparsity.data)
    assert result == SQUARE_MATRIX.shape[0]
    assert isinstance(profile, splx.SolveProfile)
    assert "init" in _ops(profile)


def test_profile_solves_enabled_false_skips_profiling(enable_x64: None) -> None:
    """`enabled=False` returns `(result, None)` and profiles nothing."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    solve = _make_solve(sparsity.indices, sparsity.shape)
    result, profile = solve(sparsity.data, enabled=False)
    assert result == SQUARE_MATRIX.shape[0]
    assert profile is None


def test_profile_solves_profiles_repeated_calls(enable_x64: None) -> None:
    """Calling a `profile_solves`-wrapped jitted function twice with the same input shape
    profiles both calls, not just the first. The first call compiles the function while its
    own profile is active, so the compiled executable carries profiling hooks. The second call,
    a cache hit, then correctly records into its own fresh profile rather than the first
    call's now-exited one."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    solve = _make_solve(sparsity.indices, sparsity.shape)

    _, profile1 = solve(sparsity.data)
    _, profile2 = solve(sparsity.data)

    assert profile1 is not None
    assert profile2 is not None
    assert profile1 is not profile2
    assert len(profile1.records) > 0
    assert len(profile2.records) > 0
    assert profile1.records is not profile2.records


def test_profile_solves_first_call_disabled_forfeits_profiling(
    enable_x64: None,
) -> None:
    """If the very first call for a shape happens with `enabled=False`, that shape's
    compiled executable never gets profiling hooks, so a later `enabled=True` call for the
    same shape returns an empty profile. This is the one sharp edge `profile_solves` cannot
    remove, inherent to JAX compiling once per shape: always let the first call for a shape
    go through with profiling enabled (the default) if you might ever want to profile it."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    solve = _make_solve(sparsity.indices, sparsity.shape)

    # First call for this shape, so no profiling hooks are ever compiled into it.
    solve(sparsity.data, enabled=False)
    # Same shape, so this is a cache hit against the executable built above.
    _, profile = solve(sparsity.data)
    assert profile is not None
    assert profile.records == []
