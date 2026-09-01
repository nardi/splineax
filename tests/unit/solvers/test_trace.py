"""Tests for `splineax.solve_trace`, the opt-in debugging log of solver actions.

`solve_trace` records the analyze, factor, refactor, solve, and iterative-refinement steps
run inside its context, grouped into state-sequences (`init` ... `release`). These tests
check that the right actions are recorded in order, that reuse and rebuild are distinguished,
that iterative refinement records its steps, and that tracing adds nothing outside the
context. They lean on the `solver`/`make_operator`/`enable_x64` fixtures from
[conftest.py](conftest.py).
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator
from lineax._solution import RESULTS

import splineax as splx
from splineax import IterativeRefinement
from splineax._trace import _active

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


def _actions(trace: splx.SolveTrace) -> list[str]:
    return [record.action for record in trace.records]


def test_no_active_trace_outside_context() -> None:
    """With no `solve_trace` open there is no active trace, so nothing is ever recorded."""
    assert _active() is None
    with splx.solve_trace():
        assert _active() is not None
    assert _active() is None


def test_empty_block_records_nothing() -> None:
    """A trace with no solves in it collects no records and renders as empty."""
    with splx.solve_trace() as trace:
        pass
    assert trace.records == []
    assert trace.sequences == []
    assert "empty" in trace.render(colour=False)


def test_records_solve_actions(
    make_operator: OperatorFactory, solver: lx.AbstractLinearSolver
) -> None:
    """A solve inside the context records an `init` boundary, a `solve`, and lands in a
    single state-sequence, for every backend."""
    operator = make_operator(SQUARE_MATRIX)
    with splx.solve_trace() as trace:
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
        state.release()
    actions = _actions(trace)
    assert actions[0] == "init"
    assert "solve" in actions
    assert len(trace.sequences) == 1


def test_klu_records_analyze_factor_solve_in_order(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`KLU` records the analyze, factor, and solve of a fresh solve, in that order."""
    operator = make_operator(SQUARE_MATRIX)
    with splx.solve_trace() as trace:
        _, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        state.release()
    actions = _actions(trace)
    assert actions.index("analyze") < actions.index("factor") < actions.index("solve")


def test_update_reuses_analysis_on_shared_pattern(enable_x64: None) -> None:
    """`update` on a shared `sparsity_pattern_tag` records a `reused` update and a
    `refactor` carrying a finite `rcond`, rather than a fresh analyze."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    first = splx.BCOOLinearOperator(sparsity, tags=tag)
    second = splx.BCOOLinearOperator(BCOO.fromdense(2.0 * SQUARE_MATRIX), tags=tag)
    with splx.solve_trace() as trace:
        _, state = splx.linear_solve(first, RIGHT_HAND_SIDE, splx.KLU())
        _, state = splx.linear_solve(second, RIGHT_HAND_SIDE, splx.KLU(), state=state)
        state.release()
    updates = [record for record in trace.records if record.action == "update"]
    assert [update.outcome for update in updates] == ["reused"]
    refactors = [record for record in trace.records if record.action == "refactor"]
    assert len(refactors) == 1
    assert refactors[0].reused is True
    assert refactors[0].rcond is not None and refactors[0].rcond > 0.0
    # The reuse is motivated.
    assert refactors[0].reason is not None and "stable" in refactors[0].reason
    # Reusing the analysis means no second analyze was recorded.
    assert sum(action == "analyze" for action in _actions(trace)) == 1


def test_symbolic_state_records_factor_reason(enable_x64: None) -> None:
    """A first `update` on a symbolic-only state factors (there is no numeric to refactor),
    and the trace says so."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    operator = splx.BCOOLinearOperator(sparsity, tags=tag)
    solver = splx.KLU()
    with splx.solve_trace() as trace:
        state = solver.init_symbolic(sparsity)
        state = solver.update(state, operator)
        state.release()
    factors = [record for record in trace.records if record.action == "factor"]
    assert len(factors) == 1
    assert factors[0].reason is not None and "symbolic-only" in factors[0].reason


def test_update_rebuilds_on_changed_pattern(enable_x64: None) -> None:
    """`update` with a different sparsity pattern records a `rebuilt` update and re-analyzes,
    so a lost reuse is explicit in the log."""
    first = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    # A different sparsity pattern (reversed rows), so the tags cannot match.
    second = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX[::-1]))
    with splx.solve_trace() as trace:
        _, state = splx.linear_solve(first, RIGHT_HAND_SIDE, splx.KLU())
        _, state = splx.linear_solve(second, RIGHT_HAND_SIDE, splx.KLU(), state=state)
        state.release()
    updates = [record for record in trace.records if record.action == "update"]
    assert [update.outcome for update in updates] == ["rebuilt"]
    # The rebuild is motivated: neither operator carries a sparsity tag to match on.
    assert updates[0].reason is not None and "tag" in updates[0].reason
    # A rebuild re-analyzes, so there are two analyze events and still one sequence.
    assert sum(action == "analyze" for action in _actions(trace)) == 2
    assert len(trace.sequences) == 1


def test_two_lineages_are_separate_sequences(enable_x64: None) -> None:
    """Two independent `init` ... `release` lineages in one block are two sequences."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    with splx.solve_trace() as trace:
        _, first = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        first.release()
        _, second = splx.linear_solve(operator, RIGHT_HAND_SIDE, splx.KLU())
        second.release()
    assert len(trace.sequences) == 2
    assert all(sequence[0].action == "init" for sequence in trace.sequences)


class _JacobiState(eqx.Module):
    """State of `_JacobiSolver`: the operator's diagonal and the operator itself."""

    diagonal: Array
    operator: AbstractLinearOperator

    def release(self) -> None:
        """No-op, since a Jacobi state owns nothing to free."""


class _JacobiSolver(lx.AbstractLinearSolver[_JacobiState]):
    """A weak stateful solver (one Jacobi sweep, `x = b / diag(A)`), only for these tests.

    Wrapped in `IterativeRefinement` the correction loop becomes a Jacobi iteration that
    needs several steps to converge, so the trace records more than one `ir_step`.
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


def test_iterative_refinement_records_steps(enable_x64: None) -> None:
    """Iterative refinement records `ir_start`, one `ir_step` per correction with an
    increasing step and a non-increasing residual, and a converged `ir_result`."""
    operator = splx.BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = IterativeRefinement(_JacobiSolver(), tol=1e-8, max_steps=50)
    with splx.solve_trace() as trace:
        solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert jnp.allclose(
        solution,
        jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE)),
        atol=1e-6,
    )
    actions = _actions(trace)
    assert actions.count("ir_start") == 1
    assert actions.count("ir_result") == 1
    steps = [record for record in trace.records if record.action == "ir_step"]
    assert len(steps) >= 2
    assert [record.step for record in steps] == list(range(1, len(steps) + 1))
    norms = [record.residual_norm for record in steps]
    assert all(later <= earlier for earlier, later in zip(norms, norms[1:]))
    result = next(record for record in trace.records if record.action == "ir_result")
    assert result.converged is True


def test_records_under_jit(enable_x64: None) -> None:
    """Tracing works through `jax.jit`: every expected action is recorded, and the ordered
    structural events keep their relative order (the solve rides on unordered callbacks)."""
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

    with splx.solve_trace() as trace:
        run(sparsity.data)
    actions = _actions(trace)
    for expected in ("init", "analyze", "factor", "solve", "track", "release"):
        assert expected in actions
    # Ordered structural events keep program order among themselves.
    ordered = [
        a for a in actions if a in ("init", "analyze", "factor", "track", "release")
    ]
    assert ordered == ["init", "analyze", "factor", "track", "release"]


def test_function_compiled_outside_context_records_nothing(enable_x64: None) -> None:
    """Tracing is applied at trace time, so a function compiled before the block emits
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

    # Compile (and run) once outside any trace, so the cached computation has no callbacks.
    run(sparsity.data)
    with splx.solve_trace() as trace:
        run(sparsity.data)
    assert trace.records == []
