"""Tests for the shared update of a solver state with a transposed operator.

A solver that can solve `A^T` through its factorization of `A` reports it with
`transposes_cheaply`. `update_for_transposed_pattern` then folds an operator with the
transposed pattern into the state through the solver's own `update` and `transpose`. These
tests use a stub solver, so they check that logic apart from any backend.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO, BCSR

import splineax as splx
from splineax import BCOOLinearOperator, BCSRLinearOperator
from splineax.operators._tags import _ContentPatternTag, find_pattern_tag
from splineax.solvers._sparse import (
    operator_pattern_tag,
    transposed_entry_order,
    update_for_transposed_pattern,
)

# A non-symmetric matrix, so a pattern differs from its transpose.
MATRIX = jnp.array(
    [
        [10.0, 2.0, 0.0, 7.0],
        [3.0, 14.0, 5.0, 0.0],
        [0.0, 6.0, 18.0, 9.0],
        [0.0, 0.0, 1.0, 12.0],
    ]
)

# An upper triangular pattern, which is not the transpose of the pattern of `MATRIX`.
TRIANGULAR_MATRIX = jnp.array(np.triu(np.ones((4, 4))) + np.eye(4))


@dataclasses.dataclass(frozen=True)
class StubState:
    """The fields of a solver state that the shared update reads."""

    operator: lx.AbstractLinearOperator | None
    transposed: bool
    sparsity_tag: object
    shape: tuple[int, ...]


class StubSolver:
    """A solver that records the operators its `update` receives and flips `transposed`."""

    def __init__(self, reports_cheap_transpose: bool) -> None:
        self.reports_cheap_transpose = reports_cheap_transpose
        self.updates: list[lx.AbstractLinearOperator] = []

    def transposes_cheaply(self, state: StubState) -> bool:
        del state
        return self.reports_cheap_transpose

    def transpose(
        self, state: StubState, options: dict[str, Any]
    ) -> tuple[StubState, dict[str, Any]]:
        del options
        return dataclasses.replace(state, transposed=not state.transposed), {}

    def update(
        self,
        state: StubState,
        operator: lx.AbstractLinearOperator,
        options: dict[str, Any],
    ) -> StubState:
        del options
        self.updates.append(operator)
        return dataclasses.replace(state, operator=operator)


def content_tag_of(operator: BCOOLinearOperator) -> _ContentPatternTag:
    tag = find_pattern_tag(operator.tags)
    assert isinstance(tag, _ContentPatternTag)
    return tag


def state_built_on(operator: BCOOLinearOperator) -> StubState:
    return StubState(operator, False, content_tag_of(operator), (4, 4))


def fold_in(
    solver: StubSolver, state: StubState, operator: lx.AbstractLinearOperator
) -> StubState | None:
    tag = operator_pattern_tag(operator)
    return update_for_transposed_pattern(solver, state, operator, tag, {}, "Stub")


def tagged_operator(matrix: jax.Array) -> BCOOLinearOperator:
    pattern = BCOO.fromdense(matrix)
    return BCOOLinearOperator(pattern, tags=splx.sparsity_pattern_tag(pattern))


def test_a_tag_and_its_transpose_have_aligned_entries() -> None:
    tag = content_tag_of(tagged_operator(MATRIX))
    assert transposed_entry_order(tag, tag.transpose()) == (True, None)


def test_a_tag_is_not_its_own_transpose_pattern() -> None:
    tag = content_tag_of(tagged_operator(MATRIX))
    assert transposed_entry_order(tag, tag) == (False, None)
    assert transposed_entry_order(None, tag) == (False, None)
    assert transposed_entry_order(tag, None) == (False, None)


def test_entries_in_another_order_are_matched_by_position() -> None:
    """The order maps each entry of the state's matrix to the operator entry that holds
    its value, whatever order the transpose is stored in."""
    state_tag = content_tag_of(tagged_operator(MATRIX))
    stored = BCSR.fromdense(MATRIX.T)
    operator_tag = splx.sparsity_pattern_tag(stored)
    is_transpose, entry_order = transposed_entry_order(state_tag, operator_tag)
    assert is_transpose and entry_order is not None
    values = np.asarray(stored.data)[entry_order]
    rows, columns = np.asarray(state_tag.indices).T
    np.testing.assert_allclose(values, np.asarray(MATRIX)[rows, columns])


def test_unrelated_patterns_do_not_match() -> None:
    state_tag = content_tag_of(tagged_operator(MATRIX))
    unrelated_tag = content_tag_of(tagged_operator(TRIANGULAR_MATRIX))
    assert transposed_entry_order(state_tag, unrelated_tag) == (False, None)


def test_a_solver_without_a_cheap_transpose_gets_no_shared_update() -> None:
    operator = tagged_operator(MATRIX)
    solver = StubSolver(reports_cheap_transpose=False)
    state = state_built_on(operator)
    assert fold_in(solver, state, operator.transpose()) is None
    assert not solver.updates


def test_the_state_is_updated_with_the_transposed_matrix_and_then_transposed() -> None:
    """The solver's `update` receives the matrix the state analyzed, built from the new
    operator's values, and the result is a transposed state."""
    operator = tagged_operator(MATRIX)
    other_operator = tagged_operator(1.5 * MATRIX)
    solver = StubSolver(reports_cheap_transpose=True)
    updated_state = fold_in(
        solver, state_built_on(operator), other_operator.transpose()
    )
    assert updated_state is not None
    (received,) = solver.updates
    np.testing.assert_allclose(received.as_matrix(), 1.5 * MATRIX)
    assert updated_state.transposed


def test_values_in_another_order_reach_the_update_in_the_state_order() -> None:
    operator = tagged_operator(MATRIX)
    stored = BCSR.fromdense((2.0 * MATRIX).T)
    transposed_operator = BCSRLinearOperator(
        stored, tags=splx.sparsity_pattern_tag(stored)
    )
    solver = StubSolver(reports_cheap_transpose=True)
    updated_state = fold_in(solver, state_built_on(operator), transposed_operator)
    assert updated_state is not None
    (received,) = solver.updates
    np.testing.assert_allclose(received.as_matrix(), 2.0 * MATRIX)
    assert operator_pattern_tag(received) == operator_pattern_tag(operator)
    assert updated_state.transposed


def test_the_transpose_of_the_state_operator_needs_no_update() -> None:
    operator = tagged_operator(MATRIX)
    solver = StubSolver(reports_cheap_transpose=True)
    updated_state = fold_in(solver, state_built_on(operator), operator.transpose())
    assert updated_state is not None
    assert not solver.updates
    assert updated_state.transposed
    # A state that is transposed already is returned as it is.
    assert fold_in(solver, updated_state, operator.transpose()) is updated_state


def test_a_transposed_state_is_updated_in_its_own_orientation() -> None:
    """A transposed state is flipped back before its update, since the operator to fold in
    is the matrix it factorized."""
    operator = tagged_operator(MATRIX)
    solver = StubSolver(reports_cheap_transpose=True)
    transposed_state = dataclasses.replace(state_built_on(operator), transposed=True)
    updated_state = fold_in(
        solver, transposed_state, tagged_operator(2.0 * MATRIX).transpose()
    )
    assert updated_state is not None
    assert updated_state.transposed
    assert len(solver.updates) == 1


def test_a_pattern_that_is_not_a_transpose_is_left_to_the_solver() -> None:
    operator = tagged_operator(MATRIX)
    unrelated_operator = tagged_operator(TRIANGULAR_MATRIX)
    solver = StubSolver(reports_cheap_transpose=True)
    assert fold_in(solver, state_built_on(operator), unrelated_operator) is None


@pytest.mark.parametrize("reports_cheap_transpose", [True, False])
def test_the_solver_capability_is_read_for_every_state(
    reports_cheap_transpose: bool,
) -> None:
    operator = tagged_operator(MATRIX)
    solver = StubSolver(reports_cheap_transpose=reports_cheap_transpose)
    result = fold_in(solver, state_built_on(operator), operator.transpose())
    assert (result is not None) == reports_cheap_transpose
