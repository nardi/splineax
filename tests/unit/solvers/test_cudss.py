"""CuDSS-specific tests for availability, mtype selection, conversion, and (via a fake
`spineax.cudss` module) the stateful dispatch/reuse contract, plus real-GPU tests.

`CuDSS()` requires the optional cuDSS dependency, which needs a CUDA GPU to even import
(it dlopens a CUDA extension), so it can never be genuinely installed in ordinary CPU CI.
Two tiers of coverage follow from that:

- **CPU-runnable, always on:** availability/`ImportError`, tag-to-mtype selection,
  operator-to-CSR conversion, and the square/type checks, none of which touch
  `spineax.cudss` at all, plus the init/update/transpose/conj/refactorize logic in
  `_cudss.py`, exercised against a small fake `spineax.cudss` module (`FakeCuDSS` below)
  that reproduces its documented phase contract (analyze -> factorize/refactorize ->
  solve, phase checks, dtype/nnz checks) with a real dense `jnp.linalg.solve` underneath.
  This is what actually proves `_cudss.py`'s state machine is wired correctly, without a
  GPU.
- **GPU-only, skipped everywhere else:** the real `spineax.cudss` module against real
  CUDA, checking the things a fake can't stand in for (real registry/eviction behaviour,
  real dtype support, real gradients).

The solver-agnostic stateful-reuse contract (correctness, reuse, transpose, threading a
state through jit) lives in `test_factorization.py`, the generic solve suite in
`test_solvers.py`, and `AutoSparseLinearSolver`'s GPU dispatch in `test_auto.py`.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO, BCSR

import splineax as splx
import splineax.solvers._cudss as _cudss_module
from splineax import (
    BCOOLinearOperator,
    BCSRLinearOperator,
    CuDSS,
    SparseJacobianLinearOperator,
)
from splineax.solvers import CuDSSReordering
from splineax.solvers._auto import _cuda_backend_available
from splineax.solvers._cudss import _cudss_available, _CuDSSState, _mtype_id

from ...fake_cudss import FakeCuDSS
from .conftest import COMPLEX_MATRIX, RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


def _dense_from_token(state: _CuDSSState) -> jax.Array:
    """Reconstruct the dense matrix from a state's factorized token's CSR arrays."""
    token = state.token
    return BCSR(
        (token.values, token.columns, token.offsets), shape=state.shape
    ).todense()


def _expected(dense: jax.Array, b: jax.Array = RIGHT_HAND_SIDE) -> jax.Array:
    return jnp.linalg.solve(np.asarray(dense), np.asarray(b))


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_cudss_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`CuDSS()` must raise `ImportError` with the `splineax[cudss]` install hint when the
    optional dependency isn't importable, regardless of whether it happens to be installed
    in this environment (the exact inverse trick `test_pardiso.py` uses)."""
    monkeypatch.setattr(_cudss_module, "_cudss_available", lambda: False)
    with pytest.raises(ImportError, match="splineax\\[cudss\\]"):
        CuDSS()


def test_cudss_available_survives_missing_parent_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_cudss_available` must return `False`, not raise, when `spineax` isn't installed
    at all, not just its `cudss` submodule. `importlib.util.find_spec` raises
    `ModuleNotFoundError` for a dotted name whose parent package is missing, which is the
    common case since the binding is an optional dependency.

    Forces that error rather than relying on the package being absent, so the check is the
    same on a machine that really does have the binding installed.
    """

    def raise_module_not_found(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(
        _cudss_module.importlib.util, "find_spec", raise_module_not_found
    )
    assert _cudss_available() is False


def test_ensure_gpu_matches_the_platform() -> None:
    """`_ensure_gpu` rejects every platform but CUDA, and passes values through on CUDA.
    `fake_cudss` disables this guard for the dispatch/reuse tests below, so it needs its
    own unpatched check here, asserting whichever way this machine goes."""
    if _cuda_backend_available() and jax.default_backend() == "gpu":
        assert jnp.allclose(
            jax.block_until_ready(_cudss_module._ensure_gpu(jnp.ones(3))), 1.0
        )
    else:
        with pytest.raises(Exception, match="CUDA GPU"):
            jax.block_until_ready(_cudss_module._ensure_gpu(jnp.ones(3)))


# ---------------------------------------------------------------------------
# mtype selection from tags
# ---------------------------------------------------------------------------


def test_mtype_id_general_for_untagged_operator() -> None:
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    assert _mtype_id(operator) == 0


def test_mtype_id_symmetric() -> None:
    symmetric_matrix = SQUARE_MATRIX + SQUARE_MATRIX.T
    operator = BCOOLinearOperator(BCOO.fromdense(symmetric_matrix), lx.symmetric_tag)
    assert _mtype_id(operator) == 1


def test_mtype_id_spd() -> None:
    spd_matrix = SQUARE_MATRIX @ SQUARE_MATRIX.T + 10.0 * jnp.eye(4)
    operator = BCOOLinearOperator(
        BCOO.fromdense(spd_matrix),
        frozenset({lx.symmetric_tag, lx.positive_semidefinite_tag}),
    )
    assert _mtype_id(operator) == 3


# ---------------------------------------------------------------------------
# Operator -> CSR conversion, and the square/type checks, via `init`
# ---------------------------------------------------------------------------


def test_init_converts_operator_to_csr(
    make_operator: OperatorFactory, fake_cudss: FakeCuDSS
) -> None:
    """`CuDSS.init` reads the operator into the CSR arrays its factorized token carries,
    matching the dense reference, for both `BCOO`- and `BCSR`-backed operators."""
    operator = make_operator(SQUARE_MATRIX)
    state = CuDSS().init(operator, {})
    assert isinstance(state, _CuDSSState)
    assert jnp.allclose(_dense_from_token(state), SQUARE_MATRIX)


def test_init_handles_unsorted_bcsr(fake_cudss: FakeCuDSS) -> None:
    """An unsorted `BCSR` operator round-trips through `BCOO` correctly (the same caveat
    `Pardiso`/`KLU` have to handle)."""
    bcoo = BCOO.fromdense(SQUARE_MATRIX)
    # Reverse the (already coalesced) index order to get an unsorted BCSR.
    unsorted_bcsr = BCSR.from_bcoo(
        BCOO((bcoo.data[::-1], bcoo.indices[::-1]), shape=bcoo.shape)
    )
    operator = BCSRLinearOperator(unsorted_bcsr)
    state = CuDSS().init(operator, {})
    assert jnp.allclose(_dense_from_token(state), SQUARE_MATRIX)


def test_init_materialises_sparse_jacobian(fake_cudss: FakeCuDSS) -> None:
    """A `SparseJacobianLinearOperator` is materialised into the same CSR pattern as the
    equivalent `BCOOLinearOperator`."""

    def fn(x, args):
        del args
        return x * 2.0

    operator = SparseJacobianLinearOperator(
        fn, jnp.arange(4.0), sparsity=BCOO.fromdense(jnp.eye(4))
    )
    state = CuDSS().init(operator, {})
    assert jnp.allclose(_dense_from_token(state), 2.0 * jnp.eye(4))


def test_init_rejects_non_square(fake_cudss: FakeCuDSS) -> None:
    wide = jnp.ones((2, 3))
    operator = BCOOLinearOperator(BCOO.fromdense(wide))
    with pytest.raises(ValueError, match="square"):
        CuDSS().init(operator, {})


def test_init_rejects_unsupported_operator(fake_cudss: FakeCuDSS) -> None:
    operator = lx.MatrixLinearOperator(SQUARE_MATRIX)
    with pytest.raises(TypeError, match="CuDSS"):
        CuDSS().init(operator, {})


# ---------------------------------------------------------------------------
# init / update / release / compute against the fake
# ---------------------------------------------------------------------------


def test_init_analyzes_and_factorizes_once(
    make_operator: OperatorFactory, fake_cudss: FakeCuDSS
) -> None:
    """`init` analyzes and factorizes once; `compute` then only solves, and `release`
    frees the token."""
    operator = make_operator(SQUARE_MATRIX)
    solver = CuDSS()
    state = solver.init(operator, {})
    first = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
    second = solver.compute(state, 2.0 * RIGHT_HAND_SIDE, {})[0]
    state.release()

    assert jnp.allclose(first, _expected(SQUARE_MATRIX))
    assert jnp.allclose(second, _expected(SQUARE_MATRIX, 2.0 * RIGHT_HAND_SIDE))
    assert len(fake_cudss.analyze_calls) == 1
    assert len(fake_cudss.factorize_calls) == 1
    assert len(fake_cudss.solve_calls) == 2
    assert len(fake_cudss.release_calls) == 1
    assert fake_cudss.registry_size() == 0


def test_compute_rejects_symbolic_only_state(
    fake_cudss: FakeCuDSS,
) -> None:
    """A state straight from `init_symbolic` is not solvable: `compute` must raise until
    `update` folds in an operator."""
    state = CuDSS().init_symbolic(BCOO.fromdense(SQUARE_MATRIX))
    with pytest.raises(ValueError, match="symbolic-only"):
        CuDSS().compute(state, RIGHT_HAND_SIDE, {})


def test_update_same_operator_is_a_no_op(
    make_operator: OperatorFactory, fake_cudss: FakeCuDSS
) -> None:
    """`update` with the same operator object returns the state unchanged, running no
    further analyze or factorize."""
    operator = make_operator(SQUARE_MATRIX)
    solver = CuDSS()
    state = solver.init(operator, {})
    updated = solver.update(state, operator)
    assert updated is state
    assert len(fake_cudss.analyze_calls) == 1
    assert len(fake_cudss.factorize_calls) == 1


def test_update_with_a_transposed_pattern_analyzes_again(
    fake_cudss: FakeCuDSS,
) -> None:
    """cuDSS has no transposed solve, so a factorization of `A` cannot serve an operator
    with the transposed pattern. `update` analyzes the new operator again."""
    solver = CuDSS()
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    first = BCOOLinearOperator(sparsity, tags=splx.sparsity_pattern_tag(sparsity))
    state = solver.init(first, {})
    updated = solver.update(state, first.transpose())
    solution = solver.compute(updated, RIGHT_HAND_SIDE, {})[0]
    assert len(fake_cudss.analyze_calls) == 2
    assert jnp.allclose(
        solution,
        jnp.linalg.solve(np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)),
    )


def test_update_reuses_analysis_across_shared_pattern(
    fake_cudss: FakeCuDSS,
) -> None:
    """Two operators sharing a `sparsity_pattern_tag` let `update` reuse the analysis: one
    analyze, a fresh factorize per operator, and correct solves for both.

    This is the threading the stateful API buys on cuDSS: `factorize` renames the token's
    registry entry rather than dropping the analysis, so no re-analysis is needed."""
    solver = CuDSS()
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    first = BCOOLinearOperator(sparsity, tags=tag)
    second_matrix = 2.0 * SQUARE_MATRIX
    second = BCOOLinearOperator(BCOO.fromdense(second_matrix), tags=tag)

    state = solver.init(first, {})
    first_solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
    state = solver.update(state, second)
    second_solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
    state.release()

    assert jnp.allclose(first_solution, _expected(SQUARE_MATRIX))
    assert jnp.allclose(second_solution, _expected(second_matrix))
    assert len(fake_cudss.analyze_calls) == 1, (
        "the analysis must run exactly once, reused by both operators"
    )
    assert len(fake_cudss.factorize_calls) == 2


def test_update_new_pattern_reanalyzes(
    make_operator: OperatorFactory, fake_cudss: FakeCuDSS
) -> None:
    """`update` with an operator that shares no tag re-analyzes from scratch."""
    solver = CuDSS()
    state = solver.init(make_operator(SQUARE_MATRIX), {})
    other = make_operator(SQUARE_MATRIX + SQUARE_MATRIX.T)
    state = solver.update(state, other)
    assert len(fake_cudss.analyze_calls) == 2
    solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
    state.release()
    assert jnp.allclose(solution, _expected(SQUARE_MATRIX + SQUARE_MATRIX.T))


def test_init_symbolic_then_update_reuses_analysis(
    make_operator: OperatorFactory, fake_cudss: FakeCuDSS
) -> None:
    """`init_symbolic` analyzes once; a later `update` folds in an operator sharing the
    pattern and factorizes without re-analyzing."""
    solver = CuDSS()
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    operator = BCOOLinearOperator(sparsity, tags=tag)

    state = solver.init_symbolic(sparsity)
    state = solver.update(state, operator)
    solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
    state.release()

    assert jnp.allclose(solution, _expected(SQUARE_MATRIX))
    assert len(fake_cudss.analyze_calls) == 1
    assert len(fake_cudss.factorize_calls) == 1


def _profiled_update(
    reordering: CuDSSReordering, second_dense: jax.Array
) -> tuple[splx.SolveProfile, jax.Array]:
    """Update a state built on `SQUARE_MATRIX` to `second_dense`, which shares its
    sparsity tag, and solve, all under a solve profile. Returns the profile and the
    solution."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(SQUARE_MATRIX))
    first = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX), tags=tag)
    second = BCOOLinearOperator(BCOO.fromdense(second_dense), tags=tag)
    solver = CuDSS(reordering=reordering)
    profile = splx.create_solve_profile()
    with profile:
        state = solver.init(first, {})
        state = solver.update(state, second)
        solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
        state.release()
    return profile, solution


def _cudss_records(profile: splx.SolveProfile, operation: str) -> list[Any]:
    """The profile's `CuDSS.<operation>` records."""
    return [
        record
        for record in profile.records
        if record.solver == "CuDSS" and record.operation == operation
    ]


def test_update_factorizes_under_the_default_reordering(
    fake_cudss: FakeCuDSS,
) -> None:
    """cuDSS only refactorizes under the COLAMD reorderings, so under the default one
    `update` factorizes again, keeping the analysis."""
    profile, solution = _profiled_update(CuDSSReordering.DEFAULT, 2.0 * SQUARE_MATRIX)
    assert not fake_cudss.refactorize_calls
    assert len(fake_cudss.analyze_calls) == 1
    factorizations = _cudss_records(profile, "factorize")
    assert [record.outputs.get("reason") for record in factorizations] == [
        None,
        "Reused analysis",
    ]
    assert jnp.allclose(solution, _expected(2.0 * SQUARE_MATRIX))


@pytest.mark.parametrize(
    "reordering", [CuDSSReordering.COLAMD, CuDSSReordering.BTF_COLAMD]
)
def test_update_refactorizes_under_colamd(
    fake_cudss: FakeCuDSS, reordering: CuDSSReordering
) -> None:
    """Under the COLAMD reorderings `update` refactorizes with the previous pivots, and
    keeps the result when those pivots stay well scaled for the new values."""
    profile, solution = _profiled_update(reordering, 2.0 * SQUARE_MATRIX)
    assert len(fake_cudss.refactorize_calls) == 1
    (refactorization,) = _cudss_records(profile, "refactorize")
    assert refactorization.outputs["reused"] is True
    assert refactorization.outputs["rcond"] > 1e-8
    assert "stable" in refactorization.outputs["reason"]
    assert jnp.allclose(solution, _expected(2.0 * SQUARE_MATRIX))


@pytest.mark.parametrize(
    "reordering", [CuDSSReordering.COLAMD, CuDSSReordering.BTF_COLAMD]
)
def test_update_falls_back_when_reused_pivots_go_bad(
    fake_cudss: FakeCuDSS, reordering: CuDSSReordering
) -> None:
    """When the new values leave the reused pivots badly scaled, `update` factorizes
    fresh instead. The second matrix shrinks the diagonal entry the first factorization
    pivoted on, the same case `KLU`'s fallback test uses."""
    second_dense = SQUARE_MATRIX.at[0, 0].set(1e-9)
    profile, solution = _profiled_update(reordering, second_dense)
    (refactorization,) = _cudss_records(profile, "refactorize")
    assert refactorization.outputs["reused"] is False
    assert refactorization.outputs["rcond"] <= 1e-8
    assert "unstable" in refactorization.outputs["reason"]
    assert jnp.allclose(solution, _expected(second_dense))


# ---------------------------------------------------------------------------
# track / release / profile against the fake
# ---------------------------------------------------------------------------


def _shared_pattern_operators() -> tuple[BCOOLinearOperator, BCOOLinearOperator]:
    """Two operators sharing one sparsity tag, the second with doubled values."""
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    first = BCOOLinearOperator(sparsity, tags=tag)
    second = BCOOLinearOperator(BCOO.fromdense(2.0 * SQUARE_MATRIX), tags=tag)
    return first, second


def test_track_entangles_token_with_solution(fake_cudss: FakeCuDSS) -> None:
    """`track` makes the token depend on the solution, so a later `factorize` that renames
    the registry entry is ordered after the solve under jit."""
    first, _ = _shared_pattern_operators()
    state = CuDSS().init(first, {})

    def tracked_id(solution: jax.Array) -> jax.Array:
        return state.track(solution).token.id

    jaxpr = jax.make_jaxpr(tracked_id)(RIGHT_HAND_SIDE)
    assert "entangle" in str(jaxpr)
    state.release()


def test_linear_solve_threads_a_tracked_state(fake_cudss: FakeCuDSS) -> None:
    """`splineax.linear_solve` tracks the state after each solve, and the tracked state
    still reuses the analysis for an operator sharing the pattern."""
    first, second = _shared_pattern_operators()
    solver = CuDSS()

    solution, state = splx.linear_solve(first, RIGHT_HAND_SIDE, solver)
    assert isinstance(state, _CuDSSState)
    solution_second, state = splx.linear_solve(
        second, RIGHT_HAND_SIDE, solver, state=state
    )
    state.release()

    assert jnp.allclose(solution.value, _expected(SQUARE_MATRIX))
    assert jnp.allclose(solution_second.value, _expected(2.0 * SQUARE_MATRIX))
    assert len(fake_cudss.analyze_calls) == 1
    assert len(fake_cudss.factorize_calls) == 2


def test_release_under_jit_is_skipped(
    make_operator: OperatorFactory, fake_cudss: FakeCuDSS
) -> None:
    """A traced `release` cannot reach the eager-only `spineax.cudss.release`, so it is
    skipped and the cache keeps the entry until eviction."""
    state = CuDSS().init(make_operator(SQUARE_MATRIX), {})
    eqx.filter_jit(lambda traced_state: traced_state.release())(state)
    assert not fake_cudss.release_calls
    state.release()
    assert len(fake_cudss.release_calls) == 1


def test_profile_records_cudss_operations(fake_cudss: FakeCuDSS) -> None:
    """A solve profile records the analyze, the factorize per operator, the solves, and
    the release, with the `update` marked as reusing the analysis."""
    first, second = _shared_pattern_operators()
    solver = CuDSS()

    profile = splx.create_solve_profile()
    with profile:
        _, state = splx.linear_solve(first, RIGHT_HAND_SIDE, solver)
        _, state = splx.linear_solve(second, RIGHT_HAND_SIDE, solver, state=state)
        state.release()

    operations = [
        record.operation for record in profile.records if record.solver == "CuDSS"
    ]
    assert operations == [
        "analyze",
        "factorize",
        "solve",
        "factorize",
        "solve",
        "release",
    ]
    update = next(record for record in profile.records if record.operation == "update")
    assert update.outputs["outcome"] == "reused"


# ---------------------------------------------------------------------------
# transpose / conj against the fake
# ---------------------------------------------------------------------------


def test_transpose_symmetric_reuses_factorization(fake_cudss: FakeCuDSS) -> None:
    """For a symmetric matrix, `transpose` reuses the token unchanged: no extra
    analyze/factorize calls."""
    symmetric_matrix = SQUARE_MATRIX + SQUARE_MATRIX.T
    operator = BCOOLinearOperator(BCOO.fromdense(symmetric_matrix), lx.symmetric_tag)
    solver = CuDSS()

    state = solver.init(operator, {})
    analyze_before = len(fake_cudss.analyze_calls)
    factorize_before = len(fake_cudss.factorize_calls)
    transposed_state, _ = solver.transpose(state, {})
    assert transposed_state.token is state.token
    solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0]
    state.release()

    assert len(fake_cudss.analyze_calls) == analyze_before
    assert len(fake_cudss.factorize_calls) == factorize_before
    assert jnp.allclose(solution, _expected(symmetric_matrix.T))


def test_transpose_general_refactorizes(fake_cudss: FakeCuDSS) -> None:
    """For a general (untagged) matrix, `transpose` builds and factorizes a genuinely
    transposed token: cuDSS has no native transpose solve."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = CuDSS()

    state = solver.init(operator, {})
    analyze_before = len(fake_cudss.analyze_calls)
    transposed_state, _ = solver.transpose(state, {})
    assert transposed_state.token is not state.token
    solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0]
    state.release()
    transposed_state.release()

    assert len(fake_cudss.analyze_calls) == analyze_before + 1
    assert jnp.allclose(solution, _expected(SQUARE_MATRIX.T))


def test_conj_real_is_noop(fake_cudss: FakeCuDSS) -> None:
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = CuDSS()
    state = solver.init(operator, {})
    conj_state, _ = solver.conj(state, {})
    assert conj_state is state


def test_conj_complex_refactorizes(fake_cudss: FakeCuDSS) -> None:
    """For a complex matrix, `conj` reuses the pivots via `refactorize` (same magnitudes,
    so the existing pivoting stays valid) and solves correctly."""
    operator = BCOOLinearOperator(BCOO.fromdense(COMPLEX_MATRIX))
    solver = CuDSS()
    b = jnp.array([1.0 + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j, 2.0j])

    state = solver.init(operator, {})
    conj_state, _ = solver.conj(state, {})
    assert conj_state.token is not state.token
    solution = solver.compute(conj_state, b, {})[0]
    state.release()

    assert fake_cudss.refactorize_calls, "conj on a complex state should refactorize"
    assert jnp.allclose(solution, _expected(np.asarray(COMPLEX_MATRIX).conj(), b))


# ---------------------------------------------------------------------------
# GPU-only: the real `spineax.cudss` module against real CUDA hardware.
# ---------------------------------------------------------------------------


@pytest.mark.cudss_gpu
def test_gpu_state_reuses_factorization_without_rebuilds(
    make_operator: OperatorFactory,
) -> None:
    """The reuse claim that holds, checked against cuDSS's own counter: one `factorize`,
    many right-hand sides, zero rebuilds. `solve` does not consume its token's id (only
    the numeric phases do), so the factorization stays resident for every solve."""
    from spineax import cudss as spineax_cudss

    solver = CuDSS()
    operator = make_operator(SQUARE_MATRIX)
    rebuilds_before = spineax_cudss.rebuild_count()

    state = solver.init(operator, {})
    for scale in [1.0, 2.0, 0.5, 3.0]:
        b = scale * RIGHT_HAND_SIDE
        solution = solver.compute(state, b, {})[0]
        expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(b))
        assert jnp.allclose(solution, expected, atol=1e-5)
    state.release()

    assert spineax_cudss.rebuild_count() == rebuilds_before, (
        "solving repeatedly against one factorized token must not rebuild it"
    )


@pytest.mark.cudss_gpu
def test_gpu_threading_a_state_avoids_rebuilds(
    make_operator: OperatorFactory,
) -> None:
    """Threading the state through `update` for several matrices sharing one pattern costs
    no rebuilds at all.

    A numeric phase renames its registry entry (`BatchTokenRegistry::rekey` in spineax's
    solver.cpp) rather than destroying it, so the analysis survives and only the old id
    stops resolving. This is the premise the whole stateful reuse API rests on for cuDSS,
    so it is worth pinning down on real hardware.
    """
    from spineax import cudss as spineax_cudss

    solver = CuDSS()
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    scales = [2.0, 0.5, 3.0, 1.5]
    rebuilds_before = spineax_cudss.rebuild_count()

    state = solver.init(BCOOLinearOperator(sparsity, tags=tag), {})
    solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
    assert jnp.allclose(solution, _expected(SQUARE_MATRIX), atol=1e-5)
    for scale in scales:
        operator = BCOOLinearOperator(BCOO.fromdense(scale * SQUARE_MATRIX), tags=tag)
        state = solver.update(state, operator)
        solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
        assert jnp.allclose(solution, _expected(scale * SQUARE_MATRIX), atol=1e-5)
    state.release()

    rebuilds = spineax_cudss.rebuild_count() - rebuilds_before
    assert rebuilds == 0, (
        f"threading the state should never rebuild the analysis, got {rebuilds}"
    )


@pytest.mark.cudss_gpu
def test_gpu_linear_solve_from_symbolic_state_keeps_one_registry_entry() -> None:
    """The reuse example from the GPU notebook. A state from `init_symbolic`, threaded
    through `splineax.linear_solve` for several matrices sharing one pattern, never
    rebuilds and holds a single registry entry, which `release` then retires.

    Unlike the test above, this goes through `splineax.linear_solve`, so every solve also
    runs `track`, and the state starts from an analysis with no values."""
    from spineax import cudss as spineax_cudss

    solver = CuDSS()
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    tag = splx.sparsity_pattern_tag(sparsity)
    registry_before = spineax_cudss.registry_size()
    rebuilds_before = spineax_cudss.rebuild_count()

    state = solver.init_symbolic(sparsity)
    for scale in [1.0, 2.0, 0.5, 3.0, 1.5]:
        operator = BCOOLinearOperator(BCOO.fromdense(scale * SQUARE_MATRIX), tags=tag)
        solution, state = splx.linear_solve(
            operator, RIGHT_HAND_SIDE, solver, state=state
        )
        assert jnp.allclose(solution.value, _expected(scale * SQUARE_MATRIX), atol=1e-5)
    registry_while_live = spineax_cudss.registry_size()
    state.release()

    assert registry_while_live == registry_before + 1, (
        "one analysis renamed through every factorize should be one registry entry"
    )
    assert spineax_cudss.registry_size() == registry_before, (
        "releasing the state should retire its token"
    )
    rebuilds = spineax_cudss.rebuild_count() - rebuilds_before
    assert rebuilds == 0, f"the analysis was evicted and rebuilt {rebuilds} times"


@pytest.mark.cudss_gpu
@pytest.mark.parametrize(
    "reordering", [CuDSSReordering.COLAMD, CuDSSReordering.BTF_COLAMD]
)
def test_gpu_colamd_update_refactorizes_and_falls_back(
    reordering: CuDSSReordering,
) -> None:
    """Under the COLAMD reorderings, real cuDSS refactorizes when the reused pivots stay
    well scaled, and factorizes fresh when new values shrink the entry the first
    factorization pivoted on. Every solve stays correct and nothing is rebuilt."""
    from spineax import cudss as spineax_cudss

    rebuilds_before = spineax_cudss.rebuild_count()
    profile, stable_solution = _profiled_update(reordering, 2.0 * SQUARE_MATRIX)
    (refactorization,) = _cudss_records(profile, "refactorize")
    assert refactorization.outputs["reused"] is True
    assert jnp.allclose(stable_solution, _expected(2.0 * SQUARE_MATRIX), atol=1e-5)

    shrunk_pivot = SQUARE_MATRIX.at[0, 0].set(1e-9)
    profile, fallback_solution = _profiled_update(reordering, shrunk_pivot)
    (refactorization,) = _cudss_records(profile, "refactorize")
    assert refactorization.outputs["reused"] is False
    assert jnp.allclose(fallback_solution, _expected(shrunk_pivot), atol=1e-5)

    rebuilds = spineax_cudss.rebuild_count() - rebuilds_before
    assert rebuilds == 0, f"a factorization was evicted and rebuilt {rebuilds} times"


@pytest.mark.cudss_gpu
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex128])
def test_gpu_solves_in_every_supported_dtype(
    make_operator: OperatorFactory, dtype, enable_x64: None
) -> None:
    """cuDSS supports f32/f64/complex directly, with no upcasting, unlike `Pardiso`.

    Needs `enable_x64`, without which JAX silently truncates the 64-bit cases back to
    32-bit and the test would pass while proving nothing.
    """
    matrix = SQUARE_MATRIX.astype(dtype)
    b = RIGHT_HAND_SIDE.astype(dtype)
    operator = make_operator(matrix)
    solver = CuDSS()

    state = solver.init(operator, {})
    solution = solver.compute(state, b, {})[0]
    state.release()

    assert solution.dtype == dtype
    expected = jnp.linalg.solve(np.asarray(matrix), np.asarray(b))
    assert jnp.allclose(solution, expected, atol=1e-4)


@pytest.mark.cudss_gpu
def test_gpu_general_transpose_solves_correctly(make_operator: OperatorFactory) -> None:
    operator = make_operator(SQUARE_MATRIX)
    solver = CuDSS()
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )

    state = solver.init(operator, {})
    transposed_state, _ = solver.transpose(state, {})
    solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0]
    state.release()
    transposed_state.release()

    assert jnp.allclose(solution, expected, atol=1e-5)


@pytest.mark.cudss_gpu
def test_gpu_gradients_match_dense_reference(make_operator: OperatorFactory) -> None:
    dense = np.asarray(SQUARE_MATRIX)
    b = np.asarray(RIGHT_HAND_SIDE)

    def solve_with_cudss(values):
        operator = make_operator(SQUARE_MATRIX.at[SQUARE_MATRIX != 0].set(values))
        return lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=CuDSS()).value.sum()

    def solve_dense(values):
        matrix = jnp.asarray(dense).at[jnp.asarray(dense) != 0].set(values)
        return jnp.linalg.solve(matrix, jnp.asarray(b)).sum()

    nonzero_values = SQUARE_MATRIX[SQUARE_MATRIX != 0]
    grad_cudss = jax.grad(solve_with_cudss)(nonzero_values)
    grad_dense = jax.grad(solve_dense)(nonzero_values)
    assert jnp.allclose(grad_cudss, grad_dense, atol=1e-4)
