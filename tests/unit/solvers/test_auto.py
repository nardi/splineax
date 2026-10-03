"""Tests for `AutoSparseLinearSolver` dispatch and the sparse-solver Protocol.

`AutoSparseLinearSolver` selects `Pardiso` (if the optional `pardiso-mkl-jax` dependency
is installed) or otherwise `KLU` on CPU when x64 is enabled, since both are double
precision only. On a CUDA GPU it selects `CuDSS` if the optional cuDSS dependency is
installed, and `Spsolve` otherwise. It exposes the same stateful API as
`Pardiso`/`KLU`/`CuDSS` so it can be substituted verbatim. The generic solve suite lives
in test_solvers.py and the shared reuse contract in test_factorization.py. This module
covers Auto-specific dispatch and Protocol conformance.

The dispatch tests monkeypatch the availability checks in `splineax.solvers._auto`
rather than relying on what is installed, so every branch is exercised
deterministically.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import splineax as splx
import splineax.solvers._auto as _auto_module
import splineax.solvers._cudss as _cudss_module
import splineax.solvers._pardiso as _pardiso_module
from splineax import (
    KLU,
    AutoSparseLinearSolver,
    CuDSS,
    HybridDirectIterative,
    IterativeRefinementSettings,
    Pardiso,
    Spsolve,
)
from splineax.solvers import SparseLinearSolver
from splineax.solvers._auto import _AutoDispatch, _cuda_backend_available
from splineax.solvers._cudss import _cudss_available
from splineax.solvers._iterative import HybridState
from splineax.solvers._klu import _KLUState
from splineax.solvers._pardiso import _pardiso_available

from .conftest import RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory


@pytest.fixture
def pardiso_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `Pardiso` look installed to both the places that ask.

    `_auto.py`'s copy gates the dispatch branch, and `_pardiso.py`'s gates
    `Pardiso.__init__`. Patching only the first leaves `_chosen_solver` picking `Pardiso`
    and then failing to construct it, so these dispatch tests are only
    environment-independent (the point of monkeypatching at all) with both.
    """
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    monkeypatch.setattr(_pardiso_module, "_pardiso_available", lambda: True)


@pytest.mark.cpu_only
def test_dispatch_prefers_pardiso_on_cpu_with_x64(
    make_operator: OperatorFactory, pardiso_installed: None
) -> None:
    """With no override, the platform dispatch selects `Pardiso` on CPU when x64 is
    enabled and `pardiso-mkl-jax` is installed."""
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(_AutoDispatch().select_solver(operator), Pardiso)


@pytest.mark.cpu_only
def test_dispatch_falls_back_to_klu_when_pardiso_unavailable(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `pardiso-mkl-jax` is not installed, the dispatch falls back to `KLU` on CPU
    when x64 is enabled."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: False)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(_AutoDispatch().select_solver(operator), KLU)


@pytest.mark.cpu_only
def test_dispatch_falls_back_to_spsolve_on_cpu_without_x64(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On CPU with x64 disabled, the dispatch falls back to `Spsolve`, since both
    `Pardiso` and `KLU` are double precision only."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(False):
        assert isinstance(_AutoDispatch().select_solver(operator), Spsolve)


def test_dispatch_platform_override(
    make_operator: OperatorFactory, pardiso_installed: None
) -> None:
    """An explicit `platform` override forces the corresponding direct solver, without a
    solve (so neither a real GPU nor a real CPU backend is required here).

    The "gpu" branch is `CuDSS` on a machine that can actually run it and `Spsolve`
    everywhere else, so the expectation comes from the same two predicates
    `_chosen_solver` consults rather than being hard-coded either way. Note it does not
    depend on x64, unlike the "cpu" branch.
    """
    operator = make_operator(SQUARE_MATRIX)
    gpu_expected = (
        CuDSS if _cudss_available() and _cuda_backend_available() else Spsolve
    )
    with jax.enable_x64(True):
        assert isinstance(
            _AutoDispatch(platform="cpu").select_solver(operator), Pardiso
        )
        assert isinstance(
            _AutoDispatch(platform="gpu").select_solver(operator), gpu_expected
        )
    with jax.enable_x64(False):
        assert isinstance(
            _AutoDispatch(platform="cpu").select_solver(operator), Spsolve
        )
        assert isinstance(
            _AutoDispatch(platform="gpu").select_solver(operator), gpu_expected
        )


def test_dispatch_prefers_cudss_on_gpu(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With cuDSS installed and a CUDA device visible, the dispatch selects `CuDSS` on
    the GPU platform. No x64 requirement, unlike `Pardiso`/`KLU`.

    Patches the availability check in both `_auto.py` (which gates the dispatch branch)
    and `_cudss.py` (which `CuDSS.__init__` itself checks). cuDSS's real dependency is
    not installed on a CPU test machine, so both must be patched for construction to
    succeed.
    """
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: True)
    monkeypatch.setattr(_cudss_module, "_cudss_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(False):
        assert isinstance(_AutoDispatch(platform="gpu").select_solver(operator), CuDSS)


def test_dispatch_falls_back_to_spsolve_when_cudss_unavailable(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On GPU with a CUDA device visible but the optional cuDSS dependency not installed,
    the dispatch falls back to `Spsolve`."""
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: False)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    assert isinstance(_AutoDispatch(platform="gpu").select_solver(operator), Spsolve)


def test_dispatch_falls_back_to_spsolve_on_rocm(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ROCm GPU also reports platform "gpu", but has no CUDA device. Even with cuDSS
    installed, the dispatch must not select `CuDSS`, since `spineax` only registers its
    FFI targets on the CUDA platform."""
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: False)
    operator = make_operator(SQUARE_MATRIX)
    assert isinstance(_AutoDispatch(platform="gpu").select_solver(operator), Spsolve)


@pytest.mark.cpu_only
def test_dispatch_cpu_unaffected_by_cudss_availability(
    make_operator: OperatorFactory,
    monkeypatch: pytest.MonkeyPatch,
    pardiso_installed: None,
) -> None:
    """`CuDSS` availability must not change CPU dispatch. `Pardiso`/`KLU` are still chosen
    on CPU with x64 enabled, regardless of what cuDSS reports."""
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(_AutoDispatch().select_solver(operator), Pardiso)


def test_select_solver_returns_exact_solver_with_refinement(
    make_operator: OperatorFactory, pardiso_installed: None
) -> None:
    """`AutoSparseLinearSolver.select_solver` returns the exact solver it runs: an
    `HybridDirectIterative` wrapping the chosen direct solver by default, and the direct
    dispatch itself when refinement is off."""
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        refined = AutoSparseLinearSolver().select_solver(operator)
        assert isinstance(refined, HybridDirectIterative)
        assert isinstance(refined.direct, _AutoDispatch)

        plain = AutoSparseLinearSolver(iterative=False).select_solver(operator)
        assert isinstance(plain, _AutoDispatch)


def test_auto_solve_matches_numpy(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`AutoSparseLinearSolver` produces the same solution as `numpy.linalg.solve`."""
    operator = make_operator(SQUARE_MATRIX)
    solution = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=AutoSparseLinearSolver()
    ).value
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    assert jnp.allclose(solution, expected, atol=1e-5)


def test_auto_stateful_api_solves_and_releases(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """`AutoSparseLinearSolver` routes the whole stateful API (`init_symbolic`, `update`,
    `release`, and `splineax.linear_solve`'s tuple return) through the chosen solver."""
    operator = make_operator(SQUARE_MATRIX)
    solver = AutoSparseLinearSolver()
    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))

    solution, state = splx.linear_solve(operator, RIGHT_HAND_SIDE, solver)
    assert jnp.allclose(solution.value, expected, atol=1e-5)
    state.release()

    symbolic_state = solver.update(
        solver.init_symbolic(BCOO.fromdense(SQUARE_MATRIX)), operator
    )
    reused = lx.linear_solve(
        operator, RIGHT_HAND_SIDE, solver=solver, state=symbolic_state
    ).value
    symbolic_state.release()
    assert jnp.allclose(reused, expected, atol=1e-5)


@pytest.mark.cpu_only
def test_auto_falls_back_to_klu_for_complex_when_pardiso_chosen(
    make_operator: OperatorFactory, pardiso_installed: None
) -> None:
    """`pardiso_mkl_jax` does not support complex matrices, so `init` falls back to `KLU`
    for a complex operator even when `Pardiso` was otherwise selected, and every later
    call on that state keeps using `KLU`."""
    with jax.enable_x64(True):
        # Built inside the block: `.astype(jnp.complex128)` outside it would truncate to
        # complex64, since x64 is not enabled yet at that point.
        complex_matrix = SQUARE_MATRIX.astype(jnp.complex128) * (1 + 1j)
        right_hand_side = RIGHT_HAND_SIDE.astype(jnp.complex128)
        operator = make_operator(complex_matrix)
        expected = jnp.linalg.solve(
            np.asarray(complex_matrix), np.asarray(right_hand_side)
        )

        # Disable refinement so the state is the chosen direct solver's own, which this
        # test inspects to confirm the complex fallback landed on `KLU`.
        solver = AutoSparseLinearSolver(iterative=False)
        assert isinstance(_AutoDispatch().select_solver(operator), Pardiso)

        state = solver.init(operator, {})
        assert isinstance(state, _KLUState)
        solution = lx.linear_solve(
            operator, right_hand_side, solver=solver, state=state
        ).value
        assert jnp.allclose(solution, expected, atol=1e-5)

        updated = solver.update(state, operator)
        assert isinstance(updated, _KLUState)
        reused = lx.linear_solve(
            operator, right_hand_side, solver=solver, state=updated
        ).value
        updated.release()
        assert jnp.allclose(reused, expected, atol=1e-5)


def test_auto_applies_iterative_refinement_by_default(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """By default `AutoSparseLinearSolver` combines its chosen solver with iterative
    refinement, so its state is a `HybridState`. Disabling it returns the chosen solver's
    own state instead."""
    operator = make_operator(SQUARE_MATRIX)

    refined = AutoSparseLinearSolver().init(operator, {})
    assert isinstance(refined, HybridState)

    plain = AutoSparseLinearSolver(iterative=False).init(operator, {})
    assert not isinstance(plain, HybridState)


def test_auto_refinement_settings_are_forwarded(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """An `IterativeRefinementSettings` on `AutoSparseLinearSolver` reaches the wrapping
    `IterativeRefinement`, and the configured solve is still correct."""
    operator = make_operator(SQUARE_MATRIX)
    settings = IterativeRefinementSettings(tol=1e-8, max_steps=3)
    solver = AutoSparseLinearSolver(iterative=settings)
    wrapper = solver.select_solver(operator)
    assert isinstance(wrapper, HybridDirectIterative)
    assert wrapper.tol == 1e-8
    assert wrapper.iterative.max_steps == 3

    expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(RIGHT_HAND_SIDE))
    solution = lx.linear_solve(operator, RIGHT_HAND_SIDE, solver=solver).value
    assert jnp.allclose(solution, expected, atol=1e-6)


def test_solvers_satisfy_sparse_linear_solver_protocol() -> None:
    """All solvers structurally satisfy the `SparseLinearSolver` Protocol."""
    assert isinstance(KLU(), SparseLinearSolver)
    assert isinstance(Spsolve(), SparseLinearSolver)
    assert isinstance(AutoSparseLinearSolver(), SparseLinearSolver)
    assert isinstance(HybridDirectIterative(KLU()), SparseLinearSolver)
    if _pardiso_available():
        assert isinstance(Pardiso(), SparseLinearSolver)
    if _cudss_available():
        assert isinstance(CuDSS(), SparseLinearSolver)
