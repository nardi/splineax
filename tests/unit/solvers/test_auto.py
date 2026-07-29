"""Tests for `AutoSparseLinearSolver` dispatch and the sparse-solver Protocols.

`AutoSparseLinearSolver` selects `Pardiso` (if the optional `pardiso-mkl-jax`
dependency is installed) or otherwise `KLU` on CPU when x64 is enabled, since both are
double precision only, and `Spsolve` otherwise (CPU without x64, or any other
platform). It exposes the same factorization API as `Pardiso`/`KLU` so it can be
substituted verbatim. The generic solve-correctness suite (parametrised over all
solvers) lives in `test_solvers.py`, and the shared factorization-reuse contract in
`test_factorization.py`. This module covers Auto-specific dispatch and Protocol
conformance.

The dispatch tests monkeypatch `splineax.solvers._auto._pardiso_available` rather than
relying on whether `pardiso-mkl-jax` actually happens to be installed, so both branches
are exercised deterministically regardless of the test environment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import splineax.solvers._auto as _auto_module
import splineax.solvers._cudss as _cudss_module
import splineax.solvers._pardiso as _pardiso_module
from splineax import (
    KLU,
    AutoSparseLinearSolver,
    CuDSS,
    Pardiso,
    Spsolve,
)
from splineax.solvers import (
    SparseBasicState,
    SparseLinearSolver,
    SparseSymbolicScope,
    SparseSymbolicState,
)
from splineax.solvers._cudss import _cudss_available
from splineax.solvers._klu import _KLUBasicState, _KLUNumericState, _KLUSymbolicState
from splineax.solvers._pardiso import _pardiso_available

from .conftest import (
    RIGHT_HAND_SIDE,
    SQUARE_MATRIX,
    OperatorFactory,
    requires_cpu_backend,
)

KLU_STATE_TYPES = (_KLUBasicState, _KLUSymbolicState, _KLUNumericState)


@pytest.fixture
def pardiso_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `Pardiso` look installed to both the places that ask.

    `_auto.py`'s copy gates the dispatch branch, and `_pardiso.py`'s gates
    `Pardiso.__init__`. Patching only the first leaves `_chosen_solver` picking
    `Pardiso` and then failing to construct it, so these dispatch tests are only
    environment-independent (the point of monkeypatching at all) with both.
    """
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    monkeypatch.setattr(_pardiso_module, "_pardiso_available", lambda: True)


# ---------------------------------------------------------------------------
# Solver selection
# ---------------------------------------------------------------------------


@requires_cpu_backend
def test_select_solver_prefers_pardiso_on_cpu_with_x64(
    make_operator: OperatorFactory, pardiso_installed: None
) -> None:
    """With no override, `AutoSparseLinearSolver` selects `Pardiso` on the CPU test
    platform when x64 is enabled and `pardiso-mkl-jax` is installed."""
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), Pardiso)


@requires_cpu_backend
def test_select_solver_falls_back_to_klu_when_pardiso_unavailable(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When `pardiso-mkl-jax` isn't installed, `AutoSparseLinearSolver` falls back to
    `KLU` on the CPU test platform when x64 is enabled."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: False)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), KLU)


@requires_cpu_backend
def test_select_solver_falls_back_to_spsolve_on_cpu_without_x64(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On CPU with x64 disabled, `AutoSparseLinearSolver` falls back to `Spsolve`, since
    both `Pardiso` and `KLU` are double precision only."""
    monkeypatch.setattr(_auto_module, "_pardiso_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(False):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), Spsolve)


def test_select_solver_platform_override(
    make_operator: OperatorFactory,
    monkeypatch: pytest.MonkeyPatch,
    pardiso_installed: None,
) -> None:
    """An explicit `platform` override forces the corresponding solver, without running
    a solve (so neither a real GPU nor a real CPU backend is required here).

    cuDSS is forced unavailable so the "gpu" branch means `Spsolve` even when this
    really is a cuDSS-capable GPU machine. The cuDSS branch has its own tests below.
    """
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: False)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(
            AutoSparseLinearSolver(platform="cpu").select_solver(operator), Pardiso
        )
        assert isinstance(
            AutoSparseLinearSolver(platform="gpu").select_solver(operator), Spsolve
        )
    with jax.enable_x64(False):
        assert isinstance(
            AutoSparseLinearSolver(platform="cpu").select_solver(operator), Spsolve
        )
        assert isinstance(
            AutoSparseLinearSolver(platform="gpu").select_solver(operator), Spsolve
        )


def test_select_solver_prefers_cudss_on_gpu(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With cuDSS installed and a CUDA device visible, `AutoSparseLinearSolver` selects
    `CuDSS` on the GPU platform. No x64 requirement, unlike `Pardiso`/`KLU`.

    Patches the availability check in both `_auto.py` (which gates the dispatch
    branch) and `_cudss.py` (which `CuDSS.__init__` itself checks), since `Pardiso`'s
    equivalent tests get away with patching only the former by relying on
    `pardiso-mkl-jax` actually being installed in the test environment. cuDSS's real
    dependency is never installed here (it needs a CUDA GPU), so both must be patched
    for construction to succeed.
    """
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: True)
    monkeypatch.setattr(_cudss_module, "_cudss_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(False):
        assert isinstance(
            AutoSparseLinearSolver(platform="gpu").select_solver(operator), CuDSS
        )


def test_select_solver_falls_back_to_spsolve_when_cudss_unavailable(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On GPU with a CUDA device visible but the optional cuDSS dependency not
    installed, `AutoSparseLinearSolver` falls back to `Spsolve`."""
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: False)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    assert isinstance(
        AutoSparseLinearSolver(platform="gpu").select_solver(operator), Spsolve
    )


def test_select_solver_falls_back_to_spsolve_on_rocm(
    make_operator: OperatorFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ROCm GPU also reports platform "gpu", but has no CUDA device: even with
    cuDSS installed, `AutoSparseLinearSolver` must not select `CuDSS`, since `spineax`
    only registers its FFI targets on the CUDA platform."""
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: False)
    operator = make_operator(SQUARE_MATRIX)
    assert isinstance(
        AutoSparseLinearSolver(platform="gpu").select_solver(operator), Spsolve
    )


@requires_cpu_backend
def test_select_solver_cpu_unaffected_by_cudss_availability(
    make_operator: OperatorFactory,
    monkeypatch: pytest.MonkeyPatch,
    pardiso_installed: None,
) -> None:
    """`CuDSS` availability must not change CPU dispatch: `Pardiso`/`KLU` are still
    chosen on CPU with x64 enabled, regardless of what cuDSS reports."""
    monkeypatch.setattr(_auto_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_auto_module, "_cuda_backend_available", lambda: True)
    operator = make_operator(SQUARE_MATRIX)
    with jax.enable_x64(True):
        assert isinstance(AutoSparseLinearSolver().select_solver(operator), Pardiso)


# ---------------------------------------------------------------------------
# Solve correctness through Auto
# ---------------------------------------------------------------------------


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


@requires_cpu_backend
def test_auto_falls_back_to_klu_for_complex_when_pardiso_chosen(
    make_operator: OperatorFactory, pardiso_installed: None
) -> None:
    """`pardiso_mkl_jax` doesn't support complex matrices, so `init`/`factorize` must
    fall back to `KLU` for a complex operator even when `Pardiso` was otherwise
    selected, keeping `Auto` able to solve anything `KLU` can."""

    with jax.enable_x64(True):
        # Built inside the block: `.astype(jnp.complex128)` outside it would silently
        # truncate to complex64, since x64 isn't enabled yet at that point.
        complex_matrix = SQUARE_MATRIX.astype(jnp.complex128) * (1 + 1j)
        right_hand_side = RIGHT_HAND_SIDE.astype(jnp.complex128)
        operator = make_operator(complex_matrix)
        expected = jnp.linalg.solve(
            np.asarray(complex_matrix), np.asarray(right_hand_side)
        )

        solver = AutoSparseLinearSolver()
        assert isinstance(solver.select_solver(operator), Pardiso)

        state = solver.init(operator, {})
        assert isinstance(state, KLU_STATE_TYPES)
        solution = lx.linear_solve(
            operator, right_hand_side, solver=solver, state=state
        ).value
        assert jnp.allclose(solution, expected, atol=1e-5)

        with solver.factorize(operator) as numeric_state:
            assert isinstance(numeric_state, KLU_STATE_TYPES)
            factorized_solution = lx.linear_solve(
                operator, right_hand_side, solver=solver, state=numeric_state
            ).value
        assert jnp.allclose(factorized_solution, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_solvers_satisfy_sparse_linear_solver_protocol() -> None:
    """All solvers implement the `SparseLinearSolver` Protocol."""
    assert isinstance(KLU(), SparseLinearSolver)
    assert isinstance(Spsolve(), SparseLinearSolver)
    assert isinstance(AutoSparseLinearSolver(), SparseLinearSolver)
    if _pardiso_available():
        assert isinstance(Pardiso(), SparseLinearSolver)
    if _cudss_available():
        assert isinstance(CuDSS(), SparseLinearSolver)


@requires_cpu_backend
def test_states_and_scopes_satisfy_protocols(
    make_operator: OperatorFactory, enable_x64: None
) -> None:
    """Init states, symbolic scopes, and symbolic states satisfy their Protocols, for
    KLU, Spsolve, and (if installed) Pardiso."""
    operator = make_operator(SQUARE_MATRIX)
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    solvers = [KLU(), Spsolve()]
    if _pardiso_available():
        solvers.append(Pardiso())

    for solver in solvers:
        assert isinstance(solver.init(operator, {}), SparseBasicState)
        with solver.factorize_symbolic(sparsity) as scope:
            assert isinstance(scope, SparseSymbolicScope)
            assert isinstance(scope.init(operator), SparseSymbolicState)
