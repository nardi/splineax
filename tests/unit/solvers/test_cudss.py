"""CuDSS-specific tests for availability, mtype selection, conversion, and (via a fake
`spineax.cudss` module) the dispatch/reuse contract, plus real-GPU tests.

`CuDSS()` requires the optional cuDSS dependency, which needs a CUDA GPU to even import
(it dlopens a CUDA extension), so it can never be genuinely installed in ordinary CPU CI.
Two tiers of coverage follow from that:

- **CPU-runnable, always on:** availability/`ImportError`, tag-to-mtype selection,
  operator-to-CSR conversion, and the square/type checks, none of which touch
  `spineax.cudss` at all, plus the dispatch/reuse/transpose/conj *logic* in `_cudss.py`,
  exercised against a small fake `spineax.cudss` module (`_FakeCuDSS` below) that
  reproduces its documented phase contract (analyze -> factorize/refactorize -> solve,
  phase checks, dtype/nnz checks) with a real dense `jnp.linalg.solve` underneath. This
  is what actually proves `_cudss.py`'s state machine is wired correctly, without a GPU.
- **GPU-only, skipped everywhere else:** the real `spineax.cudss` module against real
  CUDA, checking the things a fake can't stand in for (real registry/eviction behaviour,
  real dtype support, real gradients).

The solver-agnostic factorization-reuse contract (correctness, reuse, transpose, passing
states into a jitted function) lives in `test_factorization.py`, the generic solve suite
in `test_solvers.py`, and `AutoSparseLinearSolver`'s GPU dispatch in `test_auto.py`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO, BCSR

import splineax.solvers._cudss as _cudss_module
from splineax import (
    BCOOLinearOperator,
    BCSRLinearOperator,
    CuDSS,
    SparseJacobianLinearOperator,
)
from splineax.solvers._auto import _cuda_backend_available
from splineax.solvers._cudss import (
    _cudss_available,
    _CuDSSBasicState,
    _CuDSSNumericState,
    _CuDSSSymbolicState,
    _mtype_id,
)

from .conftest import COMPLEX_MATRIX, RIGHT_HAND_SIDE, SQUARE_MATRIX, OperatorFactory

# ---------------------------------------------------------------------------
# A fake `spineax.cudss` module, faithful to its documented phase contract, backed
# by a real dense solve. Lets the dispatch/state logic in `_cudss.py` be exercised on
# CPU, without a GPU or the real (CUDA-only) binding.
# ---------------------------------------------------------------------------


class _FakeFactorToken(eqx.Module):
    id: jax.Array
    values: jax.Array
    offsets: jax.Array
    columns: jax.Array
    phase: str = eqx.field(static=True)
    dtype: Any = eqx.field(static=True)
    n: int = eqx.field(static=True)
    nnz: int = eqx.field(static=True)
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    reordering_id: int = eqx.field(static=True)
    memory_id: int = eqx.field(static=True)


class _FakeCuDSS:
    """A minimal stand-in for `spineax.cudss`, tracking calls and backed by a real
    dense solve. Mirrors the real module's documented contract closely enough to
    prove `_cudss.py` calls it the right way, not to test cuDSS itself:

    - `factorize` accepts any phase, while `refactorize`/`solve` need a factorized one.
    - `factorize`/`refactorize` check the incoming values' dtype and size against the
      token they were analyzed for.
    - `solve` checks the right-hand side dtype against the token.
    - every phase call mints a fresh registry id, `release` retires one.
    """

    def __init__(self) -> None:
        self._next_id = 0
        self._live: set[int] = set()
        self.analyze_calls: list[dict[str, Any]] = []
        self.factorize_calls: list[Any] = []
        self.refactorize_calls: list[Any] = []
        self.solve_calls: list[Any] = []
        self.release_calls: list[Any] = []

    def _mint(self) -> jax.Array:
        token_id = self._next_id
        self._next_id += 1
        self._live.add(token_id)
        return jnp.array([token_id], dtype=jnp.int32)

    def analyze(
        self,
        values,
        offsets,
        columns,
        *,
        mtype_id: int,
        mview_id: int,
        device_id: int,
        reordering: int,
        memory: int,
    ) -> _FakeFactorToken:
        self.analyze_calls.append(
            dict(mtype_id=mtype_id, mview_id=mview_id, device_id=device_id)
        )
        n = offsets.shape[0] - 1
        return _FakeFactorToken(
            id=self._mint(),
            values=values,
            offsets=offsets.astype(jnp.int32),
            columns=columns.astype(jnp.int32),
            phase="analyzed",
            dtype=jnp.dtype(values.dtype),
            n=int(n),
            nnz=int(columns.shape[0]),
            mtype_id=mtype_id,
            mview_id=mview_id,
            device_id=device_id,
            reordering_id=reordering,
            memory_id=memory,
        )

    def _numeric(self, token: _FakeFactorToken, values, *, refactor: bool):
        if refactor and token.phase != "factorized":
            raise ValueError("fake cudss: refactorize requires a factorized token")
        if jnp.dtype(values.dtype) != token.dtype:
            raise ValueError("fake cudss: values dtype does not match token dtype")
        if values.shape[-1] != token.nnz:
            raise ValueError("fake cudss: values size does not match token nnz")
        # "factorize/refactorize consume their input's id and return a fresh one"
        # (the real `FactorToken`'s docstring): the old id is retired here, not left
        # behind as a second live entry, so a whole analyze -> factorize chain is one
        # registry slot, renamed as it advances, not one entry per call.
        old_id = int(jax.device_get(token.id).ravel()[0])
        self._live.discard(old_id)
        return dataclasses.replace(
            token, id=self._mint(), values=values, phase="factorized"
        )

    def factorize(self, token: _FakeFactorToken, values) -> _FakeFactorToken:
        self.factorize_calls.append(values)
        return self._numeric(token, values, refactor=False)

    def refactorize(self, token: _FakeFactorToken, values) -> _FakeFactorToken:
        self.refactorize_calls.append(values)
        return self._numeric(token, values, refactor=True)

    def solve(self, token: _FakeFactorToken, b, ir_nsteps=None):
        del ir_nsteps
        self.solve_calls.append(b)
        if token.phase != "factorized":
            raise ValueError("fake cudss: solve requires a factorized token")
        if jnp.dtype(b.dtype) != token.dtype:
            raise ValueError("fake cudss: rhs dtype does not match token dtype")
        dense = BCSR(
            (token.values, token.columns, token.offsets), shape=(token.n, token.n)
        ).todense()
        return jnp.linalg.solve(dense, b)

    def release(self, token: _FakeFactorToken) -> bool:
        self.release_calls.append(token)
        token_id = int(jax.device_get(token.id).ravel()[0])
        return self._live.discard(token_id) is None  # discard returns None either way

    def registry_size(self) -> int:
        return len(self._live)

    def rebuild_count(self) -> int:
        return 0

    def cache_capacity(self) -> int:
        return 8


@pytest.fixture
def fake_cudss(monkeypatch: pytest.MonkeyPatch) -> _FakeCuDSS:
    """Make `CuDSS()` construct successfully and every `_spineax_cudss()` lookup in
    `_cudss.py` return a fresh `_FakeCuDSS`, so the whole solver runs against it.

    Also disables `_ensure_gpu`: these tests exercise the dispatch/state logic, not
    the real CUDA-only platform guard, which this environment's real "cpu" backend
    would otherwise (correctly) trip on every `compute` call. That guard is checked
    for real, unpatched, by `test_ensure_gpu_rejects_cpu` below.
    """
    fake = _FakeCuDSS()
    monkeypatch.setattr(_cudss_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_cudss_module, "_spineax_cudss", lambda: fake)
    monkeypatch.setattr(_cudss_module, "_ensure_gpu", lambda args: args)
    return fake


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_cudss_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """`CuDSS()` must raise `ImportError` with the `splineax[cudss]` install hint when
    the optional dependency isn't importable, regardless of whether it happens to be
    installed in this environment (the exact inverse trick `test_pardiso.py` uses)."""
    monkeypatch.setattr(_cudss_module, "_cudss_available", lambda: False)
    with pytest.raises(ImportError, match="splineax\\[cudss\\]"):
        CuDSS()


def test_cudss_available_survives_missing_parent_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_cudss_available` must return `False`, not raise, when `spineax` isn't
    installed at all, not just its `cudss` submodule. `importlib.util.find_spec`
    raises `ModuleNotFoundError` for a dotted name whose parent package is missing,
    which is the common case since the binding is an optional dependency.

    Forces that error rather than relying on the package being absent, so the check
    is the same on a machine that really does have the binding installed.
    """

    def raise_module_not_found(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(
        _cudss_module.importlib.util, "find_spec", raise_module_not_found
    )
    assert _cudss_available() is False


def test_ensure_gpu_matches_the_platform() -> None:
    """`_ensure_gpu` rejects every platform but CUDA, and passes values through on
    CUDA. `fake_cudss` disables this guard for the dispatch/reuse tests below, so it
    needs its own unpatched check here, asserting whichever way this machine goes."""
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


def test_init_converts_operator_to_coo(
    make_operator: OperatorFactory, fake_cudss: _FakeCuDSS
) -> None:
    """`CuDSS.init` reads the operator into row/column/value arrays matching its
    dense reference, for both `BCOO`- and `BCSR`-backed operators."""
    operator = make_operator(SQUARE_MATRIX)
    state = CuDSS().init(operator, {})
    assert isinstance(state, _CuDSSBasicState)
    rows, cols, values = state.coo
    dense = jnp.zeros(state.shape).at[rows, cols].set(values)
    assert jnp.allclose(dense, SQUARE_MATRIX)


def test_init_handles_unsorted_bcsr(fake_cudss: _FakeCuDSS) -> None:
    """An unsorted `BCSR` operator round-trips through `BCOO` correctly (the same
    caveat `Pardiso`/`KLU` have to handle)."""
    bcoo = BCOO.fromdense(SQUARE_MATRIX)
    # Reverse the (already coalesced) index order to get an unsorted BCSR.
    unsorted_bcsr = BCSR.from_bcoo(
        BCOO((bcoo.data[::-1], bcoo.indices[::-1]), shape=bcoo.shape)
    )
    operator = BCSRLinearOperator(unsorted_bcsr)
    state = CuDSS().init(operator, {})
    rows, cols, values = state.coo
    dense = jnp.zeros(state.shape).at[rows, cols].set(values)
    assert jnp.allclose(dense, SQUARE_MATRIX)


def test_init_materialises_sparse_jacobian(fake_cudss: _FakeCuDSS) -> None:
    """A `SparseJacobianLinearOperator` is materialised into the same CSR pattern as
    the equivalent `BCOOLinearOperator`."""
    from asdex import ColoredPattern

    def fn(x, args):
        del args
        return x * 2.0

    operator = SparseJacobianLinearOperator(
        fn, jnp.arange(4.0), sparsity=BCOO.fromdense(jnp.eye(4))
    )
    del ColoredPattern  # imported only to document what `.coloring` holds
    state = CuDSS().init(operator, {})
    rows, cols, values = state.coo
    dense = jnp.zeros(state.shape).at[rows, cols].set(values)
    assert jnp.allclose(dense, 2.0 * jnp.eye(4))


def test_init_rejects_non_square(fake_cudss: _FakeCuDSS) -> None:
    wide = jnp.ones((2, 3))
    operator = BCOOLinearOperator(BCOO.fromdense(wide))
    with pytest.raises(ValueError, match="square"):
        CuDSS().init(operator, {})


def test_init_rejects_unsupported_operator(fake_cudss: _FakeCuDSS) -> None:
    operator = lx.MatrixLinearOperator(SQUARE_MATRIX)
    with pytest.raises(TypeError, match="CuDSS"):
        CuDSS().init(operator, {})


# ---------------------------------------------------------------------------
# Dispatch / reuse / transpose / conj, against the fake
# ---------------------------------------------------------------------------


def _expected(dense: jax.Array, b: jax.Array = RIGHT_HAND_SIDE) -> jax.Array:
    return jnp.linalg.solve(np.asarray(dense), np.asarray(b))


def test_basic_state_analyzes_factorizes_and_releases(
    make_operator: OperatorFactory, fake_cudss: _FakeCuDSS
) -> None:
    """A one-shot solve through `_CuDSSBasicState` analyzes, factorizes, solves, and
    releases the token, all in one `compute` call."""
    operator = make_operator(SQUARE_MATRIX)
    solver = CuDSS()
    state = solver.init(operator, {})
    solution, results, _ = solver.compute(state, RIGHT_HAND_SIDE, {})
    assert jnp.allclose(solution, _expected(SQUARE_MATRIX))
    assert len(fake_cudss.analyze_calls) == 1
    assert len(fake_cudss.factorize_calls) == 1
    assert len(fake_cudss.release_calls) == 1
    assert fake_cudss.registry_size() == 0


def test_factorize_reuses_across_solves(
    make_operator: OperatorFactory, fake_cudss: _FakeCuDSS
) -> None:
    """`solver.factorize(operator)` analyzes and factorizes once. Every solve inside
    the block reuses that same numeric token, and it is released on exit."""
    operator = make_operator(SQUARE_MATRIX)
    solver = CuDSS()
    expected = _expected(SQUARE_MATRIX)

    with solver.factorize(operator) as state:
        assert isinstance(state, _CuDSSNumericState)
        first = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
        second = solver.compute(state, 2.0 * RIGHT_HAND_SIDE, {})[0]
        assert not fake_cudss.release_calls

    assert jnp.allclose(first, expected)
    assert jnp.allclose(second, _expected(SQUARE_MATRIX, 2.0 * RIGHT_HAND_SIDE))
    assert len(fake_cudss.analyze_calls) == 1
    assert len(fake_cudss.factorize_calls) == 1
    assert len(fake_cudss.solve_calls) == 2
    assert fake_cudss.release_calls


def test_factorize_symbolic_reuses_analysis_across_operators(
    make_operator: OperatorFactory, fake_cudss: _FakeCuDSS
) -> None:
    """A `factorize_symbolic` scope analyzes once. `.init(operator)` for different
    operators sharing the pattern each refactor numerically on `compute`, reusing the
    one analysis. The token is released when the scope exits."""
    solver = CuDSS()
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    other_matrix = 2.0 * SQUARE_MATRIX
    operator = make_operator(SQUARE_MATRIX)
    other_operator = make_operator(other_matrix)

    with solver.factorize_symbolic(sparsity) as scope:
        first_state = scope.init(operator)
        assert isinstance(first_state, _CuDSSSymbolicState)
        first_solution = solver.compute(first_state, RIGHT_HAND_SIDE, {})[0]

        second_state = scope.init(other_operator)
        second_solution = solver.compute(second_state, RIGHT_HAND_SIDE, {})[0]

        assert not fake_cudss.release_calls

    assert jnp.allclose(first_solution, _expected(SQUARE_MATRIX))
    assert jnp.allclose(second_solution, _expected(other_matrix))
    assert len(fake_cudss.analyze_calls) == 1, (
        "the symbolic analysis must run exactly once, reused by both solves"
    )
    assert len(fake_cudss.factorize_calls) == 2
    assert fake_cudss.release_calls


def test_factorize_symbolic_factorize_promotes_to_numeric(
    make_operator: OperatorFactory, fake_cudss: _FakeCuDSS
) -> None:
    """`scope.factorize(operator)` (or `.init(operator).factorize()`) promotes to a
    `_CuDSSNumericState`, reusing the scope's analysis."""
    solver = CuDSS()
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    operator = make_operator(SQUARE_MATRIX)

    with solver.factorize_symbolic(sparsity) as scope:
        with scope.factorize(operator) as numeric_state:
            assert isinstance(numeric_state, _CuDSSNumericState)
            solution = solver.compute(numeric_state, RIGHT_HAND_SIDE, {})[0]

    assert jnp.allclose(solution, _expected(SQUARE_MATRIX))
    assert len(fake_cudss.analyze_calls) == 1
    assert len(fake_cudss.factorize_calls) == 1


def test_transpose_symmetric_reuses_factorization(fake_cudss: _FakeCuDSS) -> None:
    """For a symmetric matrix, `transpose()` must reuse the token unchanged: no extra
    `analyze`/`factorize` calls."""
    symmetric_matrix = SQUARE_MATRIX + SQUARE_MATRIX.T
    operator = BCOOLinearOperator(BCOO.fromdense(symmetric_matrix), lx.symmetric_tag)
    solver = CuDSS()

    with solver.factorize(operator) as state:
        analyze_calls_before = len(fake_cudss.analyze_calls)
        factorize_calls_before = len(fake_cudss.factorize_calls)
        transposed_state, _ = solver.transpose(state, {})
        assert isinstance(transposed_state, _CuDSSNumericState)
        assert transposed_state.token is state.token
        solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0]

    assert len(fake_cudss.analyze_calls) == analyze_calls_before
    assert len(fake_cudss.factorize_calls) == factorize_calls_before
    assert jnp.allclose(solution, _expected(symmetric_matrix.T))


def test_transpose_general_refactorizes(fake_cudss: _FakeCuDSS) -> None:
    """For a general (untagged) matrix, `transpose()` must build and factorize a
    genuinely transposed token: cuDSS has no native transpose solve."""
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = CuDSS()

    with solver.factorize(operator) as state:
        analyze_calls_before = len(fake_cudss.analyze_calls)
        transposed_state, _ = solver.transpose(state, {})
        assert isinstance(transposed_state, _CuDSSNumericState)
        assert transposed_state.token is not state.token
        solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0]

    assert len(fake_cudss.analyze_calls) == analyze_calls_before + 1
    assert jnp.allclose(solution, _expected(SQUARE_MATRIX.T))


def test_conj_real_is_noop(fake_cudss: _FakeCuDSS) -> None:
    operator = BCOOLinearOperator(BCOO.fromdense(SQUARE_MATRIX))
    solver = CuDSS()

    with solver.factorize(operator) as state:
        conj_state, _ = solver.conj(state, {})
        assert conj_state is state


def test_conj_complex_refactorizes(fake_cudss: _FakeCuDSS) -> None:
    """For a complex matrix, `conj()` must reuse the pivots via `refactorize` (same
    magnitudes, so the existing pivoting stays valid) and solve correctly."""
    operator = BCOOLinearOperator(BCOO.fromdense(COMPLEX_MATRIX))
    solver = CuDSS()
    b = jnp.array([1.0 + 1.0j, 2.0 + 0.0j, 3.0 - 1.0j, 2.0j])

    with solver.factorize(operator) as state:
        conj_state, _ = solver.conj(state, {})
        assert isinstance(conj_state, _CuDSSNumericState)
        assert conj_state.token is not state.token
        solution = solver.compute(conj_state, b, {})[0]

    assert fake_cudss.refactorize_calls, "conj() on a complex state should refactorize"
    assert jnp.allclose(solution, _expected(np.asarray(COMPLEX_MATRIX).conj(), b))


# ---------------------------------------------------------------------------
# GPU-only: the real `spineax.cudss` module against real CUDA hardware.
# ---------------------------------------------------------------------------

pytestmark_gpu = pytest.mark.skipif(
    not (_cudss_available() and _cuda_backend_available()),
    reason="the optional cuDSS dependency is not installed, or no CUDA GPU is visible",
)


@pytestmark_gpu
def test_gpu_numeric_state_reuses_factorization_without_rebuilds(
    make_operator: OperatorFactory,
) -> None:
    """The reuse claim that does hold, checked against cuDSS's own counter: one
    `factorize`, many right-hand sides, zero rebuilds. `solve` does not consume its
    token's id (only the numeric phases do), so the factorization stays resident for
    every solve made against it."""
    from spineax import cudss as spineax_cudss

    solver = CuDSS()
    operator = make_operator(SQUARE_MATRIX)
    rebuilds_before = spineax_cudss.rebuild_count()

    with solver.factorize(operator) as state:
        for scale in [1.0, 2.0, 0.5, 3.0]:
            b = scale * RIGHT_HAND_SIDE
            solution = solver.compute(state, b, {})[0]
            expected = jnp.linalg.solve(np.asarray(SQUARE_MATRIX), np.asarray(b))
            assert jnp.allclose(solution, expected, atol=1e-5)

    assert spineax_cudss.rebuild_count() == rebuilds_before, (
        "solving repeatedly against one factorized token must not rebuild it"
    )


@pytestmark_gpu
def test_gpu_threading_the_token_avoids_rebuilds(
    make_operator: OperatorFactory,
) -> None:
    """Chaining each numeric phase from the token the previous one returned costs no
    rebuilds at all, for several different matrices sharing one sparsity pattern.

    This is the premise the whole `factorize_symbolic` redesign rests on, so it is
    worth pinning down on real hardware. A numeric phase *renames* its registry entry
    (`BatchTokenRegistry::rekey` in spineax's solver.cpp) rather than destroying it, so
    the analysis survives and only the old id stops resolving. Contrast
    `test_gpu_symbolic_scope_solves_correctly_across_values`, which restarts from the
    scope's analyzed token every time and pays one rebuild per solve.
    """
    from spineax import cudss as spineax_cudss

    solver = CuDSS()
    scales = [2.0, 0.5, 3.0, 1.5]
    rebuilds_before = spineax_cudss.rebuild_count()

    with solver.factorize(make_operator(SQUARE_MATRIX)) as state:
        solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
        assert jnp.allclose(solution, _expected(SQUARE_MATRIX), atol=1e-5)
        for scale in scales:
            # Each refactorize starts from the token the previous call handed back,
            # which is still resident, rather than from the original analysis.
            state = solver.refactorize(state, make_operator(scale * SQUARE_MATRIX))
            solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
            assert jnp.allclose(solution, _expected(scale * SQUARE_MATRIX), atol=1e-5)

    rebuilds = spineax_cudss.rebuild_count() - rebuilds_before
    assert rebuilds == 0, (
        f"threading the token should never rebuild the analysis, got {rebuilds}"
    )


@pytestmark_gpu
def test_gpu_symbolic_scope_solves_correctly_across_values(
    make_operator: OperatorFactory,
) -> None:
    """A `factorize_symbolic` scope gives correct answers for many matrices sharing its
    pattern, and costs one rebuild per solve after the first.

    It does not currently save the analysis. cuDSS's numeric phases *consume* their
    input token's id and hand back a fresh one, so the second and later `compute` calls
    find the scope's analyzed id superseded, and cuDSS transparently rebuilds it from
    the token's own arrays. That is correct but no cheaper than re-analyzing, so this
    pins the rebuild count at its known value rather than asserting a zero it does not
    achieve. See the `CuDSS` class docstring.
    """
    from spineax import cudss as spineax_cudss

    solver = CuDSS()
    sparsity = BCOO.fromdense(SQUARE_MATRIX)
    scales = [1.0, 2.0, 0.5, 3.0]
    rebuilds_before = spineax_cudss.rebuild_count()

    with solver.factorize_symbolic(sparsity) as scope:
        for scale in scales:
            state = scope.init(make_operator(scale * SQUARE_MATRIX))
            solution = solver.compute(state, RIGHT_HAND_SIDE, {})[0]
            assert jnp.allclose(solution, _expected(scale * SQUARE_MATRIX), atol=1e-5)

    rebuilds = spineax_cudss.rebuild_count() - rebuilds_before
    assert rebuilds == len(scales) - 1, (
        "expected one rebuild per solve after the first, the known cost of driving "
        f"several numeric phases from one analyzed token; got {rebuilds}"
    )


@pytestmark_gpu
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

    with solver.factorize(operator) as state:
        solution = solver.compute(state, b, {})[0]

    assert solution.dtype == dtype
    expected = jnp.linalg.solve(np.asarray(matrix), np.asarray(b))
    assert jnp.allclose(solution, expected, atol=1e-4)


@pytestmark_gpu
def test_gpu_general_transpose_solves_correctly(make_operator: OperatorFactory) -> None:
    operator = make_operator(SQUARE_MATRIX)
    solver = CuDSS()
    expected = jnp.linalg.solve(
        np.asarray(SQUARE_MATRIX).T, np.asarray(RIGHT_HAND_SIDE)
    )

    with solver.factorize(operator) as state:
        transposed_state, _ = solver.transpose(state, {})
        solution = solver.compute(transposed_state, RIGHT_HAND_SIDE, {})[0]

    assert jnp.allclose(solution, expected, atol=1e-5)


@pytestmark_gpu
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
