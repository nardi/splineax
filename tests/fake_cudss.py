"""A fake `spineax.cudss` module, shared by the CuDSS unit tests and the GPU-notebook test.

The real binding needs a CUDA GPU even to import, so ordinary CPU CI can never run it. The
fake reproduces its phase contract on top of a dense solve, which lets everything in
`splineax` that sits above the binding run on CPU.
"""

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCSR

import splineax.solvers._cudss as _cudss_module


class FakeFactorToken(eqx.Module):
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


class FakeCuDSS:
    """A minimal stand-in for `spineax.cudss`, tracking calls and backed by a real dense
    solve. Mirrors the real module's documented contract closely enough to prove
    `_cudss.py` calls it the right way, not to test cuDSS itself:

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
        self._refactorized_ids: set[int] = set()
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
    ) -> FakeFactorToken:
        self.analyze_calls.append(
            dict(mtype_id=mtype_id, mview_id=mview_id, device_id=device_id)
        )
        n = offsets.shape[0] - 1
        return FakeFactorToken(
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

    def _numeric(self, token: FakeFactorToken, values, *, refactor: bool):
        if refactor and token.phase != "factorized":
            raise ValueError("fake cudss: refactorize requires a factorized token")
        if jnp.dtype(values.dtype) != token.dtype:
            raise ValueError("fake cudss: values dtype does not match token dtype")
        if values.shape[-1] != token.nnz:
            raise ValueError("fake cudss: values size does not match token nnz")
        # "factorize/refactorize consume their input's id and return a fresh one" (the
        # real `FactorToken`'s docstring): the old id is retired here, not left behind as
        # a second live entry, so a whole analyze -> factorize chain is one registry slot,
        # renamed as it advances, not one entry per call. A traced id, from a branch of a
        # `lax.cond`, has no concrete value to retire.
        if not isinstance(token.id, jax.core.Tracer):
            self._live.discard(int(jax.device_get(token.id).ravel()[0]))
        return dataclasses.replace(
            token, id=self._mint(), values=values, phase="factorized"
        )

    def factorize(self, token: FakeFactorToken, values) -> FakeFactorToken:
        self.factorize_calls.append(values)
        return self._numeric(token, values, refactor=False)

    def refactorize(self, token: FakeFactorToken, values) -> FakeFactorToken:
        self.refactorize_calls.append(values)
        refactorized = self._numeric(token, values, refactor=True)
        self._refactorized_ids.add(int(jax.device_get(refactorized.id).ravel()[0]))
        return refactorized

    def query(self, token: FakeFactorToken) -> dict[str, jax.Array]:
        """Return the factor's diagonal, the only `query` field `_cudss.py` reads.

        A refactorized token keeps the first factorization's pivots. The reference
        matrices are diagonally dominant, so those pivots are the diagonal itself, which
        elimination without row swaps models. Any other token gets partial pivoting.
        """
        dense = np.array(
            BCSR(
                (token.values, token.columns, token.offsets), shape=(token.n, token.n)
            ).todense()
        )
        if int(jax.device_get(token.id).ravel()[0]) not in self._refactorized_ids:
            return {"diag": jnp.diag(jax.scipy.linalg.lu(dense)[2])}
        for k in range(token.n):
            dense[k + 1 :, k:] -= np.outer(
                dense[k + 1 :, k] / dense[k, k], dense[k, k:]
            )
        return {"diag": jnp.asarray(np.diag(dense))}

    def solve(self, token: FakeFactorToken, b, ir_nsteps=None):
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

    def release(self, token: FakeFactorToken) -> bool:
        self.release_calls.append(token)
        token_id = int(jax.device_get(token.id).ravel()[0])
        # `set.discard` returns None whether or not the id was present.
        return self._live.discard(token_id) is None

    def registry_size(self) -> int:
        return len(self._live)

    def rebuild_count(self) -> int:
        return 0

    def cache_capacity(self) -> int:
        return 8


@pytest.fixture
def fake_cudss(monkeypatch: pytest.MonkeyPatch) -> FakeCuDSS:
    """Make `CuDSS()` construct successfully and every `_spineax_cudss()` lookup in
    `_cudss.py` return a fresh `FakeCuDSS`, so the whole solver runs against it.

    Also disables `_ensure_gpu`: these tests exercise the dispatch/state logic, not the
    real CUDA-only platform guard, which this environment's real "cpu" backend would
    otherwise (correctly) trip on every `compute` call. That guard is checked for real,
    unpatched, by `test_ensure_gpu_matches_the_platform` below.
    """
    fake = FakeCuDSS()
    monkeypatch.setattr(_cudss_module, "_cudss_available", lambda: True)
    monkeypatch.setattr(_cudss_module, "_spineax_cudss", lambda: fake)
    monkeypatch.setattr(_cudss_module, "_ensure_gpu", lambda args: args)
    return fake
