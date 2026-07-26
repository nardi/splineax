"""Builds the callable a benchmark times, for one (solver, mode, n_rhs) combination.

Three things live here because they are entangled and easy to get wrong separately:

- **Native operator format per (solver, mode).** Feeding a solver the format it does not
  store internally makes it pay a conversion, so each pair gets the format that costs it
  nothing. See `NATIVE_FORMAT` for the table and the evidence behind it.
- **What sits inside the timer.** Determined by the reuse mode. Anything the mode says is
  already available is set up outside the timed region, in a fixture.
- **How multiple right-hand sides are batched.** `jax.vmap` where it works, `jax.lax.map`
  where it does not, which is not a free choice: see `BATCHING_FALLBACK`.
"""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax import Array
from jax.experimental.sparse import BCSR
from scipy.sparse import coo_matrix

import splineax as splx

from . import config
from .matrices import Matrix

SOLVER_CLASSES: dict[str, type] = {
    "spsolve": splx.Spsolve,
    "klu": splx.KLU,
    "pardiso": splx.Pardiso,
}

# Operator format each (solver, mode) pair stores internally, so passing it costs no
# conversion. Verified against the source rather than assumed:
#
# - `KLU` keeps COO and `klujax.solve` consumes a COO triple (`_klu.py:278`, `:298`), so
#   BCSR input would pay a `to_bcoo()`.
# - `Spsolve` and `Pardiso` keep CSR and use a sorted BCSR directly
#   (`_spsolve.py:128-135`, `_pardiso.py:396-403`), so BCOO input pays a
#   `BCSR.from_bcoo`.
# - `Pardiso` in `symbolic` mode is the exception: its scope's `init` always round-trips
#   through BCOO (`_pardiso.py:246-267`), so BCSR input would convert twice.
NATIVE_FORMAT: dict[tuple[str, str], str] = {
    ("spsolve", "none"): "bcsr",
    ("spsolve", "symbolic"): "bcsr",
    ("spsolve", "numeric"): "bcsr",
    ("klu", "none"): "bcoo",
    ("klu", "symbolic"): "bcoo",
    ("klu", "numeric"): "bcoo",
    ("pardiso", "none"): "bcsr",
    ("pardiso", "symbolic"): "bcoo",
    ("pardiso", "numeric"): "bcsr",
}

# Pairs where `jax.vmap` over the right-hand side raises, so batching falls back to the
# sequential `jax.lax.map`. Pardiso's factorized solves go through an `ffi_call` whose
# `vmap_method` is not one of the batchable ones, and that is set inside
# `pardiso_mkl_jax`, not reachable from here. Recorded per benchmark in `extra_info`, so
# a plot never silently compares a batched series against a sequential one.
BATCHING_FALLBACK: frozenset[tuple[str, str]] = frozenset(
    {("pardiso", "symbolic"), ("pardiso", "numeric")}
)


def batching_for(solver: str, mode: str) -> str:
    """Batching strategy actually used for a pair, honouring `config.BATCHING`."""
    if config.BATCHING == "lax_map":
        return "lax_map"
    if (solver, mode) in BATCHING_FALLBACK:
        return "lax_map"
    return config.BATCHING


def build_operator(matrix: Matrix, fmt: str) -> lx.AbstractLinearOperator:
    """Wrap a generated matrix in the requested operator format.

    Built in a fixture, outside the timer, so a solver given its native format does no
    conversion at all inside the timed region.
    """
    if fmt == "bcoo":
        return splx.BCOOLinearOperator(matrix.bcoo)
    if fmt == "bcsr":
        # `from_bcoo` on an already-sorted BCOO is the cheap path, and `matrices.generate`
        # coalesces (and so sorts) every matrix.
        return splx.BCSRLinearOperator(BCSR.from_bcoo(matrix.bcoo))
    raise ValueError(f"unknown operator format {fmt!r}")


@dataclass
class Case:
    """One benchmark's timed callable, plus the metadata describing it."""

    solver_name: str
    mode: str
    n_rhs: int
    batching: str
    matrix: Matrix
    operator: lx.AbstractLinearOperator
    input_format: str
    run: Callable[[], Array]
    residual: float
    jitted: Any
    """The wrapped `jax.jit` function, kept so the caller can watch its compile cache."""
    extra: dict[str, Any] = field(default_factory=dict)

    def cache_size(self) -> int:
        """Compile-cache entries for the timed function. Constant after warm-up, so a
        change means something recompiled inside the measured loop."""
        return int(self.jitted._cache_size())


def _single_solve(
    operator: lx.AbstractLinearOperator, solver: Any, state: Any
) -> Callable[[Array], Array]:
    """Solve for one right-hand side, reusing `state` when the mode provides one."""
    if state is None:
        return lambda b: lx.linear_solve(operator, b, solver=solver).value
    return lambda b: lx.linear_solve(operator, b, solver=solver, state=state).value


def _timed_callable(
    operator: lx.AbstractLinearOperator,
    solver: Any,
    state: Any,
    rhs: Array,
    n_rhs: int,
    batching: str,
) -> tuple[Callable[[], Array], Any]:
    """The function a benchmark times, wrapped in `jax.jit`, and the jitted function.

    `state` is closed over rather than passed as an argument: a scope opened while tracing
    gets an id token that changes the jit cache key (`_handle.py:94-103`), and closing over
    it also keeps the state's static fields out of the signature.

    The result is blocked on inside the callable, because JAX dispatches asynchronously
    and an unblocked call would measure dispatch instead of compute.
    """
    one = _single_solve(operator, solver, state)

    if n_rhs == 1:
        fn = jax.jit(one)
    elif batching == "vmap":
        fn = jax.jit(jax.vmap(one))
    elif batching == "lax_map":
        # Scan-based, so no unrolling: compile time stays flat in `n_rhs`.
        fn = jax.jit(lambda bs: jax.lax.map(one, bs))
    else:
        raise ValueError(f"unknown batching strategy {batching!r}")

    def run() -> Array:
        return jax.block_until_ready(fn(rhs))

    return run, fn


def _relative_residual(
    matrix: Matrix, rhs: Array, solution: Array, n_rhs: int
) -> float:
    """Worst relative residual over the right-hand sides, computed host-side with scipy.

    The gate that stops a wrong answer from being reported as a timing. No solver here
    detects singularity, so nothing else would catch it.
    """
    indices = np.asarray(matrix.bcoo.indices)
    reference = coo_matrix(
        (np.asarray(matrix.bcoo.data), (indices[:, 0], indices[:, 1])),
        shape=(matrix.n, matrix.n),
    ).tocsr()

    x = np.atleast_2d(np.asarray(solution))
    b = np.atleast_2d(np.asarray(rhs))
    numerator = np.linalg.norm(reference @ x.T - b.T, axis=0)
    denominator = np.linalg.norm(b.T, axis=0)
    return float(np.max(numerator / np.where(denominator == 0, 1.0, denominator)))


def build_case(
    stack: ExitStack,
    solver_name: str,
    mode: str,
    n_rhs: int,
    matrix: Matrix,
    rhs: Array,
) -> Case:
    """Set up everything the mode says is already available, then warm up and verify.

    `stack` owns the factorization scope: exiting it frees the native handle, so it has to
    outlive every timed call, which is why the caller passes in a fixture-scoped stack
    rather than one opened here.
    """
    solver = SOLVER_CLASSES[solver_name]()
    fmt = NATIVE_FORMAT[(solver_name, mode)]
    operator = build_operator(matrix, fmt)

    if mode == "none":
        # Nothing reused: every timed call redoes symbolic analysis, numeric
        # factorization and the solve.
        state = None
    elif mode == "symbolic":
        # The symbolic analysis is hoisted out. Note that for KLU and Pardiso the numeric
        # refactorization is fused into each *solve*, so this tier at 100 right-hand sides
        # is 100 factorizations. See the README.
        scope = stack.enter_context(solver.factorize_symbolic(operator))
        state = scope.init(operator)
    elif mode == "numeric":
        # Fully factorized outside the timer, so only the triangular solves are measured.
        state = stack.enter_context(solver.factorize(operator))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    batching = batching_for(solver_name, mode)
    run, jitted = _timed_callable(operator, solver, state, rhs, n_rhs, batching)

    # Untimed warm-up. Absorbs jit compilation and the lazy `klujax` / `pardiso_mkl_jax`
    # shared-library load, and produces the solution the residual gate checks.
    solution = run()
    residual = _relative_residual(matrix, rhs, solution, n_rhs)

    return Case(
        solver_name=solver_name,
        mode=mode,
        n_rhs=n_rhs,
        batching=batching,
        matrix=matrix,
        operator=operator,
        input_format=fmt,
        run=run,
        residual=residual,
        jitted=jitted,
        extra={
            "solver": solver_name,
            "mode": mode,
            "n_rhs": n_rhs,
            "batching": batching,
            # True when this pair could not use the requested strategy and was forced
            # onto the sequential one, so plots can mark those series.
            "batching_forced": batching != config.BATCHING,
            "family": matrix.spec.family,
            "replicate": matrix.spec.replicate,
            "nominal_n": matrix.spec.n,
            "actual_n": matrix.n,
            "nnz": matrix.nnz,
            "max_nnz_per_row": matrix.max_nnz_per_row,
            "max_nnz_per_col": matrix.max_nnz_per_col,
            "n_components": matrix.n_components,
            "input_format": fmt,
            "residual": residual,
            "backend": jax.default_backend(),
            "x64": jnp.zeros(1).dtype == jnp.float64,
        },
    )
