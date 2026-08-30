"""Tests for `stateful_solve_transform`.

The transform is checked against the untransformed function on hand-written algorithms, so
the benchmark is exact: same output, plus a threaded state that reuses a factorization. It
covers correctness, factorization reuse, composition with `jit`/`vmap`/`jacfwd`/`jacrev`,
per-signature caching, the filter-primitive round-trip, threading through a `cond`, `scan`,
`while_loop`, and `remat`, the opt-in custom-diff pass-through, the lifecycle paths, and the
solver filter.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax import make_jaxpr
from jax.experimental.sparse import BCOO

import splineax as splx

pytestmark = pytest.mark.usefixtures("enable_x64")

# Kept as float64 numpy arrays. This module is imported before x64 is enabled, so a `jnp`
# constant here would round to float32 and then mismatch the float64 arrays the tests build
# under the fixture. The helpers convert to `jnp` inside the tests instead.
_DENSE = np.array([[10.0, 2.0, 0.0], [3.0, 14.0, 5.0], [0.0, 6.0, 18.0]])
_B1_NP = np.array([1.0, 2.0, 3.0])
_B2_NP = np.array([3.0, 2.0, 1.0])


def _dense() -> jax.Array:
    return jnp.asarray(_DENSE)


def _b1() -> jax.Array:
    return jnp.asarray(_B1_NP)


def _b2() -> jax.Array:
    return jnp.asarray(_B2_NP)


def _indices() -> jax.Array:
    return BCOO.fromdense(_dense()).indices


def _data() -> jax.Array:
    return BCOO.fromdense(_dense()).data


def _two_solve_fn(tag: object):
    """A function that solves one matrix against two right-hand sides, sharing a pattern."""
    indices = _indices()

    def fn(data: jax.Array, b1: jax.Array, b2: jax.Array):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        x1 = lx.linear_solve(operator, b1, splx.KLU()).value
        x2 = lx.linear_solve(operator, b2, splx.KLU()).value
        return x1, x2

    return fn


def _count_primitive(jaxpr, needle: str) -> int:
    """Count equations whose primitive name contains `needle`, recursing into sub-jaxprs."""
    total = 0
    for eqn in jaxpr.eqns:
        if needle in eqn.primitive.name:
            total += 1
        for value in eqn.params.values():
            inner = getattr(value, "jaxpr", value)
            if hasattr(inner, "eqns"):
                total += _count_primitive(inner, needle)
    return total


def test_output_matches_untransformed() -> None:
    """The transform reproduces the function's output, with no state in or out by default."""
    fn = _two_solve_fn(splx.sparsity_pattern_tag(BCOO.fromdense(_dense())))
    run = splx.stateful_solve_transform(fn)
    got = run(_data(), _b1(), _b2())
    expected = fn(_data(), _b1(), _b2())
    assert jnp.allclose(got[0], expected[0], atol=1e-8)
    assert jnp.allclose(got[1], expected[1], atol=1e-8)


def test_reuses_factorization_across_solves() -> None:
    """Two solves sharing a pattern analyze once, so the pruned jaxpr holds one analysis."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _two_solve_fn(tag)
    run = splx.stateful_solve_transform(fn)
    b1, b2 = _b1(), _b2()
    threaded = make_jaxpr(lambda d: run(d, b1, b2))(_data())
    plain = make_jaxpr(lambda d: fn(d, b1, b2))(_data())
    assert _count_primitive(plain.jaxpr, "analyze") == 2
    assert _count_primitive(threaded.jaxpr, "analyze") == 1


def test_composes_with_jit_vmap_and_diff() -> None:
    """`jit`, `vmap`, `jacfwd`, and `jacrev` of the wrapped function match the plain one."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _two_solve_fn(tag)
    run = splx.stateful_solve_transform(fn)
    data = _data()

    def first(bb):
        return run(data, bb, _b2())[0]

    def first_plain(bb):
        return fn(data, bb, _b2())[0]

    assert jnp.allclose(jax.jit(first)(_b1()), first_plain(_b1()), atol=1e-8)
    batch = jnp.stack([_b1(), _b2(), _b1() + 1.0])
    assert jnp.allclose(jax.vmap(first)(batch), jax.vmap(first_plain)(batch), atol=1e-8)
    assert jnp.allclose(
        jax.jacfwd(first)(_b1()), jax.jacfwd(first_plain)(_b1()), atol=1e-8
    )
    assert jnp.allclose(
        jax.jacrev(first)(_b1()), jax.jacrev(first_plain)(_b1()), atol=1e-8
    )


def test_differentiates_through_the_matrix() -> None:
    """A gradient with respect to the matrix values matches the untransformed function."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _two_solve_fn(tag)
    run = splx.stateful_solve_transform(fn)

    def loss(data, algorithm):
        return jnp.sum(algorithm(data, _b1(), _b2())[0] ** 2)

    grad_run = jax.grad(lambda d: loss(d, run))(_data())
    grad_plain = jax.grad(lambda d: loss(d, fn))(_data())
    assert jnp.allclose(grad_run, grad_plain, atol=1e-6)


def test_vmap_over_operators_agrees_both_ways() -> None:
    """Transforming a vmap and vmapping a transform agree, threading one non-batched state."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    indices = _indices()

    def fn(data, b):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        return lx.linear_solve(operator, b, splx.KLU()).value

    scales = jnp.array([1.0, 2.0, 0.5])
    batched_data = scales[:, None] * _data()[None, :]

    outer = jax.vmap(splx.stateful_solve_transform(fn))(
        batched_data, jnp.stack([_b1()] * 3)
    )
    inner = splx.stateful_solve_transform(jax.vmap(fn))(
        batched_data, jnp.stack([_b1()] * 3)
    )
    reference = jax.vmap(fn)(batched_data, jnp.stack([_b1()] * 3))
    assert jnp.allclose(outer, reference, atol=1e-8)
    assert jnp.allclose(inner, reference, atol=1e-8)


def test_caches_the_staged_jaxpr_per_signature() -> None:
    """Repeated calls with one signature stage the interpreter once; a new signature stages
    again."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    indices = _indices()
    traces: list[bool] = []

    def fn(data, b):
        traces.append(True)
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        return lx.linear_solve(operator, b, splx.KLU()).value

    run = splx.stateful_solve_transform(fn)
    run(_data(), _b1())
    run(_data(), _b2())
    after_same = len(traces)
    # A different dtype is a new signature, so it stages once more.
    run(_data().astype(jnp.float32), _b1().astype(jnp.float32))
    assert after_same == 1, "same signature staged the function more than once"
    assert len(traces) == 2, "a new signature did not stage exactly once"


def test_filter_primitive_round_trip() -> None:
    """Reconstructing and rebinding a `linear_solve_p` equation reproduces the solve, so a
    change in equinox's private encoding is caught here."""
    from lineax._solve import linear_solve_p

    from splineax._stateful_transform import (
        _reconstruct_solve_arguments,
        _StateThreadingInterpreter,
    )

    indices = _indices()

    def solve(data, b):
        op = splx.BCOOLinearOperator(BCOO((data, indices), shape=(3, 3)))
        return lx.linear_solve(op, b, splx.KLU()).value

    closed = make_jaxpr(solve)(_data(), _b1())

    # Running the interpreter reconstructs and rebinds the solve, so a matching output is the
    # round-trip through equinox's encoding.
    interpreter = _StateThreadingInterpreter(
        filter_solver=(lambda _: False), state=None
    )
    outputs = interpreter.interpret(closed.jaxpr, closed.consts, [_data(), _b1()])
    assert jnp.allclose(outputs[0], solve(_data(), _b1()), atol=1e-8)

    # The reconstruction also recovers the solver, which is static, from a solve equation.
    solve_eqns = [
        eqn
        for eqn in closed.jaxpr.eqns[0].params["jaxpr"].jaxpr.eqns
        if eqn.primitive is linear_solve_p
    ]
    _, _, _, _, solver, _ = _reconstruct_solve_arguments(
        solve_eqns[0], [None] * len(solve_eqns[0].invars)
    )
    assert isinstance(solver, splx.KLU)


def _cond_fn(tag: object):
    """A function that solves once, then solves again inside one `cond` branch."""
    indices = _indices()

    def fn(data: jax.Array, b: jax.Array, flag: jax.Array):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        first = lx.linear_solve(operator, b, splx.KLU()).value
        second = jax.lax.cond(
            flag > 0,
            lambda: lx.linear_solve(operator, b * 2.0, splx.KLU()).value,
            lambda: b,
        )
        return first + second

    return fn


def test_threads_a_solve_inside_cond() -> None:
    """A solve inside a `cond` branch is threaded, so both branches match the plain run."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _cond_fn(tag)
    run = splx.stateful_solve_transform(fn)
    for flag in (jnp.array(1.0), jnp.array(-1.0)):
        got = run(_data(), _b1(), flag)
        assert jnp.allclose(got, fn(_data(), _b1(), flag), atol=1e-8)


def test_reuses_factorization_across_a_cond() -> None:
    """The branch solve reuses the state from the solve before the `cond`, so the taken
    branch analyzes zero extra times and the pruned jaxpr holds one analysis."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    run = splx.stateful_solve_transform(_cond_fn(tag))
    b1, flag = _b1(), jnp.array(1.0)
    threaded = make_jaxpr(lambda d: run(d, b1, flag))(_data())
    assert _count_primitive(threaded.jaxpr, "analyze") == 1


def test_cond_composes_with_jit_vmap_and_grad() -> None:
    """`jit`, `vmap`, and `grad` of a function whose `cond` threads a solve match the plain
    one, so the branch rewrite composes with the outer transforms."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _cond_fn(tag)
    run = splx.stateful_solve_transform(fn)
    data, flag = _data(), jnp.array(1.0)

    def loss(algorithm, b):
        return jnp.sum(algorithm(data, b, flag) ** 2)

    assert jnp.allclose(
        jax.jit(lambda b: run(data, b, flag))(_b1()),
        fn(data, _b1(), flag),
        atol=1e-8,
    )
    batch = jnp.stack([_b1(), _b2(), _b1() + 1.0])
    assert jnp.allclose(
        jax.vmap(lambda b: run(data, b, flag))(batch),
        jax.vmap(lambda b: fn(data, b, flag))(batch),
        atol=1e-8,
    )
    assert jnp.allclose(
        jax.grad(lambda b: loss(run, b))(_b1()),
        jax.grad(lambda b: loss(fn, b))(_b1()),
        atol=1e-6,
    )


def test_cond_without_prior_state_raises() -> None:
    """Threading into a `cond` needs a state already, so a first solve inside a branch with
    no prior state raises a clear error."""
    indices = _indices()

    def fn(data, b, flag):
        operator = splx.BCOOLinearOperator(BCOO((data, indices), shape=(3, 3)))
        return jax.lax.cond(
            flag > 0,
            lambda: lx.linear_solve(operator, b, splx.KLU()).value,
            lambda: b,
        )

    with pytest.raises(NotImplementedError, match="cond"):
        splx.stateful_solve_transform(fn)(_data(), _b1(), jnp.array(1.0))


def _scan_fn(tag: object):
    """A function that solves once, then solves each step of a `scan` over right-hand sides."""
    indices = _indices()

    def fn(data: jax.Array, b: jax.Array):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        start = lx.linear_solve(operator, b, splx.KLU()).value

        def body(carry: jax.Array, rhs: jax.Array):
            solution = lx.linear_solve(operator, carry + rhs, splx.KLU()).value
            return solution, solution

        final, _ = jax.lax.scan(body, start, jnp.stack([b, b * 2.0, b * 3.0]))
        return final

    return fn


def _while_fn(tag: object):
    """A function that solves once, then solves each step of a fixed-count `while_loop`."""
    indices = _indices()

    def fn(data: jax.Array, b: jax.Array):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        start = lx.linear_solve(operator, b, splx.KLU()).value

        def body(carry):
            i, x = carry
            return i + 1, lx.linear_solve(operator, x + b, splx.KLU()).value

        _, final = jax.lax.while_loop(lambda c: c[0] < 3, body, (0, start))
        return final

    return fn


def test_threads_a_solve_inside_scan() -> None:
    """A solve in a `scan` body is threaded, so the output matches and the shared pattern
    analyzes once across the prior solve and every iteration."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _scan_fn(tag)
    run = splx.stateful_solve_transform(fn)
    assert jnp.allclose(run(_data(), _b1()), fn(_data(), _b1()), atol=1e-8)
    threaded = make_jaxpr(lambda d: run(d, _b1()))(_data())
    assert _count_primitive(threaded.jaxpr, "analyze") == 1


def test_threads_a_solve_inside_while() -> None:
    """A solve in a `while_loop` body is threaded, so the output matches and the shared
    pattern analyzes once across the prior solve and every iteration."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _while_fn(tag)
    run = splx.stateful_solve_transform(fn)
    assert jnp.allclose(run(_data(), _b1()), fn(_data(), _b1()), atol=1e-8)
    threaded = make_jaxpr(lambda d: run(d, _b1()))(_data())
    assert _count_primitive(threaded.jaxpr, "analyze") == 1


def test_scan_composes_with_jit_and_grad() -> None:
    """`jit` and `grad` of a function whose `scan` threads a solve match the plain one, so
    the loop-carry rewrite composes with the outer transforms."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _scan_fn(tag)
    run = splx.stateful_solve_transform(fn)
    data = _data()
    assert jnp.allclose(
        jax.jit(lambda b: run(data, b))(_b1()), fn(data, _b1()), atol=1e-8
    )
    assert jnp.allclose(
        jax.grad(lambda b: jnp.sum(run(data, b) ** 2))(_b1()),
        jax.grad(lambda b: jnp.sum(fn(data, b) ** 2))(_b1()),
        atol=1e-6,
    )


def _loop_only_scan_fn(tag: object):
    """A function whose only solves are inside a `scan`, with no solve before it."""
    indices = _indices()

    def fn(data: jax.Array, b: jax.Array):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )

        def body(carry: jax.Array, rhs: jax.Array):
            solution = lx.linear_solve(operator, carry + rhs, splx.KLU()).value
            return solution, solution

        final, _ = jax.lax.scan(body, b, jnp.stack([b, b * 2.0, b * 3.0]))
        return final

    return fn


def _loop_only_while_fn(tag: object):
    """A function whose only solves are inside a `while_loop`, with no solve before it."""
    indices = _indices()

    def fn(data: jax.Array, b: jax.Array):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )

        def body(carry):
            i, x = carry
            return i + 1, lx.linear_solve(operator, x + b, splx.KLU()).value

        _, final = jax.lax.while_loop(lambda c: c[0] < 3, body, (0, b))
        return final

    return fn


def test_scan_without_prior_state_threads() -> None:
    """A `scan` whose only solves are inside it has its first iteration unrolled to create the
    state, so the output matches and the shared pattern analyzes once."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _loop_only_scan_fn(tag)
    run = splx.stateful_solve_transform(fn)
    assert jnp.allclose(run(_data(), _b1()), fn(_data(), _b1()), atol=1e-8)
    threaded = make_jaxpr(lambda d: run(d, _b1()))(_data())
    assert _count_primitive(threaded.jaxpr, "analyze") == 1


def test_while_without_prior_state_threads() -> None:
    """A `while_loop` whose only solves are inside it unrolls its first iteration behind a
    guard, so the output matches and the shared pattern analyzes once."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _loop_only_while_fn(tag)
    run = splx.stateful_solve_transform(fn)
    assert jnp.allclose(run(_data(), _b1()), fn(_data(), _b1()), atol=1e-8)
    threaded = make_jaxpr(lambda d: run(d, _b1()))(_data())
    assert _count_primitive(threaded.jaxpr, "analyze") == 1


def test_while_that_runs_zero_times_keeps_its_carry() -> None:
    """When the loop condition is false at once, the unrolled solve is discarded and the
    original carry is returned, matching the untransformed function."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    indices = _indices()

    def fn(data, b):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )

        def body(carry):
            i, x = carry
            return i + 1, lx.linear_solve(operator, x + b, splx.KLU()).value

        _, final = jax.lax.while_loop(lambda c: c[0] < 0, body, (0, b))
        return final

    run = splx.stateful_solve_transform(fn)
    assert jnp.allclose(run(_data(), _b1()), fn(_data(), _b1()), atol=1e-8)


def test_incompatible_seed_state_raises_a_clear_error() -> None:
    """A state whose structure differs from the loop-carry structure, such as one from
    `init_symbolic`, cannot be carried, so threading it into a loop raises a clear error."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    run = splx.stateful_solve_transform(_loop_only_scan_fn(tag))
    symbolic_state = splx.KLU().init_symbolic(BCOO.fromdense(_dense()))
    with pytest.raises(ValueError, match="pytree structure"):
        run(_data(), _b1(), state=symbolic_state)


def test_remat_without_a_solve_passes_through() -> None:
    """`remat` is not inlined, so a `remat` with no matched solve rebinds and stays intact."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    indices = _indices()

    def fn(data, b):
        rescaled = jax.checkpoint(lambda v: v * 2.0)(b)
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        return lx.linear_solve(operator, rescaled, splx.KLU()).value

    run = splx.stateful_solve_transform(fn)
    got = run(_data(), _b1())
    assert jnp.allclose(got, fn(_data(), _b1()), atol=1e-8)
    # The transformed jaxpr still holds the rematerialisation boundary.
    threaded = make_jaxpr(lambda d: run(d, _b1()))(_data())
    assert _count_primitive(threaded.jaxpr, "remat") == 1


def _remat_fn(tag: object):
    """A function that solves once, then solves again inside a `jax.checkpoint`."""
    indices = _indices()

    def fn(data: jax.Array, b: jax.Array):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        first = lx.linear_solve(operator, b, splx.KLU()).value

        def solve(v):
            return lx.linear_solve(operator, v, splx.KLU()).value

        return first + jax.checkpoint(solve)(b * 2.0)

    return fn


def test_threads_a_solve_inside_remat() -> None:
    """A solve inside a `jax.checkpoint` is threaded, so the output matches, the shared
    pattern analyzes once, and the rematerialisation boundary survives."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _remat_fn(tag)
    run = splx.stateful_solve_transform(fn)
    assert jnp.allclose(run(_data(), _b1()), fn(_data(), _b1()), atol=1e-8)
    threaded = make_jaxpr(lambda d: run(d, _b1()))(_data())
    assert _count_primitive(threaded.jaxpr, "analyze") == 1
    assert _count_primitive(threaded.jaxpr, "remat2") == 1


def test_remat_threads_a_first_solve_without_prior_state() -> None:
    """A `remat` runs once, so a first solve inside it may create the state, unlike a loop.
    With no prior solve the output still matches the untransformed function."""
    indices = _indices()

    def fn(data, b):
        def solve(v):
            operator = splx.BCOOLinearOperator(BCOO((data, indices), shape=(3, 3)))
            return lx.linear_solve(operator, v, splx.KLU()).value

        return jax.checkpoint(solve)(b)

    run = splx.stateful_solve_transform(fn)
    assert jnp.allclose(run(_data(), _b1()), fn(_data(), _b1()), atol=1e-8)


def test_remat_composes_with_grad() -> None:
    """`grad` of a function whose `remat` threads a solve matches the plain one, so the
    rewrite keeps working under the rematerialising backward pass."""
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    fn = _remat_fn(tag)
    run = splx.stateful_solve_transform(fn)
    data = _data()
    assert jnp.allclose(
        jax.grad(lambda b: jnp.sum(run(data, b) ** 2))(_b1()),
        jax.grad(lambda b: jnp.sum(fn(data, b) ** 2))(_b1()),
        atol=1e-6,
    )


def test_return_final_state_paths() -> None:
    """`return_final_state` controls whether the state is handed back or released.

    A single-output `fn` keeps the two shapes, an array alone versus an `(array, state)`
    pair, unambiguous.
    """
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    indices = _indices()

    def fn(data, b):
        operator = splx.BCOOLinearOperator(
            BCOO((data, indices), shape=(3, 3)), tags=tag
        )
        return lx.linear_solve(operator, b, splx.KLU()).value

    expected = fn(_data(), _b1())

    # Default with no initial state: the output alone, released inside.
    out = splx.stateful_solve_transform(fn)(_data(), _b1())
    assert isinstance(out, jax.Array)
    assert jnp.allclose(out, expected, atol=1e-8)

    # Explicit true: the output paired with the state.
    keep = splx.stateful_solve_transform(fn, return_final_state=True)
    out_true, state = keep(_data(), _b1())
    assert jnp.allclose(out_true, expected, atol=1e-8)

    # Default with an initial state passed at call time: the pair again.
    out_seed, seeded = splx.stateful_solve_transform(fn)(_data(), _b1(), state=state)
    assert jnp.allclose(out_seed, expected, atol=1e-8)
    seeded.release()

    # Explicit false: the output alone even though a state was threaded.
    out_false = splx.stateful_solve_transform(fn, return_final_state=False)(
        _data(), _b1()
    )
    assert isinstance(out_false, jax.Array)
    assert jnp.allclose(out_false, expected, atol=1e-8)


def test_custom_jvp_solve_raises_by_default() -> None:
    """A matched solve inside a `custom_jvp` raises by default, since the state cannot cross
    the custom rule."""
    indices = _indices()

    @jax.custom_jvp
    def solve(data, b):
        operator = splx.BCOOLinearOperator(BCOO((data, indices), shape=(3, 3)))
        return lx.linear_solve(operator, b, splx.KLU()).value

    @solve.defjvp
    def _solve_jvp(primals, tangents):
        (data, b), (_, b_dot) = primals, tangents
        return solve(data, b), b_dot

    with pytest.raises(NotImplementedError, match="custom_jvp"):
        splx.stateful_solve_transform(solve)(_data(), _b1())


def test_custom_jvp_solve_passes_through_when_opted_in() -> None:
    """With `pass_through_custom_diff`, a solve inside a `custom_jvp` runs unthreaded, so the
    output matches and the primitive is left in the jaxpr rather than rewritten."""
    indices = _indices()

    @jax.custom_jvp
    def solve(data, b):
        operator = splx.BCOOLinearOperator(BCOO((data, indices), shape=(3, 3)))
        return lx.linear_solve(operator, b, splx.KLU()).value

    @solve.defjvp
    def _solve_jvp(primals, tangents):
        (data, b), (_, b_dot) = primals, tangents
        return solve(data, b), b_dot

    run = splx.stateful_solve_transform(solve, pass_through_custom_diff=True)
    assert jnp.allclose(run(_data(), _b1()), solve(_data(), _b1()), atol=1e-8)
    threaded = make_jaxpr(lambda d: run(d, _b1()))(_data())
    assert _count_primitive(threaded.jaxpr, "custom_jvp_call") >= 1


def test_custom_vjp_solve_passes_through_when_opted_in() -> None:
    """The pass-through covers `custom_vjp` too, so a solve inside one runs unthreaded and
    the primitive is left in the jaxpr."""
    indices = _indices()

    @jax.custom_vjp
    def solve(data, b):
        operator = splx.BCOOLinearOperator(BCOO((data, indices), shape=(3, 3)))
        return lx.linear_solve(operator, b, splx.KLU()).value

    def solve_fwd(data, b):
        return solve(data, b), None

    def solve_bwd(_, cotangent):
        return None, cotangent

    solve.defvjp(solve_fwd, solve_bwd)

    run = splx.stateful_solve_transform(solve, pass_through_custom_diff=True)
    assert jnp.allclose(run(_data(), _b1()), solve(_data(), _b1()), atol=1e-8)
    threaded = make_jaxpr(lambda d: run(d, _b1()))(_data())
    assert _count_primitive(threaded.jaxpr, "custom_vjp_call") >= 1


def test_filter_solver_skips_a_dense_solve() -> None:
    """The default filter threads only stateful solvers, so a dense `lineax.LU()` passes
    through, and a predicate filter works too."""

    def mixed(matrix, b):
        return lx.linear_solve(lx.MatrixLinearOperator(matrix), b, lx.LU()).value

    got = splx.stateful_solve_transform(mixed)(_DENSE, _b1())
    assert jnp.allclose(
        got, np.linalg.solve(np.asarray(_DENSE), np.asarray(_b1())), atol=1e-8
    )

    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    predicate = splx.stateful_solve_transform(
        _two_solve_fn(tag), filter_solver=lambda solver: isinstance(solver, splx.KLU)
    )
    out = predicate(_data(), _b1(), _b2())
    assert jnp.allclose(out[0], _two_solve_fn(tag)(_data(), _b1(), _b2())[0], atol=1e-8)


def test_explicit_state_is_replaced_for_a_matched_solve() -> None:
    """A `state=` the function passes to `lineax.linear_solve` is discarded for a threaded
    solve, and the transform substitutes its own state built from the operator.

    The function seeds a stale state factored from matrix `a`, then solves matrix `b`. The
    untransformed run reuses that stale factorization and returns `a`'s solution. The
    transform ignores the stale state, updates to `b`, and returns `b`'s solution.
    """
    indices = _indices()
    tag = splx.sparsity_pattern_tag(BCOO.fromdense(_dense()))
    data_a = _data()
    data_b = 2.0 * data_a
    matrix_a = np.asarray(_DENSE)
    matrix_b = 2.0 * matrix_a
    solution_a = np.linalg.solve(matrix_a, np.asarray(_b1()))
    solution_b = np.linalg.solve(matrix_b, np.asarray(_b1()))

    def fn(data_a, data_b, b):
        solver = splx.KLU()
        operator_a = splx.BCOOLinearOperator(
            BCOO((data_a, indices), shape=(3, 3)), tags=tag
        )
        operator_b = splx.BCOOLinearOperator(
            BCOO((data_b, indices), shape=(3, 3)), tags=tag
        )
        stale = solver.init(operator_a)
        return lx.linear_solve(operator_b, b, solver, state=stale).value

    plain = fn(data_a, data_b, _b1())
    transformed = splx.stateful_solve_transform(fn)(data_a, data_b, _b1())
    assert jnp.allclose(plain, solution_a, atol=1e-8)
    assert jnp.allclose(transformed, solution_b, atol=1e-8)


def test_explicit_state_survives_an_unthreaded_solve() -> None:
    """A `state=` on a solve the filter skips passes through untouched, so the transform
    leaves a stale dense factorization in place and returns the same as the plain run.

    This is the counterpart to a threaded solve, where the same stale state is discarded.
    """
    matrix_a = np.asarray(_DENSE)
    matrix_b = 2.0 * matrix_a
    solution_a = np.linalg.solve(matrix_a, np.asarray(_b1()))

    def fn(matrix_a, matrix_b, b):
        stale = lx.LU().init(lx.MatrixLinearOperator(matrix_a), {})
        return lx.linear_solve(
            lx.MatrixLinearOperator(matrix_b), b, lx.LU(), state=stale
        ).value

    plain = fn(matrix_a, matrix_b, _b1())
    transformed = splx.stateful_solve_transform(fn)(matrix_a, matrix_b, _b1())
    assert jnp.allclose(transformed, plain, atol=1e-8)
    assert jnp.allclose(transformed, solution_a, atol=1e-8)
