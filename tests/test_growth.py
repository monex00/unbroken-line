"""Invariants of the growth: one line, closed, never crossing itself.

The grown curves are shared between tests (module fixtures): one uniform
field, run past the point where it fills the frame, and one two-tone field.
"""

from dataclasses import replace

import numpy as np
import pytest

from unbroken_line import (
    HAVE_NUMBA,
    Growth,
    arc_positions,
    attraction,
    attraction_reference,
    crossings,
    crossings_kernel,
    grow,
    guard,
    sample,
    seed_ring,
    step,
    step_reference,
    violating_segments,
)

SMALL = Growth(width=120, r_min=3.0, r_max=3.0, max_nodes=50_000)
TWO = Growth(width=160, r_min=2.0, r_max=4.0, max_nodes=50_000)


def uniform(g, h=120):
    return np.full((h, g.width), g.r_min)


def two_tone(g):
    R = np.full((g.width, g.width), g.r_min)
    R[:, g.width // 2 :] = g.r_max
    return R


def spacing(P, x0, x1, y0, y1):
    """Mean strand spacing in a box: its area over the length of line in it."""
    B = np.roll(P, -1, axis=0)
    mid = 0.5 * (P + B)
    inside = (mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1)
    return (x1 - x0) * (y1 - y0) / np.hypot(*(B - P)[inside].T).sum()


@pytest.fixture(scope="module")
def uniform_run():
    """Never stops by itself (settle_tol=0), so the plateau is visible."""
    g = replace(SMALL, settle_tol=0)
    return grow(uniform(g), g, 5000, every=100)


@pytest.fixture(scope="module")
def two_tone_run():
    return grow(two_tone(TWO), TWO, 4000, every=100)


def test_seed_ring_is_closed_and_centred():
    P = seed_ring(SMALL, np.array([60.0, 60.0]), 3.0)
    assert P.shape == (SMALL.seed_nodes, 2)
    assert np.allclose(P.mean(0), [60.0, 60.0])
    assert np.allclose(np.hypot(*(P - 60.0).T), SMALL.seed_radius * 3.0)


def test_determinism():
    R = uniform(SMALL)
    a = grow(R, SMALL, 300, seed=1)[-1]
    b = grow(R, SMALL, 300, seed=1)[-1]
    assert a.tobytes() == b.tobytes()


def test_stays_inside_frame(uniform_run):
    P = uniform_run[-1]
    H, W = uniform(SMALL).shape
    assert P[:, 0].min() >= 0 and P[:, 0].max() <= W - 1
    assert P[:, 1].min() >= 0 and P[:, 1].max() <= H - 1


def test_node_count_plateaus(uniform_run):
    n = [len(s) for s in uniform_run]
    assert n[-1] < SMALL.max_nodes
    assert abs(n[-1] - n[-6]) / n[-1] < 0.02  # last 500 steps


def test_settles_by_itself():
    snaps = grow(uniform(SMALL), SMALL, 12000, every=100)
    assert len(snaps) < 100  # stopped well before 12000 steps


def test_detector_catches_crossings():
    t = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    for off in (0.0, 0.01):  # through a node, and between nodes
        eight = np.column_stack([np.sin(t + off), np.sin(2 * (t + off))])
        assert len(crossings(eight))
    circle = np.column_stack([np.cos(t), np.sin(t)])
    assert not len(crossings(circle, 0.01, 0.1))
    near = np.vstack([circle, circle[::-1] * 0.995])  # two loops, a hair apart
    assert len(crossings(near[::2], 0.02, 0.1))


@pytest.mark.parametrize("run", ["uniform_run", "two_tone_run"])
def test_never_self_intersects(run, request):
    P = request.getfixturevalue(run)[-1]
    g = SMALL if run == "uniform_run" else TWO
    assert not len(crossings(P, 0.2 * g.r_min, g.r_min))


def test_spacing_matches_field_on_uniform(uniform_run):
    P = uniform_run[-1]
    m = 2 * SMALL.r_max
    assert spacing(P, m, 120 - m, m, 120 - m) == pytest.approx(SMALL.r_min, rel=0.15)


def test_spacing_matches_field_on_two_tones(two_tone_run):
    P = two_tone_run[-1]
    W, m = TWO.width, 2 * TWO.r_max
    assert spacing(P, m, W / 2 - m, m, W - m) == pytest.approx(TWO.r_min, rel=0.15)
    assert spacing(P, W / 2 + m, W - m, m, W - m) == pytest.approx(TWO.r_max, rel=0.15)


def test_polish_smooths_and_never_crosses(two_tone_run):
    from unbroken_line import polish

    def turn(Q):  # mean turning angle per node
        a, b = np.roll(Q, -1, 0) - Q, Q - np.roll(Q, 1, 0)
        c = (a * b).sum(1) / np.maximum(np.hypot(*a.T) * np.hypot(*b.T), 1e-12)
        return np.arccos(np.clip(c, -1, 1)).mean()

    P = two_tone_run[-1]
    Q = polish(P, TWO)
    assert Q.shape == P.shape and turn(Q) < 0.6 * turn(P)
    assert not len(crossings(Q, 0.2 * TWO.r_min, TWO.r_min))


def test_guard_undoes_a_crossing():
    t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    P = np.column_stack([60 + 30 * np.cos(t), 60 + 10 * np.sin(t)])  # flat ellipse
    assert not len(violating_segments(P, SMALL))
    Q = P.copy()
    Q[30] = P[90]  # a node jumps to the other side
    assert len(violating_segments(Q, SMALL))
    G = guard(P, Q, SMALL)
    assert not len(violating_segments(G, SMALL))
    assert np.array_equal(G[30], P[30])


@pytest.mark.skipif(not HAVE_NUMBA, reason="numba not installed")
def test_fast_kernel_matches_reference(two_tone_run):
    R = two_tone(TWO)
    P = two_tone_run[len(two_tone_run) // 2]
    r = sample(R, P)
    s, L = arc_positions(P)
    a, b = attraction_reference(P, r, s, L, TWO), attraction(P, r, s, L, TWO)
    assert np.abs(a).max() > 0
    assert np.abs(a - b).max() < 1e-9

    Q = P + np.random.default_rng(1).normal(scale=0.3, size=P.shape)  # with defects
    ref = np.unique(crossings(Q, 0.5, 2.0))
    fast = np.flatnonzero(crossings_kernel(Q, 0.96, 0.5, 2.0, np.ones(len(Q), bool)))
    assert len(ref) and np.array_equal(ref, fast)

    p1 = step(P, R, TWO, np.random.default_rng(7))
    p2 = step_reference(P, R, TWO, np.random.default_rng(7))
    assert p1.shape == p2.shape and np.abs(p1 - p2).max() < 1e-9


def test_step_returns_closed_polyline_shape():
    rng = np.random.default_rng(0)
    R = uniform(SMALL)
    P = step(seed_ring(SMALL, np.array([60.0, 60.0]), 3.0), R, SMALL, rng)
    assert P.ndim == 2 and P.shape[1] == 2 and np.isfinite(P).all()


def test_numpy_path_without_numba(monkeypatch):
    """Under Pyodide there is no numba: the same step must run on numpy.
    (Over many steps the two paths part ways: the growth is chaotic, and a
    1e-15 difference eventually flips a split. One step must agree.)"""
    import unbroken_line

    R = uniform(SMALL)
    P = grow(R, SMALL, 150, seed=3)[-1]
    fast = step(P, R, SMALL, np.random.default_rng(5))
    monkeypatch.setattr(unbroken_line, "HAVE_NUMBA", False)
    slow = step(P, R, SMALL, np.random.default_rng(5))
    assert fast.shape == slow.shape and np.abs(fast - slow).max() < 1e-9
    Q = grow(R, SMALL, 100, seed=4)[-1]
    assert np.isfinite(Q).all() and not len(violating_segments(Q, SMALL))


def test_flow_offset_sways_and_never_crosses(two_tone_run):
    from unbroken_line import Flow

    P = two_tone_run[-1]
    flow = Flow(TWO, np.random.default_rng(3))
    for t in (0, 60, 120, 180):
        Q = flow.offset(P, t, 100.0)  # far too much: capped at max_offset()
        moved = np.hypot(*(Q - P).T)
        assert 0.1 * flow.max_offset() < moved.max() <= 1.001 * np.sqrt(6) * flow.max_offset()
        assert not len(crossings(Q, 0.2 * TWO.r_min, TWO.r_min))


def test_pack_growth_round_trip(two_tone_run):
    from unbroken_line import pack_growth, unpack_growth

    curves = [two_tone_run[0], two_tone_run[len(two_tone_run) // 2], two_tone_run[-1]]
    curves2, shape, g = unpack_growth(pack_growth(curves, (160, 160), TWO))
    assert shape == (160, 160) and g == TWO and len(curves2) == len(curves)
    for P, Q in zip(curves, curves2):
        assert P.shape == Q.shape and np.abs(P - Q).max() <= 0.5 / 32 + 1e-9
