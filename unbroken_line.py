# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo", "numpy", "scipy", "matplotlib", "pillow", "numba"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Unbroken Line",
    css_file="notebook.css",
)

with app.setup:
    # Everything defined here and in @app.function / @app.class_definition
    # cells is importable:  from unbroken_line import Growth, grow, draw, ...
    # animate.py, figures.py and the tests import from this file; the
    # algorithm and the look of the line live here and nowhere else.
    import math
    import os
    import tomllib
    import urllib.request
    from dataclasses import asdict, dataclass, replace
    from io import BytesIO
    from pathlib import Path

    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter, map_coordinates
    from scipy.spatial import cKDTree

    try:  # the fast kernels; in the browser (Pyodide) numpy does it all
        import numba
        from numba import njit, prange

        HAVE_NUMBA = True
        # not every core by default: hours at full load are hard on a laptop
        THREADS = min(int(os.environ.get("UNBROKEN_THREADS", 8)), numba.config.NUMBA_NUM_THREADS)
        numba.set_num_threads(THREADS)
    except ImportError:
        HAVE_NUMBA = False
        prange = range

        def njit(*args, **kwargs):
            return lambda f: f

    try:
        ROOT = Path(__file__).parent
    except NameError:
        ROOT = Path.cwd()
    PUBLIC = ROOT / "public"  # paintings and catalog; served next to the web page

    # Two palettes, strictly MONOCHROME: tone comes from how tightly the line
    # is packed, never from a colormap.
    PAPER = dict(bg="#f4f0e8", ink="#383b3e")
    NIGHT = dict(bg="#1d252b", ink="#fbfcfc")
    # Line width, in units of r_min: with strands r_min apart in the darks,
    # this is the share of the paper the ink covers there.
    INK = 0.55
    # Measured: grown strands settle this much tighter than r.
    PACKING = 0.93


@app.class_definition
@dataclass(frozen=True)
class Growth:
    """Every knob of the growth, in work units (1 unit = 1 px of the tone map).

    Spacing between neighbouring strands is what makes tone: tight in the
    darks, loose in the lights. Every wall of the line pushes the walls
    closer than the local spacing r(x) and pulls the ones a little further,
    and r(x) is read off the image. Lengths are in units of the local r
    unless they say px.
    """

    width: int = 600  # tone-map width, px; height follows the image
    r_min: float = 2.4  # spacing in the darkest tone, px
    r_max: float = 12.0  # spacing in the lightest tone, px
    gamma: float = 1.8  # tone curve: >1 keeps more of the image dense
    night: bool = False  # light ink on dark (NIGHT): dense strands in the lights
    blur: float = 1.0  # tone-map blur, in units of r_min
    clip_dark: float = 1.0  # percentile of the image mapped to r_min
    clip_light: float = 99.0  # percentile mapped to r_max
    local: float = 0.8  # local contrast: how much each tone is pushed away
    local_size: float = 6.0  # from the mean around it, over this many r_max
    flatten: float = 0.0  # lift plain, detail-less ground towards the light (0: off)
    flatten_size: float = 1.0  # detail measured over this many r_max
    # Pedersen & Singh (2006), with every length in units of the local r:
    sample: float = 0.4  # node spacing D, in r
    sample_cap: float = 2.0  # but D never above sample * sample_cap * r_min
    k_max: float = 1.2  # split an edge longer than k_max * D
    k_min: float = 0.2  # delete a node closer than k_min * D to its neighbour
    cutoff: float = 2.0  # walls attract up to cutoff * r (paper: k1 / k0)
    n_min: int = 1  # a node ignores the segments ending this close (nodes)
    fold: float = 0.9  # and those not folded back: distance > fold * arc
    f_a: float = 0.02  # attraction-repulsion strength
    lj_floor: float = 0.85  # closer than this pushes as hard as at this
    f_f: float = 0.1  # fairing: pull towards the neighbours' mean
    f_b: float = 0.1  # Brownian motion, in units of D
    max_move: float = 0.25  # no node moves further than this in one step
    edge_move: float = 0.5  # nor further than this times its shorter edge
    gap: float = 0.25  # folded strands never closer than gap * r_min (enforced)
    seed_radius: float = 1.0  # radius of the initial ring
    seed_nodes: int = 20
    max_nodes: int = 400_000
    settle: int = 300  # stop once the node count grew by less than
    settle_tol: float = 0.001  # settle_tol over the last `settle` steps (0: never)
    # for video, after the growth: a slow fluid drift, so the finished line moves
    flow: float = 0.0  # in r_min per step (0: none)
    flow_scale: float = 8.0  # its swirls are this many r_max across
    flow_period: int = 240  # and change over this many steps
    flow_waves: int = 6
    steady_reach: float = 0.35  # video frames: average a node with the nearest point
    steady_window: int = 2  # of the curve this many frames back and on, if within reach * r
    # for drawing only
    polish: int = 10  # smoothing passes on the curve that gets drawn
    polish_f: float = 0.5  # each pass moves a node this far to its neighbours' mean


@app.function
def public_bytes(name):
    """The bytes of public/<name>: from disk, or, when the notebook runs in
    the browser, over HTTP from the folder served next to the page."""
    local = PUBLIC / name
    if local.exists():
        return local.read_bytes()
    import marimo as mo

    with urllib.request.urlopen(str(mo.notebook_location() / "public" / name)) as f:
        return f.read()


@app.function
def load_catalog():
    """public/artworks.toml -> {slug: entry}."""
    return {a["slug"]: a for a in tomllib.loads(public_bytes("artworks.toml").decode())["artwork"]}


@app.function
def load_grey(data):
    """Any image (a path, or its bytes) as float grey in [0, 1]."""
    img = Image.open(BytesIO(data) if isinstance(data, bytes) else data)
    return np.asarray(img.convert("L"), dtype=np.float64) / 255.0


@app.function
def load_image(slug):
    """A painting of the catalog, public/art/<slug>.jpg, as float grey."""
    return load_grey(public_bytes(f"art/{slug}.jpg"))


@app.function
def spacing_field(grey, g):
    """Grey image -> r(x): local strand spacing, in work units.

    Resized to g.width and blurred, so the line reads tone rather than film
    grain. Local contrast (g.local > 0) pushes each tone away from the mean
    of its surroundings, so a dark eye in a lit face stays darker than the
    face around it. Then the tones between the clip_dark and clip_light
    percentiles are stretched to [0, 1] and mapped through the tone curve.
    """
    h = round(grey.shape[0] * g.width / grey.shape[1])
    img = Image.fromarray((grey * 255).astype(np.uint8)).resize(
        (g.width, h), Image.Resampling.LANCZOS
    )
    t = np.asarray(img, dtype=np.float64) / 255.0
    t = gaussian_filter(t, g.blur * g.r_min)
    if g.local:
        t = t + g.local * (t - gaussian_filter(t, g.local_size * g.r_max))
    lo, hi = np.percentile(t, [g.clip_dark, g.clip_light])
    t = np.clip((t - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    # how open the strands are: on paper the lights are open; on the night
    # palette the ink is light, so the darks are
    t = 1.0 - t if g.night else t
    if g.flatten:
        t = t + g.flatten * flatness(t, g) * (1.0 - t)
    return g.r_min + (g.r_max - g.r_min) * t**g.gamma


@app.function
def flatness(t, g):
    """1 where the tone map is flat (a plain ground), 0 where it has detail.
    Detail: the fine grain of the image (t minus t blurred over a quarter
    of g.flatten_size * r_max), its strength averaged over g.flatten_size
    * r_max, against the image's own 80th percentile of it. spacing_field
    opens the flat parts up by g.flatten, so a textured subject on a plain
    ground (Medusa's snakes on her olive shield) stands out as dense
    strands on open loops, on either palette."""
    s = g.flatten_size * g.r_max
    fine = t - gaussian_filter(t, s / 4)
    sd = np.sqrt(gaussian_filter(fine * fine, s))
    return np.clip(1.0 - sd / max(np.percentile(sd, 80), 1e-9), 0.0, 1.0)


@app.function
def params(entry=None, **knobs):
    """The Growth of one artwork: the defaults, then its [artwork.growth]
    changes from artworks.toml, then `knobs` (sliders, command line)."""
    return replace(Growth(), **{**(entry or {}).get("growth", {}), **knobs})


@app.function
def sample(field, P):
    """Bilinear read of a (H, W) field at points P (N, 2) given as (x, y)."""
    return map_coordinates(field, [P[:, 1], P[:, 0]], order=1, mode="nearest")


@app.function
def node_spacing(r, g):
    """D, how far apart the nodes sit where the strands are r apart: the
    paper's D * delta, g.sample * r, capped in the lights so that open loops
    are still drawn as curves, not as polygons."""
    return g.sample * np.minimum(r, g.sample_cap * g.r_min)


@app.function
def seed_ring(g, center, r):
    """A small closed ring, sized from the spacing r at its centre: the curve
    every growth starts from."""
    a = np.linspace(0.0, 2.0 * np.pi, g.seed_nodes, endpoint=False)
    return np.column_stack([np.cos(a), np.sin(a)]) * g.seed_radius * r + center


@app.function
def arc_positions(P):
    """Arc length from node 0 to each node of the closed polyline, and the
    total length L. Two points lie on the same stretch of line when their arc
    distance min(|s_i - s_j|, L - |s_i - s_j|) is short."""
    e = np.hypot(*(np.roll(P, -1, axis=0) - P).T)
    s = np.concatenate([[0.0], np.cumsum(e[:-1])])
    return s, float(s[-1] + e[-1])  # same running sum as s, not e.sum()


@app.function
def lennard_jones(rho, g):
    """w(rho) = rho^-12 - rho^-6: pushes below rho = 1, pulls up to the
    cutoff. Bounded: closer than g.lj_floor pushes as hard as at lj_floor."""
    q = 1.0 / np.maximum(rho, g.lj_floor) ** 6
    return q * q - q


@app.function
def attraction_reference(P, r, s, L, g):
    """Attraction-repulsion A_i of Pedersen & Singh (2006), numpy reference.

    Node p_i feels every segment (p_j, p_j+1) through the closest point x_ij
    on it: f_ij = (p_i - x_ij)/|p_i - x_ij| * w(|p_i - x_ij| / r_ij) with
    r_ij = min(r_i, r(x_ij)), when |p_i - x_ij| < cutoff * r_ij and the
    segment does not end within n_min nodes of p_i. The paper scales w by
    r_i alone; then where dark meets light the two sides want different
    distances, the pushes are not mutual, and dark strands creep into the
    light as combs. With r_ij both sides agree. One more condition: the
    line must have folded back between p_i and x_ij, |p_i - x_ij| <
    fold * a with a their distance along the line. A straight or gently bent
    stretch (distance = arc) never pulls or pushes itself; the two sides of a
    U-turn do, and so does a kink. The paper needs no such rule because its
    nodes sit further apart than the cutoff. The zero of w sits at r: walls
    settle r apart.
    """
    n = len(P)
    nxt = np.roll(P, -1, axis=0)
    seg = nxt - P
    seglen2 = np.maximum((seg * seg).sum(1), 1e-300)
    reach = g.cutoff * g.r_max + np.sqrt(seglen2.max())
    i, j = cKDTree(P).query_pairs(reach, output_type="ndarray").T
    i, j = np.concatenate([i, j]), np.concatenate([j, i])  # node i, segment j
    t = np.clip(((P[i] - P[j]) * seg[j]).sum(1) / seglen2[j], 0.0, 1.0)
    x = P[j] + t[:, None] * seg[j]
    d = P[i] - x
    dist = np.hypot(d[:, 0], d[:, 1])
    rx = r[j] + t * (np.roll(r, -1)[j] - r[j])
    arc = np.abs(s[i] - (s[j] + t * np.sqrt(seglen2[j])))
    arc = np.minimum(arc, L - arc)
    gap = np.maximum(np.minimum(np.abs(j - i), n - np.abs(j - i)),
                     np.minimum(np.abs(j + 1 - i) % n, n - np.abs(j + 1 - i) % n))
    rp = np.minimum(r[i], rx)  # the scale of the pair, the same seen from both sides
    hit = (dist < g.cutoff * rp) & (dist > 1e-12)
    hit &= (gap > g.n_min) & (dist < g.fold * arc)
    i, d, dist, rp = i[hit], d[hit], dist[hit], rp[hit]
    f = (g.f_a * rp * lennard_jones(dist / rp, g) / dist)[:, None] * d
    return np.column_stack([np.bincount(i, f[:, k], n) for k in (0, 1)])


@app.function
@njit(cache=True)
def grid_bins(X, cell):
    """Points X (N, 2) sorted into a uniform grid of square cells (counting
    sort): the points of cell c are order[start[c]:start[c + 1]], and a
    point (x, y) falls in cell column (x - x0) // cell, row (y - y0) // cell."""
    n = X.shape[0]
    x0 = X[:, 0].min()
    y0 = X[:, 1].min()
    nx = int((X[:, 0].max() - x0) / cell) + 1
    ny = int((X[:, 1].max() - y0) / cell) + 1
    cx = np.empty(n, np.int64)
    cy = np.empty(n, np.int64)
    start = np.zeros(nx * ny + 1, np.int64)
    for i in range(n):
        cx[i] = int((X[i, 0] - x0) / cell)
        cy[i] = int((X[i, 1] - y0) / cell)
        start[cy[i] * nx + cx[i] + 1] += 1
    for c in range(nx * ny):
        start[c + 1] += start[c]
    fill = start.copy()
    order = np.empty(n, np.int64)
    for i in range(n):
        c = cy[i] * nx + cx[i]
        order[fill[c]] = i
        fill[c] += 1
    return x0, y0, nx, ny, cx, cy, start, order


@app.function
@njit(cache=True)
def segment_table(P):
    """For the kernels, in one pass: end point B, length e, midpoint M and
    arc position s of every segment (P[i], P[i+1]), and the total length L."""
    n = P.shape[0]
    B = np.empty_like(P)
    M = np.empty_like(P)
    e = np.empty(n)
    s = np.empty(n)
    acc = 0.0
    for i in range(n):
        j = i + 1 if i + 1 < n else 0
        B[i, 0] = P[j, 0]
        B[i, 1] = P[j, 1]
        M[i, 0] = 0.5 * (P[i, 0] + P[j, 0])
        M[i, 1] = 0.5 * (P[i, 1] + P[j, 1])
        e[i] = math.hypot(P[j, 0] - P[i, 0], P[j, 1] - P[i, 1])
        s[i] = acc
        acc += e[i]
    return B, e, M, s, acc


@app.function
@njit(cache=True, parallel=True)
def attraction_kernel(P, r, s, L, cell, cutoff, n_min, fold, lj_floor, f_a):
    """attraction_reference in one numba pass. Segments are binned by their
    midpoint on a grid of cell = cutoff * r_min. A segment can reach
    node i only if its midpoint is within cutoff * r_i + half its length,
    so node i scans the cells within cutoff * r_i + half the longest edge:
    a node in the darks looks at a few cells, one in the lights at more.
    Every node writes only its own force: threads never race and the sum
    does not depend on how many there are."""
    n = P.shape[0]
    _, e, M, _, _ = segment_table(P)
    half = 0.5 * e.max()
    x0, y0, nx, ny, _, _, start, order = grid_bins(M, cell)
    F = np.zeros((n, 2))
    for i in prange(n):
        reach = cutoff * r[i]
        span = int((reach + half) / cell) + 1
        ix = math.floor((P[i, 0] - x0) / cell)
        iy = math.floor((P[i, 1] - y0) / cell)
        fx = 0.0
        fy = 0.0
        for gy in range(max(iy - span, 0), min(iy + span + 1, ny)):
            for gx in range(max(ix - span, 0), min(ix + span + 1, nx)):
                c = gy * nx + gx
                for k in range(start[c], start[c + 1]):
                    j = order[k]
                    lim = reach + 0.5 * e[j]
                    if (P[i, 0] - M[j, 0]) ** 2 + (P[i, 1] - M[j, 1]) ** 2 > lim * lim:
                        continue
                    j1 = j + 1 if j + 1 < n else 0
                    g0 = abs(j - i)
                    g1 = abs(j1 - i)
                    if max(min(g0, n - g0), min(g1, n - g1)) <= n_min:
                        continue
                    sx = P[j1, 0] - P[j, 0]
                    sy = P[j1, 1] - P[j, 1]
                    l2 = max(sx * sx + sy * sy, 1e-300)
                    t = ((P[i, 0] - P[j, 0]) * sx + (P[i, 1] - P[j, 1]) * sy) / l2
                    t = min(max(t, 0.0), 1.0)
                    dx = P[i, 0] - (P[j, 0] + t * sx)
                    dy = P[i, 1] - (P[j, 1] + t * sy)
                    dist = math.hypot(dx, dy)
                    rx = r[j] + t * (r[j1] - r[j])
                    rp = min(r[i], rx)
                    if dist >= cutoff * rp or dist <= 1e-12:
                        continue
                    a = abs(s[i] - (s[j] + t * math.sqrt(l2)))
                    a = min(a, L - a)
                    if dist >= fold * a:
                        continue
                    q = 1.0 / max(dist / rp, lj_floor) ** 6
                    f = f_a * rp * (q * q - q) / dist
                    fx += f * dx
                    fy += f * dy
        F[i, 0] = fx
        F[i, 1] = fy
    return F


@app.function
def attraction(P, r, s, L, g):
    """The numba kernel when numba is there, the numpy reference otherwise
    (Pyodide, the notebook in the browser)."""
    if not HAVE_NUMBA:
        return attraction_reference(P, r, s, L, g)
    return attraction_kernel(
        P, r, s, L, g.cutoff * g.r_min, g.cutoff, g.n_min, g.fold, g.lj_floor, g.f_a
    )


@app.function
@njit(cache=True, parallel=True)
def crossings_kernel(P, cell, min_gap, arc_skip, only):
    """crossings() in one numba pass, as a flag per segment: True when the
    segment is in some bad pair. Segments are binned by midpoint; two
    segments can touch only if their midpoints are closer than half their
    lengths plus min_gap, so segment i scans the cells within e_i / 2 +
    longest edge / 2 + min_gap.
    Only the segments where `only` is True are tested (against all); both
    segments of a bad pair are flagged. Two threads may flag the same
    segment, but they both write True, so the result does not depend on
    the order."""
    n = P.shape[0]
    B, e, M, s, L = segment_table(P)
    emax = e.max()
    _, _, nx, ny, cx, cy, start, order = grid_bins(M, cell)
    bad = np.zeros(n, np.bool_)
    for i in prange(n):
        if not only[i]:
            continue
        ax, ay, bx, by = P[i, 0], P[i, 1], B[i, 0], B[i, 1]
        span = int((0.5 * (e[i] + emax) + min_gap) / cell) + 1
        for gy in range(max(cy[i] - span, 0), min(cy[i] + span + 1, ny)):
            for gx in range(max(cx[i] - span, 0), min(cx[i] + span + 1, nx)):
                c = gy * nx + gx
                for k in range(start[c], start[c + 1]):
                    j = order[k]
                    gap = abs(i - j)
                    if gap == 0:
                        continue
                    nextdoor = min(gap, n - gap) == 1  # share a node
                    reach = 0.5 * (e[i] + e[j]) + min_gap
                    if (M[i, 0] - M[j, 0]) ** 2 + (M[i, 1] - M[j, 1]) ** 2 > reach * reach:
                        continue
                    qx, qy, wx, wy = P[j, 0], P[j, 1], B[j, 0], B[j, 1]
                    o1 = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
                    o2 = (bx - ax) * (wy - ay) - (by - ay) * (wx - ax)
                    o3 = (wx - qx) * (ay - qy) - (wy - qy) * (ax - qx)
                    o4 = (wx - qx) * (by - qy) - (wy - qy) * (bx - qx)
                    if not nextdoor and o1 * o2 < 0 and o3 * o4 < 0:
                        bad[i] = True
                        bad[j] = True
                        continue
                    touch = np.inf if nextdoor else min(
                        segment_distance(ax, ay, qx, qy, wx, wy),
                        segment_distance(bx, by, qx, qy, wx, wy),
                        segment_distance(qx, qy, ax, ay, bx, by),
                        segment_distance(wx, wy, ax, ay, bx, by),
                    )
                    if touch < 1e-9:
                        bad[i] = True
                        bad[j] = True
                        continue
                    dist = folded_gap(ax, ay, bx, by, qx, qy, wx, wy,
                                      s[i], s[j], e[i], e[j], L, arc_skip)
                    if dist < min_gap:
                        bad[i] = True
                        bad[j] = True
                        continue
    return bad


@app.function
@njit(cache=True)
def segment_distance(px, py, ux, uy, vx, vy):
    """Distance from point p to segment (u, v), for crossings_kernel."""
    wx = vx - ux
    wy = vy - uy
    t = ((px - ux) * wx + (py - uy) * wy) / max(wx * wx + wy * wy, 1e-300)
    t = min(max(t, 0.0), 1.0)
    return math.hypot(ux + t * wx - px, uy + t * wy - py)


@app.function
def violating_segments(P, g, only=None):
    """Indices of the segments that break the promise of the title:
    crossing, touching, or closer than g.gap * r_min where the line has
    folded back on itself (more than r_min apart along it). None, after
    every step. `only`: test just these segments (default all); the
    segments they clash with are returned too."""
    if not HAVE_NUMBA:
        pairs = crossings(P, g.gap * g.r_min, g.r_min)
        if only is not None:
            pairs = pairs[np.isin(pairs, only).any(1)]
        return np.unique(pairs)
    mask = np.ones(len(P), bool)
    if only is not None:
        mask[:] = False
        mask[only] = True
    cell = g.k_max * g.sample * g.r_min
    return np.flatnonzero(crossings_kernel(P, cell, g.gap * g.r_min, g.r_min, mask))


@app.function
def guard(P_old, P_new, g):
    """Put back to its old place both ends of every segment that breaks the
    promise (violating_segments), until none is left. P_old was clean, so
    this always ends: at worst the step does not move those nodes at all."""
    n = len(P_new)
    P_new = P_new.copy()
    reverted = np.zeros(n, bool)
    bad = violating_segments(P_new, g)
    while len(bad):
        idx = np.unique(np.concatenate([bad, bad + 1]) % n)
        idx = idx[~reverted[idx]]
        if not len(idx):  # cannot happen from a clean P_old; last resort
            return P_old.copy()
        P_new[idx] = P_old[idx]
        reverted[idx] = True
        # only the segments next to the nodes just put back have changed
        bad = violating_segments(P_new, g, np.unique(np.concatenate([idx - 1, idx]) % n))
    return P_new


@app.function
def resample(P, r, g):
    """Keep the nodes about D = node_spacing(r) apart (Pedersen & Singh):
    delete a node closer than k_min * D to the one before it, then split
    every edge longer than k_max * D at its midpoint. A split never changes
    the shape; a deletion cuts a corner, so it is dropped if the cut would
    break the promise (violating_segments). r: the spacing at each node."""
    e = np.hypot(*(P - np.roll(P, 1, axis=0)).T)  # edge ending at node i
    close = e < g.k_min * node_spacing(0.5 * (r + np.roll(r, 1)), g)
    close &= ~np.roll(close, 1)  # never two neighbours at once
    if close.any() and len(P) - close.sum() >= 3:
        # a deleted node i leaves the segment that starts at the node before
        # it; undo the deletions whose new segment breaks the promise
        merged = np.cumsum(~close) - 1
        kept = P[~close]
        flag = np.zeros(len(P), bool)
        flag[violating_segments(kept, g, merged[close])] = True
        bad = flag[merged] & close
        if bad.any():
            close &= ~bad
            kept = P[~close]
            if len(violating_segments(kept, g, (np.cumsum(~close) - 1)[close])):
                close[:] = False
                kept = P
        P, r = kept, r[~close]
    nxt = np.roll(P, -1, axis=0)
    e = np.hypot(*(nxt - P).T)
    long = np.flatnonzero(e > g.k_max * node_spacing(0.5 * (r + np.roll(r, -1)), g))
    long = long[: max(g.max_nodes - len(P), 0)]
    if len(long):
        P = np.insert(P, long + 1, 0.5 * (P[long] + nxt[long]), axis=0)
    return P


@app.function
def move_reference(P, F, r, Z, shape, g):
    """The rest of the update, numpy reference: add fairing, Brownian motion
    (Z: standard normal, (N, 2)) and the frame walls to the attraction F,
    cap each move, keep the nodes in the frame. Returns the moved nodes."""
    prv, nxt = np.roll(P, 1, axis=0), np.roll(P, -1, axis=0)
    rp, rn = np.roll(r, 1)[:, None], np.roll(r, -1)[:, None]
    F = F + g.f_f * ((prv * rn + nxt * rp) / (rp + rn) - P)

    H, W = shape
    for k, hi in ((0, W - 1.0), (1, H - 1.0)):
        for dist, sign in ((P[:, k], 1.0), (hi - P[:, k], -1.0)):
            near = (dist < g.cutoff * r) & (dist > 1e-12)
            F[near, k] += sign * g.f_a * r[near] * lennard_jones(dist[near] / r[near], g)

    F = F + g.f_b * node_spacing(r, g)[:, None] * Z
    e = np.hypot(*(nxt - P).T)
    cap = np.minimum(g.max_move * r, g.edge_move * np.minimum(e, np.roll(e, 1)))
    F *= np.minimum(1.0, cap / np.maximum(np.hypot(F[:, 0], F[:, 1]), 1e-12))[:, None]
    Q = P + F
    Q[:, 0] = np.clip(Q[:, 0], 0.0, W - 1.0)
    Q[:, 1] = np.clip(Q[:, 1], 0.0, H - 1.0)
    return Q


@app.function
@njit(cache=True, parallel=True)
def move_kernel(P, F, r, D, Z, W, H, f_f, f_b, cutoff, f_a, lj_floor, max_move, edge_move):
    """move_reference in one numba pass, node by node."""
    n = P.shape[0]
    Q = np.empty_like(P)
    for i in prange(n):
        ip = i - 1 if i > 0 else n - 1
        jn = i + 1 if i + 1 < n else 0
        rp = r[ip]
        rn = r[jn]
        fx = F[i, 0] + f_f * ((P[ip, 0] * rn + P[jn, 0] * rp) / (rp + rn) - P[i, 0])
        fy = F[i, 1] + f_f * ((P[ip, 1] * rn + P[jn, 1] * rp) / (rp + rn) - P[i, 1])
        for k in range(2):
            hi = W - 1.0 if k == 0 else H - 1.0
            for side in range(2):
                dist = P[i, k] if side == 0 else hi - P[i, k]
                if dist < cutoff * r[i] and dist > 1e-12:
                    q = 1.0 / max(dist / r[i], lj_floor) ** 6
                    w = (1.0 if side == 0 else -1.0) * f_a * r[i] * (q * q - q)
                    if k == 0:
                        fx += w
                    else:
                        fy += w
        fx += f_b * D[i] * Z[i, 0]
        fy += f_b * D[i] * Z[i, 1]
        e_out = math.hypot(P[jn, 0] - P[i, 0], P[jn, 1] - P[i, 1])
        e_in = math.hypot(P[i, 0] - P[ip, 0], P[i, 1] - P[ip, 1])
        cap = min(max_move * r[i], edge_move * min(e_out, e_in))
        m = max(math.hypot(fx, fy), 1e-12)
        c = min(1.0, cap / m)
        Q[i, 0] = min(max(P[i, 0] + fx * c, 0.0), W - 1.0)
        Q[i, 1] = min(max(P[i, 1] + fy * c, 0.0), H - 1.0)
    return Q


@app.function
def move(P, F, r, Z, shape, g):
    """The numba kernel when numba is there, the numpy reference otherwise."""
    if not HAVE_NUMBA:
        return move_reference(P, F, r, Z, shape, g)
    H, W = shape
    return move_kernel(
        P, F, r, node_spacing(r, g), Z, float(W), float(H), g.f_f, g.f_b, g.cutoff, g.f_a,
        g.lj_floor, g.max_move, g.edge_move,
    )


@app.class_definition
class Flow:
    """A slow, fluid drift for the finished line (the last seconds of a
    video): the curl of a stream function made of g.flow_waves plane waves,
    each g.flow_scale * r_max long, drifting in phase over g.flow_period
    steps. A curl has no divergence, so the strands slide past each other
    like a liquid instead of bunching up; and it changes smoothly in space
    and time, so the line flows instead of trembling. Waves from `rng`."""

    def __init__(self, g, rng):
        n = g.flow_waves
        a = rng.uniform(0.0, 2.0 * np.pi, n)
        self.k = (2.0 * np.pi / (g.flow_scale * g.r_max)) * np.column_stack([np.cos(a), np.sin(a)])
        self.phase = rng.uniform(0.0, 2.0 * np.pi, n)
        self.omega = (2.0 * np.pi / g.flow_period) * rng.choice([-1.0, 1.0], n)
        # each wave's stream function has amplitude 1 / |k|, so its speed is
        # 1; dividing by sqrt(n) keeps the typical speed at g.flow * r_min
        self.speed = g.flow * g.r_min / np.sqrt(n)
        self.t = 0

    def field(self, P, t):
        """The flow at the points P and time t (steps), of unit typical size."""
        c = np.cos(P @ self.k.T + self.phase + self.omega * t)  # (N, waves)
        unit = self.k / np.hypot(*self.k.T)[:, None]
        # v = (d psi / dy, -d psi / dx) for psi = sum sin(k.x + ...) / |k|
        return np.column_stack([c @ unit[:, 1], -(c @ unit[:, 0])]) / np.sqrt(len(self.k))

    def __call__(self, P):
        """Displacement (N, 2) of the nodes P for this step; advances time."""
        self.t += 1
        return self.speed * np.sqrt(len(self.k)) * self.field(P, self.t - 1)

    def max_offset(self):
        """The largest amplitude offset() takes with the map x -> x + offset
        staying one-to-one, so a curve that does not cross itself still does
        not: the field's gradient is at most |k| sqrt(waves) per unit of
        amplitude, and the map is one-to-one while amplitude * gradient < 1.
        Half of that, so strands are never squeezed by more than half."""
        return 0.5 / (np.hypot(*self.k[0]) * np.sqrt(len(self.k)))

    def offset(self, P, t, amplitude):
        """P moved by the flow field itself, `amplitude` long (capped at
        max_offset()): a slow, fluid sway of the whole drawing that needs no
        simulation, for the frames of a growth already computed."""
        return P + min(amplitude, self.max_offset()) * self.field(P, t)


@app.function
def step(P, R, g, rng, attract=attraction, move=move, drift=None):
    """One step of the labyrinth growth of Pedersen & Singh (2006):
    p_i' = p_i + B_i + F_i + A_i, then resampling. `drift`, if given (a
    Flow), adds its displacement to the forces before the moves are capped.

    B  Brownian motion, f_b * z * D: roughens the line, which is what makes
       it longer; the resampling turns the extra length into new nodes
    F  fairing, f_f times the pull towards the r-weighted mean of the two
       neighbours (weighted so nodes do not drift towards coarse regions)
    A  attraction-repulsion between the line and every other stretch of
       itself (attraction_reference): walls settle r apart and stay packed
    The frame edge acts as one more wall (move_reference). No node moves
    further than g.max_move * r in one step, nor further than g.edge_move
    times its shorter edge: with many walls near, the pushes add up, and a
    node that jumps further than its own edges can fold the line over itself.
    Last, guard() undoes any move that would make the line cross or touch
    itself: the forces keep that rare, the guard makes it impossible.
    """
    r = sample(R, P)
    s, L = arc_positions(P)
    F = attract(P, r, s, L, g)
    if drift is not None:
        F = F + drift(P)
    Z = rng.standard_normal((len(P), 2))
    Q = move(P, F, r, Z, R.shape, g)
    return resample(guard(P, Q, g), r, g)


@app.function
def step_reference(P, R, g, rng):
    """step() with the numpy attraction and move: the yardstick for the
    numba kernels."""
    return step(P, R, g, rng, attract=attraction_reference, move=move_reference)


@app.function
def grow(R, g, steps, every=10, seed=0, center=None, on_step=None, log=None):
    """Run the growth. Returns the list of snapshots, one every `every` steps.

    The first snapshot is the seed ring, the last is the final curve. The
    growth stops before `steps` once it has settled (Growth.settle).
    `center` is in normalised (x, y) of the tone map; default the middle.
    `log`, if a list, gets (step, nodes) for every step.
    """
    rng = np.random.default_rng(seed)
    H, W = R.shape
    cx, cy = center or (0.5, 0.5)
    c = np.array([cx * (W - 1), cy * (H - 1)])
    P = seed_ring(g, c, sample(R, c[None])[0])
    snaps = [P.copy()]
    counts = [len(P)]
    for s in range(1, steps + 1):
        P = step(P, R, g, rng)
        counts.append(len(P))
        if log is not None:
            log.append((s, counts[-1]))
        settled = (
            g.settle_tol > 0
            and s >= g.settle
            and counts[-1] - counts[-1 - g.settle] <= g.settle_tol * counts[-1]
        )
        if s % every == 0 or s == steps or settled:
            snaps.append(P.copy())
        if on_step:
            on_step(s, P)
        if settled:
            break
    return snaps


@app.function
@njit(cache=True)
def point_to_run(px, py, ox, oy, dx, dy, t0, t1):
    """Distance from p to the run o + t d, t in [t0, t1] (d a unit vector);
    inf if the run is empty."""
    if t0 > t1:
        return np.inf
    t = min(max((px - ox) * dx + (py - oy) * dy, t0), t1)
    return math.hypot(ox + t * dx - px, oy + t * dy - py)


@app.function
@njit(cache=True)
def folded_gap(ax, ay, bx, by, qx, qy, wx, wy, si, sj, ei, ej, L, skip):
    """Smallest distance between a point x of segment i (a to b) and a point
    y of segment j (q to w) that lie more than `skip` apart along the line;
    inf if there is no such pair. s, e: arc position and length of each.

    Put the segment that comes first (the short way round) as i, with a0 the
    arc between them. With x a distance u back from b and y a distance v on
    from q, the arc is a0 + u + v, so the allowed pairs fill the box
    [0, ei] x [0, ej] cut by u + v > c = skip - a0. The squared distance is
    convex in (u, v) and the two segments do not cross (checked first), so
    its minimum lies on the border: four point-to-run distances and the cut.
    """
    a0 = (sj - si - ei) % L
    if L - a0 < 1e-9 * L:  # neighbours: 0 that rounded to L
        a0 = 0.0
    if a0 > 0.5 * L:  # j comes first: swap
        ax, ay, bx, by, qx, qy, wx, wy = qx, qy, wx, wy, ax, ay, bx, by
        si, sj, ei, ej = sj, si, ej, ei
        a0 = (sj - si - ei) % L
        if L - a0 < 1e-9 * L:
            a0 = 0.0
    c = skip - a0
    if c >= ei + ej:
        return np.inf
    dix = (bx - ax) / max(ei, 1e-300)
    diy = (by - ay) / max(ei, 1e-300)
    djx = (wx - qx) / max(ej, 1e-300)
    djy = (wy - qy) / max(ej, 1e-300)
    best = min(
        point_to_run(ax, ay, qx, qy, djx, djy, max(c - ei, 0.0), ej),  # u = ei
        point_to_run(wx, wy, bx, by, -dix, -diy, max(c - ej, 0.0), ei),  # v = ej
        point_to_run(bx, by, qx, qy, djx, djy, max(c, 0.0), ej),  # u = 0
        point_to_run(qx, qy, bx, by, -dix, -diy, max(c, 0.0), ei),  # v = 0
    )
    u0 = max(c - ej, 0.0)
    u1 = min(c, ei)
    if c > 0.0 and u0 <= u1:  # the cut u + v = c
        mx = bx - qx - c * djx
        my = by - qy - c * djy
        kx = dix - djx
        ky = diy - djy
        t = min(max((mx * kx + my * ky) / max(kx * kx + ky * ky, 1e-300), u0), u1)
        best = min(best, math.hypot(mx - t * kx, my - t * ky))
    return best


@app.function
def folded_gap_np(a, b, q, w, si, sj, ei, ej, L, skip):
    """folded_gap for arrays of segment pairs (a, b, q, w: (m, 2)), numpy."""
    def ahead(si, sj, ei):
        a0 = np.mod(sj - si - ei, L)
        return np.where(L - a0 < 1e-9 * L, 0.0, a0)  # neighbours: 0 that rounded to L

    swap = ahead(si, sj, ei) > 0.5 * L
    sw = swap[:, None]
    a, b, q, w = np.where(sw, q, a), np.where(sw, w, b), np.where(sw, a, q), np.where(sw, b, w)
    si, sj, ei, ej = (np.where(swap, sj, si), np.where(swap, si, sj),
                      np.where(swap, ej, ei), np.where(swap, ei, ej))
    c = skip - ahead(si, sj, ei)
    di = (b - a) / np.maximum(ei, 1e-300)[:, None]
    dj = (w - q) / np.maximum(ej, 1e-300)[:, None]

    def run(p, o, d, t0, t1):
        t = np.clip(((p - o) * d).sum(1), t0, t1)
        out = np.hypot(*(o + t[:, None] * d - p).T)
        return np.where(t0 > t1, np.inf, out)

    zero = np.zeros_like(c)
    best = np.min([
        run(a, q, dj, np.maximum(c - ei, zero), ej),
        run(w, b, -di, np.maximum(c - ej, zero), ei),
        run(b, q, dj, np.maximum(c, zero), ej),
        run(q, b, -di, np.maximum(c, zero), ei),
    ], axis=0)
    u0, u1 = np.maximum(c - ej, zero), np.minimum(c, ei)
    m = b - q - c[:, None] * dj
    k = di - dj
    t = np.clip((m * k).sum(1) / np.maximum((k * k).sum(1), 1e-300), u0, u1)
    cut = np.hypot(*(m - t[:, None] * k).T)
    best = np.where((c > 0) & (u0 <= u1), np.minimum(best, cut), best)
    return np.where(c >= ei + ej, np.inf, best)


@app.function
def crossings(P, min_gap=0.0, arc_skip=0.0):
    """Pairs of segments (i, j) of the closed polyline that cross, touch, or
    hold two points closer than min_gap that are more than arc_skip apart
    along the line (folded_gap: exact, so splitting a segment in two
    changes nothing). Neighbouring segments share a node, so for them only
    the gap counts: a needle-sharp turn is a fold too.
    Empty for a valid curve: one line, closed, never touching itself.

    Candidates from a cKDTree on the segment midpoints, then an exact test:
    orientation signs for a crossing, point-segment distances for the gap;
    two segments that touch (a crossing through a node) count too.
    """
    n = len(P)
    A = P
    B = np.roll(P, -1, axis=0)
    e = np.hypot(*(B - A).T)
    i, j = cKDTree(0.5 * (A + B)).query_pairs(e.max() + min_gap, output_type="ndarray").T
    gap = np.abs(i - j)
    nextdoor = np.minimum(gap, n - gap) == 1  # share a node: they always touch
    a, b, c, d = A[i], B[i], A[j], B[j]

    def orient(p, q, t):
        return (q - p)[:, 0] * (t - p)[:, 1] - (q - p)[:, 1] * (t - p)[:, 0]

    cross = (orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0)

    def to_segment(p, u, v):
        w = v - u
        t = np.clip(((p - u) * w).sum(1) / np.maximum((w * w).sum(1), 1e-300), 0.0, 1.0)
        return np.hypot(*(u + t[:, None] * w - p).T)

    touch = np.min([to_segment(a, c, d), to_segment(b, c, d), to_segment(c, a, b),
                    to_segment(d, a, b)], axis=0) < 1e-9
    s, L = arc_positions(P)
    near = folded_gap_np(a, b, c, d, s[i], s[j], e[i], e[j], L, arc_skip) < min_gap
    bad = ((cross | touch) & ~nextdoor) | near
    return np.column_stack([i[bad], j[bad]])


@app.function
def polish(P, g):
    """The curve as it is drawn: g.polish passes of smoothing, each node
    moving g.polish_f of the way to the mean of its two neighbours, then
    guard() against the curve as grown, which was clean. The growth keeps
    its Brownian tremor (that is what makes it grow); this irons out the
    node-scale wobble it leaves and barely touches the folds, which are
    many nodes wide."""
    Q = P
    for _ in range(g.polish):
        Q = Q + g.polish_f * (0.5 * (np.roll(Q, 1, axis=0) + np.roll(Q, -1, axis=0)) - Q)
    return guard(P, Q, g)


@app.function
def nearest_on_curve(X, Q, tree):
    """For each point of X, the nearest point on the closed polyline Q (its
    nearest node's two segments) and the distance to it. tree: cKDTree(Q)."""
    _, j = tree.query(X)
    n = len(Q)
    best, dist = None, None
    for a, b in ((Q[j - 1], Q[j]), (Q[j], Q[(j + 1) % n])):
        w = b - a
        t = np.clip(((X - a) * w).sum(1) / np.maximum((w * w).sum(1), 1e-300), 0.0, 1.0)
        Y = a + t[:, None] * w
        d = np.hypot(*(X - Y).T)
        if best is None:
            best, dist = Y, d
        else:
            closer = d < dist
            best[closer], dist[closer] = Y[closer], d[closer]
    return best, dist


@app.function
def steady(P, others, R, g):
    """A video frame without the shiver. In the packed labyrinth each strand
    jitters around a place its neighbours hold it to, a little differently
    every frame; the growth needs that jitter, the eye does not. So each node
    of P is averaged with the nearest point of the curve in the frames
    around it (`others`: [(curve, cKDTree, weight)]), wherever that point is
    within g.steady_reach * r: the growing front and anything new, further
    off, keeps its own position. Then guard() against P, which was clean."""
    r = sample(R, P)
    acc = P.copy()
    tot = np.ones(len(P))
    for Q, tree, w in others:
        Y, d = nearest_on_curve(P, Q, tree)
        near = d < g.steady_reach * r
        acc[near] += w * Y[near]
        tot[near] += w
    return guard(P, acc / tot[:, None], g)


@app.function
def curve_length(P):
    """Length of the closed polyline, in work units."""
    return float(np.hypot(*(np.roll(P, -1, axis=0) - P).T).sum())


@app.function
def pack_growth(curves, shape, g, quantum=32):
    """A growth (a list of closed curves) as a small compressed file, bytes.
    Every point is rounded to 1 / quantum px and stored as the step from
    the one before, which is a small integer: the file shrinks to a few
    MB for a full painting. `g` (the Growth) travels with it, so the curves
    are drawn with the same pen and smoothing they were grown with."""
    import json
    first, steps, off = [], [], [0]
    for P in curves:
        q = np.round(np.asarray(P) * quantum).astype(np.int64)
        d = np.diff(q, axis=0)
        assert np.abs(d).max() < 2**15, "points too far apart for int16 steps"
        first.append(q[0])
        steps.append(d.astype(np.int16))
        off.append(off[-1] + len(q))
    buf = BytesIO()
    np.savez_compressed(buf, first=np.array(first), steps=np.concatenate(steps),
                        offsets=np.array(off), shape=np.array(shape), quantum=quantum,
                        growth=json.dumps(asdict(g)))
    return buf.getvalue()


@app.function
def load_growth(name):
    """A growth packed by pack_growth, read from public/<name>, as
    (curves, shape, Growth); None if there is no such file (in the browser
    a missing file can come back as an error page, so anything unreadable
    counts as missing)."""
    try:
        return unpack_growth(public_bytes(name))
    except Exception:
        return None


@app.function
def unpack_growth(data):
    """The bytes of pack_growth back to (curves, shape, Growth)."""
    import json
    z = np.load(BytesIO(data))
    first, steps, off, quantum = z["first"], z["steps"].astype(np.int64), z["offsets"], int(z["quantum"])
    curves = []
    for k in range(len(off) - 1):
        d = steps[off[k] - k:off[k + 1] - k - 1]  # each curve has one step fewer than points
        q = np.concatenate([first[k][None], first[k] + np.cumsum(d, axis=0)])
        curves.append(q / quantum)
    return curves, tuple(int(v) for v in z["shape"]), Growth(**json.loads(str(z["growth"])))


@app.function
def draw(ax, P, shape, pal=PAPER, width=None, lw=None):
    """The one way the line is drawn: notebook, stills and video frames.

    `shape` is the tone map's (H, W); y points down, as in the image.
    `width`: line width in tone-map px, so it scales with the picture (use
    INK * r_min); `lw` in points overrides it, for thumbnails. Alpha 1: the
    line never overlaps itself, so tone comes only from the spacing.
    """
    H, W = shape
    Q = np.vstack([P, P[:1]])
    ax.set_facecolor(pal["bg"])
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    if lw is None:
        box = ax.get_window_extent()  # display px at the figure's dpi
        per_px = min(box.width / (W - 1), box.height / (H - 1))
        lw = (width if width is not None else 1.0) * per_px * 72.0 / ax.figure.dpi
    ax.plot(Q[:, 0], Q[:, 1], color=pal["ink"], lw=lw, solid_joinstyle="round")


@app.function
def ink_cmap(pal):
    """Coverage 0 -> the ground, 1 -> the ink: the only greys there are."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ink", [pal["bg"], pal["ink"]])


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _(mo):
    # the palette follows the page's theme; on the dark one the ink is light,
    # so the line must be dense where the picture is LIGHT (Growth.night)
    PAL = NIGHT if mo.app_meta().theme == "dark" else PAPER
    night = PAL is NIGHT

    def public_src(name):
        """A file of public/ for mo.image: a path here, a URL in the browser."""
        loc = mo.notebook_location() / "public" / name
        return loc if isinstance(loc, Path) and loc.exists() else str(loc)

    return PAL, night, public_src


@app.cell
def _(mo, public_src):
    mo.md(rf"""
    # Unbroken Line

    ### a painting drawn by one line that grows and never crosses itself

    {mo.image(public_src("growth.gif"), width=360, style={"margin": "0 auto"})}

    Take a small ring of thread and let it grow. Every point on it trembles
    a little, is pulled smooth by its two neighbours, and is pushed away from
    any other part of the thread that comes too close. The ring gets longer,
    buckles, folds into a labyrinth and fills the page. It never crosses
    itself.

    Now read "too close" off a painting: a couple of pixels in the shadows,
    more in the lights. The thread packs tight where the painting is dark and
    stays loose where it is bright, and the picture appears in the gaps. From
    start to finish it is still one closed line.

    $$r(\mathbf x) = r_{{\min}} + (r_{{\max}} - r_{{\min}})\,T(\mathbf x)^{{\gamma}}$$

    The method is the labyrinth growth of H. Pedersen and K. Singh,
    [*Organic Labyrinths and Mazes*](https://www.dgp.toronto.edu/~karan/artexhibit/mazes.pdf)
    (NPAR 2006). Try it first: pick a painting and watch it grow. Then the
    sections below take the method apart one idea at a time. No maths needed
    beyond "push" and "pull".
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Try it

    Pick a painting, or drop in your own picture. Three of them were grown
    in advance, on a map 600 pixels wide and in the palette they were made
    for: drag the slider and watch the line fill the frame.
    """)
    return


@app.cell
def _(mo):
    catalog = load_catalog()
    art = mo.ui.dropdown(
        options={a["title"]: s for s, a in catalog.items()},
        value=next(iter(catalog.values()))["title"],
        label="painting",
    )
    upload = mo.ui.file(filetypes=["image/*"], label="or your own picture")
    mo.hstack([art, upload], justify="start")
    return art, catalog, upload


@app.cell
def _(art, catalog, upload):
    # the picture, its credit line, and the start of the ring
    if upload.value:
        grey = load_grey(upload.contents())
        entry = {"title": upload.name(), "start": [0.5, 0.5]}
        credit = upload.name()
    else:
        entry = catalog[art.value]
        grey = load_image(art.value)
        credit = f"**{entry['title']}**, {entry['artist']}, {entry['year']}, {entry['where']}"
    return credit, entry, grey


@app.cell
def _(art, upload):
    # grown in advance by figures.py; None for your own picture
    ready = None if upload.value else load_growth(f"growth/{art.value}.npz")
    return (ready,)


@app.cell
def _(mo, ready):
    if ready:
        stage = mo.ui.slider(0, len(ready[0]) - 1, value=len(ready[0]) // 2, label="growth",
                             full_width=True)
        _out = stage
    else:
        stage = None
        _out = mo.md("_Nothing grown in advance for this picture: grow it yourself below._")
    _out
    return (stage,)


@app.cell
def _(mo, plt, ready, stage):
    mo.stop(ready is None)
    _curves, _shape, _g = ready
    _pal = NIGHT if _g.night else PAPER
    _P = _curves[stage.value]  # already smoothed for drawing (polish)
    _fig = plt.figure(figsize=(5.4, 5.4 * _shape[0] / _shape[1]), facecolor=_pal["bg"])
    draw(_fig.add_axes([0, 0, 1, 1]), _P, _shape, _pal, width=INK * _g.r_min)
    mo.vstack([
        _fig,
        mo.md(f"{len(_P):,} points · line {curve_length(_P):,.0f} px long "
              f"on a {_shape[1]} px map"),
    ], align="center")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Tune the tone

    How close the strands may sit is read off the picture: tight in the
    darks, loose in the lights (section 7 tells how). Move the sliders and
    the map changes; the circle is where the ring starts.
    """)
    return


@app.cell
def _(entry, mo):
    _g = params(entry)
    r_min = mo.ui.slider(1.0, 6.0, step=0.1, value=_g.r_min, label="r min", show_value=True)
    r_max = mo.ui.slider(4.0, 24.0, step=0.5, value=_g.r_max, label="r max", show_value=True)
    gamma = mo.ui.slider(0.5, 4.0, step=0.1, value=_g.gamma, label="gamma", show_value=True)
    local = mo.ui.slider(0.0, 2.0, step=0.1, value=_g.local, label="local contrast", show_value=True)
    blur = mo.ui.slider(0.0, 3.0, step=0.25, value=_g.blur, label="blur", show_value=True)
    start_x = mo.ui.slider(0.0, 1.0, step=0.01, value=entry["start"][0], label="start x", show_value=True)
    start_y = mo.ui.slider(0.0, 1.0, step=0.01, value=entry["start"][1], label="start y", show_value=True)
    mo.vstack([mo.hstack([r_min, r_max, gamma]), mo.hstack([local, blur]), mo.hstack([start_x, start_y])])
    return blur, gamma, local, r_max, r_min, start_x, start_y


@app.cell
def _(blur, entry, gamma, grey, local, night, r_max, r_min, width):
    g = params(
        entry, width=width.value, r_min=r_min.value, r_max=r_max.value, gamma=gamma.value,
        local=local.value, blur=blur.value, night=night,
    )
    R = spacing_field(grey, g)
    return R, g


@app.cell
def _(PAL, R, credit, g, grey, mo, night, plt, start_x, start_y):
    _fig, _ax = plt.subplots(1, 3, figsize=(10, 4.4), facecolor=PAL["bg"])
    _cmap = ink_cmap(PAL)
    # every panel as ink coverage, so all three read the same way
    _ax[0].imshow(grey if night else 1 - grey, cmap=_cmap, vmin=0, vmax=1)
    _ax[1].imshow((g.r_max - R) / (g.r_max - g.r_min), cmap=_cmap, vmin=0, vmax=1)
    # the tone the line will make: pen width over strand spacing
    _ax[2].imshow(np.clip(INK * g.r_min / (PACKING * R), 0, 1), cmap=_cmap, vmin=0, vmax=1)
    _h, _w = grey.shape
    _ax[0].plot(start_x.value * _w, start_y.value * _h, "o", ms=7, mfc="none", mec=PAL["ink"], mew=1.2)
    for _a, _t in zip(_ax, ["picture", "spacing r(x): dark = tight", "the tone the line will make"]):
        _a.set_title(_t, color=PAL["ink"], fontsize=9)
        _a.axis("off")
    _fig.tight_layout()
    mo.vstack([mo.md(credit + "  · the circle is where the ring starts"), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Grow it yourself

    The map above, with your sliders, grown here and now. The width is the
    size of the map in pixels: the bigger, the finer the line and the longer
    the wait. On a laptop 350 takes under a minute and 600 a few minutes.
    """)
    return


@app.cell
def _(mo):
    width = mo.ui.slider(100, 1200, step=25, value=350, label="width", show_value=True)
    steps = mo.ui.slider(500, 10000, step=500, value=6000, label="at most steps", show_value=True)
    run = mo.ui.run_button(label="grow")
    # In the browser (Pyodide) there are no compiled kernels and one core:
    # a whole picture would take the best part of an hour, so growing is
    # left to the notebook run on a computer.
    if HAVE_NUMBA:
        _out = mo.hstack([width, steps, run], justify="start")
    else:
        _out = mo.callout(mo.md(
            "This page runs in your browser, without the compiled code that makes the "
            "growth fast, so growing a whole picture here would take the best part of "
            "an hour. Run the notebook on your computer to grow your own: "
            "`uv run marimo edit unbroken_line.py` in a copy of the "
            "[repository](https://github.com/monex00/unbroken-line)."), kind="info")
    _out
    return run, steps, width


@app.cell
def _(R, g, mo, run, start_x, start_y, steps):
    mo.stop(not HAVE_NUMBA)
    mo.stop(not run.value, mo.md("Press **grow**."))
    with mo.status.progress_bar(total=steps.value, title="growing", remove_on_exit=True) as _bar:
        snaps = grow(R, g, steps.value, every=50, center=(start_x.value, start_y.value),
                     on_step=lambda s, P: _bar.update())
    return (snaps,)


@app.cell
def _(mo, snaps):
    frame = mo.ui.slider(0, len(snaps) - 1, value=len(snaps) - 1, label="snapshot", show_value=True)
    frame
    return (frame,)


@app.cell
def _(PAL, R, frame, g, mo, plt, snaps):
    _P = snaps[frame.value]
    _fig = plt.figure(figsize=(5, 5 * R.shape[0] / R.shape[1]), facecolor=PAL["bg"])
    draw(_fig.add_axes([0, 0, 1, 1]), polish(_P, g), R.shape, PAL, width=INK * g.r_min)
    mo.vstack([
        _fig,
        mo.md(f"{len(_P):,} points · line {curve_length(_P):,.0f} px long on a {g.width} px map"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How it works

    Seven ideas, one at a time, most with a slider of their own: a line,
    three nudges that make it grow, a rule that keeps it from crossing, and
    a painting that tells it where to pack tight.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. One line

    The line is nothing but a list of points, $\mathbf p_0, \mathbf p_1,
    \dots, \mathbf p_{n-1}$, each joined to the next by a straight piece and
    the last joined back to the first. It is drawn with a pen of one fixed
    width, in one colour, with no shading. Every grey you will see comes from
    one thing: how close together its strands run.

    It starts as a small ring of 20 points. Then, over and over, every point
    takes a step made of three small nudges,

    $$\mathbf p_i \;\leftarrow\; \mathbf p_i + \mathbf B_i + \mathbf F_i + \mathbf A_i$$

    and the line is tidied up: points are added where it has stretched,
    removed where they bunch up. The next four sections take the three
    nudges and the tidying one at a time.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Tremble

    The first nudge is pure chance. Every point is kicked a short distance in
    a random direction:

    $$\mathbf B_i = f_B\, D\, \mathbf z_i, \qquad \mathbf z_i \sim \mathcal N(0, 1)$$

    $D$ is the distance between neighbouring points, so the kick is a small
    fraction $f_B$ of it (0.1 here). This is the engine. A thread that
    wobbles is longer than a straight one, and in section 5 that extra length
    turns into new points. With $f_B = 0$ the ring stays a ring.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Smooth

    The second nudge pulls every point a little way towards the midpoint of
    its two neighbours:

    $$\mathbf F_i = f_F \left( \tfrac12\,(\mathbf p_{i-1} + \mathbf p_{i+1}) - \mathbf p_i \right)$$

    On its own this would shrink the ring to a dot. Against the trembling it
    irons out the jitter and keeps the line a curve instead of a scribble. A
    larger $f_F$ gives smooth, wide meanders; a smaller one a frayed line
    that grows like coral.

    Where the spacing changes along the line, the midpoint is weighted by the
    spacing at each neighbour, $(\mathbf p_{i-1} r_{i+1} + \mathbf p_{i+1}
    r_{i-1}) / (r_{i-1} + r_{i+1})$, so that points do not slide towards the
    open, light parts of the picture.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Keep your distance

    The third nudge is what makes a labyrinth. Every stretch of the line is
    pushed away by any other stretch closer than a distance $r$, and pulled,
    gently, by one a little further off. How hard depends only on the
    distance $d$ counted in units of $r$, $\rho = d / r$:

    $$w(\rho) = \rho^{-12} - \rho^{-6}$$

    This is the Lennard-Jones curve, the textbook rule for two atoms: a steep
    wall of push below $\rho = 1$, a shallow dip of pull beyond it, nothing
    past $\rho = 2$. At $\rho = 1$ push and pull cancel, so two strands side
    by side settle exactly $r$ apart. Remember that $r$: in section 7 it
    becomes the brush.

    The distance is measured from a point to the nearest spot on each piece
    of the line, not to the other points, so a strand feels its neighbour as
    a smooth wall rather than a row of beads. Two changes to the textbook
    curve. Closer than $\rho = 0.85$ the push stops growing: the true curve
    shoots up so fast that a step overshoots and the points rattle. And a
    point feels only the parts of the line that have folded back towards it
    (closer in space than 0.9 times their distance along the line), never
    the stretch it sits on, or a straight line would push on itself.
    """)
    return


@app.cell
def _(mo):
    lj_floor = mo.ui.slider(0.6, 1.0, step=0.01, value=Growth.lj_floor, label="push stops growing at",
                            show_value=True)
    lj_floor
    return (lj_floor,)


@app.cell
def _(PAL, lj_floor, plt):
    _g = replace(Growth(), lj_floor=lj_floor.value)
    _rho = np.linspace(0.5, 2.2, 600)
    _fig, _ax = plt.subplots(figsize=(6.4, 2.8), facecolor=PAL["bg"])
    _ax.set_facecolor(PAL["bg"])
    _pure = _rho**-12 - _rho**-6
    _ax.plot(_rho, _pure, color=PAL["ink"], lw=1, ls=(0, (2, 3)), alpha=0.6)
    _w = np.where(_rho < _g.cutoff, lennard_jones(_rho, _g), 0.0)
    _ax.plot(_rho, _w, color=PAL["ink"], lw=1.8)
    _ax.axhline(0, color=PAL["ink"], lw=0.6, alpha=0.5)
    _ax.axvline(1, color=PAL["ink"], lw=0.6, alpha=0.3)
    _ax.set_ylim(-0.4, 2.0)
    _ax.set_xlim(0.5, 2.2)
    _ax.text(0.62, 1.75, "push", color=PAL["ink"], fontsize=9)
    _ax.text(1.35, -0.32, "pull", color=PAL["ink"], fontsize=9)
    _ax.text(1.02, 1.75, "ρ = 1: strands settle here", color=PAL["ink"], fontsize=8, alpha=0.8)
    _ax.set_xlabel("ρ = distance / r", color=PAL["ink"], fontsize=9)
    _ax.tick_params(colors=PAL["ink"], labelsize=8)
    for _s in ("top", "right"):
        _ax.spines[_s].set_visible(False)
    for _s in ("left", "bottom"):
        _ax.spines[_s].set_color(PAL["ink"])
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Split and merge

    Points are kept about $D = 0.4\,r$ apart. After every step a piece longer
    than $1.2\,D$ is cut in two at its middle, and a point closer than
    $0.2\,D$ to its neighbour is removed.

    Now the loop closes. Trembling makes the line longer; the long pieces
    split; every new point pushes on its neighbours; the line has nowhere to
    go but sideways, so it buckles and folds, and each fold is held $r$ from
    the next by the push and pull of section 4. When the frame is full there
    is nothing left to buckle into: the count of points stops climbing and
    the growth stops by itself.

    ### Try it: a ring on blank paper

    The same spacing everywhere, $r = 3$ pixels, on a small square. Change
    the three nudges and grow: no trembling, no growth; little smoothing, a
    ragged coral; strong push and pull, a tidy maze.
    """)
    return


@app.cell
def _(mo):
    demo_fb = mo.ui.slider(0.0, 0.3, step=0.01, value=Growth.f_b, label="tremble  f_B", show_value=True)
    demo_ff = mo.ui.slider(0.01, 0.4, step=0.01, value=Growth.f_f, label="smooth  f_F", show_value=True)
    demo_fa = mo.ui.slider(0.005, 0.08, step=0.005, value=Growth.f_a, label="push-pull  f_A",
                           show_value=True)
    demo_size = mo.ui.slider(50, 160, step=10, value=100 if HAVE_NUMBA else 70, label="square, px",
                             show_value=True)
    demo_run = mo.ui.run_button(label="grow")
    mo.vstack([mo.hstack([demo_fb, demo_ff, demo_fa]), mo.hstack([demo_size, demo_run], justify="start")])
    return demo_fa, demo_fb, demo_ff, demo_run, demo_size


@app.cell
def _(demo_fa, demo_fb, demo_ff, demo_run, demo_size, mo):
    mo.stop(not demo_run.value, mo.md("Press **grow**."
                                      + ("" if HAVE_NUMBA else " In the browser this takes a minute.")))
    demo_g = Growth(width=demo_size.value, r_min=3.0, r_max=3.0, f_b=demo_fb.value, f_f=demo_ff.value,
                    f_a=demo_fa.value, max_nodes=60_000)
    demo_R = np.full((demo_size.value, demo_size.value), 3.0)
    demo_log = []
    with mo.status.progress_bar(total=4000, title="growing", remove_on_exit=True) as _bar:
        demo_snaps = grow(demo_R, demo_g, 4000, every=20, log=demo_log, on_step=lambda s, P: _bar.update())
    return demo_R, demo_g, demo_log, demo_snaps


@app.cell
def _(demo_snaps, mo):
    demo_frame = mo.ui.slider(0, len(demo_snaps) - 1, value=len(demo_snaps) - 1, label="snapshot",
                              show_value=True)
    demo_frame
    return (demo_frame,)


@app.cell
def _(PAL, demo_R, demo_frame, demo_g, demo_log, demo_snaps, plt):
    _P = demo_snaps[demo_frame.value]
    _fig = plt.figure(figsize=(8.4, 4.0), facecolor=PAL["bg"])
    _ax = _fig.add_axes([0.0, 0.0, 0.48, 1.0])
    draw(_ax, polish(_P, demo_g), demo_R.shape, PAL, width=INK * demo_g.r_min)
    _ax2 = _fig.add_axes([0.58, 0.16, 0.4, 0.74])
    _s, _n = np.array(demo_log).T
    _ax2.plot(_s, _n, color=PAL["ink"], lw=1.4)
    _ax2.axvline(min(demo_frame.value * 20, _s[-1]), color=PAL["ink"], lw=0.6, alpha=0.4)
    _ax2.set_facecolor(PAL["bg"])
    _ax2.set_xlabel("step", color=PAL["ink"], fontsize=9)
    _ax2.set_title("points on the line", color=PAL["ink"], fontsize=9, loc="left")
    _ax2.tick_params(colors=PAL["ink"], labelsize=8)
    for _k in ("top", "right"):
        _ax2.spines[_k].set_visible(False)
    for _k in ("left", "bottom"):
        _ax2.spines[_k].set_color(PAL["ink"])
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Never crossing

    The promise in the title is the one thing the original method does not
    make: the short-range push keeps strands apart nearly always, not always.
    Here it is kept by construction, three ways.

    - **Short steps.** No point moves further than $r/4$ in one step, nor
      further than half its shorter neighbouring piece. Deep in a fold a
      point can feel a dozen walls at once; without the cap it sums all their
      pushes, jumps over its own neighbours and ties a knot.
    - **A check after every step.** Any piece that crosses another, touches
      it, or comes within $r_{\min}/4$ of a strand that has folded back, is
      put back where it was before the step, both ends, and checked again.
      The line was clean before the step, so this always ends.
    - **Careful removals.** Removing a point (section 5) cuts a corner; if
      the shortcut would pass across another strand, the point stays.

    The forces keep the check rare; the check makes crossing impossible. The
    tests grow lines and count crossings on every final curve. On the line
    you grew above:
    """)
    return


@app.cell
def _(demo_g, demo_snaps, mo):
    _P = demo_snaps[-1]
    _bad = crossings(_P, demo_g.gap * demo_g.r_min, demo_g.r_min)
    mo.md(
        f"**{len(_bad)}** crossings or near-touches among **{len(_P):,}** pieces, "
        f"line length **{curve_length(_P):,.0f} px** on a {demo_g.width} px square."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. From paint to spacing

    So far $r$ was one number. Now it changes from place to place, and it is
    read off a picture. Each point of the picture gets a brightness $T$
    between 0 (its darkest tone) and 1 (its lightest), and the strands there
    are asked to sit $r$ pixels apart:

    $$r(\mathbf x) = r_{\min} + (r_{\max} - r_{\min})\,T(\mathbf x)^{\gamma}$$

    Dark means $r_{\min}$, tight; light means $r_{\max}$, open. The exponent
    $\gamma$ bends the curve in between: above 1, more of the picture stays
    dense.

    A pen of fixed width can only reach about half black, where strands run
    side by side. So the lights do the talking: the wider $r_{\max}$, the
    more the face stands out. Before the curve, two touches. A blur of about
    $r_{\min}$, so the line follows tone and not the cracks in the varnish;
    and local contrast, which pushes every tone away from the average of its
    surroundings, so an eye stays darker than the lit cheek around it.

    The sliders under *Tune the tone*, at the top, move exactly these knobs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. What differs from the paper

    Pedersen and Singh give every length in units of a base distance and a
    field read from the image. The method here is theirs; these are the
    changes, each made because the original came apart at these sizes.

    - **Lengths in units of the local spacing $r$.** Push and pull cancel at
      $r$, the cutoff is $2r$, points sit $0.4\,r$ apart, but never more than
      $0.8\,r_{\min}$, so the open loops in the lights are drawn as curves
      and not as polygons.
    - **One scale per pair.** Two pieces of line use the smaller of their two
      spacings. With each point's own, where dark meets light the two sides
      want different distances, the pushes are not mutual, and dark strands
      creep into the light like the teeth of a comb.
    - **Only folded parts interact** and **the push has a ceiling** (section 4).
    - **Short steps and a check** (section 6): the paper does not promise a
      line that never crosses; this one does.
    - **Smoothing at drawing time only:** ten passes that move each point
      halfway to its neighbours' midpoint, then the check again. It removes
      the point-to-point wobble the trembling leaves and keeps the folds,
      which are many points wide. The growth itself keeps its tremble: that
      is what makes it grow.

    What it is not: a pen of fixed width never reaches solid black, so the
    contrast comes from opening the lights, not from filling the darks.
    """)
    return


if __name__ == "__main__":
    app.run()
