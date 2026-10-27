"""Growth videos: one line grows from a small ring into a painting.

    uv run python animate.py pearl-earring                  # out/pearl-earring/growth.mp4
    uv run python animate.py pearl-earring --size 2160      # 2160 px wide
    uv run python animate.py mona-lisa --reveal             # ends on the painting, and back
    uv run python animate.py medusa --palette night
    uv run python animate.py --image photo.jpg --start 0.5 0.35
    uv run python animate.py pearl-earring --gif out/pearl.gif --size 480 --fps 15
    uv run python animate.py pearl-earring --still out/pearl.png

Everything about the line, how it grows and how it is drawn, is imported from
the notebook, unbroken_line.py. This file decides which snapshot each frame
shows and writes the frames out. The growth is cached in out/<name>/, so the
second render of the same picture starts drawing at once. Video needs ffmpeg
on the PATH.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from unbroken_line import (
    INK,
    NIGHT,
    PACKING,
    PAPER,
    ROOT,
    Flow,
    curve_length,
    draw,
    grow,
    load_catalog,
    load_grey,
    load_image,
    params,
    polish,
    spacing_field,
    steady,
    step,
)

OUT = ROOT / "out"
PALETTES = {"paper": PAPER, "night": NIGHT}
# H.264, near-lossless, the colour tags players expect; frames are full-range
# RGB, the file is limited-range BT.709
ENCODE = [
    "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709",
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "slow",
    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
    "-color_range", "tv", "-movflags", "+faststart",
]


@dataclass(frozen=True)
class Animation:
    """Every knob of the timing. Seconds unless said otherwise."""

    fps: int = 30
    grow: float = 11.0  # from the ring to the full frame
    hold: float = 3.0  # the finished line, still moving
    fade: float = 1.0  # with --reveal: line -> painting
    painting: float = 1.5  # the painting
    back: float = 1.0  # painting -> line
    tail: float = 1.0  # the line again, to the end
    ease: float = 1.5  # line length goes as 1 - (1 - t)^ease: quick start, slow finish
    snapshots: int = 600  # kept during the growth, evenly spaced in line length ...
    snapshot_steps: int = 5  # ... and at least one every this many steps
    step_share: float = 0.3  # frames advance by line length and, this much, by step
    motion: int = 2  # growth steps per frame once the frame is full
    sway: float = 1.0  # every frame swayed by a slow flow this many r_min long
    flow: float = 0.03  # after the growth the line drifts like a liquid (Growth.flow) ...
    flow_ramp: float = 1.5  # ... coming in from zero over this long ...
    calm: float = 0.3  # ... no trembling, attraction and fairing scaled by this
    max_steps: int = 20_000

    def frames(self, seconds):
        return round(seconds * self.fps)

    def seconds(self, reveal):
        tail = self.fade + self.painting + self.back + self.tail if reveal else 0.0
        return self.grow + self.hold + tail


def estimate_length(R):
    """Line length that fills the tone map: each pixel holds 1 / spacing of it."""
    return float((1.0 / (PACKING * R)).sum())


# --- the growth, once, cached ---------------------------------------------


def grown(R, g, start, anim, folder, regrow=False):
    """The growth of R, as (snapshots, their line lengths, their steps,
    final curve).

    A snapshot is kept every time the line has grown by 1 / anim.snapshots
    of the length that fills the frame, so every frame, spaced by length,
    finds one close by; and at least every anim.snapshot_steps steps, since
    near the end the line hardly grows while it still settles, and with
    length alone the last frames would freeze and then jump. Cached in
    folder/growth.npz while the picture, the growth knobs and the start are
    the same."""
    path = folder / f"growth{'_night' if g.night else ''}.npz"
    key = json.dumps({"growth": asdict(g), "start": list(start), "snapshots": anim.snapshots,
                      "snapshot_steps": anim.snapshot_steps, "max_steps": anim.max_steps,
                      "map": hashlib.sha1(R.tobytes()).hexdigest()}, sort_keys=True)
    if path.exists() and not regrow:
        z = np.load(path)
        if str(z["key"]) == key:
            off, points = z["offsets"], z["points"]  # read once, then slice
            pts = [points[off[i]:off[i + 1]] for i in range(len(off) - 1)]
            print(f"  cached growth: {path.name}, {len(pts)} snapshots")
            return pts, z["lengths"], z["steps"], z["final"]

    dL = estimate_length(R) / anim.snapshots
    pts, lengths, steps, target = [], [], [], [0.0]

    def keep(s, P):
        L = curve_length(P)
        if L >= target[0] or s - (steps[-1] if steps else 0) >= anim.snapshot_steps:
            pts.append(P.astype(np.float32))
            lengths.append(L)
            steps.append(s)
            target[0] = max(target[0], L + dL)
        if s % 500 == 0:
            print(f"  step {s}  {len(P):,} points", flush=True)

    snaps = grow(R, g, anim.max_steps, every=10**9, center=start, on_step=keep)
    ring, final = snaps[0], snaps[-1]
    pts = [ring.astype(np.float32), *pts]
    lengths, steps = [curve_length(ring), *lengths], [0, *steps]
    if not np.array_equal(pts[-1], final.astype(np.float32)):
        pts.append(final.astype(np.float32))
        lengths.append(curve_length(final))
        steps.append(steps[-1] + 1)
    lengths, steps = np.array(lengths), np.array(steps)
    print(f"  grown: {len(final):,} points, line {lengths[-1]:,.0f} px, {steps[-1]} steps")
    folder.mkdir(parents=True, exist_ok=True)
    off = np.concatenate([[0], np.cumsum([len(p) for p in pts])])
    np.savez(path, points=np.concatenate(pts), offsets=off, lengths=lengths, steps=steps,
             final=final, key=key)
    return pts, lengths, steps, final


def still(final, R, g, anim, n):
    """The n frames after the growth. The simulation goes on, anim.motion
    steps per frame, so the finished line never freezes; but the trembling
    that drove the growth would only make it shiver now. So the trembling
    is off, attraction and fairing take a smaller step (anim.calm), and a
    slow Flow takes over: the strands glide like a liquid. The Flow comes
    in over anim.flow_ramp, so the line starts to drift gently."""
    calm = replace(g, f_b=0.0, f_a=g.f_a * anim.calm, f_f=g.f_f * anim.calm, flow=anim.flow)
    drift = Flow(calm, np.random.default_rng(2))
    speed, ramp = drift.speed, anim.frames(anim.flow_ramp)
    rng = np.random.default_rng(1)
    P = final
    for i in range(n):
        drift.speed = speed * smoothstep((i + 1) / ramp) if ramp else speed
        for _ in range(anim.motion):
            P = step(P, R, calm, rng, drift=drift)
        yield P


# --- from snapshots to frames ---------------------------------------------


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def frame_plan(lengths, steps, anim, reveal):
    """[(snapshot index or None, painting weight)] for every frame; None
    means the next still() frame.

    Growth frames are spaced mostly by line length, not by step: the filled
    area goes with the length, so the picture spreads at an even pace
    instead of sitting as a tiny blob for the first seconds and then
    exploding. A share anim.step_share goes by step, so the last settling,
    when the length hardly moves, still moves on frame after frame. The
    ease-out slows it into the finish."""
    def span(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x[0]) / max(x[-1] - x[0], 1e-12)

    q = (1 - anim.step_share) * span(lengths) + anim.step_share * span(steps)
    n = anim.frames(anim.grow)
    plan = []
    for f in range(n):
        u = f / (n - 1)
        plan.append((int(np.argmin(np.abs(q - (1.0 - (1.0 - u) ** anim.ease)))), 0.0))
    if not reveal:
        return plan + [(None, 0.0)] * anim.frames(anim.hold)
    marks = np.cumsum([anim.hold, anim.fade, anim.painting, anim.back, anim.tail])
    for k in range(anim.frames(marks[-1])):
        t = k / anim.fps
        if t < marks[0]:
            w = 0.0
        elif t < marks[1]:
            w = smoothstep((t - marks[0]) / anim.fade)
        elif t < marks[2]:
            w = 1.0
        elif t < marks[3]:
            w = 1.0 - smoothstep((t - marks[2]) / anim.back)
        else:
            w = 0.0
        plan.append((None, float(w)))
    return plan


def painting_layer(grey, size, pal):
    """The painting in the palette's ink (every grey is the ink at some
    coverage: dark paint on paper, light paint on night), float RGB."""
    W, H = size
    img = Image.fromarray((grey * 255).astype(np.uint8)).resize((W, H), Image.Resampling.LANCZOS)
    grey = np.asarray(img, dtype=np.float64)[..., None] / 255.0
    cover = grey if pal is NIGHT else 1.0 - grey
    bg = np.array(matplotlib.colors.to_rgb(pal["bg"])) * 255
    ink = np.array(matplotlib.colors.to_rgb(pal["ink"])) * 255
    return bg * (1 - cover) + ink * cover


def steadied(curves, R, g):
    """The frames' curves through steady(): each one averaged with the
    g.steady_window frames before and after it, nearer frames weighing more
    (1 - |d| / (window + 1)). A sliding window, so only 2 * window + 1
    curves are held at a time."""
    m = g.steady_window
    buf = []  # (curve, tree)
    head = 0  # frame index of buf[0]
    k = 0  # next frame to give out
    for P in curves:
        buf.append((P, cKDTree(P)))
        while k + m < head + len(buf):  # frame k has its window
            yield steady_frame(buf, head, k, m, R, g)
            k += 1
            if k - m > head:
                buf.pop(0)
                head += 1
    while k < head + len(buf):
        yield steady_frame(buf, head, k, m, R, g)
        k += 1


def steady_frame(buf, head, k, m, R, g):
    others = [(*buf[j - head], 1.0 - abs(j - k) / (m + 1))
              for j in range(max(k - m, head), min(k + m + 1, head + len(buf))) if j != k]
    return steady(buf[k - head][0], others, R, g)


def frames(grey, R, g, start, anim, size, pal, folder, reveal=False, regrow=False):
    """Every frame, RGB uint8: the growth from the cache, the still frames,
    and the painting blended in by frame_plan. Every frame, growth and still
    alike, is swayed by one slow Flow (Flow.offset, anim.sway), so the line
    flows from the first frame to the last instead of standing still while
    it grows. The sway is one-to-one: it never makes the line cross."""
    W, H = size
    pts, lengths, steps, final = grown(R, g, start, anim, folder, regrow)
    plan = frame_plan(lengths, steps, anim, reveal)
    later = still(final, R, g, anim, sum(idx is None for idx, _ in plan))
    paint = painting_layer(grey, size, pal) if reveal else None
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=pal["bg"])
    dither = np.random.default_rng(0)
    curves = (next(later) if idx is None else pts[idx].astype(np.float64) for idx, _ in plan)
    sway = Flow(g, np.random.default_rng(3))
    for k, ((_, w), P) in enumerate(zip(plan, steadied(curves, R, g), strict=True)):
        P = sway.offset(P, k * anim.motion, anim.sway * g.r_min)
        fig.clear()
        draw(fig.add_axes([0, 0, 1, 1]), polish(P, g), R.shape, pal, width=INK * g.r_min)
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        if w > 0:
            # a blend holds gradients: one level of dither before rounding
            mix = frame * (1 - w) + paint * w + dither.random((H, W, 1))
            frame = np.clip(np.floor(mix), 0, 255).astype(np.uint8)
        if k % 30 == 0:
            print(f"  frame {k}/{len(plan)}", flush=True)
        yield frame
    plt.close(fig)


# --- writing ----------------------------------------------------------------


def write_mp4(frames, size, fps, out):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on the PATH (needed for .mp4; --gif works without)")
    W, H = size
    enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                            *ENCODE, str(out)], stdin=subprocess.PIPE)
    for frame in frames:
        enc.stdin.write(np.ascontiguousarray(frame).tobytes())
    enc.stdin.close()
    if enc.wait():
        raise SystemExit("ffmpeg failed")


def write_gif(frames, fps, out, pal, colors=16):
    """A looping GIF, every frame on one fixed palette: the ground, the ink
    and the mixes between them, nothing else."""
    bg = np.array(matplotlib.colors.to_rgb(pal["bg"])) * 255
    ink = np.array(matplotlib.colors.to_rgb(pal["ink"])) * 255
    ramp = np.linspace(0.0, 1.0, colors)[:, None]
    lut = np.round(bg * (1 - ramp) + ink * ramp).astype(np.uint8)
    board = Image.new("P", (1, 1))
    board.putpalette(lut.flatten().tolist() + [0] * (768 - 3 * colors))
    imgs = [Image.fromarray(f).quantize(palette=board, dither=Image.Dither.NONE) for f in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0,
                 optimize=True, disposal=1)


def write_still(final, R, g, size, pal, out):
    W, H = size
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=pal["bg"])
    draw(fig.add_axes([0, 0, 1, 1]), polish(final, g), R.shape, pal, width=INK * g.r_min)
    fig.savefig(out, dpi=100, facecolor=pal["bg"])
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slug", nargs="?", help="a painting of public/artworks.toml")
    ap.add_argument("--image", type=Path, help="any picture instead of a painting of the catalog")
    ap.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"),
                    help="where the ring starts, fractions of the picture (default: the "
                         "painting's own, or the middle)")
    ap.add_argument("--palette", choices=sorted(PALETTES), help="default: the painting's own")
    ap.add_argument("--width", type=int, help="growth width in px (default 600): finer line, slower")
    ap.add_argument("--size", type=int, default=1080, help="output width in px (default 1080)")
    ap.add_argument("--fps", type=int, default=Animation.fps)
    ap.add_argument("--seconds", type=float, default=Animation.grow, help="length of the growth")
    ap.add_argument("--reveal", action="store_true", help="fade to the painting and back at the end")
    ap.add_argument("--gif", type=Path, help="write a GIF here instead of the mp4")
    ap.add_argument("--still", type=Path, help="write only the finished line, as a PNG, here")
    ap.add_argument("--out", type=Path, help="mp4 path (default out/<name>/growth.mp4)")
    ap.add_argument("--regrow", action="store_true", help="ignore the cached growth")
    a = ap.parse_args()
    if bool(a.slug) == bool(a.image):
        ap.error("give a painting's slug or --image, one of the two")

    if a.image:
        name, grey = a.image.stem, load_grey(a.image)
        entry = {"start": [0.5, 0.5], "palette": "paper"}
    else:
        catalog = load_catalog()
        if a.slug not in catalog:
            ap.error(f"unknown painting {a.slug!r}; one of: {', '.join(catalog)}")
        name, grey, entry = a.slug, load_image(a.slug), catalog[a.slug]
    palette = a.palette or entry.get("palette", "paper")
    start = tuple(a.start or entry["start"])
    knobs = {"width": a.width} if a.width else {}
    g = params(entry, night=palette == "night", **knobs)
    anim = replace(Animation(), fps=a.fps, grow=a.seconds)
    R = spacing_field(grey, g)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    W = a.size + a.size % 2
    size = (W, round(W * R.shape[0] / R.shape[1] / 2) * 2)  # even, for the encoder
    pal = PALETTES[palette]
    (folder / "params.json").write_text(json.dumps(
        {"growth": asdict(g), "animation": asdict(anim), "palette": palette, "start": start,
         "reveal": a.reveal, "size": size}, indent=2))
    print(f"{name}: tone map {R.shape[1]}x{R.shape[0]}, output {size[0]}x{size[1]}, {palette}")

    if a.still:
        final = grown(R, g, start, anim, folder, a.regrow)[3]
        a.still.parent.mkdir(parents=True, exist_ok=True)
        write_still(final, R, g, size, pal, a.still)
        print(f"  wrote {a.still}")
        return
    video = frames(grey, R, g, start, anim, size, pal, folder, a.reveal, a.regrow)
    if a.gif:
        a.gif.parent.mkdir(parents=True, exist_ok=True)
        write_gif(video, anim.fps, a.gif, pal)
        print(f"  wrote {a.gif}")
    else:
        out = a.out or folder / f"growth{'_night' if palette == 'night' else ''}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_mp4(video, size, anim.fps, out)
        print(f"  wrote {out}, {anim.seconds(a.reveal):.1f} s")


if __name__ == "__main__":
    main()
