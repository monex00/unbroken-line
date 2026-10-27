"""The pictures of the README and the notebook, in public/ (committed).

    uv run python figures.py

Grows nothing itself: it reads the growths that animate.py cached in out/,
so run animate.py first for the paintings below (`--still` is enough):

    uv run python animate.py <slug> --still public/line/<slug>.png

Writes
    public/growth/<slug>.npz     a growth at 24 moments, for the notebook to show
                                 at once instead of growing it (pack_growth)
    public/growth.gif            the Mona Lisa growing, for the top of the page
    public/figures/process.png   picture -> spacing -> line
    public/figures/stages.png    one growth at four moments
    public/figures/detail.png    a piece of the line at full size
"""

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from animate import PALETTES, Animation, frame_plan, frames, grown, write_gif
from unbroken_line import (
    INK,
    ROOT,
    draw,
    ink_cmap,
    load_catalog,
    load_image,
    pack_growth,
    params,
    polish,
    spacing_field,
)

PUBLIC = ROOT / "public"
FIG = PUBLIC / "figures"


def piece(slug):
    """(grey, R, g, start, pal, folder) of a painting, as animate.py grows it."""
    entry = load_catalog()[slug]
    palette = entry.get("palette", "paper")
    g = params(entry, night=palette == "night")
    grey = load_image(slug)
    return grey, spacing_field(grey, g), g, tuple(entry["start"]), PALETTES[palette], \
        ROOT / "out" / slug


def save(fig, name, pal):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=100, facecolor=pal["bg"])
    plt.close(fig)
    print(f"  wrote public/figures/{name}")


def process(slug, h=675):
    """Three panels in the palette's ink: the picture, the spacing map
    (dark = tight) and the finished line."""
    grey, R, g, start, pal, folder = piece(slug)
    final = grown(R, g, start, Animation(), folder)[3]
    w = round(h * R.shape[1] / R.shape[0])
    gap = 24
    fig = plt.figure(figsize=((3 * w + 2 * gap) / 100, h / 100), dpi=100, facecolor=pal["bg"])
    cmap = ink_cmap(pal)
    W = 3 * w + 2 * gap
    for k, img in enumerate([grey if g.night else 1 - grey, (g.r_max - R) / (g.r_max - g.r_min)]):
        ax = fig.add_axes([k * (w + gap) / W, 0, w / W, 1])
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation="lanczos")
        ax.axis("off")
    draw(fig.add_axes([2 * (w + gap) / W, 0, w / W, 1]), polish(final, g), R.shape, pal,
         width=INK * g.r_min)
    save(fig, "process.png", pal)


def stages(slug, h=500, at=(0.12, 0.35, 0.65, 1.0)):
    """One growth at four moments, by share of the final line length."""
    grey, R, g, start, pal, folder = piece(slug)
    pts, lengths, _, final = grown(R, g, start, Animation(), folder)
    w = round(h * R.shape[1] / R.shape[0])
    gap = 16
    W = len(at) * w + (len(at) - 1) * gap
    fig = plt.figure(figsize=(W / 100, h / 100), dpi=100, facecolor=pal["bg"])
    for k, f in enumerate(at):
        P = final if f >= 1 else pts[int(np.argmin(np.abs(lengths - f * lengths[-1])))]
        ax = fig.add_axes([k * (w + gap) / W, 0, w / W, 1])
        draw(ax, polish(P.astype(np.float64), g), R.shape, pal, width=INK * g.r_min)
    save(fig, "stages.png", pal)


def detail(slug, box=(0.18, 0.22, 0.62, 0.48), scale=4.0):
    """A window of the finished line drawn `scale` times the tone map's
    size, so single strands show: box = (x0, y0, x1, y1), fractions."""
    grey, R, g, start, pal, folder = piece(slug)
    final = polish(grown(R, g, start, Animation(), folder)[3], g)
    H, W = R.shape
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    fw, fh = (x1 - x0) * scale, (y1 - y0) * scale
    fig = plt.figure(figsize=(fw / 100, fh / 100), dpi=100, facecolor=pal["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    draw(ax, final, R.shape, pal, lw=INK * g.r_min * scale * 0.72)  # px -> pt at dpi 100
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    save(fig, "detail.png", pal)


def hero(slug, width=360, fps=10):
    """The growth as a looping GIF: GitHub plays nothing else inline."""
    grey, R, g, start, pal, folder = piece(slug)
    anim = replace(Animation(), fps=fps, grow=9.0, hold=2.0)
    size = (width, round(width * R.shape[0] / R.shape[1] / 2) * 2)
    write_gif(frames(grey, R, g, start, anim, size, pal, folder), fps, PUBLIC / "growth.gif", pal)
    print("  wrote public/growth.gif")


def ready(slug, n=24):
    """The growth at n moments, chosen the way the video's frames are (even
    in line length, slower into the finish), smoothed for drawing, packed
    small: the notebook shows these instead of growing the painting."""
    grey, R, g, start, pal, folder = piece(slug)
    pts, lengths, steps, final = grown(R, g, start, Animation(), folder)
    plan = frame_plan(lengths, steps, replace(Animation(), fps=1, grow=float(n)), reveal=False)
    curves = [polish(pts[i].astype(np.float64), g) for i, _ in plan if i is not None]
    curves[-1] = polish(final, g)
    out = PUBLIC / "growth" / f"{slug}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pack_growth(curves, R.shape, g))
    print(f"  wrote public/growth/{slug}.npz, {out.stat().st_size / 1e6:.1f} MB")


def main():
    for slug in ("pearl-earring", "mona-lisa", "creation-of-adam"):
        ready(slug)
    process("creation-of-adam")
    stages("mona-lisa")
    detail("pearl-earring")
    hero("mona-lisa")


if __name__ == "__main__":
    main()
