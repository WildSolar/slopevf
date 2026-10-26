"""Shared figure style for the slopevf follow-up paper (publication-figures rules).

Target document: Solar Energy, elsarticle ``[review, 5p, twocolumn]``; text width 522 pt
(7.22 in), column width 252 pt (3.49 in), body 10 pt, captions 8 pt (measured with
measure_layout.tex). Figures are produced at their final printed size and included at 100 %
(``\\includegraphics{<name>_pub.pdf}`` without width): double-wide DOUBLE, single SINGLE
(square, same height). Fonts: Times New Roman, black; axis labels/titles FS_LABEL (9 pt),
ticks/legends FS_TICK (8 pt), annotations FS_NOTE (7 pt).

Encoding table (identical in every figure of the paper):

- panel side -> categorical hue: front SIDE["front"] (orange), back SIDE["back"] (green)
- data source -> line / marker: model (slopevf) solid line, ray tracing (Radiance) open
  circles, measurement filled markers
- model variant -> line style: current model solid, legacy (MATLAB) dashed, other references
  (e.g. pvlib) dotted
- panel tilt beta_p -> marker shape (MARKER_TILT): 30 deg triangle, 45 deg diamond, 60 deg
  circle, 90 deg square
- ground slope beta_g (ordered, 0..30 deg) -> lightness of a blue ramp (slope_color)
- row spacing / shading angle (ordered) -> lightness of a grey ramp (RAMP_GREY)
- heat-map magnitude -> violet ramp (RAMP_VIOLET)
- other unordered categories (e.g. sky types) -> CAT hues not used for the panel sides

Usage: ``import pubstyle as ps; ps.apply_style(); fig, ax = plt.subplots(figsize=ps.SINGLE)``;
``ps.footnote(fig, ...)``; ``ps.save(fig, path)`` (review PNG/PDF + ``<name>_pub.pdf``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- sizes [in]
TEXTWIDTH_IN = 7.21  # a hair below \textwidth (522 pt = 7.223 in) to avoid overfull boxes
FIG_H = TEXTWIDTH_IN / 2.5
DOUBLE = (TEXTWIDTH_IN, FIG_H)
SINGLE = (FIG_H, FIG_H)
FS_LABEL, FS_TICK, FS_NOTE = 9, 8, 7  # body 10 pt, captions 8 pt

# ---------------------------------------------------------------- colours
SURFACE = "#ffffff"
INK = "#000000"  # all text
MUTED = "#7f7f7f"  # reference lines and secondary data only, never text
GRID = "#e3e3e3"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SIDE = {"front": CAT[1], "back": CAT[2]}
RAMP_BLUE = ["#9ec5f4", "#5598e7", "#1c5cab", "#0d366b"]
RAMP_GREY = ["#cfcfcf", "#9a9a9a", "#5f5f5f", "#1f1f1f"]
RAMP_VIOLET = ["#ece9f7", "#cfc8ee", "#ada1e2", "#8a79d4", "#6a56c2", "#4a3aa7", "#30256f"]
SLOPES = (0, 10, 20, 30)  # ground slopes [deg] mapped onto RAMP_BLUE

# ---------------------------------------------------------------- markers and lines
MARKER_TILT = {30: "^", 45: "D", 60: "o", 90: "s"}
LINESTYLE = {"model": "-", "legacy": "--", "reference": ":"}


def slope_color(beta_g: float) -> str:
    """Blue-ramp colour for a ground slope [deg] (nearest of SLOPES)."""
    i = int(np.argmin(np.abs(np.asarray(SLOPES) - beta_g)))
    return RAMP_BLUE[i]


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "font.size": FS_LABEL,
        "text.color": INK,
        "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.labelsize": FS_LABEL, "axes.titlesize": FS_LABEL, "axes.titlelocation": "center",
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "xtick.color": INK, "ytick.color": INK, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.fontsize": FS_TICK, "legend.title_fontsize": FS_TICK, "legend.frameon": True,
        "legend.framealpha": 0.9, "legend.edgecolor": "#bdbdbd", "legend.fancybox": False,
        "legend.borderpad": 0.3, "legend.handlelength": 1.6,
        "lines.linewidth": 1.1, "lines.markersize": 4, "patch.linewidth": 0.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
        "savefig.dpi": 600, "savefig.bbox": None,  # keep the exact printed size
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def frame_on_top(ax: plt.Axes) -> None:
    """Draw spines and ticks above filled areas (terrain, bands); grid stays below."""
    ax.grid(False)
    for x in ax.get_xticks():
        ax.axvline(x, color=GRID, linewidth=0.5, zorder=0.5)
    for y in ax.get_yticks():
        ax.axhline(y, color=GRID, linewidth=0.5, zorder=0.5)
    for sp in ax.spines.values():
        sp.set_zorder(10)
    ax.xaxis.set_zorder(10)
    ax.yaxis.set_zorder(10)


def label_on_line(ax: plt.Axes, x0: float, slope: float, text: str, intercept: float = 0.0) -> None:
    """Put `text` on the line y = intercept + slope * x at x0, rotated along it.

    Call after the axes limits are final (draws the canvas once to get the display angle).
    """
    ax.figure.canvas.draw()
    p0 = ax.transData.transform((x0, intercept + slope * x0))
    dx = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    p1 = ax.transData.transform((x0 + dx, intercept + slope * (x0 + dx)))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(x0, intercept + slope * x0, text, color=INK, fontsize=FS_NOTE, ha="center",
            va="center", rotation=angle, rotation_mode="anchor", zorder=4,
            bbox={"boxstyle": "square,pad=0.1", "fc": SURFACE, "ec": "none"})


def footnote(fig: plt.Figure, text: str, y: float = -0.06) -> None:
    """Explanatory note for the review version; omitted in the publication version."""
    t = fig.text(0.01, y, text, color=INK, fontsize=FS_NOTE, va="top")
    t.set_gid("footnote")


def save(fig: plt.Figure, path: Path) -> None:
    """Review version (PNG + PDF, trimmed, with footnotes) and publication version
    (``<name>_pub.pdf``, exact printed size, without footnotes)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    for t in [t for t in fig.texts if t.get_gid() == "footnote"]:
        t.remove()
    fig.savefig(path.with_name(path.stem + "_pub.pdf"))
    plt.close(fig)
