"""Figure of the Totalp validation (manuscript figure "validationFigure").

(a) Ratio of predicted and measured irradiation as the model extensions are added step by step:
    mean over the configurations per channel, with min-max bars.
(b) Vertical backside per configuration against the row spacing, after the base-plate correction
    and with the diffuse panel reflection (fitted effective reflectance).

Reads results/totalp_summary.csv (run totalp_validation.py first).
Usage:  python validation_figure.py      (results/validation_steps[_pub].pdf, .png)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import pubstyle as ps  # noqa: E402

RESULTS = HERE / "results"
STEPS = [("legacy", "Original"), ("corrected", "Corrected"), ("array", "Three rows"),
         ("array_glass", "+ Specular refl."), ("array_glass_tau", "+ Transmission"), ("full", "+ Horizon"),
         ("full_thick", "+ Thickness"), ("plate", "+ Plate, snow"), ("plate_dfit", "+ Diffuse refl.")]
CHANNELS = [("front", 60), ("front", 90), ("back", 60), ("back", 90)]
OFFSET = {("front", 60): -0.21, ("front", 90): -0.07, ("back", 60): 0.07, ("back", 90): 0.21}


def main():
    s = pd.read_csv(RESULTS / "totalp_summary.csv")
    s = s[s.n >= 20]
    ps.apply_style()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=ps.DOUBLE, gridspec_kw={"width_ratios": [1.55, 1]})

    # (a) step by step
    x = np.arange(len(STEPS))
    ax.axhline(1.0, color=ps.MUTED, lw=0.7, zorder=1)
    for side, tilt in CHANNELS:
        g = s[(s.side == side) & (s.beta_p == tilt)]
        r = np.array([[g[f"ratio_{v}"].mean(), g[f"ratio_{v}"].min(), g[f"ratio_{v}"].max()] for v, _ in STEPS])
        xx = x + OFFSET[(side, tilt)]
        ax.plot(xx, r[:, 0], color=ps.SIDE[side], lw=0.8, zorder=3)
        ax.errorbar(xx, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], fmt=ps.MARKER_TILT[tilt],
                    color=ps.SIDE[side], mfc=ps.SIDE[side], mec=ps.SIDE[side], ms=3.5, elinewidth=0.7, capsize=0,
                    zorder=4)
    ax.set_xticks(x, [lab for _, lab in STEPS], rotation=35, ha="right", rotation_mode="anchor")
    ax.set(xlim=(-0.5, len(STEPS) - 0.5), ylim=(0.0, 1.2), ylabel="Predicted / measured [-]",
           title="(a) Model extensions added step by step")
    ax.grid(axis="x", visible=False)
    side_handles = [Line2D([], [], color=ps.SIDE["front"], label="Front side"),
                    Line2D([], [], color=ps.SIDE["back"], label="Back side")]
    tilt_handles = [Line2D([], [], color=ps.INK, ls="none", marker=ps.MARKER_TILT[t], ms=3.5,
                           label=f"Tilt {t}°") for t in (60, 90)]
    leg = ax.legend(handles=side_handles, loc="lower right", bbox_to_anchor=(1.0, 0.12))
    ax.add_artist(leg)
    ax.legend(handles=tilt_handles, loc="lower right", bbox_to_anchor=(0.66, 0.12))

    # (b) vertical backside against the row spacing
    b = s[(s.side == "back") & (s.beta_p == 90)].sort_values("rho_bar")
    bx.axhline(1.0, color=ps.MUTED, lw=0.7, zorder=1)
    for bg in sorted(b.beta_g.unique()):
        g = b[b.beta_g == bg]
        c = ps.slope_color(bg)
        bx.plot(g.rho_bar, g.ratio_plate, ls="--", lw=0.8, color=c, marker="s", ms=3.5, mfc="none", mec=c, zorder=3)
        bx.plot(g.rho_bar, g.ratio_plate_dfit, ls="-", lw=0.8, color=c, marker="s", ms=3.5, mfc=c, mec=c, zorder=4)
    bx.set(xlim=(0.8, 7.5), ylim=(0.8, 1.05), xlabel=r"Row spacing $\overline{\rho}$ [-]",
           ylabel="Predicted / measured [-]", title="(b) Back side, tilt 90°")
    slope_handles = [Line2D([], [], color=ps.slope_color(bg), lw=0.8, label=f"{bg:.0f}°")
                     for bg in sorted(b.beta_g.unique())]
    var_handles = [Line2D([], [], color=ps.INK, ls="--", lw=0.8, marker="s", ms=3.5, mfc="none",
                          label="+ Plate, snow"),
                   Line2D([], [], color=ps.INK, ls="-", lw=0.8, marker="s", ms=3.5, label="+ Diffuse refl.")]
    leg = bx.legend(handles=slope_handles, title="Ground slope", loc="lower right", ncol=2, columnspacing=0.8)
    bx.add_artist(leg)
    bx.legend(handles=var_handles, loc="upper left")

    ps.footnote(fig, "Totalp miniature array, 1 Oct 2024 - 31 Mar 2025, 28 configurations. (a) mean over the 14 "
                     "configurations per channel, bars: min-max. (b) one marker per configuration; fitted effective "
                     "diffuse reflectance 0.068.", y=-0.02)
    ps.save(fig, RESULTS / "validation_steps")


if __name__ == "__main__":
    main()
