"""Plot of the irradiance distribution on the panel (equivalent of visualizeResult.m)."""

from __future__ import annotations

import numpy as np

_STYLE = {
    "front_direct": ("Front direct", "#AA3939"),
    "front_nearground": ("Front near-ground reflection", "#7A9F35"),
    "front_farground": ("Front far-ground reflection", "#226666"),
    "back_direct": ("Back direct", "#D46A6A"),
    "back_nearground": ("Back near-ground reflection", "#A5C663"),
    "front_reflected": ("Front panel reflection", "#8C6BB1"),
    "back_reflected": ("Back panel reflection", "#B9A3D3"),
}


def plot_distribution(result, time=0, ax=None, legend=True):
    """Stacked irradiance components along the panel; front to the right, back to the left.

    ``time`` is an integer position or a label of ``result.index``.
    """
    import matplotlib.pyplot as plt

    i = time if isinstance(time, (int, np.integer)) else result.index.get_loc(time)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    chi = result.chi
    for side, comps, sign in (("front", ("front_direct", "front_nearground", "front_farground",
                                         "front_reflected"), 1),
                              ("back", ("back_direct", "back_nearground", "back_reflected"), -1)):
        base = np.zeros_like(chi)
        for c in comps:
            values = getattr(result, c)[i]
            if c.endswith("_reflected") and not np.any(values):
                continue
            top = base + values
            label, color = _STYLE[c]
            ax.fill_betweenx(chi, sign * base, sign * top, color=color, label=label, linewidth=0)
            base = top
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel(r"Position $\chi$ on panel [-]")
    ax.set_xlabel(r"Irradiance in panel plane [W/m$^2$]  (back $\leftarrow$ | $\rightarrow$ front)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):g}"))
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")
    return ax
