"""
visualization.py
=================
Matplotlib dashboards shared by main.py (CLI) and gui_tkinter.py.
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def build_dashboard(ocf, yn, yc, y_curve, E_curve, e_min_point, gvf_result, jump):
    """
    Build a dual-panel Matplotlib figure:
      left  panel: specific-energy (E-y) diagram with yn/yc marked
      right panel: water-surface profile along the reach, with a jump
                   marker if one was located
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel 1: specific energy diagram -------------------------------
    ax1.plot(E_curve, y_curve, color="steelblue", label="E-y curve")
    ax1.plot(e_min_point[0], yc, "ro", label=f"Critical yc={yc:.3f} m")
    from .energy import specific_energy
    E_at_yn = specific_energy(yn, ocf.Q, ocf.g, ocf.section)
    ax1.plot(E_at_yn, yn, "o", color="orange", label=f"Normal yn={yn:.3f} m")
    ax1.set_xlabel("Specific energy E (m)")
    ax1.set_ylabel("Depth y (m)")
    ax1.set_title("Specific Energy Diagram")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    # --- Panel 2: water surface profile ----------------------------------
    xs = gvf_result.x
    S0 = ocf.S0
    bed = [S0 * x for x in xs]
    surface = [S0 * x + y for x, y in zip(xs, gvf_result.y)]
    normal_line = [S0 * x + yn for x in xs]
    critical_line = [S0 * x + yc for x in xs]

    ax2.plot(xs, bed, color="saddlebrown", label="Bed")
    ax2.plot(xs, normal_line, "k--", linewidth=1, label="Normal depth line")
    ax2.plot(xs, critical_line, "k:", linewidth=1, label="Critical depth line")
    ax2.plot(xs, surface, color="royalblue", linewidth=2.5, label="Water surface")

    if jump:
        ax2.axvline(jump["x_jump"], color="crimson", linestyle="-.", label="Hydraulic jump")

    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("Elevation (m)")
    ax2.set_title(f"Water Surface Profile ({gvf_result.curve_type})")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_roughness_sensitivity(results):
    """
    results: list of {"n": ..., "yn": ...} dicts from
    optimize.roughness_sensitivity().
    """
    ns = [r["n"] for r in results]
    yns = [r["yn"] for r in results]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, yns, "o-", color="teal")
    ax.set_xlabel("Manning's n")
    ax.set_ylabel("Normal depth yn (m)")
    ax.set_title("Roughness Sensitivity")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
