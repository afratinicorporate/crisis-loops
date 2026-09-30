"""Schematic: what 'clockwise' vs 'counter-clockwise' means for a crisis loop. No data."""
import os, sys
import numpy as np
from matplotlib.patches import FancyArrowPatch

import figure_style as qr

def bez(p0, p1, p2, p3, n=200):
    t = np.linspace(0, 1, n)[:, None]
    P = (1-t)**3*np.array(p0) + 3*(1-t)**2*t*np.array(p1) + 3*(1-t)*t**2*np.array(p2) + t**3*np.array(p3)
    return P[:, 0], P[:, 1]

def y_at(x, y, x0):
    i = np.argmin(np.abs(x - x0)); return y[i]

fig = qr.frame("Same price, two moments: which road is higher?",
               "In a crisis the market goes down one road and comes back up another. The direction of the loop says which road sits on top.",
               "SCHEMATIC",
               "Illustrative shapes, no data. Horizontal axis: distance below the all-time high. Vertical axis: volatility. "
               "The dashed line compares the same price level (15% below the peak) on the way down and on the way back.")
panels = [
    ("A coincident measure of volatility", qr.OXBLOOD, "counter-clockwise",
     bez((0, 15), (-8, 22), (-22, 45), (-30, 55)), bez((-30, 55), (-31, 30), (-18, 22), (0, 15)),
     "on the way back volatility is\nalready LOWER at the same price"),
    ("The same volatility, measured late", qr.GREEN, "clockwise",
     bez((0, 15), (-10, 17), (-24, 28), (-30, 45)), bez((-30, 45), (-26, 60), (-8, 52), (0, 15)),
     "on the way back the lagging measure\nis still HIGHER at the same price"),
]
for k, (title, c, orient, down, up, note) in enumerate(panels):
    ax = fig.add_axes([0.075 + k * 0.465, 0.20, 0.405, 0.56])
    (xd, yd), (xu, yu) = down, up
    ax.plot(xd, yd, color=c, lw=2.4); ax.plot(xu, yu, color=c, lw=2.4, ls=(0, (5, 2.5)))
    for (x, y) in (down, up):
        for f in (0.35, 0.7):
            i = int(f * len(x)); ax.add_patch(FancyArrowPatch((x[i], y[i]), (x[i + 6], y[i + 6]), arrowstyle="-|>", mutation_scale=20, color=c, lw=0))
    a, b = y_at(xd, yd, -15), y_at(xu, yu, -15)
    ax.plot([-15, -15], [min(a, b) - 6, max(a, b) + 6], color=qr.STRONG_GRAY, lw=0.9, ls=(0, (3, 3)))
    ax.text(-15, min(a, b) - 7.5, "same price\n(−15%)", fontsize=8.5, ha="center", va="top", color=qr.STRONG_GRAY, style="italic", linespacing=1.1)
    ax.plot([-15, -15], [a, b], "o", color=qr.TEXT_DARK, ms=6, zorder=5)
    ax.text(-14.3, a, "going down", fontsize=9.5, va="center", color=qr.TEXT_MID)
    ax.text(-14.3, b, "coming back", fontsize=9.5, va="center", color=qr.TEXT_MID)
    ax.text(-0.5, 12.2, "1  calm, at the peak", fontsize=9, ha="right", color=qr.TEXT_LIGHT, style="italic")
    ax.text(-29.5, 63.5, orient.upper(), fontsize=11, color=c, weight="semibold")
    ax.text(-29.5, 4.5, note, fontsize=9.5, color=c, style="italic", linespacing=1.15)
    ax.set_xlim(-32, 1); ax.set_ylim(0, 68); ax.set_xticks([-30, -20, -10, 0]); ax.set_xticklabels(["−30%", "−20%", "−10%", "peak"])
    ax.set_yticks([]); ax.set_xlabel("distance below the all-time high", fontsize=10)
    ax.set_ylabel("volatility  (higher = more agitated)", fontsize=10) if k == 0 else None
    ax.set_title(title, loc="left", fontsize=11.5, color=qr.TEXT_MID, pad=8); qr.tidy(ax, grid_y=False)
fig.text(0.955, 0.80, "solid = the fall     dashed = the recovery", fontsize=9.5, color=qr.TEXT_LIGHT, style="italic", ha="right")
qr.save(fig, "fig3_loop_direction_schematic", os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
