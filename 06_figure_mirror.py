"""Figure 4: every crisis, both loops, and the average loop they add up to.
Top: one pair of bars per crisis. Bar = signed loop area rescaled by the episode's own ranges
(> 0 counter-clockwise, < 0 clockwise). Bottom: the average crisis loop, from the same 21 episodes:
volatility (rescaled 0-1 within each episode) against depth of the fall (0 = start, 1 = deepest point),
averaged separately over the fall and over the recovery."""
import os
import numpy as np
from matplotlib.patches import FancyArrowPatch
from common import load, episodes, signed_area, HERE
import figure_style as qr

d = load(); eps = episodes(d.VIX, 30, 20)
def shape(x, y):
    return signed_area(x, y) / ((x.max() - x.min()) * (y.max() - y.min()))
V = np.array([shape(d.dd.values[a:b + 1], d.VIX.values[a:b + 1]) for a, b in eps])
R = np.array([shape(d.dd.values[a:b + 1], d.rv21.values[a:b + 1]) for a, b in eps])
labels = [d.index[a].strftime("%b\n%Y") for a, b in eps]

EDGES = np.linspace(0, 1, 9); CTR = 0.5 * (EDGES[1:] + EDGES[:-1])
def average_loop(col):
    fall, rec = [], []
    for a, b in eps:
        x = d.dd.values[a:b + 1]; y = d[col].values[a:b + 1]
        u = np.clip((x[0] - x) / (x[0] - x.min()), 0, 1); v = (y - y.min()) / (y.max() - y.min())
        k = int(np.argmin(x))
        for part, store in ((slice(0, k + 1), fall), (slice(k, None), rec)):
            uu, vv = u[part], v[part]; idx = np.digitize(uu, EDGES[1:-1])
            store.append([vv[idx == j].mean() if np.any(idx == j) else np.nan for j in range(len(CTR))])
    return np.nanmean(fall, axis=0), np.nanmean(rec, axis=0)

fig = qr.frame("Twenty-one crises. Two clocks. Opposite directions.",
               "Top: one pair of bars per crisis since 2000 (up = counter-clockwise loop, down = clockwise). Bottom: the average loop those crises add up to.",
               "OBSERVED",
               "S&P 500 and VIX daily closes, Yahoo Finance, 3 Jan 2000 - 29 Sep 2026. Crisis = VIX above 30, from the last to the first close below 20. Bars: signed loop area over the "
               "episode's own ranges. Average loop: volatility rescaled 0-1 per episode, averaged at equal depth of the fall. Trailing volatility = returns of the last 21 days (lags by about 10 days).")
# ---- top: evidence
ax = fig.add_axes([0.06, 0.575, 0.90, 0.215])
x = np.arange(len(eps)); w = 0.38
ax.bar(x - w / 2, V, width=w, color=qr.OXBLOOD, zorder=3); ax.bar(x + w / 2, R, width=w, color=qr.GREEN, zorder=3)
ax.axhline(0, color=qr.TEXT_DARK, lw=1.0, zorder=4)
lim = 1.08 * max(np.abs(V).max(), np.abs(R).max()); ax.set_ylim(-lim, lim); ax.set_xlim(-0.8, len(eps) - 0.2)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.2, linespacing=1.0); ax.set_yticks([])
for s in ("left", "bottom"): ax.spines[s].set_visible(False)
ax.tick_params(length=0, pad=3)
# ---- bottom: the structure
for k, (col, c, head, note, x0) in enumerate([
        ("VIX", qr.OXBLOOD, f"VIX: counter-clockwise in {int((V > 0).sum())} of {len(V)}", "a coincident measure: already lower on the way back", 0.085),
        ("rv21", qr.GREEN, f"Trailing 21-day volatility: clockwise in {int((R < 0).sum())} of {len(R)}", "a lagging measure: it still shows the fall on the way back", 0.555)]):
    f, r = average_loop(col)
    ax = fig.add_axes([x0, 0.175, 0.37, 0.21])
    X = -CTR                                           # deepest point on the left, like the map
    ax.fill_between(X, f, r, color=c, alpha=0.13, lw=0)
    ax.plot(X, f, color=c, lw=2.4); ax.plot(X, r, color=c, lw=2.4, ls=(0, (5, 2.5)))
    ax.add_patch(FancyArrowPatch((X[2], f[2]), (X[3], f[3]), arrowstyle="-|>", mutation_scale=22, color=c, lw=0, zorder=5))   # fall: start -> deepest
    ax.add_patch(FancyArrowPatch((X[5], r[5]), (X[4], r[4]), arrowstyle="-|>", mutation_scale=22, color=c, lw=0, zorder=5))   # recovery: deepest -> start
    up = f[4] > r[4]
    ax.annotate("the fall", (X[5], f[5]), xytext=(0, 9 if up else -9), textcoords="offset points", color=c, fontsize=9.5, ha="center", va="bottom" if up else "top")
    ax.annotate("the recovery", (X[2], r[2]), xytext=(0, -9 if up else 9), textcoords="offset points", color=c, fontsize=9.5, ha="center", va="top" if up else "bottom")
    ax.set_xlim(-1.02, 0.02); ax.set_ylim(-0.08, 1.0); ax.set_yticks([]); ax.set_xticks([-0.94, -0.5, -0.06]); ax.set_xticklabels(["deepest point", "halfway down", "start"], fontsize=8.5)
    ax.set_ylabel("volatility", fontsize=9) if k == 0 else None
    fig.text(x0, 0.445, head, color=c, fontsize=13, weight="semibold", va="bottom")
    fig.text(x0, 0.415, note, color=qr.TEXT_LIGHT, fontsize=9.5, style="italic", va="bottom")
    ax.spines["left"].set_visible(False); ax.tick_params(length=0, pad=4)
qr.save(fig, "fig4_twenty_one_crises", os.path.join(HERE, "figures"))
