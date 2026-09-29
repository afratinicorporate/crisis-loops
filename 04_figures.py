"""The two figures. Requires the results of scripts 01-03."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.patches import FancyArrowPatch
from common import load, OUT, HERE
import figure_style as qr

FIG = os.path.join(HERE, "figures"); d = load()
R1 = json.load(open(os.path.join(OUT, "01_loop_orientation.json")))
R2 = json.load(open(os.path.join(OUT, "02_null_model_gjr.json"))); null = np.load(os.path.join(OUT, "02_null_ccw_fractions.npy"))
R3 = json.load(open(os.path.join(OUT, "03_step_scale_test.json")))

# ---------------- Figure 1: cycle scale ----------------
fig = qr.frame("Same crisis. Opposite loops.",
               "The price leaves a crisis slowly. The price of fear leaves it at once. Same 21 crises, same dates, two orbits turning opposite ways.",
               "OBSERVED",
               "S&P 500 and VIX daily closes, Yahoo Finance, 3 Jan 2000 - 29 Sep 2026. Crisis = VIX above 30, from the last to the first close below 20 (21 episodes). "
               "Realized volatility = 21-day rolling, annualised. Orientation = sign of the signed area of each closed orbit. Null model: GJR-GARCH(1,1)-t fitted to the same returns, 200 simulated histories.")
ax = fig.add_axes([0.075, 0.20, 0.44, 0.57])
seg = d.loc["2020-02-21":"2021-02-12"].rolling(5, min_periods=1).mean()
for col, c, lab, pos in [("VIX", qr.OXBLOOD, "VIX\n(price of fear)", (-6, 70)), ("rv21", qr.GREEN, "realized volatility\n(the price itself)", (-8, 88))]:
    ax.plot(seg.dd, seg[col], color=c, lw=2)
    n = len(seg)
    for f in (0.03, 0.06, 0.09, 0.13, 0.18, 0.25, 0.35, 0.50):
        k = int(f * n)
        ax.add_patch(FancyArrowPatch((seg.dd.iloc[k], seg[col].iloc[k]), (seg.dd.iloc[k + 2], seg[col].iloc[k + 2]),
                                     arrowstyle="-|>", mutation_scale=22, color=c, lw=0, zorder=4))
    ax.text(*pos, lab, color=c, fontsize=10.5, weight="semibold", ha="center", linespacing=1.1)
ax.text(-33.5, 20, "counter-clockwise: fear drains\nbefore the price recovers", color=qr.OXBLOOD, fontsize=9.5, style="italic", ha="left", linespacing=1.15)
ax.text(-1, 45, "clockwise: the price\nstays agitated\nwhile it climbs back", color=qr.GREEN, fontsize=9.5, style="italic", ha="right", linespacing=1.15)
ax.set_xlim(-36, 1); ax.set_ylim(10, 100)
ax.set_xlabel("distance below the all-time high, %", fontsize=10); ax.set_ylabel("volatility, %", fontsize=10)
ax.set_title("One crisis, two orbits: Feb 2020 - Feb 2021", loc="left", fontsize=11.5, color=qr.TEXT_MID, pad=8); qr.tidy(ax)

ax = fig.add_axes([0.665, 0.20, 0.28, 0.57])
rows = [("VIX", R1["paired_vix"], qr.OXBLOOD), ("realized, 21 days", R1["paired_realized_21d"], qr.GREEN),
        ("realized, 10 days", R1["paired_realized_10d"], qr.GREEN), ("GARCH-filtered", R1["paired_garch_filtered"], qr.GREEN)]
for i, (lab, s, c) in enumerate(rows):
    y = len(rows) - i
    ax.barh(y, 100 * s["ccw"] / s["n"], color=c, height=0.55)
    ax.text(100 * s["ccw"] / s["n"] + 2, y, f'{s["ccw"]}/{s["n"]}', va="center", fontsize=10, color=c, weight="semibold")
    ax.text(-2, y, lab, va="center", ha="right", fontsize=10, color=qr.TEXT_MID)
lo, hi = np.percentile(null, [5, 95])
ax.plot([100 * lo, 100 * hi], [0, 0], color=qr.STRONG_GRAY, lw=2); ax.plot(100 * null.mean(), 0, "o", color=qr.STRONG_GRAY, ms=7)
ax.text(-2, 0, "GARCH, simulated", va="center", ha="right", fontsize=10, color=qr.TEXT_MID)
ax.text(100 * hi + 2, 0, f"mean {100*null.mean():.0f}%\n0 of 200 runs reach 18/21", va="center", fontsize=8.5, color=qr.STRONG_GRAY, linespacing=1.1)
ax.axvline(50, color=qr.MID_GRAY, lw=0.9, ls=(0, (4, 3))); ax.text(50, 4.65, "no preferred direction", ha="center", fontsize=8.5, color=qr.STRONG_GRAY, style="italic")
ax.set_xlim(0, 100); ax.set_ylim(-0.7, 4.9); ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0", "25", "50", "75", "100%"])
ax.set_xlabel("share of the 21 crises turning counter-clockwise", fontsize=10)
ax.set_title("Which way does each orbit turn?", loc="left", fontsize=11.5, color=qr.TEXT_MID, pad=8)
qr.tidy(ax, grid_y=False); ax.xaxis.grid(True, color=qr.LIGHT_GRAY, lw=0.6)
qr.save(fig, "fig1_same_crisis_opposite_loops", FIG)

# ---------------- Figure 2: step scale ----------------
H, K = 10, 20
d["lv"] = np.log(d.VIX); d["vel"] = d.lv - d.lv.shift(10); d["mem"] = d.lv.shift(21)
d["fv"] = d.VIX.shift(-H) - d.VIX; d["fdd"] = d.dd.shift(-H) - d.dd
x = d.dropna(subset=["lv", "dd", "vel", "mem", "fv", "fdd"]); Z = x[["lv", "dd", "vel", "mem"]].values
odd = np.asarray(x.index.year % 2 == 1); tr, te = odd, ~odd
m, s = Z[tr].mean(0), Z[tr].std(0); _, idx = cKDTree((Z[tr] - m) / s).query((Z[te] - m) / s, k=K)
up = (x.fv.values[tr][idx] > 0).mean(1); agree = np.maximum(up, 1 - up)
j = np.argsort(np.abs(agree - np.median(agree)))[0]            # a typical day: median agreement, not a chosen one
day = x.index[te][j]; nb = idx[j]
fig = qr.frame("Same state today. Any direction in ten days.",
               "Days that sit on the same point of the map do not move together over the next two weeks. The structure lives in the cycle, not in the step.",
               "OBSERVED",
               f"S&P 500 and VIX, Yahoo Finance, 2000-2026 ({R3['days']:,} days). State = log VIX, drawdown, 10-day VIX velocity, VIX 21 days earlier. "
               "Neighbours = 20 closest days from years of opposite parity. Error = squared error of the neighbours' forecast of the next-10-day VIX move, relative to the plain average (1 = no information). "
               "Control = last coordinate shuffled in time.")
ax = fig.add_axes([0.075, 0.20, 0.40, 0.57])
for k in nb:
    ax.add_patch(FancyArrowPatch((0, 0), (x.fdd.values[tr][k], x.fv.values[tr][k]), arrowstyle="-|>", mutation_scale=11,
                                 color=qr.OXBLOOD if x.fv.values[tr][k] > 0 else qr.GREEN, lw=1.2, alpha=0.85))
ax.plot(0, 0, "o", color=qr.TEXT_DARK, ms=5)
lim_x = max(3, 1.15 * np.abs(x.fdd.values[tr][nb]).max()); lim_y = max(3, 1.15 * np.abs(x.fv.values[tr][nb]).max())
ax.set_xlim(-lim_x, lim_x); ax.set_ylim(-lim_y, lim_y); ax.axhline(0, color=qr.MID_GRAY, lw=0.7); ax.axvline(0, color=qr.MID_GRAY, lw=0.7)
ax.text(0.03, 0.96, f"{int((x.fv.values[tr][nb] > 0).sum())} of 20: VIX up", transform=ax.transAxes, color=qr.OXBLOOD, fontsize=10, weight="semibold", va="top")
ax.text(0.03, 0.06, f"{int((x.fv.values[tr][nb] <= 0).sum())} of 20: VIX down", transform=ax.transAxes, color=qr.GREEN, fontsize=10, weight="semibold")
ax.set_xlabel("change in drawdown over next 10 days, points", fontsize=10); ax.set_ylabel("change in VIX over next 10 days, points", fontsize=10)
ax.set_title(f"The 20 nearest states to {day:%d %b %Y}, ten days later", loc="left", fontsize=11.5, color=qr.TEXT_MID, pad=8); qr.tidy(ax)
ax = fig.add_axes([0.585, 0.20, 0.36, 0.57])
res = R3["results"]; xs = np.arange(1, 5)
ax.bar(xs - 0.18, [r["error_vix"] for r in res], width=0.34, color=qr.OXBLOOD, label="state")
ax.bar(xs[1:] + 0.18, [r["control_error_vix"] for r in res[1:]], width=0.34, color=qr.MID_GRAY)
ax.axhline(1, color=qr.TEXT_DARK, lw=1, ls=(0, (4, 3))); 
for xi, r in zip(xs, res):
    ax.text(xi - 0.18, r["error_vix"] + 0.006, f'{r["error_vix"]:.2f}', ha="center", fontsize=9, color=qr.OXBLOOD)
ax.text(0.52, 1.133, "red: real coordinates\ngrey: last coordinate shuffled in time", fontsize=9, color=qr.TEXT_LIGHT, style="italic", va="top", linespacing=1.2)
ax.set_xticks(xs); ax.set_xticklabels(["VIX", "+ drawdown", "+ velocity", "+ memory"], fontsize=9.5)
ax.set_xlim(0.5, 4.5); ax.set_ylim(0.9, 1.135); ax.set_yticks([0.9, 0.95, 1.0, 1.05, 1.1]);  ax.set_ylabel("forecast error  (1 = no information)", fontsize=10)
ax.set_title("Adding coordinates does not help", loc="left", fontsize=11.5, color=qr.TEXT_MID, pad=8); qr.tidy(ax)
qr.save(fig, "fig2_same_state_any_direction", FIG)
json.dump(dict(example_day=str(day.date()), median_neighbour_agreement=round(float(np.median(agree)), 3),
               mean_neighbour_agreement=round(float(agree.mean()), 3)), open(os.path.join(OUT, "04_fig2_example.json"), "w"), indent=1)
print(day.date(), np.median(agree), agree.mean())
