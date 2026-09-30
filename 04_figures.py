"""The two figures. Requires the results of scripts 01-03."""
import json, os, sys
import numpy as np
from scipy.spatial import cKDTree
from matplotlib.patches import FancyArrowPatch
from common import load, OUT, HERE
import figure_style as qr

FIG = os.path.join(HERE, "figures"); d = load()
R1 = json.load(open(os.path.join(OUT, "01_loop_orientation.json")))
R3 = json.load(open(os.path.join(OUT, "03_step_scale_test.json")))

# ---------------- Figure 1: one crisis, three clocks ----------------
from common import load_ohlc, garman_klass
R6 = json.load(open(os.path.join(OUT, "07_measurement_lag.json")))
d["range_c11"] = (100 * np.sqrt(252 * garman_klass(load_ohlc()).rolling(11, center=True).mean())).reindex(d.index)
fig = qr.frame("Same crisis, opposite loops: the difference is the clock of the instrument.",
               "Volatility measured as it happens turns one way. Volatility measured on a trailing window turns the other way.",
               "OBSERVED",
               "S&P 500 and VIX, Yahoo Finance, 3 Jan 2000 - 29 Sep 2026. Crisis = VIX above 30, from the last to the first close below 20 (21 episodes). "
               "No-lag volatility = Garman-Klass daily range, centred average. Trailing volatility = close-to-close returns of the last m days. Lag of a trailing window = (m - 1) / 2 days.")
ax = fig.add_axes([0.075, 0.20, 0.44, 0.57]); seg = d.loc["2020-02-21":"2021-02-12"].rolling(5, min_periods=1).mean()
for col, c, lab, pos in [("VIX", qr.OXBLOOD, "VIX", (-33.2, 70)), ("range_c11", qr.AMBER, "price volatility,\nno lag", (-25.5, 13)),
                         ("rv21", qr.GREEN, "price volatility,\nlast 21 days", (-8.5, 86))]:
    ax.plot(seg.dd, seg[col], color=c, lw=2); n = len(seg)
    for f in (0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.48):
        k = int(f * n)
        ax.add_patch(FancyArrowPatch((seg.dd.iloc[k], seg[col].iloc[k]), (seg.dd.iloc[k + 2], seg[col].iloc[k + 2]),
                                     arrowstyle="-|>", mutation_scale=20, color=c, lw=0, zorder=4))
    ax.text(*pos, lab, color=c, fontsize=10.5, weight="semibold", ha="center", linespacing=1.1)
ax.set_xlim(-36, 1); ax.set_ylim(6, 112)
ax.set_xlabel("distance below the all-time high, %", fontsize=10); ax.set_ylabel("volatility, %", fontsize=10)
ax.set_title("One crisis, three clocks: Feb 2020 - Feb 2021", loc="left", fontsize=11.5, color=qr.TEXT_MID, pad=8); qr.tidy(ax)

ax = fig.add_axes([0.70, 0.20, 0.245, 0.57])
g = {(x["window"], x["centred"]): x["ccw"] for x in R6["range_volatility"]}
rows = [("VIX", R1["paired_vix"]["ccw"], qr.OXBLOOD), ("range, centred 11d  (lag 0)", g[(11, True)], qr.AMBER), ("range, centred 5d  (lag 0)", g[(5, True)], qr.AMBER),
        ("range, last 3d  (lag 1)", g[(3, False)], qr.GREEN), ("returns, last 5d  (lag 2)", R1["paired_realized_5d"]["ccw"], qr.GREEN),
        ("returns, last 10d  (lag 4.5)", R1["paired_realized_10d"]["ccw"], qr.GREEN), ("returns, last 21d  (lag 10)", R1["paired_realized_21d"]["ccw"], qr.GREEN)]
for i, (lab, k, c) in enumerate(rows):
    y = len(rows) - i; ax.barh(y, k, color=c, height=0.58)
    ax.text(k + 0.4, y, f"{k}/21", va="center", fontsize=10, color=c, weight="semibold"); ax.text(-0.5, y, lab, va="center", ha="right", fontsize=9.5, color=qr.TEXT_MID)
ax.axvline(10.5, color=qr.MID_GRAY, lw=0.9, ls=(0, (4, 3))); ax.set_xlim(0, 23.5); ax.set_ylim(0.3, 7.7); ax.set_yticks([]); ax.set_xticks([0, 7, 14, 21])
ax.set_xlabel("counter-clockwise loops out of 21", fontsize=10)
ax.set_title("Direction by lag of the measure", loc="left", fontsize=11.5, color=qr.TEXT_MID, pad=8); qr.tidy(ax, grid_y=False); ax.xaxis.grid(True, color=qr.LIGHT_GRAY, lw=0.6)
qr.save(fig, "fig1_same_crisis_opposite_loops", FIG)

# ---------------- Figure 5: direction versus lag, and the null model ----------------
fa = np.load(os.path.join(OUT, "02_null_ccw_fractions_dated_at_close.npy")); fb = np.load(os.path.join(OUT, "02_null_ccw_fractions_one_day_late.npy"))
fig = qr.frame("A few days of lag turn the crisis loop backwards.",
               "Left: counter-clockwise loops out of 21 as the volatility window is moved in time. Right: a standard GARCH reproduces the direction once it is dated correctly.",
               "OBSERVED",
               "S&P 500 and VIX, Yahoo Finance, 2000-2026, 21 crises. Left: 21-day realized volatility with the window shifted; lag = distance between window centre and today. "
               "Right (simulated): GJR-GARCH(1,1)-t fitted to the same returns, 200 histories; proxy dated at the close (black) or one day late (grey).")
ax = fig.add_axes([0.075, 0.22, 0.44, 0.55]); sh = R6["shifted_window"]
ax.axvspan(0, 10.5, color=qr.ACCENT_SOFT, lw=0); ax.axhline(18, color=qr.OXBLOOD, lw=1.2, ls=(0, (4, 3))); ax.axhline(10.5, color=qr.MID_GRAY, lw=0.8)
ax.plot([r["lag_of_window_centre"] for r in sh], [r["ccw"] for r in sh], color=qr.TEXT_DARK, lw=2, marker="o", ms=4)
ax.text(-10.8, 18.4, "VIX: 18 of 21", color=qr.OXBLOOD, fontsize=10.5, weight="semibold")
ax.text(5.3, 20.0, "lagging instrument", color=qr.OXBLOOD, fontsize=10, style="italic", ha="center"); ax.text(-5.5, 1.0, "leading instrument\n(uses future days)", color=qr.STRONG_GRAY, fontsize=10, style="italic", ha="center", linespacing=1.15)
ax.set_xlim(-11.5, 10.8); ax.set_ylim(0, 22); ax.set_xticks(range(-10, 11, 5)); ax.set_yticks([0, 3, 7, 14, 18, 21])
ax.set_xlabel("lag of the instrument, trading days (0 = coincident)", fontsize=10); ax.set_ylabel("counter-clockwise loops out of 21", fontsize=10); qr.tidy(ax)
ax = fig.add_axes([0.60, 0.22, 0.345, 0.55]); bins = np.linspace(0, 1, 26)
ax.hist(fb, bins=bins, color=qr.MID_GRAY); ax.hist(fa, bins=bins, histtype="step", color=qr.TEXT_DARK, lw=2); top = ax.get_ylim()[1]; ax.set_ylim(0, top * 1.55)
ax.axvline(18 / 21, color=qr.OXBLOOD, lw=1.6, ls=(0, (4, 3)))
ax.text(0.30, top * 1.5, f"one day late\nmean {100*fb.mean():.0f}%\n{int((fb >= 18/21).sum())} of 200 reach 18/21", color=qr.STRONG_GRAY, fontsize=9.5, ha="center", va="top", linespacing=1.2)
ax.text(0.66, top * 1.5, f"dated at the close\nmean {100*fa.mean():.0f}%\n{int((fa >= 18/21).sum())} of 200 reach 18/21", color=qr.TEXT_DARK, fontsize=9.5, ha="center", va="top", weight="semibold", linespacing=1.2)
ax.text(18 / 21 + 0.012, top * 0.95, "observed", color=qr.OXBLOOD, fontsize=9.5, weight="semibold")
ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50", "100%"]); ax.set_yticks([]); ax.set_xlabel("share of counter-clockwise loops in a simulated history", fontsize=10); qr.tidy(ax, grid_y=False)
qr.save(fig, "fig5_direction_versus_lag", FIG)

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
