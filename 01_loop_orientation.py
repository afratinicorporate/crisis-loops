"""Cycle scale. Does the crisis orbit in the (drawdown, volatility) plane turn the same way every time?
Test 1: episodes defined on the VIX, loop measured with the VIX.
Test 2 (paired): the SAME episode windows, loop measured with backward realized volatility (21, 10, 5 days)
and with GJR-GARCH filtered volatility dated at the close (and, for the record, dated one day late)."""
import json, os
import numpy as np
from scipy.stats import binomtest
from common import load, episodes, loop_areas, gjr_fit, gjr_filter, gjr_next_variance, gjr_month_vol, OUT

d = load(); out = {}
def summary(A):
    k = int((A > 0).sum()); n = len(A)
    return dict(ccw=k, n=n, p_one_sided=float(binomtest(k, n, 0.5, alternative="greater").pvalue),
                ccw_area_share=round(float(A[A > 0].sum() / np.abs(A).sum()), 3))

for hi, lo in [(25, 18), (30, 20), (35, 22), (40, 25)]:
    out[f"vix_{hi}_{lo}"] = summary(loop_areas(d.dd, d.VIX, episodes(d.VIX, hi, lo)))

eps = episodes(d.VIX, 30, 20)
r = 100 * np.log(d.S).diff().dropna().values
mu, om, a, g, b, nu = gjr_fit(r)
e, h = gjr_filter(r, mu, om, a, g, b)
garch_vol = np.r_[np.nan, gjr_month_vol(gjr_next_variance(e, h, om, a, g, b), om, a, g, b)]      # dated at the close
garch_vol_one_day_late = np.r_[np.nan, gjr_month_vol(h, om, a, g, b)]                           # the earlier, wrong dating
A_vix = loop_areas(d.dd, d.VIX, eps)
paired = {"vix": A_vix, "realized_21d": loop_areas(d.dd, d.rv21, eps),
          "realized_10d": loop_areas(d.dd, d.rv10, eps), "realized_5d": loop_areas(d.dd, d.rv5, eps),
          "garch_filtered": loop_areas(d.dd, garch_vol, eps), "garch_filtered_one_day_late": loop_areas(d.dd, garch_vol_one_day_late, eps)}
for k, A in paired.items():
    out[f"paired_{k}"] = summary(A)
out["paired_opposite_vix_vs_realized_21d"] = int(np.sum(np.sign(A_vix) != np.sign(paired["realized_21d"])))
out["episodes"] = [dict(start=str(d.index[a].date()), end=str(d.index[b].date()),
                        area_vix=round(float(A_vix[i]), 1), area_realized_21d=round(float(paired["realized_21d"][i]), 1))
                   for i, (a, b) in enumerate(eps)]
d90 = load("1990-01-01", "1999-12-31")
for hi, lo in [(25, 18), (30, 20), (35, 22)]:
    out[f"vix_1990s_{hi}_{lo}"] = summary(loop_areas(d90.dd, d90.VIX, episodes(d90.VIX, hi, lo)))
out["gjr_params"] = dict(zip(["mu", "omega", "alpha", "gamma", "beta", "nu"], np.round([mu, om, a, g, b, nu], 4).tolist()))
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, "01_loop_orientation.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "episodes"}, indent=1))
