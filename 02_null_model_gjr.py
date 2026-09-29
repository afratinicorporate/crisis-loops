"""Null model. A GJR-GARCH(1,1)-t fitted to the same S&P 500 returns is simulated over the same number
of days (200 runs). Its forward-looking volatility, scaled to VIX units, plays the role of the VIX.
Question: does a standard asymmetric volatility model reproduce the counter-clockwise VIX loop?"""
import json, os
import numpy as np
from common import load, episodes, loop_areas, gjr_fit, gjr_filter, gjr_month_vol, gjr_simulate, OUT

d = load(); r = 100 * np.log(d.S).diff().dropna().values
mu, om, a, g, b, nu = gjr_fit(r); _, h = gjr_filter(r, mu, om, a, g, b)
fv = gjr_month_vol(h, om, a, g, b); vix = d.VIX.values[1:]
c = float(np.sum(fv * vix) / np.sum(fv * fv))           # least-squares scale from model vol to VIX units
rng = np.random.default_rng(1); frac, share, n_eps = [], [], []
for _ in range(200):
    ret, hh = gjr_simulate(len(r), mu, om, a, g, b, nu, h[0], rng)
    S = np.exp(np.cumsum(ret / 100)); dd = 100 * (S / np.maximum.accumulate(S) - 1)
    V = c * gjr_month_vol(hh, om, a, g, b); eps = episodes(V, 30, 20)
    if eps:
        A = loop_areas(dd, V, eps); frac.append((A > 0).mean()); share.append(A[A > 0].sum() / np.abs(A).sum()); n_eps.append(len(A))
out = dict(scale_to_vix=round(c, 3), corr_model_vol_vix=round(float(np.corrcoef(fv, vix)[0, 1]), 3), runs=200,
           median_episodes=float(np.median(n_eps)), ccw_fraction_mean=round(float(np.mean(frac)), 3),
           ccw_fraction_p5_p95=[round(float(np.percentile(frac, q)), 3) for q in (5, 95)],
           runs_at_or_above_observed_18_of_21=int(np.sum(np.array(frac) >= 18 / 21)),
           ccw_area_share_mean=round(float(np.mean(share)), 3))
os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "02_null_model_gjr.json"), "w"), indent=1)
np.save(os.path.join(OUT, "02_null_ccw_fractions.npy"), np.array(frac))
print(json.dumps(out, indent=1))
