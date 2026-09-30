"""Null model. A GJR-GARCH(1,1)-t fitted to the same S&P 500 returns is simulated over the same number
of days (200 runs). Its one-month expected volatility, scaled to VIX units, plays the role of the VIX.
Question: does a standard asymmetric volatility model reproduce the counter-clockwise VIX loop?
Dating matters: the proxy at the close of day t uses h[t+1] (known at that close). Using h[t] instead
shifts the proxy one day late and reverses the answer; both are reported."""
import json, os
import numpy as np
from common import load, episodes, loop_areas, gjr_fit, gjr_filter, gjr_next_variance, gjr_month_vol, gjr_simulate, OUT

d = load(); r = 100 * np.log(d.S).diff().dropna().values
mu, om, a, g, b, nu = gjr_fit(r); e, h = gjr_filter(r, mu, om, a, g, b)
vix = d.VIX.values[1:]; out = {}
for name in ("dated_at_close", "one_day_late"):
    hh_real = gjr_next_variance(e, h, om, a, g, b) if name == "dated_at_close" else h
    fv = gjr_month_vol(hh_real, om, a, g, b); c = float(np.sum(fv * vix) / np.sum(fv * fv))   # least-squares scale to VIX units
    rng = np.random.default_rng(1); frac, n_eps = [], []
    for _ in range(200):
        ret, hs = gjr_simulate(len(r), mu, om, a, g, b, nu, h[0], rng); es = ret - mu
        hs = gjr_next_variance(es, hs, om, a, g, b) if name == "dated_at_close" else hs
        S = np.exp(np.cumsum(ret / 100)); dd = 100 * (S / np.maximum.accumulate(S) - 1)
        V = c * gjr_month_vol(hs, om, a, g, b); eps = episodes(V, 30, 20)
        if eps:
            A = loop_areas(dd, V, eps); frac.append((A > 0).mean()); n_eps.append(len(A))
    frac = np.array(frac)
    out[name] = dict(scale_to_vix=round(c, 3), corr_model_vol_vix=round(float(np.corrcoef(fv, vix)[0, 1]), 3), runs=len(frac),
                     median_episodes=float(np.median(n_eps)), ccw_fraction_mean=round(float(frac.mean()), 3),
                     ccw_fraction_p5_p95=[round(float(np.percentile(frac, q)), 3) for q in (5, 95)],
                     runs_at_or_above_observed_18_of_21=int(np.sum(frac >= 18 / 21)))
    np.save(os.path.join(OUT, f"02_null_ccw_fractions_{name}.npy"), frac)
os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "02_null_model_gjr.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
