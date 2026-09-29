"""Step scale. Pre-declared test: do days that sit in the same state move the same way over the next
10 trading days? Coordinates added in order: log VIX, drawdown, 10-day VIX velocity, log VIX 21 days
earlier. For each day, the 20 nearest days (standardised coordinates) taken ONLY from years of the
opposite parity (odd years predict even years and vice versa) forecast the next-10-day move.
Score: squared error relative to the plain training mean (1.0 = no information).
Control: the last added coordinate randomly permuted in time (20 draws)."""
import json, os
import numpy as np
from scipy.spatial import cKDTree
from common import load, OUT

d = load(); H, K = 10, 20
d["lv"] = np.log(d.VIX); d["vel"] = d.lv - d.lv.shift(10); d["mem"] = d.lv.shift(21)
d["f_lv"] = d.lv.shift(-H) - d.lv; d["f_dd"] = d.dd.shift(-H) - d.dd
x = d.dropna(subset=["lv", "dd", "vel", "mem", "f_lv", "f_dd"])
COORDS = ["lv", "dd", "vel", "mem"]; Y = x[["f_lv", "f_dd"]].values; odd = np.asarray(x.index.year % 2 == 1)

def evaluate(Z):
    pred = np.empty_like(Y); base = np.empty_like(Y)
    for tr, te in [(odd, ~odd), (~odd, odd)]:
        m, s = Z[tr].mean(0), Z[tr].std(0)
        _, idx = cKDTree((Z[tr] - m) / s).query((Z[te] - m) / s, k=K)
        pred[te] = Y[tr][idx].mean(1); base[te] = Y[tr].mean(0)
    e = ((Y - pred)**2).sum(0) / ((Y - base)**2).sum(0)
    return dict(error_vix=round(float(e[0]), 3), error_drawdown=round(float(e[1]), 3),
                wrong_vix_direction=round(float(np.mean(np.sign(pred[:, 0]) != np.sign(Y[:, 0]))), 3))

rng = np.random.default_rng(0); res = []
for dim in range(1, 5):
    Z = x[COORDS[:dim]].values; r = dict(coordinates=COORDS[:dim], **evaluate(Z))
    if dim > 1:
        sur = []
        for _ in range(20):
            Zp = Z.copy(); Zp[:, -1] = rng.permutation(Zp[:, -1]); sur.append(evaluate(Zp))
        r["control_error_vix"] = round(float(np.mean([s["error_vix"] for s in sur])), 3)
        r["control_wrong_vix_direction"] = round(float(np.mean([s["wrong_vix_direction"] for s in sur])), 3)
    res.append(r); print(r)
os.makedirs(OUT, exist_ok=True); json.dump(dict(days=len(x), horizon_days=H, neighbours=K, results=res),
                                           open(os.path.join(OUT, "03_step_scale_test.json"), "w"), indent=1)
