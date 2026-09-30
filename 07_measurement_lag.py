"""What the loop direction measures: the lead or lag of the volatility measure relative to the price.
(a) Backward 21-day realized volatility shifted forward by k days: counter-clockwise loops out of 21 for each k.
(b) Garman-Klass range volatility, which needs no past days: backward windows (lagging) versus centred windows.
(c) Peak timing: trading days between the volatility peak and the price trough, per crisis."""
import json, os
import numpy as np
from common import load, load_ohlc, garman_klass, episodes, loop_areas, OUT

full = load("1999-06-01", "2026-09-29"); d = full.loc["2000-01-01":]; eps = episodes(d.VIX, 30, 20)
gk = garman_klass(load_ohlc()); out = {"vix_ccw": int((loop_areas(d.dd, d.VIX, eps) > 0).sum()), "n": len(eps)}
out["shifted_window"] = [dict(shift_days=k, lag_of_window_centre=10 - k,
                              ccw=int((loop_areas(d.dd, full.rv21.shift(-k).reindex(d.index), eps) > 0).sum())) for k in range(0, 22)]
rng = []
for m in (1, 3, 5, 11, 21):
    for centred in (False, True):
        if m == 1 and centred: continue
        v = (100 * np.sqrt(252 * gk.rolling(m, center=centred).mean())).reindex(d.index).bfill().ffill()
        rng.append(dict(window=m, centred=centred, lag_days=0.0 if centred else (m - 1) / 2, ccw=int((loop_areas(d.dd, v, eps) > 0).sum())))
out["range_volatility"] = rng
series = {"vix": d.VIX, "realized_back21": d.rv21, "realized_back10": d.rv10,
          "range_centred11": (100 * np.sqrt(252 * gk.rolling(11, center=True).mean())).reindex(d.index)}
rows = []
for a, b in eps:
    seg = d.iloc[a:b + 1]; t = int(np.argmin(seg.dd.values))
    row = dict(start=str(seg.index[0].date()), trough=str(seg.index[t].date()), end=str(seg.index[-1].date()))
    for k, s in series.items():
        row[f"peak_minus_trough_{k}"] = int(np.nanargmax(s.iloc[a:b + 1].values) - t)
    rows.append(row)
out["peak_timing"] = {k: dict(median_days=float(np.median([r[f"peak_minus_trough_{k}"] for r in rows])),
                              peak_at_or_before_trough=int(sum(r[f"peak_minus_trough_{k}"] <= 0 for r in rows))) for k in series}
out["peak_timing_by_episode"] = rows
os.makedirs(OUT, exist_ok=True); json.dump(out, open(os.path.join(OUT, "07_measurement_lag.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "peak_timing_by_episode"}, indent=1))
