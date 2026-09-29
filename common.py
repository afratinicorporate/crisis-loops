"""Shared functions: data, drawdown, realized volatility, crisis episodes, loop orientation, GJR-GARCH."""
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "spx_vix.csv")
OUT = os.path.join(HERE, "results")
START, END = "2000-01-01", "2026-09-29"


def load(start=START, end=END):
    """S&P 500 and VIX daily closes, drawdown (%) and realized volatility (%, annualised)."""
    d = pd.read_csv(DATA, index_col=0, parse_dates=True).dropna()
    d.columns = ["S", "VIX"]
    d["dd"] = 100 * (d.S / d.S.cummax() - 1)                 # distance below the running all-time high
    lr = np.log(d.S).diff()
    for w in (10, 21):
        d[f"rv{w}"] = 100 * np.sqrt(252 * (lr**2).rolling(w).mean())
    return d.loc[start:end]


def episodes(v, hi=30.0, lo=20.0):
    """Crisis episodes on series v: v crosses above `hi`; the episode runs from the last value below
    `lo` before the crossing to the first value below `lo` after it. Overlapping episodes are merged."""
    v = np.asarray(v); eps = []; i = 0
    while i < len(v):
        if v[i] > hi:
            a = i
            while a > 0 and v[a] >= lo:
                a -= 1
            b = i
            while b < len(v) - 1 and v[b] >= lo:
                b += 1
            if eps and a <= eps[-1][1]:
                eps[-1][1] = max(eps[-1][1], b)
            else:
                eps.append([a, b])
            i = b + 1
        else:
            i += 1
    return eps


def signed_area(x, y):
    """Shoelace formula on the closed orbit: A > 0 counter-clockwise, A < 0 clockwise."""
    x = np.append(x, x[0]); y = np.append(y, y[0])
    return 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])


def loop_areas(dd, vol, eps):
    dd, vol = np.asarray(dd), np.asarray(vol)
    return np.array([signed_area(dd[a:b + 1], vol[a:b + 1]) for a, b in eps])


# ---------- GJR-GARCH(1,1) with Student-t innovations (Glosten, Jagannathan & Runkle 1993) ----------
def gjr_filter(r, mu, om, a, g, b):
    e = r - mu; h = np.empty_like(e); h[0] = e.var()
    for t in range(1, len(e)):
        h[t] = om + (a + g * (e[t - 1] < 0)) * e[t - 1]**2 + b * h[t - 1]
    return e, h


def gjr_fit(r):
    """r: daily log returns in percent. Returns (mu, omega, alpha, gamma, beta, nu)."""
    def nll(p):
        mu, om, a, g, b, nu = p
        if om <= 0 or a < 0 or g < 0 or b < 0 or a + g / 2 + b >= 0.999 or nu <= 2.1:
            return 1e10
        e, h = gjr_filter(r, mu, om, a, g, b)
        ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2) * h)
              - (nu + 1) / 2 * np.log1p(e**2 / (h * (nu - 2))))
        return -ll.sum()
    res = minimize(nll, [0.05, 0.02, 0.01, 0.12, 0.88, 7], method="Nelder-Mead",
                   options=dict(maxiter=6000, xatol=1e-6, fatol=1e-6))
    return res.x


def gjr_month_vol(h, om, a, g, b):
    """Model expectation of annualised volatility (%) over the next 21 trading days."""
    p = a + g / 2 + b; lr = om / (1 - p)
    return np.sqrt(252 * (lr + (h - lr) * np.mean(p ** np.arange(1, 22))))


def gjr_simulate(n, mu, om, a, g, b, nu, h0, rng):
    z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2))
    h = np.empty(n); e = np.empty(n); h[0] = h0
    for t in range(n):
        if t > 0:
            h[t] = om + (a + g * (e[t - 1] < 0)) * e[t - 1]**2 + b * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
    return mu + e, h
