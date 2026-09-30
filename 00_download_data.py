"""Download S&P 500 (^GSPC) and VIX (^VIX) daily data from Yahoo Finance:
data/spx_vix.csv (closes) and data/spx_ohlc.csv (open, high, low, close of the index)."""
import os
import yfinance as yf
from common import DATA, HERE
os.makedirs(os.path.dirname(DATA), exist_ok=True)
d = yf.download(["^GSPC", "^VIX"], start="1990-01-01", end="2026-09-30", progress=False, auto_adjust=False)["Close"]
d[["^GSPC", "^VIX"]].dropna().to_csv(DATA)
o = yf.download("^GSPC", start="1999-01-01", end="2026-09-30", progress=False, auto_adjust=False)
o.columns = [c[0] for c in o.columns]
o[["Open", "High", "Low", "Close"]].to_csv(os.path.join(HERE, "data", "spx_ohlc.csv"))
print("saved", DATA, len(d), len(o))
