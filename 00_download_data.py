"""Download S&P 500 (^GSPC) and VIX (^VIX) daily closes from Yahoo Finance into data/spx_vix.csv."""
import os
import yfinance as yf
from common import DATA
os.makedirs(os.path.dirname(DATA), exist_ok=True)
d = yf.download(["^GSPC", "^VIX"], start="1990-01-01", end="2026-09-30", progress=False, auto_adjust=False)["Close"]
d[["^GSPC", "^VIX"]].dropna().to_csv(DATA)
print("saved", DATA, len(d))
