# Crisis loops: the direction of the loop is the clock of the instrument

S&P 500 and VIX, daily, 3 Jan 2000 – 29 Sep 2026. Every number below is produced by the scripts in this folder from public data.

> **Corrected on 30 September 2026.** The first version of this repository contained three statements that later checks did not support. They are listed under [Corrections](#corrections). The counts were right; two interpretations and one comparison were wrong.

## Question

Place each trading day in a two-dimensional state space: **drawdown** (distance of the S&P 500 below its running all-time high) and **volatility**. During a crisis the state traces a closed orbit.

1. **Cycle scale.** Does the orbit turn the same way in every crisis, and what decides the direction? The sign of the signed area, `A = ½ Σ (xₜ·yₜ₊₁ − xₜ₊₁·yₜ)`, gives the direction: `A > 0` counter-clockwise, `A < 0` clockwise.
2. **Step scale.** Do days that sit on the same point of the state space move the same way over the next 10 trading days?

## How to read the direction of a loop

Take one price level, say 15% below the peak, and compare the volatility there on the way down and on the way back. If it is lower on the way back, the loop turns counter-clockwise. If it is higher, the loop turns clockwise. Equivalently, the direction is the sign of the lead or lag of the volatility measure relative to the price: a measure that peaks before the price trough turns counter-clockwise, one that peaks after it turns clockwise. See `figures/fig3_loop_direction_schematic.png`.

## Definitions (fixed before looking at results)

- **Crisis episode:** the VIX closes above 30. The episode runs from the last close below 20 before that to the first close below 20 after it. Overlapping episodes are merged. This gives 21 episodes, 2000–2026.
- **Volatility measures, all on the same episode windows:**
  - the VIX;
  - trailing realized volatility from close-to-close returns of the last 5, 10 or 21 days, which lags by about half the window;
  - the same 21-day window shifted forward in time;
  - Garman–Klass range volatility, computed from the open, high, low and close of the same day, averaged over trailing or centred windows;
  - GJR-GARCH filtered volatility.
- **Null model:** GJR-GARCH(1,1) with Student-t innovations, fitted by maximum likelihood to the same returns and simulated 200 times. Its one-month expected volatility, scaled to VIX units, stands in for the VIX. The proxy at the close of day *t* uses the variance of day *t+1*, which is known at that close.
- **Step-scale test:** coordinates are added in a fixed order: log VIX; drawdown; 10-day VIX velocity; log VIX 21 days earlier. For each day, the 20 nearest days (standardised coordinates) are taken only from years of opposite parity, and they forecast the next-10-day move. The score is squared error relative to the plain training mean, so 1.0 means no information. Control: the last coordinate is permuted in time (20 draws).

## Results

### Cycle scale

| Volatility measure | Lag (days) | Counter-clockwise loops out of 21 |
|---|--:|--:|
| VIX | – | **18** |
| Range volatility, centred 11-day average | 0 | 20 |
| Range volatility, centred 5-day average | 0 | 19 |
| Close-to-close returns, centred 21-day window | 0 | 18 |
| Range volatility, last 3 days | 1 | 8 |
| Close-to-close returns, last 5 days | 2 | 8 |
| Close-to-close returns, last 10 days | 4.5 | 4 |
| Close-to-close returns, last 21 days | 10 | 3 |
| GJR-GARCH filtered, dated at the close | – | 15 |

- **The VIX loop is counter-clockwise in 18 of 21 crises.** At the same price level the VIX is lower on the way back than on the way down. The count is stable across thresholds: 26/30 (25/18), 12/14 (35/22), 7/9 (40/25).
- **The direction depends on the lag of the measure.** Volatility of the price measured without lag turns the same way as the VIX. Trailing windows turn the other way, and a lag of one to five days is enough to flip the direction (`figures/fig1_*.png`, `figures/fig5_*.png`).
- **Peak timing.** The VIX peaks on the day of the price trough or before it in 19 of 21 crises (median: one day before). Trailing 21-day volatility peaks after the trough (median: six days after) and precedes it in only 5.
- **Null model.** Dated at the close, the simulated GJR-GARCH gives 65% counter-clockwise loops on average (5th–95th percentile 44–85%), and 10 of 200 histories reach 18/21. The standard model therefore reproduces most of the observed direction; the VIX exceeds it only slightly.
- **1990–1999:** no conclusion is possible (three shallow episodes).

### Step scale

| Forecast error of the next-10-day VIX move | real | control |
|---|--:|--:|
| log VIX | 1.00 | – |
| + drawdown | 1.07 | 1.00 |
| + VIX velocity | 1.02 | 1.01 |
| + VIX 21 days earlier | 1.02 | 0.98 |

Ten days ahead, the state (up to four coordinates) carries no usable information about the next move.

## Reading

- **In a crisis, volatility peaks at the price trough or before it, and the recovery takes place at lower volatility than the fall.** This is the counter-clockwise loop.
- **A risk measure built on a trailing window reports the peak of risk after the trough has passed.** Its loop turns clockwise. The opposition is between coincident and lagging instruments, not between implied and realized volatility.
- **A standard asymmetric GARCH, dated correctly, produces mostly the same direction.** Leverage effect and mean reversion are enough for most of it.
- **The regularity lives at the scale of the episode, not of the day.**

## Corrections

The first version of this repository stated the following. Each statement is followed by what the checks showed.

1. *"The VIX loop and the loop of the price's own volatility turn opposite ways; the price is still agitated while it climbs back."* The counts (18/21 counter-clockwise for the VIX, 18/21 clockwise for trailing 21-day volatility) are correct. The interpretation is not. A trailing 21-day window lags current volatility by about ten days. Measured without lag, the price's own volatility turns the same way as the VIX (15 to 20 of 21, depending on the measure). Script `07_measurement_lag.py`.
2. *"A GJR-GARCH fails on the VIX loop: 34% counter-clockwise on average, 0 of 200 histories reach 18/21; GARCH-filtered volatility is clockwise (3/21 counter-clockwise)."* These figures came from dating the model's variance one day late. With the correct dating the numbers are 65%, 10 of 200, and 15/21. Scripts `01_loop_orientation.py` and `02_null_model_gjr.py` now report both datings.
3. *"A null model that explains one loop and fails on the other."* Withdrawn, as a consequence of points 1 and 2.

The step-scale result is unchanged.

## What is not new

The asymmetric response of volatility to returns (Black 1976; Glosten, Jagannathan & Runkle 1993), its fast decay for indices (Bouchaud, Matacz & Potters 2001), the asymmetric response of implied-volatility indexes (Giot 2005), and the information content of implied volatility beyond past returns (Christensen & Prabhala 1998; Jiang & Tian 2005) are established results. The contribution here is a geometric, episode-by-episode measurement of the order of events in a crisis, and the demonstration that its sign is set by the lag of the instrument.

## Limitations

- 21 episodes, and they are not independent: merging episodes less than six months apart leaves about 11 crises (10 counter-clockwise). No significance level computed on 21 independent trials is reported.
- 2008 and 2020 account for more than half of the total loop area, so counts are more informative than area shares.
- The VIX is not ATM implied volatility: it includes skew and the variance premium. Values before 22 September 2003 are back-calculated.
- Daily closes cannot order events within a day. Since the direction flips with a lag of one day, higher-frequency data would be the right tool to measure the flip precisely.
- The centred measures use future days and are not available in real time. They are used only to establish what the price's volatility does when observed without lag.
- The step-scale test uses one horizon, one neighbour method and four coordinates chosen in advance.
- The example crisis in Figure 1 (Feb 2020 – Feb 2021) is illustrative. The counts are the evidence.

## Run

```
pip install -r requirements.txt
python 00_download_data.py     # S&P 500 closes and OHLC, VIX closes, from Yahoo Finance -> data/
python 01_loop_orientation.py  # cycle scale: VIX, trailing realized, GARCH-filtered, thresholds, 1990s
python 02_null_model_gjr.py    # null model: 200 GJR-GARCH histories, dated at the close and one day late
python 03_step_scale_test.py   # step scale: nearest-neighbour test with controls
python 07_measurement_lag.py   # direction versus lag of the measure, range volatility, peak timing
python 04_figures.py           # figures/fig1_*, fig2_*, fig5_*
python 05_schematic.py         # figures/fig3_*: what clockwise / counter-clockwise means (no data)
python 06_figure_mirror.py     # figures/fig4_*: all 21 crises as paired bars, plus the average loop
```

Results are written to `results/` as JSON. Raw data are not redistributed: run the download script. Runtime is a few minutes, dominated by the GARCH simulations.

## References

- Black, F. (1976). Studies of Stock Price Volatility Changes. *Proceedings of the 1976 Meetings of the American Statistical Association, Business and Economic Statistics Section*, 177–181.
- Bouchaud, J.-P., Matacz, A. & Potters, M. (2001). Leverage Effect in Financial Markets: The Retarded Volatility Model. *Physical Review Letters*, 87(22), 228701.
- Carr, P. & Wu, L. (2006). A Tale of Two Indices. *The Journal of Derivatives*, 13(3), 13–29.
- Christensen, B. J. & Prabhala, N. R. (1998). The Relation between Implied and Realized Volatility. *Journal of Financial Economics*, 50(2), 125–150.
- Garman, M. B. & Klass, M. J. (1980). On the Estimation of Security Price Volatilities from Historical Data. *The Journal of Business*, 53(1), 67–78.
- Giot, P. (2005). Relationships between Implied Volatility Indexes and Stock Index Returns. *The Journal of Portfolio Management*, 31(3), 92–100.
- Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *The Journal of Finance*, 48(5), 1779–1801.
- Jiang, G. J. & Tian, Y. S. (2005). The Model-Free Implied Volatility and Its Information Content. *The Review of Financial Studies*, 18(4), 1305–1342.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md). Free for research, study, teaching and other noncommercial use. Commercial use requires permission from the author.
