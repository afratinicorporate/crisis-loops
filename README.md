# Crisis loops: structure at the scale of the cycle, noise at the scale of the step

S&P 500 and VIX, daily, 3 Jan 2000 – 29 Sep 2026. Every number below is produced by the scripts in this folder from public data.

## Question

Place each trading day in a two-dimensional state space: **drawdown** (distance of the S&P 500 below its running all-time high) and **volatility**. During a crisis the state traces a closed orbit. Two questions:

1. **Cycle scale.** Does the orbit turn the same way in every crisis? The sign of its signed area, `A = ½ Σ (xₜ·yₜ₊₁ − xₜ₊₁·yₜ)`, gives the direction: `A > 0` counter-clockwise, `A < 0` clockwise.
2. **Step scale.** Do days that sit on the same point of the state space move the same way over the next 10 trading days?

## Definitions (fixed before looking at results)

- **Crisis episode:** the VIX closes above 30. The episode runs from the last close below 20 before that to the first close below 20 after it. Overlapping episodes are merged. This gives 21 episodes, 2000–2026.
- **Volatility, two measures, same episode windows:** (a) the VIX, which is implied volatility and so includes the variance risk premium; (b) volatility of the price itself, measured three ways: 21-day realized, 10-day realized, and GJR-GARCH filtered volatility (no rolling window).
- **Null model:** GJR-GARCH(1,1) with Student-t innovations, fitted by maximum likelihood to the same returns. It is simulated 200 times over the same number of days. Its forward one-month volatility, scaled to VIX units, stands in for the VIX.
- **Step-scale test:** coordinates are added in a fixed order: log VIX; drawdown; 10-day VIX velocity; log VIX 21 days earlier. For each day, the 20 nearest days (standardised coordinates) are taken only from years of opposite parity, and they forecast the next-10-day move. The score is squared error relative to the plain training mean, so 1.0 means no information. Control: the last coordinate is permuted in time (20 draws).

## Results

| Cycle scale: share of the 21 crises turning counter-clockwise | |
|---|--:|
| VIX | **18/21** (97% of total loop area; one-sided binomial p = 0.0007) |
| Realized volatility, 21 days | 3/21 |
| Realized volatility, 10 days | 4/21 |
| GJR-GARCH filtered volatility | 3/21 |
| GJR-GARCH simulated, 200 histories | 34% on average (5th–95th percentile 14–54%); **0 of 200 reach 18/21** |

- On the same windows, the VIX loop and the realized-volatility loop turn in opposite directions in **15 of 21** crises.
- VIX robustness to thresholds: 26/30 (25/18), 18/21 (30/20), 12/14 (35/22), 7/9 (40/25).
- 1990–1999: **no signal** (4/5, 2/3 and 1/4 counter-clockwise across thresholds), with few and shallow episodes.

| Step scale: forecast error of the next-10-day VIX move | real | control |
|---|--:|--:|
| log VIX | 1.00 | – |
| + drawdown | 1.07 | 1.00 |
| + VIX velocity | 1.02 | 1.01 |
| + VIX 21 days earlier | 1.02 | 0.98 |

## How to read the direction of a loop

Take one price level, say 15% below the peak, and compare the volatility there on the way down and on the way back. If it is lower on the way back, the recovery road runs below the fall and the loop turns counter-clockwise: this is the VIX, where fear is priced out before the price recovers. If it is higher on the way back, the loop turns clockwise: this is the price's own volatility, where the market is still agitated while it climbs back. See `figures/fig3_loop_direction_schematic.png`. The average loop measured on the 21 crises is in `figures/fig4_twenty_one_crises.png`: for the VIX the recovery runs below the fall at every depth, for realized volatility it runs above.

## Reading

- **Cycle scale: structure.** The orbit of the VIX turns counter-clockwise: fear is priced out before the price recovers. The orbit of the price's own volatility turns clockwise: the market stays agitated while it climbs back. A standard asymmetric GARCH reproduces the second orbit and fails on the first.
- **Step scale: noise.** Ten days ahead, the state (up to four coordinates) carries no usable information about the next move.

## What is not new

The asymmetric response of volatility to returns (Black 1976; Glosten, Jagannathan & Runkle 1993) and the time-varying variance risk premium (Carr & Wu 2009; Bollerslev, Tauchen & Zhou 2009) are established results. Time-reversal asymmetry of volatility is documented by Zumbach (2009). The contribution here is a geometric, episode-by-episode measurement: two loops with opposite orientation on the same crises, and a null model that explains one and not the other.

## Limitations

- 21 episodes, and they are not independent: 2000–2002 and 2022 each contain several.
- The VIX is not ATM implied volatility. It includes skew and the variance premium, and that is part of what the loop measures.
- The step-scale test uses one horizon (10 days), one neighbour method and four coordinates chosen in advance. Other state variables could change that result.
- The example crisis in Figure 1 (Feb 2020 – Feb 2021) is illustrative. The counts in the table are the evidence.
- The signal is absent in the 1990s.

## Run

```
pip install -r requirements.txt
python 00_download_data.py     # S&P 500 and VIX from Yahoo Finance -> data/spx_vix.csv
python 01_loop_orientation.py  # cycle scale: VIX, realized, GARCH-filtered, thresholds, 1990s
python 02_null_model_gjr.py    # null model: 200 GJR-GARCH histories
python 03_step_scale_test.py   # step scale: nearest-neighbour test with controls
python 04_figures.py           # figures/fig1_*.png, figures/fig2_*.png
python 05_schematic.py         # figures/fig3_*.png: what clockwise / counter-clockwise means (no data)
python 06_figure_mirror.py     # figures/fig4_*.png: all 21 crises as paired bars, plus the average loop
```

Results are written to `results/` as JSON. Raw data are not redistributed: run the download script. Runtime is a few minutes, dominated by the GARCH simulations.

## References

- Black, F. (1976). Studies of Stock Price Volatility Changes. *Proceedings of the 1976 Meeting of the Business and Economic Statistics Section, American Statistical Association*, 177–181.
- Glosten, L. R., Jagannathan, R. & Runkle, D. E. (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance*, 48(5), 1779–1801.
- Carr, P. & Wu, L. (2009). Variance Risk Premiums. *Review of Financial Studies*, 22(3), 1311–1341.
- Bollerslev, T., Tauchen, G. & Zhou, H. (2009). Expected Stock Returns and Variance Risk Premia. *Review of Financial Studies*, 22(11), 4463–4492.
- Zumbach, G. (2009). Time Reversal Invariance in Finance. *Quantitative Finance*, 9(5), 505–515.
- Ewing, J. A. (1885). Experimental Researches in Magnetism. *Philosophical Transactions of the Royal Society of London*, 176, 523–640.

## License

[PolyForm Noncommercial 1.0.0](LICENSE.md). Free for research, study, teaching and other noncommercial use. Commercial use requires permission from the author.
