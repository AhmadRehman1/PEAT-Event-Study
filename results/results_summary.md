_Generated 2026-09-23. Source: yfinance prices, yfinance earnings._

**Sample: 626 usable events (556 beats, 70 misses) across 40 companies** (of 40 in the universe), reactions from 2021-09-24 to 2026-08-20. A further 163 in-line reports (|surprise| <= 2%) were excluded by design.

**Table 1 - Post-announcement DRIFT (starts at the close of day 0; the headline result)**  
Average cumulative abnormal return from the day after the reaction. `***` p<0.01, `**` p<0.05, `*` p<0.10.

| Window | Beats: avg | t-stat | p | Misses: avg | t-stat | p | Beat minus miss | t-stat | p |
|---|---|---|---|---|---|---|---|---|---|
| CAR[1,1] | +0.00% | 0.04 | 0.967 | -0.04% | -0.14 | 0.886 | +0.04% | 0.15 | 0.881 |
| CAR[1,5] | -0.02% | -0.15 | 0.879 | -0.05% | -0.08 | 0.933 | +0.03% | 0.04 | 0.967 |
| CAR[1,10] | +0.08% | 0.36 | 0.720 | -0.24% | -0.35 | 0.730 | +0.31% | 0.44 | 0.664 |
| CAR[1,20] | -0.13% | -0.43 | 0.667 | +0.29% | 0.32 | 0.751 | -0.42% | -0.44 | 0.663 |

**Table 2 - Full reaction window (includes the day-0 jump)**  
Shown for context: this mostly measures the immediate price reaction, which is NOT the anomaly.

| Window | Beats: avg | t-stat | p | Misses: avg | t-stat | p | Beat minus miss | t-stat | p |
|---|---|---|---|---|---|---|---|---|---|
| CAR[0,1] | +0.85% | 3.06 | 0.002*** | -3.03% | -3.25 | 0.002*** | +3.88% | 3.99 | <0.001*** |
| CAR[0,5] | +0.83% | 2.61 | 0.009*** | -3.04% | -2.64 | 0.010** | +3.87% | 3.23 | 0.002*** |
| CAR[0,10] | +0.93% | 2.66 | 0.008*** | -3.23% | -2.82 | 0.006*** | +4.16% | 3.47 | <0.001*** |
| CAR[0,20] | +0.72% | 1.73 | 0.084* | -2.70% | -2.01 | 0.049** | +3.42% | 2.43 | 0.017** |

**Plain-English verdict (generated automatically from the numbers above)**

- Over the 20 trading days after the reaction, beats moved -0.13% versus the market-adjusted norm (t = -0.43, p = 0.667) - NOT statistically distinguishable from zero at the 5% level.
- Misses moved +0.29% (t = 0.32, p = 0.751) - NOT statistically distinguishable from zero at the 5% level.
- The beat-minus-miss gap was -0.42% (t = -0.44, p = 0.663), i.e. we cannot conclude the two groups drift differently at the 5% level.
- Robustness - clustering by earnings season (20 quarters): the 20-day gap is +0.21% (t = 0.18, p = 0.856) - not significant at 5%.

**Robustness checks** (20-day and 5-day drift, beat minus miss)

| Check | Variant | Window | n beats | n misses | Beat minus miss | t-stat | p |
|---|---|---|---|---|---|---|---|
| abnormal-return definition | market model (beta-adjusted S&P 500) | car1_5 | 556 | 70 | +0.03% | 0.04 | 0.967 |
| abnormal-return definition | market model (beta-adjusted S&P 500) | car1_20 | 556 | 70 | -0.42% | -0.44 | 0.663 |
| abnormal-return definition | market-adjusted (stock minus S&P 500) | car1_5 | 556 | 70 | +0.11% | 0.18 | 0.858 |
| abnormal-return definition | market-adjusted (stock minus S&P 500) | car1_20 | 556 | 70 | -0.39% | -0.38 | 0.706 |
| abnormal-return definition | sector-adjusted (stock minus sector ETF) | car1_5 | 556 | 70 | +0.06% | 0.13 | 0.899 |
| abnormal-return definition | sector-adjusted (stock minus sector ETF) | car1_20 | 556 | 70 | -0.77% | -0.98 | 0.328 |
| beat/miss threshold | +/-2% | car1_20 | 556 | 70 | -0.42% | -0.44 | 0.663 |
| beat/miss threshold | +/-5% | car1_20 | 373 | 38 | -0.67% | -0.47 | 0.643 |
| beat/miss threshold | +/-10% | car1_20 | 203 | 28 | -0.81% | -0.64 | 0.529 |
| sub-period | first half (to/from 2024-03-15) | car1_20 | 276 | 46 | -1.09% | -1.05 | 0.298 |
| sub-period | second half (to/from 2024-03-15) | car1_20 | 280 | 24 | +0.68% | 0.34 | 0.739 |
| timing | only events with a known announcement time | car1_20 | 556 | 70 | -0.42% | -0.44 | 0.663 |

**Data audit** (how many events were removed, and why)

- raw earnings rows: 800
- dropped: missing estimate or actual (incl. future reports): 1
- dropped: |consensus EPS| < $0.10: 6
- dropped: fewer than 20 trading days of data after day 0: 4
- events kept for analysis (all labels): 789
