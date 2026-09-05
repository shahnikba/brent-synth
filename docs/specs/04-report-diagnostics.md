# Report — SPEC 2, statistical diagnostics

## Status
Complete. 19 diagnostics tests, 27 passing in total. No matplotlib in the
module.

## Moments (n=4754 at the time of writing; 4753 after the data fix)

| Statistic | Value |
|---|---|
| mean | 0.000050 |
| std | 0.024345 |
| skew | -0.780 |
| excess kurtosis | 11.708 |
| annualised vol | 38.6% |

## Dependence
- ACF(r) lag 1 = -0.019 -> no linear predictability in the mean.
- ACF(r^2) lag 1 = 0.204, still 0.097 at lag 40 -> slow decay.
- Ljung-Box on r^2, lag 10: p = 1.7e-213.

## Tails

| | Left (losses) | Right (gains) |
|---|---|---|
| GPD threshold | 0.0519 | 0.0447 |
| xi (shape) | 0.212 | 0.127 |
| beta (scale) | 0.0200 | 0.0188 |
| exceedances | 114 | 123 |
| 1/xi | 4.72 | 7.91 |
| Hill alpha (median k in [50,500]) | 2.86 | 2.94 |

## Estimator validation
Hill recovers alpha = 3.76-4.09 across five seeds on t(4) (true 4);
separates t(4) from t(10), recovering ~10 within 35%; POT recovers
xi = 0.25 +/- 0.08 from an exact GPD sample.

## Findings
1. Returns are near-unpredictable in the mean but volatility clusters
   overwhelmingly and decays slowly.
2. Excess kurtosis 11.7, Jarque-Bera p underflows to 0 — Gaussian
   innovations are ruled out.
3. Tails are asymmetric: xi = 0.212 on losses vs 0.127 on gains.
4. Worst day (2020-04-21, -0.2798 log return) is 11.5 sigma. Under a
   Gaussian that is 7.1e-31 per day.

## Caveats
- Hill and POT disagree (alpha ~2.9 vs 1/xi ~4.7). Different effective
  thresholds, not a bug. Do not quote either as *the* tail index without
  stating estimator and threshold.
- The Hill plateau read (median over k in [50,500]) is a judgement call,
  not specified. A single-k assertion was seed-fragile.

## Note
The right tail's 1/xi = 7.91 was later withdrawn: the significance gate
added in review 11/12 found xi/se = 1.25, short of the 95% mark.
