# Report — diagnostics fixes

## Status
All seven items fixed. 94 tests passing (17 new). Every claim reproduced
before fixing.

## 1. mean_excess ties — fixed
Reproduced exactly. Exceedances now selected by value via
`searchsorted(..., side="right")`. Thresholds equal to the maximum are
dropped (undefined there, and with ties at the max the old slice would
divide by zero). Impact figures confirmed: left tail 2 tied pairs, max
deviation 1.54e-05; right tail 1 pair, 1.19e-05.

## 2. Hill guard — fixed, with a correction
Exactly-identical losses already returned NaN on this build. It is
**near-constant** input that slips through: noise 1e-15 -> alpha 6.5e15,
1e-13 -> 1.8e13, 1e-11 -> 2.9e11. Guard is now
`MIN_MEAN_LOG_EXCESS = 1e-6`.

## 3. Gap splicing — fixed
`_as_ordered_array` raises on non-finite input for acf and Ljung-Box.
The docstring states the assumption: observations are treated as
consecutive; weekend and holiday gaps are expected and not flagged.

## 4. threshold_quantile — documented
Confirmed: u = 0.0519 is the 97.60th percentile of all returns, 2.40%
tail probability, 114 exceedances against the ~240 a reader would expect.

## 5. tail_index on significance — fixed, changes a published number
Gated on `xi_se = (1 + xi)/sqrt(n_exceedances)`, one-sided 95%.

| tail | xi | SE | xi/SE | tail_index |
|---|---|---|---|---|
| left | 0.2117 | 0.1135 | 1.87 | 4.723 kept |
| right | 0.1269 | 0.1016 | 1.25 | now NaN |

My SPEC 2 report quoted 1/xi = 7.91 for the right tail with confidence it
did not have.

## Minor — all five fixed
