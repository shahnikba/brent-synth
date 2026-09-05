# Report — SPEC 5, model ladder

## Status
Complete. 195 tests passing (75 new). 84 (model, origin) pairs in 2m50s
cold (budget 15 min), seconds warm from cache.

## Result
**Champion selected on 2011-2018: `gjr_skewt`. It holds on the held-out
2019-2024 confirmation origins.** FIGARCH is a close second on both
(2.266 vs 2.156 on confirmation — a gap of 0.11 ranks).

## Three caveats
1. `gjr_skewt_vt` is scored on 3 origins, not 8 — it raises at 5 of 8
   selection origins (fitted persistence >= 1). The pre-registered rule
   does not handle unbalanced availability, and I did not invent a
   correction. Its 2.958 is not comparable.
2. The variance-targeting experiment did not pay off (3.391 vs 2.156).
3. 2020 is labelled `normal`, not `stress`: origin 2019's window IS
   calendar 2020 and is correctly `stress`. In that window every model
   put the realised year at rank ~0.99 for drawdown and ~0.00 for worst
   day; `iid_t` took 14 exceedances at 99% out of 252.

## Two defects found while building
- arch names the skew-t degrees of freedom `eta`, not `nu` — the new
  wrapper raised `KeyError: 'nu'`. Fixed with an explicit alias map.
- `regime_labels` crashed on fewer than three distinct origins
  (`pd.qcut` "Bin edges must be unique"). Replaced with rank-based
  terciles.

## Deviations
`MarkovRegression.fit` takes `rng`, not `search_rng`. `initial_value_vol`
works for FIGARCH, so the burn=1000 fallback was unnecessary. The
pre-registration hash covers the candidate list, so scoring a subset is
refused.

## Caveat on density filtering
`forecast_density` builds on the concatenated sample and calls `.fix()`;
arch's backcast then sees the test data. Effect decays as beta^t; the
hand-recursion test confirms agreement to 1e-10 across 252 days.
