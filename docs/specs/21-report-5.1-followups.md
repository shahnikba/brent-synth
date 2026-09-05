# Report — SPEC 5.1

## Status
All four complete and pushed. 209 tests passing.

## 1. Amendment 1
`failure_rule` defaults to `"worst_rank"`; `"drop"` retained for
comparison. `gjr_skewt_vt` moves from **3rd (2.958, scored on 3 of 8
origins) to last (5.088, all 8)**. Champion unchanged under both rulings;
confirmation agrees under both.

## 2. Backcast leak — removed, and smaller than it looked
`fix()` takes no backcast argument, so the recursion is driven through
`volatility.compute_variance` with a train-only backcast. Day one now
opens at exactly sigma^2_{T+1}: bit-identical for garch_normal, one ulp
(2.7e-20) for gjr_skewt and figarch where the raw/percent arithmetic
paths differ.

**Correction to my SPEC 5 caveat:** old and new agree bit-for-bit at 2500
training days — the backcast's influence falls below float64 resolution
long before the test window. The leak was real in principle and had zero
numerical effect. The fix makes it structural.

## 3. Headline numbers — they complicate the champion
- **Persistence >= 1 at five origins** (2014-2018, 1.00012 to 1.00066).
- **The champion is not well calibrated**: PIT hump, p = 0.0004. The only
  model reading flat is `gjr_skewt_vt`, just moved to last.
- **The champion does not significantly beat FIGARCH**: DM statistics are
  negative on both nll (-0.910, p 0.363) and tail CRPS (-1.344, p 0.179).

## 4. Tags
`spec5-preregistered` -> 5891ab1, `spec5-run` -> 358b216. Both annotated
and pushed.
