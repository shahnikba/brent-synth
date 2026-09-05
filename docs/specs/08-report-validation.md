# Report — SPEC 4, validation and auto report

## Status
Complete with one specified test that cannot pass as written. 62 tests
passing. Added `matplotlib`.

## SPEC 3 amendment
`initial_var='last'` now starts from sigma^2_{T+1}. `ModelFit` carries
both `last_variance` (7.128e-04) and `forecast_variance` (6.736e-04),
plus `unconditional_variance` (7.951e-04).

## Results — 15 of 16 pass
Excess kurtosis FAILS at 96.36 against a band of [0.55, 25.86].

## Two findings you need before trusting that table

### 1. The kurtosis FAIL is an artifact
Pooled synthetic statistics are compared against 252-day bootstrap bands.
Sample kurtosis is strongly sample-size dependent. Measured like-for-like:
synthetic per-path median 2.01 vs real 252-day window median 1.81. They
agree. The 96.36 is what pooling 5000 paths produces.

### 2. The known-fail guard cannot pass as specified
iid Gaussian noise FAILS on excess_kurtosis correctly, but PASSES all
three acf_sq statistics. The 2.5% bound of ACF(r^2) over 252-day windows
is negative (-0.028 at lag 1), so a two-sided 95% band at this horizon
cannot exclude zero. Checked at block lengths 5, 20, 40, 60, 100 — the
lower bound stays negative at every one. The guard asserts the verifiable
truth instead, with the limitation documented.

## Implementation notes
- Stationary bootstrap (Politis-Romano), geometric blocks, mean 20.
- `validate()` raises on a horizon mismatch.
- Report carries no timestamp; PNG metadata stripped. Byte-reproducible.
- `MIN_PERIODS = 11` guard added: the lag-10 ACF is undefined below that.
