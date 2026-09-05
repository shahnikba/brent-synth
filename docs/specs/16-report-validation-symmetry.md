# Report — validation symmetry fix

## Status
Main bug and all four smaller ones fixed. 114 tests passing (13 new).
**Both previously reported failures were artifacts. All 14 independent
statistics now pass on location, and the real defect is dispersion.**

## Confirmed
37x vol spread (0.0103 to 0.3781); pooled excess kurtosis 99.55 vs
per-path median 2.03; ACF lag 10 pooled 0.2331 vs per-path 0.0498;
var/quantile aliases bit-identical; `bootstrap_bands` accepted 5 NaNs in
605 observations.

## The fix
`compute_path_stats(paths)` computes every statistic per path;
`compute_stats` is the median across paths. A 1-D bootstrap sample and a
1-row array are now literally the same computation. ACF is computed
row-wise and vectorised, never across a concatenation.

## Corrected results
14 of 14 independent statistics pass on location. **5 fail on spread:**
daily volatility 2.13x, 5th percentile 2.11x, 95th percentile 2.60x
(over-dispersed); skewness 0.50x, excess kurtosis 0.34x (under-dispersed).

`validate` now reports `synth_lo`/`synth_hi` and `dispersion_ratio`
against bounds [0.5, 2.0].

## Four smaller ones
Duplicates marked as aliases and excluded from the headline; the
bootstrap vectorised into one pass and reused rather than recomputed —
**the pipeline went from 17.9s to 2.6s**; the persistence label follows
the corrected formula; `make_report` honours `boot_seed`;
`bootstrap_bands` refuses non-finite input.

## Verification
Reverting `compute_stats` to pooled fails two regression tests built so
the artifact is unmistakable: two groups of Gaussian paths, one 10x the
vol of the other.
