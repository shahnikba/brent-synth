# Review — the two sides compute different statistics

`compute_stats` ravels whatever it is given. On the synthetic side that
is `paths.ravel()` — 1,260,000 values pooled across 5,000 paths. On the
bootstrap side each sample is a 1-D 252-day series, so it is 252 values.
The band is built from a statistic on 252 observations and compared
against a statistic on 1.26M pooled observations. For anything nonlinear
those are not the same quantity.

Pooling mixes 5,000 paths whose realised volatility spans 0.0103 to
0.3781 — a 37x spread. A mixture at wildly different vol levels is far
more leptokurtic than any single path, and between-path level differences
masquerade as autocorrelation.

Both reported failures are artifacts:

| statistic | pooled | per-path median | real band | pooled | per-path |
|---|---|---|---|---|---|
| excess_kurtosis | 99.55 | 2.04 | [0.63, 24.05] | FAIL | PASS |
| acf_sq_lag10 | 0.2331 | 0.0531 | [-0.065, 0.216] | FAIL | PASS |
| acf_sq_lag1 | 0.3414 | 0.0805 | [-0.028, 0.430] | PASS | PASS |
| acf_sq_lag5 | 0.2064 | 0.0633 | [-0.038, 0.263] | PASS | PASS |

The module's opening docstring is right that comparing a 252-day path
against full-history statistics would be a category error — the code
commits the mirror-image error. `_plot_acf` averages ACF per path (the
correct way) while the acceptance table pools: the plot and the table in
the same document disagree by 4x.

There is also a power problem: the pooled synthetic value has essentially
zero sampling noise, so the test only asks whether the central value
lands in the band. It has no power to detect wrong dispersion — and the
per-path spread suggests that is where the real issue is (synthetic std
[0.014, 0.065] against a real band [0.016, 0.037]).

## Four smaller ones
- var_95 = -left_q05 and var_99 = -left_q01, bit-identical. "14 of 16" is
  really 14 of 14 independent checks.
- The bootstrap runs twice: `run_validation` calls `bootstrap_bands`, then
  `validate` calls it again with the same seed.
- A stale label: the report renders "persistence (a + g/2 + b)" after the
  model fix changed the formula.
- `make_report` hardcodes the bootstrap seed to 7 regardless of the caller.
- `bootstrap_bands` still does `values[np.isfinite(values)]` — the same
  silent gap-splicing refused in diagnostics and model.
