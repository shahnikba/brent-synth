# SPEC 4 — Validation + Auto Report

## Goal
Compare synthetic paths to real Brent returns under sampling
uncertainty. Bootstrap the real data to get acceptance bands, test
whether synthetic statistics fall inside, emit an HTML report with
plots and explicit pass/fail.

## Amend SPEC 3 first
Change `initial_var='last'` to start paths from the one-step-ahead
forecast variance sigma^2_{T+1} (apply one GJR recursion step from the
last observed shock), not sigma^2_T.

## Core idea
Real data is ONE realisation; its sample statistics are random. So the
threshold is the real data's own bootstrap sampling band, not a
hand-set tolerance. Both sides MUST be computed on the same 252-day
horizon.

## Module (src/brent_synth/validation.py)
- `bootstrap_bands(returns, horizon=252, n_boot=2000, block=20, seed=7)`
  — stationary block bootstrap, NOT iid.
- `compute_stats(paths_or_sample)` — mean, std, skew, excess_kurtosis,
  var_95/99, es_95/99, left_q01/q05, right_q95/q99, acf_sq_lag1/5/10,
  max_drawdown.
- `validate(synth_paths, returns, **kwargs) -> pd.DataFrame` — synthetic
  value, real (lo, median, hi), PASS/FAIL, band-width position.
- `make_report(...)` — self-contained HTML, base64 PNGs, five plots,
  a "Failure modes" section with a marked TODO and numbers pre-filled.
- `run_validation(seed=42) -> str` — the whole pipeline in one call.

## Tests
Band ordering; compute_stats keys on both shapes; real 252-blocks mostly
PASS; iid Gaussian noise FAILS on excess_kurtosis and acf_sq; one row per
statistic with a boolean pass column; determinism under fixed seeds.

## Notes
- Both sides on the SAME horizon (252).
- Expect acf_sq at long lag to be borderline/FAIL — GARCH geometric
  decay vs oil's slow decay. That is the honest failure mode.

## Done when
`run_validation()` writes a self-contained HTML report with the pass/fail
table and all five plots, every test passes INCLUDING the Gaussian-noise
known-fail guard, and output is deterministic.
