# SPEC 5 — Model ladder and temporal validation

## Goal
Turn the single-model, in-sample validation into a pre-registered,
out-of-time comparison of several candidate scenario generators. Every
candidate sits behind one interface; every candidate is fitted only on
data before an origin date and scored only on the year after it; a
champion is selected on the early origins and confirmed on the late ones.

Existing modules are not modified except where stated. The current
GJR-GARCH model is wrapped, not rewritten, and a regression test pins its
output.

## Package layout
    src/brent_synth/candidates/{__init__,base,iid_t,arch_backed,gjr_skewt,ms_variance}.py
    src/brent_synth/{scoring,backtest,comparison_report}.py
    tests/{test_candidates,test_scoring,test_backtest}.py
    docs/preregistration.md

## Interface
`ScenarioModel` (name, step, fit) and `FittedModel` (name, params,
n_params, loglik, simulate, forecast_density). `DensityForecast` with
`logpdf`/`cdf`/`ppf`, concrete `LocationScaleForecast` and
`MixtureForecast` (per-day mixture of K Gaussians, ppf by vectorised
bisection).

## Candidates (ladder step)
iid_t (0), garch_normal (1), gjr_skewt (2), gjr_skewt_vt (3),
figarch_skewt (4), ms_variance (5). All refuse non-finite input and
fewer than 500 observations. Simulation always starts from sigma^2_{T+1}.

## Scoring
`TAU_GRID = linspace(0.0005, 0.9995, 999)`. nll, crps (quantile-integral
form), tail_crps (tau <= 0.05), pit, var_exceedances, kupiec_pvalue,
christoffersen_pvalue, sign_split_nll, diebold_mariano (Newey-West HAC).

## Path-level scores
`percentile_ranks` with mid-rank for ties; `path_coverage_loss` = mean
|rank - 0.5|. Adds `worst_day` without touching `validation.STAT_NAMES`.

## Backtest
ORIGIN_YEARS 2011-2024; SELECTION 2011-2018; CONFIRMATION 2019-2024;
TEST_HORIZON 252; BACKTEST_N_PATHS 2000; BACKTEST_SEED 2024;
STRESS_WEIGHT 2.0. Regime = tercile of realised test-window volatility.
Four ranking losses: mean nll, mean tail_crps, |exc99 - 0.01|,
path_coverage_loss. Per-origin ranks, weighted mean, stress weight 2.0.
Ties to lower step. Per-(candidate, origin) parquet cache keyed by the
pre-registration hash.

## Pre-registration
`preregister` writes docs/preregistration.md with a SHA-256; `run_backtest`
refuses to run if it is absent or differs, unless --allow-unregistered.

## Report
Nine sections: pre-registration, scoreboard, scores by regime, crisis
years, PIT histograms, path-rank histograms, parameter drift,
Diebold-Mariano, failure modes.

## Done when
preregister writes and commits the plan; run completes for all six
candidates at all 14 origins; report writes both formats with all nine
sections and the four figures; every test passes including the leakage
test and the GJR regression test.
