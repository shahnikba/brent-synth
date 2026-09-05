# Pre-registration — Brent scenario-generator comparison

Fixed before any origin was scored. `run_backtest` recomputes the hash
below from the live constants and refuses to run if they have drifted.

## Candidates

| Step | Name | Description |
|---|---|---|
| 0 | `iid_t` | Independent Student-t returns, fitted by maximum likelihood. |
| 1 | `garch_normal` | Step 1: GARCH(1,1) with Gaussian innovations. |
| 2 | `gjr_skewt` | GJR-GARCH(1,1,1) with skewed-t innovations — the current model. |
| 3 | `gjr_skewt_vt` | Step 3: GJR-skewt with omega pinned to the sample variance. |
| 4 | `figarch_skewt` | Step 4: FIGARCH(1,d,1) with skewed-t innovations. |
| 5 | `ms_variance` | Two-regime Gaussian mixture with Markov-persistent variance. |

## Ranking losses

Each is computed per (model, origin), ranked across models at that
origin (1 = best), then averaged with the stress weighting below.

1. `mean_nll`
2. `mean_tail_crps`
3. `exc99_abs_error`
4. `path_coverage_loss`

## Origins and split

- Origin for year Y: last trading day on or before Y-12-31.
- Training: every return up to and including the origin.
- Test: the next 252 returns.
- Origins: [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
- Selection: [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
- Confirmation: [2019, 2020, 2021, 2022, 2023, 2024]

## Simulation

- Paths per (model, origin): 2000
- Seed: 2024
- Path statistics ranked: ['std', 'skew', 'excess_kurtosis', 'left_q01', 'left_q05', 'right_q95', 'right_q99', 'es_99', 'acf_sq_lag1', 'max_drawdown', 'worst_day']

## Weighting

- Stress-labelled origins carry weight 2.0; others 1.
- Regimes are terciles of realised test-window volatility: ['calm', 'normal', 'stress'].

## Selection rule

Champion = lowest weighted mean rank on the selection origins; ties
broken by lower ladder step. The same table is then recomputed on the
confirmation origins and reported verbatim, whether or not it agrees.

## Hash

`sha256:c2701bd8e122b5f56aa70752f595b2ffee98b843c5c6d317227f368f5c661c8c`
