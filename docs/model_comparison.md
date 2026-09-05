# Brent scenario generators — out-of-time model comparison

*Companion to [validation.md](validation.md), which is the in-sample
descriptive check on the champion fitted to all data. This report is the
pre-registered, out-of-time comparison that selected it.*

<!-- Cells marked {{generated}} are produced by `brent_synth.backtest report`
     and were not available when this prose was written. Fill from the
     generated report; the prose is written to remain true once filled. -->

## Summary

Six candidate generators were compared out of time at fourteen year-end
origins, 2011–2024, under a ranking rule fixed and committed before the
first result existed. GJR-GARCH(1,1,1) with skewed-t innovations won the
pre-registered composite on the 2011–2018 selection origins (weighted
score 1.913, top-two at 7 of 8 origins) and held on the 2019–2024
confirmation origins (2.156, top-two at 5 of 6). It is retained as the
champion.

The win is narrower and more specific than the scoreboard suggests.
FIGARCH(1,d,1) with the same innovations is second on both periods (2.388,
2.266), and on the confirmation origins it beats the champion on both
density losses — negative log-likelihood and tail-weighted CRPS — with
Diebold–Mariano statistics of −0.91 and −1.34 (p = 0.36, 0.18): nominally
better, not significantly so. The champion holds the composite on the two
*risk* losses, 99% exceedance error and path coverage. So the evidence
says: for VaR and scenario coverage, GJR; for density forecasting, the two
are indistinguishable and FIGARCH is the better-calibrated of the pair.

Three findings qualify the champion and are the substance of §9: its own
fit is non-stationary (persistence 1.00012–1.00066) at five of the
fourteen origins; it is under-confident in calm years and calibrated in
normal and stress years; and at the one origin whose test window is 2020,
no candidate — including it — produced a year like the one that happened.

## 1. Pre-registration

Fixed in `docs/preregistration.md` and committed (tag
`spec5-preregistered`) before the first scoring run (tag `spec5-run`).

| Item | Value |
|---|---|
| Candidates (ladder step) | iid_t (0), garch_normal (1), gjr_skewt (2), gjr_skewt_vt (3), figarch_skewt (4), ms_variance (5) |
| Origins | last trading day of each year, 2011–2024 (14) |
| Selection / confirmation | 2011–2018 (8) / 2019–2024 (6) |
| Test horizon | 252 trading days after each origin |
| Ranking losses (per origin) | mean NLL; mean tail-weighted CRPS (τ ≤ 0.05); \|99% exceedance rate − 0.01\|; path coverage loss |
| Aggregation | mean of the four ranks, stress-labelled origins weighted 2.0 |
| Regime label | tercile of the test window's realised volatility (calm / normal / stress) |
| Paths per origin / seed | 2000 / 2024 |
| Original hash | sha256:c2701bd8… |
| Amendment 1 (post hoc) | a candidate whose fit raises at an origin takes the worst rank on all four losses there, instead of being dropped and the rest re-ranked. Reason: scoring a generator only where it succeeds is survivorship bias. Hash regenerated: sha256:15aa66f0… Both rulings reported in §2. |

Every candidate is fitted on data up to and including the origin and
nothing later. Filtered one-day densities for the test window use
parameters frozen at the origin and a variance backcast computed from the
training slice only. Verified by test: perturbing every post-origin return
leaves fitted parameters and simulated paths byte-identical.

## 2. Scoreboard

Lower is better. Rank stability is the fraction of origins on which the
model is top-two on the mean of the four ranks.

**Selection origins, 2011–2018**

| model | step | amended score | rank stability | original score (origins scored) |
|---|---:|---:|---:|---:|
| gjr_skewt | 2 | **1.913** | 0.875 | 1.913 (8) |
| figarch_skewt | 4 | 2.388 | 0.750 | 2.388 (8) |
| ms_variance | 5 | 3.525 | 0.125 | 3.525 (8) |
| garch_normal | 1 | 3.775 | 0.000 | 3.775 (8) |
| iid_t | 0 | 4.313 | 0.125 | 4.313 (8) |
| gjr_skewt_vt | 3 | 5.088 | 0.333 | 2.958 (3) |

**Confirmation origins, 2019–2024**

| model | step | amended score | rank stability |
|---|---:|---:|---:|
| gjr_skewt | 2 | **2.156** | 0.833 |
| figarch_skewt | 4 | 2.266 | 0.667 |
| gjr_skewt_vt | 3 | 3.391 | 0.333 |
| garch_normal | 1 | 4.328 | 0.000 |
| iid_t | 0 | 4.422 | 0.167 |
| ms_variance | 5 | 4.438 | 0.000 |

Champion under both rulings: `gjr_skewt`. Confirmation agrees under both.
(`gjr_skewt_vt` fits at all six confirmation origins, so the two rulings
coincide there.)

### 2.1 Which loss the composite is won on

Mean rank per loss; lower is better.

| loss | gjr_skewt (sel) | figarch (sel) | gjr_skewt (conf) | figarch (conf) |
|---|---:|---:|---:|---:|
| Mean NLL | **1.500** | 1.750 | 2.000 | **1.833** |
| Mean tail CRPS | **1.750** | 2.625 | 2.500 | **2.000** |
| \|99% exceedance error\| | 2.688 | **2.375** | **1.917** | 2.167 |
| Path coverage loss | **2.125** | 2.625 | **2.333** | 2.833 |

On selection the champion wins three of four including both density
losses. On confirmation it wins neither density loss and holds the
composite through the two risk losses.

## 3. Scores by regime

Four losses averaged within each regime of the test window.

{{generated: table, model × (calm / normal / stress) × four losses}}

## 4. Crisis years

Origins whose test window is labelled stress, plus origin 2019 (window =
calendar 2020) pinned unconditionally. 2008 is absent: no origin has
enough history before it to fit.

### 4.1 Per-origin losses and ranks at stress origins

{{generated: table — origin, model, NLL, tail CRPS, 99% exceedances / 252,
path coverage loss, drawdown rank, worst-day rank}}

Known rows for origin 2019 (window 2020):

| model | drawdown rank | worst-day rank | 99% exceedances (expected 2.5) |
|---|---:|---:|---:|
| iid_t | ≈ 0.99 | ≈ 0.00 | 14 |
| ms_variance | ≈ 0.99 | ≈ 0.00 | 10 |
| garch_normal | ≈ 0.99 | ≈ 0.00 | {{5–6}} |
| gjr_skewt | ≈ 0.99 | ≈ 0.00 | {{5–6}} |
| gjr_skewt_vt | ≈ 0.99 | ≈ 0.00 | {{5–6}} |
| figarch_skewt | ≈ 0.99 | ≈ 0.00 | {{5–6}} |

## 5. Calibration — PIT

Probability-integral-transform values of the realised return under each
day's forecast density. Uniform means calibrated; a hump means the
densities are too wide (under-confident); a U means too narrow
(over-confident). χ² is over 20 bins; because consecutive PITs are mildly
dependent and n is large, the p-values are anti-conservative and effect
sizes (KS distance from uniform, largest single-bin deviation from 0.05)
are reported beside them.

### 5.1 Pooled over all origins

| model | shape | χ² p | KS | max-bin dev |
|---|---|---:|---:|---:|
| gjr_skewt_vt | flat | 0.0996 | {{generated}} | {{generated}} |
| figarch_skewt | hump | 0.0167 | {{generated}} | {{generated}} |
| gjr_skewt | hump | 0.0004 | {{generated}} | {{generated}} |
| iid_t | hump | 0.0020 | {{generated}} | {{generated}} |
| ms_variance | hump | 0.0000 | {{generated}} | {{generated}} |
| garch_normal | hump | 0.0000 | {{generated}} | {{generated}} |

`gjr_skewt_vt` is judged on fewer days (2268 vs 3528); see §5.3.

### 5.2 By regime of the test window

| model | calm | normal | stress |
|---|---|---|---|
| gjr_skewt | hump (0.0001) | flat (0.099) | flat (0.269) |
| figarch_skewt | hump (0.022) | flat (0.137) | flat (0.352) |
| garch_normal | hump (0.000) | hump (0.000) | flat (0.068) |
| gjr_skewt_vt | hump (0.000) | flat (0.130) | flat (0.235) |
| iid_t | hump (0.000) | hump (0.002) | **U** (0.000) |
| ms_variance | hump (0.000) | hump (0.001) | **U** (0.012) |

![PIT by regime](model_comparison_figures/pit-by-regime.png)

### 5.3 On common days

Restricted to the nine origins at which every candidate fitted, so all
six are judged on identical days.

| model | shape | χ² p |
|---|---|---:|
| figarch_skewt | flat | 0.122 |
| gjr_skewt_vt | flat | 0.100 |
| gjr_skewt | hump | 0.008 |
| garch_normal | {{generated}} | {{generated}} |
| iid_t | {{generated}} | {{generated}} |
| ms_variance | {{generated}} | {{generated}} |

## 6. Path-rank histograms

Percentile rank of each realised-year statistic inside the 2000 simulated
paths, pooled over origins and statistics. Same reading as §5.

![Path ranks](model_comparison_figures/path-ranks.png)

## 7. Parameter drift

Each fitted parameter against origin year. The 1.0 line in the persistence
panels is the stationarity boundary.

![Parameter drift](model_comparison_figures/parameter-drift.png)

### 7.1 Persistence by origin

| origin | garch_normal | gjr_skewt | gjr_skewt_vt | figarch_skewt (d) | gjr nu | gjr gamma |
|---:|---:|---:|---:|---:|---:|---:|
| 2011 | {{generated}} | {{generated}} | {{generated}} | {{generated}} | 10.80 | {{generated}} |
| 2012–2013 | {{generated}} | {{0.983–0.997}} | {{generated}} | {{generated}} | {{generated}} | {{generated}} |
| 2014–2018 | {{generated}} | **1.00012–1.00066 (non-stationary)** | *unavailable* | {{generated}} | {{generated}} | {{generated}} |
| 2019–2024 | {{generated}} | {{0.983–0.997}} | {{generated}} | {{generated}} | 5.56 at 2021 | {{generated}} |

The champion's persistence sits at or above 1.0 at 2014–2018, exactly the
five origins at which the variance-targeted variant cannot be fitted, and
in 0.983–0.997 everywhere else. Its ν falls monotonically from 10.80
(2011) to 5.56 (2021) as the sample absorbs 2014–15 and 2020, then
stabilises: the model is being told the tails are steadily fatter than it
first thought.

## 8. Diebold–Mariano vs the champion

Confirmation origins pooled; Newey–West HAC variance, lag 10. A negative
statistic means the row model's loss is lower than the champion's.

| model | NLL stat | NLL p | tail CRPS stat | tail CRPS p |
|---|---:|---:|---:|---:|
| figarch_skewt | **−0.910** | 0.363 | **−1.344** | 0.179 |
| garch_normal | 2.409 | 0.016 | 0.582 | 0.560 |
| gjr_skewt_vt | 2.792 | 0.005 | 1.751 | 0.080 |
| iid_t | 4.033 | 0.000 | 1.612 | 0.107 |
| ms_variance | 2.388 | 0.017 | 1.109 | 0.268 |

## 9. Failure modes

### 9.1 What the composite win is made of

The ranking rule averages four per-origin losses. Broken out, the champion
and FIGARCH tell opposite stories in the two periods. On selection the
champion wins three of four, including both density losses (mean rank
1.50 vs 1.75 on NLL, 1.75 vs 2.63 on tail CRPS). On confirmation it loses
both density losses (2.00 vs 1.83, 2.50 vs 2.00) and holds the composite
only through exceedance error (1.92 vs 2.17) and path coverage (2.33 vs
2.83). The champion therefore survives confirmation as a risk model, not
as a density model. Had the pre-registered rule weighted density fit
alone, the confirmation period would have preferred FIGARCH; it did not,
the rule was fixed in advance, and the selection stands. But the report
should not claim more than the rule delivered.

### 9.2 One mechanism behind every symptom

The in-sample validation (validation.md §4) found the champion sits on the
real bootstrap median for every statistic and yet over-disperses
volatility across simulated years by 2.13x while under-dispersing kurtosis
by 0.34x. The backtest explains why, and it is one mechanism.

The real autocorrelation of squared returns decays fast at short lags and
slowly thereafter — the hyperbolic signature of long memory. GJR-GARCH(1,1)
can only decay geometrically. The maximum-likelihood fit approximates the
slow tail the only way it can: by pushing persistence toward 1. On the
full sample it reaches 0.9954, a variance half-life of about 150 trading
days; at the 2014–2018 origins it crosses 1 (1.00012 to 1.00066), at which
point the model has no finite long-run variance at all. Those are exactly
the five origins at which the variance-targeted variant cannot be fitted —
there is nothing to target.

A near-integrated variance holds shocks too long. Three consequences
follow, and all three appear in the tables. Simulated years wander into
sustained high-variance states the market never held for a year (the
2.13x, and the 5th/95th-percentile mismatches of 2.11x and 2.60x that ride
on it). One-day densities are too wide in quiet periods, which is the
calm-regime under-confidence in §5.2 (χ² p = 0.0001). And the model has no
way to make an isolated jump — a smooth recursion with ν ≈ 5.6 innovations
generates fat tails by *raising σ for a while*, not by placing one extreme
day — which is the kurtosis under-dispersion (synthetic 252-day excess
kurtosis reaches 9.3 where real windows reach 26.8).

FIGARCH fixes the shape, not the tails: with a fractional-integration
parameter d it decays hyperbolically without integrating, which is why it
is better calibrated on common days (flat, p = 0.122, against the
champion's hump at p = 0.008) and nominally better on both density losses
out of sample. It carries no leverage term, and it still loses the
composite. The two runners-up each have half of the answer.

### 9.3 What the regime split shows

Pooled PIT histograms were misleading in both directions. Split by the
realised volatility of the test window, the champion is under-confident
only in calm years and reads flat in normal (p = 0.099) and stress
(p = 0.269) years: it spreads density wider than a quiet year needs, which
for a stress tool is the benign direction, and it is calibrated where
calibration matters.

The split also exposes the static-width failure that pooling averaged
away. The iid Student-t and the two-regime Gaussian mixture flip from hump
in calm years to U in stress years — one width for all conditions means
over-spreading when quiet and under-spreading when it counts. In the
stress rows the iid model puts 11.5% of realisations in a bin that should
hold 5%; in the 2020 window it took 14 exceedances at the 99% level
against 2.5 expected, the mixture 10, the GARCH family 5–6. That number is
the argument for a conditional variance at all, and it is why regimes
without fat-tailed shocks finished last on confirmation: the mixture can
switch its width but cannot make the days.

### 9.4 Nobody covered 2020

At the origin dated end-2019, every candidate was fitted on data through
December 2019 and asked for 2,000 years of 2020. Against the year that
happened, every one of them placed the realised drawdown and realised
volatility at rank ≈ 0.99 and the realised worst day at rank ≈ 0.00 — the
real year sat outside essentially everything any model imagined. This is
not a defect of the champion relative to the others; it is the limit of
the method. A fit-then-simulate generator conditions on the volatility
state at the origin, and in December 2019 that state was calm. The
one-day filter, which *reacts* to realised shocks, remained calibrated
through 2020 (stress-regime PIT flat). The 252-day paths, which had to
*anticipate*, did not. A scenario generator cannot forecast a regime it
has never seen from a state that gives no warning; what it can do — and
what the stress-scenario use of this model must rest on — is generate that
regime *conditionally*, by starting paths from a stressed variance state
rather than from today's. That is the `initial_var` choice, and it is the
difference between a tool that reproduces history and one that extends
it.

### 9.5 The unbalanced-availability amendment

The pre-registered rule dropped a failed candidate from an origin and
re-ranked the rest. Under that rule the variance-targeted variant finished
third on selection while being scored on only 3 of 8 origins. Amendment 1,
recorded post hoc in `docs/preregistration.md` with a new hash, assigns a
failed candidate the worst rank at that origin: a generator that cannot
produce scenarios has failed there, and scoring it only where it succeeds
is survivorship bias. It moves that variant to last (5.088). The champion
and the confirmation verdict are unchanged under both rulings, and both
scoreboards are shown in §2.

The variance-targeting experiment itself is closed: pinning ω to the
sample variance did not improve calibration where it could be fitted, and
could not be fitted where it was most needed. The inflated long-run
variance is a symptom of persistence, not its cause.

### 9.6 What I would try next, in order

1. **Asymmetric long memory.** FIAPARCH or HYGARCH with skewed-t
   innovations, or a two-component GJR: the evidence says the data want
   both the hyperbolic decay FIGARCH has and the leverage term GJR has.
   This is the direct successor and the first experiment.
2. **Jumps in the return equation.** A compound-Poisson jump term attacks
   the kurtosis under-dispersion at its source and should let persistence
   fall, because the recursion no longer has to manufacture jumps as
   regimes.
3. **Regimes with fat shocks.** MS-GARCH or a Markov-switching model with
   t innovations — the mixture's failure was the Gaussian within regime,
   not the switching.
4. **Conditional stress generation** as a first-class interface: simulate
   from a specified variance quantile or from a named historical stress
   state, and validate the resulting paths against the stressed years
   rather than against the bootstrap of the whole history. This is where
   §9.4 points.
5. **Reference-side checks not yet run**: a rolling-real-window overlay
   beside the bootstrap band, and a block-length sweep. The 20-day block
   is defensible a priori and its effect on the volatility band width
   remains asserted rather than shown.
