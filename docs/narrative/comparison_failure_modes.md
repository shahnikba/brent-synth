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
