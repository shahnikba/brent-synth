### 4.1 What "14 of 14" means

Every independent statistic passes the location test, and that is weak
evidence. At 252 days the bootstrap bands are wide because one year of
Brent is a noisy sample: excess kurtosis from 0.55 to 26.8, skewness from
−2.90 to +0.84, lag-1 ACF(r²) from −0.03 to 0.35. Any model with the right
centre and positive clustering passes. The location test is necessary, not
sufficient; the Spread column is where the information is, and it fails
five times.

### 4.2 One mechanism, two symptoms

Volatility and the central tail quantiles are over-dispersed (2.13x,
2.11x, 2.60x); skewness and kurtosis are under-dispersed (0.50x, 0.34x).
Same cause.

Persistence is 0.9954, a variance half-life of about 150 trading days, and
the level the recursion reverts to — an unconditional daily volatility of
3.24%, 51% annualised — is roughly double the median realised variance of
a real year. A path that drifts into high variance stays there for most of
the horizon: synthetic years span 1.36% to 6.18% daily volatility against
a real band of 1.60% to 3.86%. The 1st/99th percentiles and both ES rows
sit in the same direction (1.83x, 1.65x, 1.86x, 1.39x) and pass only
because they fall under the 2.0x bound.

The under-dispersion is the same recursion seen from the other side. The
real 97.5% bounds — kurtosis 26.8, skew −2.90 — come from windows
containing one day far outside the volatility around it. With ν = 5.75 the
innovation's excess kurtosis is about 3.4, and σ_t moves smoothly, so the
model's only route to an extreme return is to raise σ_t and hold it. It
converts what are jumps in the data into regimes in the simulation: too
many years of sustained extreme volatility, too few years containing an
isolated extreme day.

Why persistence sits where it does is answered out of time: the real
ACF(r²) decays hyperbolically and a GARCH(1,1) can only approximate that
by integrating. At five of the fourteen backtest origins the fit crosses
1. Variance targeting was tried as a fix and did not help; the long-memory
alternative (FIGARCH) fixes the shape and is better calibrated but has no
leverage term. See model_comparison.md §9.2.

### 4.3 The reference is not innocent

A stationary bootstrap with expected block length 20 stitches roughly a
dozen blocks from anywhere in nineteen years into each 252-day sample, so
a replicate's realised volatility averages across regimes and its
year-to-year dispersion is understated relative to actual consecutive
years. Some of the 2.13x is the denominator. The bootstrap remains the
right primary reference — it has the correct sampling variability — but
the empirical set of overlapping real 252-day windows should be shown
beside it as an overlay, and the band's sensitivity to block length
demonstrated with a sweep. Neither was run; both are listed as next steps.

### 4.4 The ACF decay, and what the model is for

Real ACF(r²) falls from 0.126 at lag 1 to 0.041 at lag 10 — fast then
slow. The synthetic profile starts lower (0.078) and is flatter (0.050):
geometric decay at ≈ 0.995 per lag barely decays over ten lags. Both pass
a band that proves little at this horizon.

For one-year stress scenarios the quantity that matters is the path
functional, and the maximum-drawdown distribution is the best-matched row
in the table: median 0.300 vs 0.305, spread 1.14x. The clustering misfit
does not visibly propagate into drawdowns at 252 days. It would at longer
horizons, and it understates lag-1 persistence by about 40% for anyone
reading the model's ACF as a short-run forecast. Scope limit, not pass.

### 4.5 Next steps

In order: asymmetric long memory (FIAPARCH / HYGARCH / component-GJR); a
jump term; regimes with fat-tailed shocks; conditional stress generation
from a specified variance state; rolling-window overlay and block sweep on
the reference side. Rationale for each in model_comparison.md §9.6.
