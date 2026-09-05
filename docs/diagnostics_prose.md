# Diagnostics — interpretation

*For the diagnostics section of the README / report. Numbers are from
`diagnostics.run_all` on 4754 daily log returns, 2007-07-30 to 2026-09-04
[check: rerun after the later diagnostics edits; the Hill guard and GPD
significance gate may move the reported tail figures slightly].*

---

**Nothing to forecast in the level; everything to forecast in the scale.**
The autocorrelation of returns is −0.019 at lag 1 and indistinguishable
from zero at every lag after it: there is no linear structure in the mean
worth a parameter, which is what a liquid, heavily traded contract should
show, since any persistent pattern in the level is traded away. The
autocorrelation of *squared* returns is a different object entirely — 0.204
at lag 1, still 0.097 at lag 40, with a Ljung–Box p-value at lag 10 that
underflows (1.7e−213). Volatility is forecastable and stays forecastable,
because variance cannot be arbitraged the way a drift can. Two further
features of that decay matter for what follows. It is slow — a fast
initial drop then a long, nearly flat tail, the shape associated with long
memory rather than the geometric decay of a single-lag recursion — and it
means any model of this series must carry a conditional variance that
moves with the recent past. That is the first modelling decision: constant
mean, time-varying variance, and everything interesting lives in the
variance equation.

**The tails are far from Gaussian, and the two tails are not the same.**
Sample excess kurtosis is 11.7; Jarque–Bera's p-value underflows to zero.
The worst day in the sample, 2020-04-21, is a −28.0% log return — 11.5
standard deviations, an event a Gaussian assigns probability 7e−31 per
day. Peaks-over-threshold fits above the 95th percentile of each tail's
sub-sample give a generalised-Pareto shape of ξ = 0.212 on losses (114
exceedances) against 0.127 on gains (123), i.e. a loss tail with roughly
half the effective power-law index of the gain tail (1/ξ ≈ 4.7 vs 7.9).
The mean-excess function of losses rises with the threshold [check: read
off the plot; a rising, roughly linear mean-excess is the GPD signature,
a flat one would say exponential], consistent with a genuinely heavy left
tail rather than a merely wide one. Sample skewness of −0.78 tells the
same story from the body of the distribution. Two honest caveats travel
with these numbers. The Hill estimator, read at its plateau, gives a tail
index near 2.9 on both sides, lower than the POT figures — not a bug but a
threshold effect, and a reminder that no single number is "the" tail
index; each estimate should be quoted with its estimator and threshold.
And the asymmetry in ξ is estimated from about a hundred exceedances a
side, so its sign is well supported and its magnitude is not precise.

**What the diagnostics ask for, and what the model choice gives up.** Three
findings, three model components. Clustering with slow decay calls for a
conditional-variance recursion — GARCH — and rules out any iid parametric
Monte Carlo, which cannot cluster at all. Excess kurtosis of 11.7, well
beyond what Gaussian-innovation GARCH generates on its own, calls for
fat-tailed innovations: a Student-t family. And the asymmetric tails and
negative skew call for two things — a skewed innovation, and an asymmetric
variance response so that a negative shock raises tomorrow's variance by
more than a positive one of the same size, the leverage effect. Together
that is GJR-GARCH(1,1,1) with skewed-t innovations, each term traceable to
a diagnostic. It is chosen with its limits stated. A skewed-t assigns both
tails the same power-law rate ν; it reproduces the skew and the heavier
loss tail in the body of the distribution, not the difference in tail
*index* the POT fit reports, so the 0.212-vs-0.127 asymmetry is only
partly captured. A single-lag GARCH decays geometrically and can only
approximate the slow squared-return decay by pushing persistence toward
one — the validation shows exactly that, and it is the model's principal
failure mode. And a smooth variance recursion has no way to place an
isolated 11σ day; it can only raise variance for a while. The generator
is built to reproduce clustering, fat tails, leverage and the distribution
of one-year drawdowns; it is not built to reproduce jumps, long memory, or
regime shifts in the long-run variance level, and the validation is
designed to show where each of those exclusions costs.
