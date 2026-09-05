The generator is GJR-GARCH(1,1,1) with standardised skewed-t innovations.
It is built to reproduce four things over a one-year horizon: the
unconditional fat tails of daily Brent returns; volatility clustering; the
leverage asymmetry (negative shocks raising variance more than positive
ones of the same size); and the resulting distribution of path-level
losses, in particular maximum drawdown. Those are the quantities the
diagnostics identified, and each has a parameter that owns it (nu and
lambda for the tails and skew, alpha and beta for clustering, gamma for
leverage).

It is not built to reproduce: jumps — isolated single-day moves far outside
the prevailing volatility, which a smooth variance recursion cannot
generate; long-memory decay of the squared-return autocorrelation, which a
GARCH(1,1) can only approximate geometrically; regime shifts in the
long-run variance level, which the model has no state for and instead
absorbs into a persistence near one — the out-of-time comparison in
model_comparison.md measures how much each of these exclusions costs; or
any dependence on the price level, calendar, or exogenous drivers. The
validation below is designed to show exactly where those exclusions bite.
