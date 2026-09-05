# Report — SPEC 3, generative model

## Status
Complete. 16 model tests, 43 passing in total. Added `arch==8.0.0`.

## Fit on Brent

| Param | Value |
|---|---|
| mu | 0.000101 |
| omega | 4.894e-06 |
| alpha | 0.0634 |
| gamma | 0.0344 |
| beta | 0.9132 |
| nu | 5.750 |
| lambda | -0.1117 |

persistence 0.99385 (as computed at the time, with the gamma/2 rule —
later corrected, see 13/14). loglik -9937.34, AIC 19888.67, BIC 19933.94.

Consistent with SPEC 2: gamma > 0 confirms the leverage asymmetry that
showed up as skew -0.78, and nu = 5.75 is the fat tail that showed up as
excess kurtosis 11.7.

## Simulate
`fit_and_simulate(load_returns())` returns a (5000, 252) array, all
finite, raw log-return units. Simulated daily std 0.02922 vs real
0.02435 — ratio 1.20, so no percent/raw scale bug. Same seed gives
byte-identical arrays.

## Round-trip recovery (n=20,000)

| Param | True | Recovered |
|---|---|---|
| mu | 0.0005 | 0.000420 |
| omega | 4.0e-06 | 3.742e-06 |
| alpha | 0.06 | 0.0622 |
| gamma | 0.05 | 0.0455 |
| beta | 0.90 | 0.9001 |
| nu | 6.0 | 6.317 |
| lambda | -0.10 | -0.0947 |

## Implementation notes
- The `*100` rescaling is confined to `fit()`: mu divides by 100 and
  omega by 100^2 (it is a variance). `simulate()` runs entirely in raw
  units.
- Simulation runs the GJR recursion by hand, drawing z from
  `arch.univariate.SkewStudent.ppf` by inverse transform off a
  `default_rng(seed)` stream. That is what makes the seed authoritative.
- loglik/AIC/BIC are on the percent scale the optimiser used.
