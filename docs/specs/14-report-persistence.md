# Report — persistence fix

## Status
Fixed. 101 tests passing (7 new).

## Confirmed
`arch.univariate.SkewStudent.partial_moment(2, 0.0)` computes E[z^2 1{z<0}]
directly. Verified against a 40M-draw Monte Carlo: exactly 0.500000 at
lambda = 0, 0.542570 at the fitted lambda = -0.112 (MC 0.542126), with
P(z<0) = 0.477244. The low-persistence table reproduces, weights matching
to five decimals (so nu = 6.0 was used).

## Impact

| | before | after |
|---|---|---|
| leverage weight | 0.5 assumed | 0.54240 |
| persistence | 0.99392 | 0.99538 |
| unconditional variance | 8.001e-04 | 1.052e-03 |
| annualised long-run vol | 44.9% | 51.5% |

## Verification
Reverting the formula to gamma/2 fails three tests, including two
parametrised unconditional-variance checks driven off the module's own
persistence property. The lambda = 0 case correctly still passes — it is
the control. My first attempt at that test passed the corrected value in
and therefore tested the test; it now derives the claim from the module.

## Also
`fit()` refuses non-finite input, matching diagnostics.

## Not fixed
Extreme simulated days confirmed (worst -99.5% in price terms, 615 of
10,000,000 days below -50%) but left alone: adding a statistic changes
SPEC 4's specified list.
