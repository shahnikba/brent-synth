# Review — persistence uses the wrong weight on gamma

`model.py` computes alpha + gamma/2 + beta, justified as "the half-weight
on gamma is the probability that a symmetric shock is negative".

Two problems. The weight in the variance recursion isn't a probability —
taking expectations of the GJR recursion gives

    E[sigma^2] = omega / (1 - alpha - gamma*E[z^2 1{z<0}] - beta)

the partial second moment, not P(z<0). For a symmetric unit-variance z
that happens to equal 1/2, which is why the rule is written that way —
but these innovations are deliberately skewed. For the fitted
lambda = -0.112, E[z^2 1{z<0}] = 0.5424 while P(z<0) = 0.4774: the
probability moves down and the weight moves up, so reasoning from the
probability gets even the sign wrong.

Isolated from Monte-Carlo noise with a fast-mixing parameter set
(persistence 0.89):

| lambda | E[z^2 1{z<0}] | measured E[sigma^2] | gamma/2 | err | corrected | err |
|---|---|---|---|---|---|---|
| 0.00 | 0.50000 | 3.997e-05 | 4.000e-05 | 0.1% | 4.000e-05 | 0.1% |
| -0.10 | 0.53719 | 4.150e-05 | 4.000e-05 | 3.8% | 4.155e-05 | 0.1% |
| -0.30 | 0.60483 | 4.461e-05 | 4.000e-05 | 11.5% | 4.468e-05 | 0.2% |
| -0.50 | 0.65618 | 4.726e-05 | 4.000e-05 | 18.2% | 4.740e-05 | 0.3% |
| -0.70 | 0.68925 | 4.913e-05 | 4.000e-05 | 22.8% | 4.934e-05 | 0.4% |

Impact on the actual fit is amplified by high persistence:
persistence 0.99392 -> 0.99537; unconditional variance 8.001e-04 ->
1.052e-03; annualised long-run vol 44.9% -> 51.5%.

The recovery test cannot catch it: it asserts persistence against the
same gamma/2 expression, and the discrepancy is inside the tolerance.

## Two things to be aware of
- `fit` still splices NaN gaps — the pattern just replaced in diagnostics.
  A GARCH likelihood is more adjacency-dependent than an ACF.
- The fitted model generates economically impossible days: worst
  simulated single day -98.6%, terminal multiples 5.0e-10 to 1.6e+05.
  Honest consequence of nu=5.75 with persistence 0.994, but the
  validation layer should be flagging it.
