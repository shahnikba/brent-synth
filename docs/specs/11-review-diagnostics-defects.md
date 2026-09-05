# Review — diagnostics defects

## Two genuine bugs

### 1. mean_excess contradicts its own definition on ties
The docstring promises E[X - u | X > u], but the vectorised form takes
everything positionally after index i, which for tied losses includes
observations equal to u contributing zero excess. On [1,2,2,2,5] it
returns [1.75, 1.0, 1.5, 3.0] where the definition gives [1.75, 3, 3, 3].
Real Brent has 2 tied loss pairs and 1 tied gain pair, so 3 points on the
mean-excess plots are understated — max deviation 1.5e-5. Small, but it
is the plot you read to choose the POT threshold.

### 2. Hill's divide-by-zero guard doesn't guard
The test `mean_log_excess > 0.0` intends NaN when the top-k losses are
identical. Floating-point noise lands just above zero and slips through —
alpha = 2.25e15 instead of NaN. Comparing against a small epsilon closes it.

## Three things that will bite later
- `_as_array` silently splices NaN gaps. A 100-day hole in a 2000-point
  series moved ACF(1) from 0.00905 to 0.01026 and Ljung-Box p from 0.957
  to 0.819. The module discards the DatetimeIndex, so it cannot notice a
  gap even in principle.
- `threshold_quantile` is the 95th percentile of the loss sub-sample, not
  of the return distribution — u = 0.0519 is the 97.60th percentile of all
  returns, a 2.40% tail probability. Defensible, but undocumented.
- `tail_index` is guarded on sign, not significance. Exponential losses
  fit xi = +0.0082 and yield tail_index = 121.6.

## Minor
acf silently returns fewer than nlags+1 values when nlags >= n;
ljung_box dies with a raw numpy broadcast error; moments on one
observation returns all-NaN with a bare warning while moments([]) raises;
`losses = losses[losses > 0.0]` is dead; `annualised_vol` is np.float64.
