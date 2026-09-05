"""Statistical diagnostics for Brent log returns.

Pure computation: every function returns plain numbers, dicts, or numpy
arrays. No plotting lives here — the report layer consumes
:func:`run_all` and draws from it.

The diagnostics answer three questions that drive the model choice:

1. How far from Gaussian are the returns? (:func:`moments`)
2. Is there volatility clustering? (:func:`acf_squared`,
   :func:`ljung_box_squared`)
3. How heavy is the loss tail? (:func:`hill_estimator`,
   :func:`pot_gpd_fit`, :func:`mean_excess`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

TRADING_DAYS = 252

#: Fewest observations for which every moment exists. Sample skew needs
#: 3 and unbiased kurtosis needs 4, so below this ``moments`` would
#: return NaNs rather than numbers.
MIN_OBS_MOMENTS = 4

#: Floor on the Hill estimator's mean log excess. Below this the top-k
#: losses are numerically indistinguishable and the reciprocal is not a
#: tail index but a division by noise — a real tail index is single or
#: low double digits, so anything implying alpha > 1e6 is degenerate.
#: Testing against exact zero does not close this: near-constant input
#: lands just above zero and slips through.
MIN_MEAN_LOG_EXCESS = 1e-6

#: One-sided normal critical value at 95%, used to decide whether the
#: fitted GPD shape is distinguishable from zero.
XI_SIGNIFICANCE_Z = 1.645

#: Smallest number of order statistics used by the Hill estimator. Below
#: roughly this many exceedances the estimate is too noisy to read.
HILL_MIN_K = 10

#: Largest fraction of the tail sample used by the Hill estimator. Beyond
#: this the estimator drifts into the bulk and picks up bias.
HILL_MAX_FRACTION = 0.10


def _as_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Coerce a return series to a finite 1-D float array.

    Non-finite values are dropped. That is safe here because every
    caller of this helper computes an order-independent statistic — a
    moment or a tail quantile — where removing a hole does not move the
    answer. Order-dependent statistics must use
    :func:`_as_ordered_array` instead.
    """
    values = np.asarray(returns, dtype="float64").ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Return series is empty after dropping non-finite values.")
    return values


def _as_ordered_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Coerce to a 1-D float array, refusing to close gaps silently.

    Autocorrelation and the Ljung-Box statistic read adjacency: they ask
    what happened on the day *after* each day. Dropping a NaN run
    silently splices its two sides together and makes them lag-1
    neighbours, which moves the answer — punching a 100-day hole in a
    2000-point series shifts ACF(1) and the Ljung-Box p-value by
    percentage points. The array carries no calendar, so a gap cannot be
    detected after the fact; it has to be refused at the door.

    Observations are treated as consecutive. Weekend and holiday gaps in
    a daily series are expected and are not flagged — this rejects
    missing *values*, not missing calendar days, so resample or
    interpolate deliberately before calling.
    """
    values = np.asarray(returns, dtype="float64").ravel()
    if values.size == 0:
        raise ValueError("Return series is empty.")
    n_missing = int((~np.isfinite(values)).sum())
    if n_missing:
        raise ValueError(
            f"Return series contains {n_missing} non-finite value(s). "
            "Autocorrelation statistics depend on adjacency, so the gaps "
            "cannot be dropped silently — handle them explicitly first."
        )
    return values


def _tail_losses(returns: pd.Series | np.ndarray, tail: str) -> np.ndarray:
    """Positive-valued tail sample.

    ``tail='left'`` takes losses: the negative returns, sign-flipped, so
    that a large loss is a large positive number. ``tail='right'`` takes
    the gains as they are. Both come back as positive magnitudes, which
    is what the extreme-value estimators expect.
    """
    values = _as_array(returns)
    if tail == "left":
        losses = -values[values < 0.0]
    elif tail == "right":
        losses = values[values > 0.0]
    else:
        raise ValueError(f"tail must be 'left' or 'right', got {tail!r}.")

    if losses.size == 0:
        raise ValueError(f"No observations in the {tail} tail.")
    return losses


def moments(returns: pd.Series) -> dict:
    """Return {'mean','std','skew','kurtosis','excess_kurtosis',
    'annualised_vol','n_obs'}.

    Skew and kurtosis are sample statistics; ``excess_kurtosis`` is the
    Fisher definition (0 for a Gaussian) and ``kurtosis`` is the Pearson
    definition (3 for a Gaussian).
    """
    values = _as_array(returns)
    if values.size < MIN_OBS_MOMENTS:
        raise ValueError(
            f"Need at least {MIN_OBS_MOMENTS} observations for every moment to "
            f"exist, got {values.size}."
        )
    std = float(np.std(values, ddof=1))
    excess_kurtosis = float(stats.kurtosis(values, fisher=True, bias=False))

    return {
        "mean": float(np.mean(values)),
        "std": std,
        "skew": float(stats.skew(values, bias=False)),
        "kurtosis": excess_kurtosis + 3.0,
        "excess_kurtosis": excess_kurtosis,
        "annualised_vol": float(std * np.sqrt(TRADING_DAYS)),
        "n_obs": int(values.size),
    }


def _check_lags(n_obs: int, lags: int, name: str) -> None:
    """Reject a lag count the sample cannot support.

    statsmodels truncates silently, handing back a shorter array than
    asked for; a caller indexing the lag it wanted then reads the wrong
    one or raises IndexError far from the cause.
    """
    if lags < 1:
        raise ValueError(f"{name} must be >= 1, got {lags}.")
    if lags >= n_obs:
        raise ValueError(
            f"{name}={lags} needs more than {lags} observations, got {n_obs}."
        )


def acf_returns(returns: pd.Series, nlags: int = 40) -> np.ndarray:
    """ACF of r_t. Expected ~0 at all lags.

    Returns ``nlags + 1`` values, lag 0 first.
    """
    values = _as_ordered_array(returns)
    _check_lags(values.size, nlags, "nlags")
    return acf(values, nlags=nlags, fft=True)


def acf_squared(returns: pd.Series, nlags: int = 40) -> np.ndarray:
    """ACF of r_t^2. Expected positive, slow decay (clustering).

    Returns ``nlags + 1`` values, lag 0 first.
    """
    values = _as_ordered_array(returns)
    _check_lags(values.size, nlags, "nlags")
    return acf(values**2, nlags=nlags, fft=True)


def ljung_box_squared(returns: pd.Series, lags: int = 10) -> float:
    """Ljung-Box p-value on r_t^2 at ``lags``.

    A small p-value rejects "no autocorrelation in squared returns",
    i.e. it is evidence of volatility clustering.
    """
    values = _as_ordered_array(returns)
    _check_lags(values.size, lags, "lags")
    squared = values**2
    result = acorr_ljungbox(squared, lags=[lags], return_df=True)
    return float(result["lb_pvalue"].iloc[0])


def hill_estimator(returns: pd.Series, tail: str = "left") -> dict:
    """Hill tail-index estimates across a range of k.

    Returns {'k': np.ndarray, 'alpha': np.ndarray} for a Hill plot.
    ``tail='left'`` uses losses (negative returns, sign-flipped);
    ``tail='right'`` uses gains.

    For order statistics X_(1) >= ... >= X_(n), the Hill estimator on the
    top k is

        alpha_k = k / sum_{i=1..k} log(X_(i) / X_(k+1))

    Small k is unbiased but noisy; large k is stable but biased toward
    the bulk. Reading the plateau in between is the point of the plot.
    """
    losses = _tail_losses(returns, tail)
    ordered = np.sort(losses)[::-1]
    n = ordered.size

    k_max = max(int(HILL_MAX_FRACTION * n), HILL_MIN_K + 1)
    k_max = min(k_max, n - 1)
    if k_max < HILL_MIN_K:
        raise ValueError(
            f"Not enough {tail}-tail observations for a Hill plot: n={n}."
        )

    k_values = np.arange(HILL_MIN_K, k_max + 1)
    log_ordered = np.log(ordered)

    # cumsum[j] is sum of the j largest log order statistics, so the mean
    # log excess over the (k+1)-th is a single vectorised difference.
    cumulative = np.cumsum(log_ordered)
    top_k_sum = cumulative[k_values - 1]
    threshold_logs = log_ordered[k_values]  # the (k+1)-th, zero-indexed
    mean_log_excess = top_k_sum / k_values - threshold_logs

    # Guard against dividing by numerical noise, not just by exact zero:
    # on near-constant losses the mean log excess lands just above zero
    # and a bare `> 0` test lets alpha ~ 1e15 through as if it meant
    # something. See MIN_MEAN_LOG_EXCESS.
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(
            mean_log_excess > MIN_MEAN_LOG_EXCESS, 1.0 / mean_log_excess, np.nan
        )

    return {"k": k_values, "alpha": alpha}


def pot_gpd_fit(
    returns: pd.Series,
    tail: str = "left",
    threshold_quantile: float = 0.95,
) -> dict:
    """Peaks-over-threshold GPD fit on exceedances.

    Returns {'threshold','xi','xi_se','beta','n_exceedances','tail_index'}.

    ``threshold_quantile`` is a quantile **of the tail sub-sample**, not
    of the return distribution: with ``tail='left'`` the 0.95 default is
    the 95th percentile of the losses only, i.e. of the roughly half of
    all days that were down. On real Brent that lands at u = 0.0519,
    which is the 97.6th percentile of *all* returns — a 2.4% tail
    probability, not 5%, and about 114 exceedances rather than the ~240 a
    reader assuming "top 5% of returns" would expect. Quantiling the
    thing actually being fitted is the defensible choice, but it is not
    the reading the parameter name invites.

    The generalised Pareto is fitted to the exceedances over the
    threshold with the location pinned at zero, the standard POT
    parameterisation.

    ``tail_index`` is 1/xi, but only when xi is distinguishable from
    zero: a light tail whose shape noises just above zero would otherwise
    yield a confident-looking absurdity (an exponential sample fitting
    xi = +0.008 reports a tail index of 122). The test is one-sided at
    95% against the asymptotic standard error
    ``xi_se = (1 + xi) / sqrt(n_exceedances)``; anything short of that
    returns NaN, since the data cannot support a power-law claim.
    """
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError(
            f"threshold_quantile must be in (0, 1), got {threshold_quantile}."
        )

    losses = _tail_losses(returns, tail)
    threshold = float(np.quantile(losses, threshold_quantile))
    exceedances = losses[losses > threshold] - threshold

    if exceedances.size < 2:
        raise ValueError(
            f"Too few exceedances above the {threshold_quantile:.0%} threshold "
            f"to fit a GPD: n={exceedances.size}."
        )

    xi, _, beta = stats.genpareto.fit(exceedances, floc=0.0)
    xi = float(xi)
    n_exceedances = int(exceedances.size)

    # Asymptotic standard error of the GPD shape (Smith 1987), valid for
    # xi > -0.5. Used to gate the tail index on significance rather than
    # on sign alone.
    xi_se = float((1.0 + xi) / np.sqrt(n_exceedances))
    significant = xi > XI_SIGNIFICANCE_Z * xi_se

    return {
        "threshold": threshold,
        "xi": xi,
        "xi_se": xi_se,
        "beta": float(beta),
        "n_exceedances": n_exceedances,
        "tail_index": 1.0 / xi if significant else float("nan"),
    }


def mean_excess(returns: pd.Series, tail: str = "left") -> dict:
    """Mean-excess function for the mean-excess plot.

    Returns {'thresholds': np.ndarray, 'mean_excess': np.ndarray}, where
    ``mean_excess[i]`` is E[X - u | X > u] at ``u = thresholds[i]``.

    A mean-excess function that rises roughly linearly in u is the
    signature of a heavy (generalised Pareto) tail; a flat one points to
    an exponential tail.

    Exceedances are selected by **value**, not by position. On a sorted
    sample, taking everything after index i quietly includes any
    observations tied with u, which contribute a zero excess and drag
    the mean down: on [1, 2, 2, 2, 5] that yields [1.75, 1.0, 1.5, 3.0]
    where the definition gives [1.75, 3, 3, 3]. Ties are rare in returns
    but not absent — real Brent has a handful — and this is the plot the
    POT threshold gets read off.

    Thresholds equal to the sample maximum are omitted: nothing exceeds
    them, so the conditional mean is undefined there.
    """
    losses = np.sort(_tail_losses(returns, tail))
    n = losses.size
    maximum = losses[-1]

    thresholds = losses[losses < maximum]
    if thresholds.size == 0:
        raise ValueError(
            f"All {n} {tail}-tail observations are identical; the mean-excess "
            "function is undefined."
        )

    # suffix_sum[k] is the sum of losses[k:], with a trailing zero so the
    # empty suffix is addressable.
    suffix_sum = np.concatenate([np.cumsum(losses[::-1])[::-1], [0.0]])

    # First index holding a value strictly greater than each threshold.
    first_above = np.searchsorted(losses, thresholds, side="right")
    counts = n - first_above  # > 0, since every threshold is below the max
    excess_sum = suffix_sum[first_above] - thresholds * counts

    return {"thresholds": thresholds, "mean_excess": excess_sum / counts}


def run_all(returns: pd.Series) -> dict:
    """Run every diagnostic, return one nested dict of results.

    This is what the report layer will consume.
    """
    return {
        "moments": moments(returns),
        "autocorrelation": {
            "acf_returns": acf_returns(returns),
            "acf_squared": acf_squared(returns),
            "ljung_box_squared_pvalue": ljung_box_squared(returns),
        },
        "tails": {
            "left": {
                "hill": hill_estimator(returns, tail="left"),
                "pot_gpd": pot_gpd_fit(returns, tail="left"),
                "mean_excess": mean_excess(returns, tail="left"),
            },
            "right": {
                "hill": hill_estimator(returns, tail="right"),
                "pot_gpd": pot_gpd_fit(returns, tail="right"),
                "mean_excess": mean_excess(returns, tail="right"),
            },
        },
    }
