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

#: Smallest number of order statistics used by the Hill estimator. Below
#: roughly this many exceedances the estimate is too noisy to read.
HILL_MIN_K = 10

#: Largest fraction of the tail sample used by the Hill estimator. Beyond
#: this the estimator drifts into the bulk and picks up bias.
HILL_MAX_FRACTION = 0.10


def _as_array(returns: pd.Series | np.ndarray) -> np.ndarray:
    """Coerce a return series to a finite 1-D float array."""
    values = np.asarray(returns, dtype="float64").ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Return series is empty after dropping non-finite values.")
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
    std = float(np.std(values, ddof=1))
    excess_kurtosis = float(stats.kurtosis(values, fisher=True, bias=False))

    return {
        "mean": float(np.mean(values)),
        "std": std,
        "skew": float(stats.skew(values, bias=False)),
        "kurtosis": excess_kurtosis + 3.0,
        "excess_kurtosis": excess_kurtosis,
        "annualised_vol": std * np.sqrt(TRADING_DAYS),
        "n_obs": int(values.size),
    }


def acf_returns(returns: pd.Series, nlags: int = 40) -> np.ndarray:
    """ACF of r_t. Expected ~0 at all lags."""
    return acf(_as_array(returns), nlags=nlags, fft=True)


def acf_squared(returns: pd.Series, nlags: int = 40) -> np.ndarray:
    """ACF of r_t^2. Expected positive, slow decay (clustering)."""
    return acf(_as_array(returns) ** 2, nlags=nlags, fft=True)


def ljung_box_squared(returns: pd.Series, lags: int = 10) -> float:
    """Ljung-Box p-value on r_t^2 at ``lags``.

    A small p-value rejects "no autocorrelation in squared returns",
    i.e. it is evidence of volatility clustering.
    """
    squared = _as_array(returns) ** 2
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
    losses = losses[losses > 0.0]
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

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(mean_log_excess > 0.0, 1.0 / mean_log_excess, np.nan)

    return {"k": k_values, "alpha": alpha}


def pot_gpd_fit(
    returns: pd.Series,
    tail: str = "left",
    threshold_quantile: float = 0.95,
) -> dict:
    """Peaks-over-threshold GPD fit on exceedances.

    Returns {'threshold','xi','beta','n_exceedances','tail_index'}.
    ``tail_index`` is 1/xi when xi > 0, else NaN — a non-positive shape
    means no power-law tail, so the tail index is undefined.

    The generalised Pareto is fitted to the exceedances over the
    threshold with the location pinned at zero, which is the standard
    POT parameterisation.
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

    return {
        "threshold": threshold,
        "xi": xi,
        "beta": float(beta),
        "n_exceedances": int(exceedances.size),
        "tail_index": 1.0 / xi if xi > 0.0 else float("nan"),
    }


def mean_excess(returns: pd.Series, tail: str = "left") -> dict:
    """Mean-excess function for the mean-excess plot.

    Returns {'thresholds': np.ndarray, 'mean_excess': np.ndarray}, where
    ``mean_excess[i]`` is E[X - u | X > u] at ``u = thresholds[i]``.

    A mean-excess function that rises roughly linearly in u is the
    signature of a heavy (generalised Pareto) tail; a flat one points to
    an exponential tail.
    """
    losses = np.sort(_tail_losses(returns, tail))
    n = losses.size

    # Every point but the last can serve as a threshold; the largest
    # observation has no exceedances above it.
    thresholds = losses[:-1]
    reverse_sum = np.cumsum(losses[::-1])[::-1]
    counts = np.arange(n, 0, -1)

    # Exceedances above thresholds[i] are losses[i+1:], so shift by one.
    excess_sum = reverse_sum[1:] - thresholds * counts[1:]
    mean_excess_values = excess_sum / counts[1:]

    return {"thresholds": thresholds, "mean_excess": mean_excess_values}


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
