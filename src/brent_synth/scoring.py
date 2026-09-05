"""Day-level scores for density forecasts.

Every score here is a **loss**: lower is better, always, so the ranking
rule in :mod:`brent_synth.backtest` can treat them uniformly and nobody
has to remember which way round a particular number reads.

The CRPS is computed from the quantile function rather than the density,
using the Gneiting-Ranjan (2011) form

    CRPS(F, y) = 2 * int_0^1 (1{y <= q(tau)} - tau) (q(tau) - y) dtau

which needs only ``ppf``. That matters: it is the one formulation that
works identically for a location-scale forecast and for a per-day
Gaussian mixture, so the regime-switching candidate is scored by exactly
the same code as the GARCH family rather than by a special case.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps
from scipy.special import xlogy

from brent_synth.candidates.base import DensityForecast

#: Quantile grid for the CRPS integral. Fixed as a module constant
#: because it is part of the pre-registered scoring rule — changing it
#: changes every number in the comparison.
TAU_GRID = np.linspace(0.0005, 0.9995, 999)


def nll(forecast: DensityForecast, y: np.ndarray) -> np.ndarray:
    """Negative log predictive density, per day."""
    return -forecast.logpdf(y)


def _crps_from_quantiles(
    quantiles: np.ndarray, y: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """Trapezoid integration of the quantile-loss integrand."""
    y = np.asarray(y, dtype="float64").ravel()[:, None]
    integrand = (np.where(y <= quantiles, 1.0, 0.0) - tau[None, :]) * (
        quantiles - y
    )
    if tau.size == 1:
        return 2.0 * integrand[:, 0]
    return 2.0 * np.trapezoid(integrand, x=tau, axis=1)


def crps(
    forecast: DensityForecast, y: np.ndarray, tau: np.ndarray = TAU_GRID
) -> np.ndarray:
    """Continuous ranked probability score, per day."""
    tau = np.asarray(tau, dtype="float64").ravel()
    return _crps_from_quantiles(forecast.ppf(tau), y, tau)


def tail_crps(
    forecast: DensityForecast,
    y: np.ndarray,
    tau_max: float = 0.05,
    tau: np.ndarray = TAU_GRID,
) -> np.ndarray:
    """CRPS restricted to the left tail, per day.

    The same integrand as :func:`crps` with an indicator weight on
    ``tau <= tau_max``: a quantile-weighted CRPS that only pays attention
    to how well the model describes losses. A model can win on the whole
    distribution by being right about the quiet middle; this is the score
    that asks whether it is right where it matters for risk.
    """
    tau = np.asarray(tau, dtype="float64").ravel()
    selected = tau[tau <= tau_max]
    if selected.size == 0:
        raise ValueError(
            f"No grid points at or below tau_max={tau_max}; the grid starts "
            f"at {tau.min()}."
        )
    return _crps_from_quantiles(forecast.ppf(selected), y, selected)


def pit(forecast: DensityForecast, y: np.ndarray) -> np.ndarray:
    """Probability integral transform, F_t(y_t), per day.

    Uniform if the forecast densities are right. Not a loss — it is read
    as a histogram, U-shaped meaning over-confident and hump-shaped
    meaning under-confident.
    """
    return forecast.cdf(y)


def var_exceedances(
    forecast: DensityForecast, y: np.ndarray, level: float
) -> np.ndarray:
    """Boolean per day: did the loss breach the VaR at ``level``?

    ``level=0.99`` is the 99% VaR, i.e. the 1% quantile of the return.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}.")
    threshold = forecast.ppf(np.array([1.0 - level]))[:, 0]
    return np.asarray(y, dtype="float64").ravel() < threshold


def kupiec_pvalue(n_exceed: int, n: int, p: float) -> float:
    """Unconditional coverage test: is the exceedance *rate* right?

    ``xlogy`` handles the boundary cases the naive formula cannot —
    zero exceedances, or every day an exceedance — where a term is
    0 * log(0) and should contribute nothing.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}.")
    x = int(n_exceed)
    if not 0 <= x <= n:
        raise ValueError(f"n_exceed must be in [0, {n}], got {x}.")

    rate = x / n
    null = xlogy(n - x, 1.0 - p) + xlogy(x, p)
    alternative = xlogy(n - x, 1.0 - rate) + xlogy(x, rate)
    statistic = -2.0 * null + 2.0 * alternative
    return float(sps.chi2.sf(max(statistic, 0.0), df=1))


def christoffersen_pvalue(exceed: np.ndarray) -> float:
    """Independence test: are exceedances clustered?

    A model can get the exceedance rate right and still fail badly by
    delivering all its breaches in one week. This is the likelihood ratio
    against a first-order Markov chain on the indicator.

    Returns NaN with fewer than two exceedances — there is no transition
    structure to test.
    """
    exceed = np.asarray(exceed, dtype=bool).ravel()
    if exceed.sum() < 2:
        return float("nan")

    previous, current = exceed[:-1], exceed[1:]
    n00 = int(np.sum(~previous & ~current))
    n01 = int(np.sum(~previous & current))
    n10 = int(np.sum(previous & ~current))
    n11 = int(np.sum(previous & current))

    total = n00 + n01 + n10 + n11
    if total == 0:
        return float("nan")

    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0

    null = xlogy(n00 + n10, 1.0 - pi) + xlogy(n01 + n11, pi)
    alternative = (
        xlogy(n00, 1.0 - pi01)
        + xlogy(n01, pi01)
        + xlogy(n10, 1.0 - pi11)
        + xlogy(n11, pi11)
    )
    statistic = -2.0 * null + 2.0 * alternative
    return float(sps.chi2.sf(max(statistic, 0.0), df=1))


def sign_split_nll(
    forecast: DensityForecast, y: np.ndarray, y_prev: np.ndarray
) -> tuple[float, float]:
    """Mean NLL split by the sign of the *previous* day's return.

    A leverage term claims the market behaves differently after a down
    day. This is the direct check: if the claim is doing work, the two
    numbers differ, and a model without leverage should show a wider gap
    than one with it.
    """
    losses = nll(forecast, y)
    previous = np.asarray(y_prev, dtype="float64").ravel()
    if previous.size != losses.size:
        raise ValueError(
            f"y_prev must have {losses.size} values, got {previous.size}."
        )
    after_down = previous < 0.0
    mean_down = float(losses[after_down].mean()) if after_down.any() else float("nan")
    mean_up = float(losses[~after_down].mean()) if (~after_down).any() else float("nan")
    return mean_down, mean_up


def diebold_mariano(
    loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 10
) -> tuple[float, float]:
    """Test whether two loss series differ, accounting for serial correlation.

    Newey-West HAC variance with Bartlett weights: daily losses are
    strongly autocorrelated through the volatility state, so treating
    them as independent would overstate significance badly.

    Returns ``(statistic, two-sided p)``. Identical inputs give
    ``(nan, nan)`` rather than raising — there is no difference to test.
    """
    a = np.asarray(loss_a, dtype="float64").ravel()
    b = np.asarray(loss_b, dtype="float64").ravel()
    if a.size != b.size:
        raise ValueError(f"Loss series differ in length: {a.size} vs {b.size}.")
    if a.size < 2:
        raise ValueError("Need at least 2 observations for a DM test.")
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}.")

    difference = a - b
    n = difference.size
    mean = float(difference.mean())
    centred = difference - mean

    variance = float(centred @ centred) / n
    for j in range(1, min(lag, n - 1) + 1):
        covariance = float(centred[j:] @ centred[:-j]) / n
        variance += 2.0 * (1.0 - j / (lag + 1.0)) * covariance

    if not np.isfinite(variance) or variance <= 0.0:
        return float("nan"), float("nan")

    statistic = mean / np.sqrt(variance / n)
    p_value = float(2.0 * sps.norm.sf(abs(statistic)))
    return float(statistic), p_value
