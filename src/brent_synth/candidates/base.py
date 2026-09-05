"""Interfaces every scenario generator on the ladder implements.

A candidate is two objects: a :class:`ScenarioModel` that knows how to
fit, and a :class:`FittedModel` that knows how to do the two things the
backtest scores — draw whole paths, and state a one-day-ahead
conditional density for each day of a realised stretch.

The density side is what makes the comparison sharp. A path draw says
what the model thinks a *year* looks like; a density says what it
thinks *tomorrow* looks like given everything up to today, with the
parameters frozen at the fit. Only the second can be scored day by day
against what actually happened.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from arch.univariate import SkewStudent
from scipy import stats as sps
from scipy.special import logsumexp

#: Fewest observations any candidate will fit on.
MIN_FIT_OBS = 500


@runtime_checkable
class ScenarioModel(Protocol):
    name: str
    step: int

    def fit(self, returns: pd.Series) -> FittedModel: ...


@runtime_checkable
class FittedModel(Protocol):
    name: str
    params: dict[str, float]
    n_params: int
    loglik: float

    def simulate(self, horizon: int, n_paths: int, seed: int) -> np.ndarray: ...

    def forecast_density(self, realised: pd.Series) -> DensityForecast: ...


def check_fit_input(returns: pd.Series, name: str) -> np.ndarray:
    """Shared validation: refuse gaps, refuse a too-short sample.

    Gaps are refused rather than dropped for the reason the rest of the
    package refuses them — every model here is a recursion over adjacent
    days, so closing a hole silently glues non-neighbours together.
    """
    values = np.asarray(returns, dtype="float64").ravel()
    n_missing = int((~np.isfinite(values)).sum())
    if n_missing:
        raise ValueError(
            f"{name}: return series contains {n_missing} non-finite value(s). "
            "These models are recursions over adjacent days, so the gaps "
            "cannot be dropped silently — handle them explicitly first."
        )
    if values.size < MIN_FIT_OBS:
        raise ValueError(
            f"{name}: need at least {MIN_FIT_OBS} returns to fit, got "
            f"{values.size}."
        )
    return values


def check_simulate_args(horizon: int, n_paths: int) -> None:
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}.")


# --- standardised innovation laws ------------------------------------------
#
# Each exposes logpdf/cdf/ppf for a z with mean 0 and variance 1, so a
# LocationScaleForecast can shift and scale it without knowing which it has.


class NormalStd:
    """Standard normal."""

    name = "normal"

    def logpdf(self, z: np.ndarray) -> np.ndarray:
        return sps.norm.logpdf(z)

    def cdf(self, z: np.ndarray) -> np.ndarray:
        return sps.norm.cdf(z)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return sps.norm.ppf(q)


class StudentTStd:
    """Student-t rescaled to unit variance.

    ``scipy``'s t has variance df/(df-2), so the scale factor
    sqrt((df-2)/df) is what makes z comparable with the other laws here.
    """

    name = "student_t"

    def __init__(self, df: float) -> None:
        if df <= 2.0:
            raise ValueError(f"Student-t needs df > 2 for unit variance, got {df}.")
        self.df = float(df)
        self.scale = float(np.sqrt((df - 2.0) / df))

    def logpdf(self, z: np.ndarray) -> np.ndarray:
        return sps.t.logpdf(z / self.scale, self.df) - np.log(self.scale)

    def cdf(self, z: np.ndarray) -> np.ndarray:
        return sps.t.cdf(z / self.scale, self.df)

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self.scale * sps.t.ppf(q, self.df)


class SkewStudentStd:
    """Hansen skew-t as arch standardises it (mean 0, variance 1)."""

    name = "skewt"

    def __init__(self, nu: float, lam: float) -> None:
        self.nu = float(nu)
        self.lam = float(lam)
        self._params = np.array([self.nu, self.lam])
        self._dist = SkewStudent()

    def logpdf(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype="float64")
        return np.asarray(
            self._dist.loglikelihood(
                self._params, z, np.ones_like(z), individual=True
            )
        )

    def cdf(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(self._dist.cdf(np.asarray(z, "float64"), self._params))

    def ppf(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(self._dist.ppf(np.asarray(q, "float64"), self._params))


# --- density forecasts -----------------------------------------------------


class DensityForecast:
    """One-day-ahead conditional densities for n consecutive days."""

    n: int

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cdf(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ppf(self, tau: np.ndarray) -> np.ndarray:
        """Quantiles, shape (n, len(tau))."""
        raise NotImplementedError

    def _check(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype="float64").ravel()
        if y.size != self.n:
            raise ValueError(
                f"Expected {self.n} realised values, got {y.size}."
            )
        return y


class LocationScaleForecast(DensityForecast):
    """y_t = mu_t + sigma_t * z, with z from a standardised law."""

    def __init__(
        self,
        mu: np.ndarray,
        sigma: np.ndarray,
        dist: NormalStd | StudentTStd | SkewStudentStd,
    ) -> None:
        self.mu = np.asarray(mu, dtype="float64").ravel()
        self.sigma = np.asarray(sigma, dtype="float64").ravel()
        if self.mu.size != self.sigma.size:
            raise ValueError("mu and sigma must have the same length.")
        if not np.all(self.sigma > 0.0):
            raise ValueError("All conditional volatilities must be positive.")
        self.dist = dist
        self.n = int(self.mu.size)

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        y = self._check(y)
        return self.dist.logpdf((y - self.mu) / self.sigma) - np.log(self.sigma)

    def cdf(self, y: np.ndarray) -> np.ndarray:
        y = self._check(y)
        return self.dist.cdf((y - self.mu) / self.sigma)

    def ppf(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype="float64").ravel()
        z = self.dist.ppf(tau)
        return self.mu[:, None] + self.sigma[:, None] * z[None, :]


class MixtureForecast(DensityForecast):
    """Per-day mixture of K Gaussians.

    Used by the regime-switching candidate, where each day is a mixture
    over the regime the market might be in. The mixture is where its fat
    tails come from — within a regime the shock is Gaussian.
    """

    #: Bisection settings for the quantile function. The mixture cdf has
    #: no closed-form inverse, but it is monotone, so bisection on a
    #: bracket wide enough to contain every quantile is exact to machine
    #: noise in this many halvings.
    PPF_ITERATIONS = 60
    PPF_BRACKET_SIGMAS = 12.0

    def __init__(
        self, weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray
    ) -> None:
        self.weights = np.asarray(weights, dtype="float64")
        self.mu = np.asarray(mu, dtype="float64")
        self.sigma = np.asarray(sigma, dtype="float64")
        if not (self.weights.shape == self.mu.shape == self.sigma.shape):
            raise ValueError("weights, mu and sigma must share a shape (n, K).")
        if self.weights.ndim != 2:
            raise ValueError("Mixture arrays must be 2-D, shape (n, K).")
        if not np.all(self.sigma > 0.0):
            raise ValueError("All mixture components must have positive sigma.")
        if not np.allclose(self.weights.sum(axis=1), 1.0):
            raise ValueError("Mixture weights must sum to 1 on each day.")
        self.n = int(self.weights.shape[0])

    def logpdf(self, y: np.ndarray) -> np.ndarray:
        y = self._check(y)
        component = sps.norm.logpdf(y[:, None], loc=self.mu, scale=self.sigma)
        return logsumexp(component + np.log(self.weights), axis=1)

    def cdf(self, y: np.ndarray) -> np.ndarray:
        y = self._check(y)
        component = sps.norm.cdf(y[:, None], loc=self.mu, scale=self.sigma)
        return (self.weights * component).sum(axis=1)

    def _cdf_at(self, x: np.ndarray) -> np.ndarray:
        """cdf evaluated at a per-day, per-tau grid of shape (n, m)."""
        component = sps.norm.cdf(
            x[:, :, None], loc=self.mu[:, None, :], scale=self.sigma[:, None, :]
        )
        return (self.weights[:, None, :] * component).sum(axis=2)

    def ppf(self, tau: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau, dtype="float64").ravel()
        spread = self.PPF_BRACKET_SIGMAS * self.sigma.max(axis=1)
        lo = np.repeat((self.mu.min(axis=1) - spread)[:, None], tau.size, axis=1)
        hi = np.repeat((self.mu.max(axis=1) + spread)[:, None], tau.size, axis=1)
        target = np.broadcast_to(tau[None, :], lo.shape)

        for _ in range(self.PPF_ITERATIONS):
            mid = 0.5 * (lo + hi)
            below = self._cdf_at(mid) < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)
