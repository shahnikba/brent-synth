"""GJR-GARCH(1,1,1) with skewed-t innovations: fit and simulate.

The model is

    r_t        = mu + eps_t
    eps_t      = sigma_t * z_t,          z_t ~ standardised skew-t(nu, lambda)
    sigma_t^2  = omega + (alpha + gamma * I[eps_{t-1} < 0]) * eps_{t-1}^2
                 + beta * sigma_{t-1}^2

The leverage term ``gamma`` is what makes this GJR rather than plain
GARCH: a negative shock raises tomorrow's variance by more than a
positive shock of the same size.

Units
-----
Everything crossing the public boundary of this module — the returns
you pass in, the parameters on :class:`ModelFit`, the array
:func:`simulate` hands back — is in **raw log-return units**.

``arch`` expects returns in percent and is numerically poor if given
raw daily returns (it warns and can converge badly), so :func:`fit`
multiplies by 100 internally and converts the estimated parameters
back on the way out. That rescaling is confined to :func:`fit`:
``mu`` divides by 100 and ``omega`` by 100^2 because it is a variance,
while ``alpha``, ``gamma``, ``beta``, ``nu`` and ``lambda`` are
dimensionless and pass through untouched. :func:`simulate` then runs
entirely in raw units and never needs the factor at all. Getting this
wrong is silent — the fit still converges, the simulated paths are
just 100x too big — so it is deliberately kept in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent

#: arch fits on percent returns; the public interface is raw log returns.
PERCENT = 100.0

PARAM_NAMES = ("mu", "omega", "alpha", "gamma", "beta", "nu", "lambda")


@lru_cache(maxsize=256)
def _leverage_weight(nu: float, lam: float) -> float:
    """E[z^2 1{z<0}] for the standardised skew-t: the weight on gamma.

    Taking expectations through the GJR recursion,

        E[sigma^2] = omega + alpha E[sigma^2] + gamma E[sigma^2] E[z^2 1{z<0}]
                     + beta E[sigma^2]

    so the coefficient gamma carries is the innovation's **partial second
    moment** below zero, not the probability of a negative shock. The two
    coincide at 1/2 for any symmetric unit-variance z, which is why the
    textbook rule is written as gamma/2 — but these innovations are
    deliberately skewed, and there the shortcut fails in both magnitude
    and direction. At the fitted lambda = -0.112 the probability falls to
    0.477 while this weight rises to 0.543, so reasoning from the
    probability moves the weight the wrong way.
    """
    return float(SkewStudent().partial_moment(2, 0.0, parameters=np.array([nu, lam])))


def _persistence(params: dict[str, float]) -> float:
    """GJR persistence, alpha + gamma * E[z^2 1{z<0}] + beta."""
    return (
        params["alpha"]
        + params["gamma"] * _leverage_weight(params["nu"], params["lambda"])
        + params["beta"]
    )


@dataclass
class ModelFit:
    """A fitted GJR-GARCH(1,1,1)-skewt, with enough state to simulate.

    ``params`` and every variance are in raw log-return units.

    Two starting variances are carried, and they are not the same
    thing. ``last_variance`` is sigma_T^2, the conditional variance
    *of* the final observed day. ``forecast_variance`` is
    sigma_{T+1}^2, one GJR recursion step further on, formed from the
    final observed shock — that is the variance tomorrow opens at, and
    it is what :func:`simulate` starts from under
    ``initial_var='last'``.
    ``loglik``, ``aic`` and ``bic`` are reported on the percent scale
    the optimiser actually worked on, so they compare across models
    fitted through this module but not against raw-scale likelihoods.
    """

    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    last_variance: float
    forecast_variance: float
    unconditional_variance: float
    result: Any | None = field(default=None, repr=False)

    @property
    def persistence(self) -> float:
        """GJR persistence, alpha + gamma * E[z^2 1{z<0}] + beta.

        The expected decay rate of a variance shock, and the quantity
        that decides stationarity: below 1 the variance process is
        stationary and has the finite long-run level reported as
        ``unconditional_variance``. Oil sits very close to 1.

        The weight on gamma is the innovation's partial second moment
        below zero — see :func:`_leverage_weight`. Under skewed
        innovations it is not 1/2, and using 1/2 understates persistence
        and so understates the long-run variance, by 21% on the Brent
        fit.
        """
        return _persistence(self.params)

    @property
    def leverage_weight(self) -> float:
        """The weight gamma carries in :attr:`persistence`."""
        return _leverage_weight(self.params["nu"], self.params["lambda"])

    @property
    def is_stationary(self) -> bool:
        return self.persistence < 1.0


def fit(returns: pd.Series) -> ModelFit:
    """Fit GJR-GARCH(1,1,1)-skewt by MLE.

    Returns a ModelFit holding: params (mu, omega, alpha, gamma, beta,
    nu, lambda), loglik, aic, bic, the last conditional variance, the
    unconditional variance, and the fitted arch result object.
    """
    values = np.asarray(returns, dtype="float64").ravel()
    n_missing = int((~np.isfinite(values)).sum())
    if n_missing:
        # A GARCH likelihood is built entirely from adjacency: every
        # conditional variance is a function of the previous one. Dropping
        # a gap splices its two sides together and the recursion carries
        # the error forward through the whole sample, so refuse it here
        # exactly as the diagnostics module does for autocorrelation.
        raise ValueError(
            f"Return series contains {n_missing} non-finite value(s). "
            "The GARCH recursion depends on adjacency, so the gaps cannot "
            "be dropped silently — handle them explicitly first."
        )
    if values.size < 100:
        raise ValueError(f"Need at least 100 returns to fit, got {values.size}.")

    # --- the one place the percent rescaling happens ---
    scaled = values * PERCENT

    model = arch_model(
        scaled, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewt"
    )
    result = model.fit(disp="off", show_warning=False)
    raw = result.params

    # mu is a location (divide by 100); omega is a variance (divide by
    # 100^2). The rest are dimensionless.
    params = {
        "mu": float(raw["mu"]) / PERCENT,
        "omega": float(raw["omega"]) / PERCENT**2,
        "alpha": float(raw["alpha[1]"]),
        "gamma": float(raw["gamma[1]"]),
        "beta": float(raw["beta[1]"]),
        "nu": float(raw["eta"]),
        "lambda": float(raw["lambda"]),
    }

    persistence = _persistence(params)
    if persistence < 1.0:
        unconditional = params["omega"] / (1.0 - persistence)
    else:
        # Non-stationary fit: no finite stationary variance exists.
        unconditional = float("nan")

    last_variance = float(result.conditional_volatility[-1] ** 2) / PERCENT**2

    # One GJR step past the sample: tomorrow's conditional variance,
    # built from the final observed shock. Done in raw units so it needs
    # no rescaling of its own.
    last_shock = float(values[-1]) - params["mu"]
    leverage = params["alpha"] + params["gamma"] * (last_shock < 0.0)
    forecast_variance = (
        params["omega"] + leverage * last_shock**2 + params["beta"] * last_variance
    )

    return ModelFit(
        params=params,
        loglik=float(result.loglikelihood),
        aic=float(result.aic),
        bic=float(result.bic),
        last_variance=last_variance,
        forecast_variance=float(forecast_variance),
        unconditional_variance=float(unconditional),
        result=result,
    )


def _initial_variance(fit: ModelFit, initial_var: str) -> float:
    if initial_var == "last":
        return fit.forecast_variance
    if initial_var == "unconditional":
        variance = fit.unconditional_variance
        if not np.isfinite(variance):
            raise ValueError(
                "Fit is non-stationary, so it has no unconditional variance; "
                "use initial_var='last'."
            )
        return variance
    raise ValueError(
        f"initial_var must be 'last' or 'unconditional', got {initial_var!r}."
    )


def simulate(
    fit: ModelFit,
    horizon: int = 252,
    n_paths: int = 5000,
    seed: int = 42,
    initial_var: str = "last",
) -> np.ndarray:
    """Simulate returns. Returns array shape (n_paths, horizon) in raw
    log-return units.

    ``initial_var`` picks the variance day 1 opens at:

    - ``'last'`` — the one-step-ahead forecast sigma_{T+1}^2, i.e. one
      GJR recursion step applied to the last observed shock. Paths
      continue from today's vol state, which is what you want when the
      question is "what happens next". Note this is deliberately *not*
      sigma_T^2, the variance of the final observed day; carrying that
      forward would replay the last day instead of moving past it.
    - ``'unconditional'`` — the stationary variance
      omega / (1 - alpha - gamma * E[z^2 1{z<0}] - beta). Paths start from the
      long-run vol level, ignoring where the market happens to sit
      today.

    The GJR recursion is path-dependent, so it is stepped through time
    with all ``n_paths`` advanced together at each step. Innovations
    come from the fitted skew-t by inverse-transform sampling off a
    ``default_rng(seed)`` stream, which is what makes the output
    reproducible: the same seed gives byte-identical arrays.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be >= 1, got {n_paths}.")

    p = fit.params
    mu, omega = p["mu"], p["omega"]
    alpha, gamma, beta = p["alpha"], p["gamma"], p["beta"]

    start_variance = _initial_variance(fit, initial_var)

    # Standardised skew-t draws (mean 0, variance 1) by inverse
    # transform, so the seeded uniform stream fully determines them.
    rng = np.random.default_rng(seed)
    uniforms = rng.random((n_paths, horizon))
    dist = SkewStudent()
    z = np.asarray(
        dist.ppf(uniforms.ravel(), parameters=np.array([p["nu"], p["lambda"]]))
    ).reshape(n_paths, horizon)

    returns = np.empty((n_paths, horizon), dtype="float64")
    variance = np.full(n_paths, start_variance, dtype="float64")

    for t in range(horizon):
        eps = np.sqrt(variance) * z[:, t]
        returns[:, t] = mu + eps
        # Tomorrow's variance from today's shock, with the extra
        # leverage weight only on negative shocks.
        leverage = alpha + gamma * (eps < 0.0)
        variance = omega + leverage * eps**2 + beta * variance

    return returns


def fit_and_simulate(
    returns: pd.Series, **kwargs: Any
) -> tuple[ModelFit, np.ndarray]:
    """Fit then simulate in one call, for the report pipeline."""
    fitted = fit(returns)
    return fitted, simulate(fitted, **kwargs)
