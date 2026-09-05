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
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import SkewStudent

#: arch fits on percent returns; the public interface is raw log returns.
PERCENT = 100.0

PARAM_NAMES = ("mu", "omega", "alpha", "gamma", "beta", "nu", "lambda")


@dataclass
class ModelFit:
    """A fitted GJR-GARCH(1,1,1)-skewt, with enough state to simulate.

    ``params`` and both variances are in raw log-return units.
    ``loglik``, ``aic`` and ``bic`` are reported on the percent scale
    the optimiser actually worked on, so they compare across models
    fitted through this module but not against raw-scale likelihoods.
    """

    params: dict[str, float]
    loglik: float
    aic: float
    bic: float
    last_variance: float
    unconditional_variance: float
    result: Any | None = field(default=None, repr=False)

    @property
    def persistence(self) -> float:
        """GJR persistence, alpha + gamma/2 + beta.

        The half-weight on gamma is the probability that a symmetric
        shock is negative, so this is the expected decay rate of a
        variance shock. Values below 1 mean the variance process is
        stationary; oil sits very close to 1.
        """
        p = self.params
        return p["alpha"] + p["gamma"] / 2.0 + p["beta"]

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
    values = values[np.isfinite(values)]
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

    persistence = params["alpha"] + params["gamma"] / 2.0 + params["beta"]
    if persistence < 1.0:
        unconditional = params["omega"] / (1.0 - persistence)
    else:
        # Non-stationary fit: no finite stationary variance exists.
        unconditional = float("nan")

    last_variance = float(result.conditional_volatility[-1] ** 2) / PERCENT**2

    return ModelFit(
        params=params,
        loglik=float(result.loglikelihood),
        aic=float(result.aic),
        bic=float(result.bic),
        last_variance=last_variance,
        unconditional_variance=float(unconditional),
        result=result,
    )


def _initial_variance(fit: ModelFit, initial_var: str) -> float:
    if initial_var == "last":
        return fit.last_variance
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

    ``initial_var``: 'last' (start from last fitted conditional
    variance — paths continue from today's vol state) or
    'unconditional' (start from stationary variance).

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
