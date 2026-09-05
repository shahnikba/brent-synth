"""Step 2: the incumbent GJR-GARCH(1,1,1)-skewt, wrapped not rewritten.

This candidate delegates to :mod:`brent_synth.model` for both fitting
and simulation, so the ladder scores exactly the model the rest of the
package already validated. A regression test pins ``simulate`` to
``model.simulate`` byte-for-byte; if this file ever starts doing its own
thing, that test fails.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from arch import arch_model

from brent_synth import model as gjr
from brent_synth.candidates.arch_backed import filtered_sigma
from brent_synth.candidates.base import (
    DensityForecast,
    LocationScaleForecast,
    SkewStudentStd,
    check_fit_input,
    check_simulate_args,
)

PERCENT = gjr.PERCENT


class FittedGjrSkewT:
    name = "gjr_skewt"
    n_params = 7

    def __init__(self, fit: gjr.ModelFit, train: pd.Series) -> None:
        self.fit = fit
        self.params = dict(fit.params)
        self.loglik = fit.loglik
        self._train = pd.Series(np.asarray(train, dtype="float64"))

    @property
    def forecast_variance(self) -> float:
        """sigma^2_{T+1}: the variance day one of the test window opens at."""
        return self.fit.forecast_variance

    def simulate(self, horizon: int, n_paths: int, seed: int) -> np.ndarray:
        check_simulate_args(horizon, n_paths)
        return gjr.simulate(
            self.fit,
            horizon=horizon,
            n_paths=n_paths,
            seed=seed,
            initial_var="last",
        )

    def forecast_density(self, realised: pd.Series) -> DensityForecast:
        sigma, binding = filtered_sigma(
            self._train,
            realised,
            self.fit.result.params,
            dist="skewt",
            spec_kwargs={"vol": "GARCH", "p": 1, "o": 1, "q": 1},
        )
        self.variance_bounds_binding = binding
        return LocationScaleForecast(
            mu=np.full(len(realised), self.params["mu"]),
            sigma=sigma,
            dist=SkewStudentStd(self.params["nu"], self.params["lambda"]),
        )


class GjrSkewT:
    """GJR-GARCH(1,1,1) with skewed-t innovations — the current model."""

    name = "gjr_skewt"
    step = 2

    def fit(self, returns: pd.Series) -> FittedGjrSkewT:
        check_fit_input(returns, self.name)
        return FittedGjrSkewT(gjr.fit(returns), returns)


class FittedGjrSkewTVarianceTarget(FittedGjrSkewT):
    name = "gjr_skewt_vt"

    def __init__(
        self, fit: gjr.ModelFit, train: pd.Series, target_variance: float
    ) -> None:
        super().__init__(fit, train)
        self.target_variance = target_variance


class GjrSkewTVarianceTarget:
    """Step 3: GJR-skewt with omega pinned to the sample variance.

    Fits the unconstrained model, then replaces omega alone so that the
    implied unconditional variance equals the training-sample variance:

        omega_vt = var(train) * (1 - persistence)

    **No other parameter is re-estimated.** That is deliberate and it is
    the experiment: SPEC 4's validation found the incumbent
    over-disperses volatility across paths (2.13x the real band), and its
    fitted long-run variance sits well above the sample's. Changing one
    parameter and nothing else isolates whether that inflated long-run
    level is the cause. Re-estimating everything under the constraint
    would answer a different question.

    A non-stationary fit has no finite long-run variance to target, so
    this candidate raises there and the backtest records it as
    unavailable at that origin.
    """

    name = "gjr_skewt_vt"
    step = 3

    def fit(self, returns: pd.Series) -> FittedGjrSkewTVarianceTarget:
        values = check_fit_input(returns, self.name)
        base = gjr.fit(returns)
        persistence = base.persistence
        if persistence >= 1.0:
            raise ValueError(
                f"{self.name}: fitted persistence {persistence:.6f} >= 1, so "
                "there is no finite unconditional variance to target."
            )

        target = float(np.var(values, ddof=1))
        omega_vt = target * (1.0 - persistence)

        params = dict(base.params)
        params["omega"] = omega_vt

        last_shock = float(values[-1]) - params["mu"]
        leverage = params["alpha"] + params["gamma"] * (last_shock < 0.0)
        forecast_variance = (
            omega_vt + leverage * last_shock**2 + params["beta"] * base.last_variance
        )

        # Re-fix the arch result at the retargeted omega so the filtered
        # volatility and reported loglik describe the model being scored.
        percent_params = base.result.params.copy()
        percent_params["omega"] = omega_vt * PERCENT**2
        spec = arch_model(
            values * PERCENT, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="skewt"
        )
        refixed = spec.fix(percent_params)

        retargeted = replace(
            base,
            params=params,
            loglik=float(refixed.loglikelihood),
            forecast_variance=float(forecast_variance),
            unconditional_variance=target,
            result=refixed,
        )
        return FittedGjrSkewTVarianceTarget(retargeted, returns, target)
