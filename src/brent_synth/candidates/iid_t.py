"""Step 0: iid Student-t. The floor of the ladder.

No volatility dynamics at all — every day draws from the same
distribution. It is on the ladder to establish what fat tails alone buy
you, so that any higher rung has to earn its extra parameters against a
model that knows nothing about clustering. Its forecast density is
constant by construction; that is the point, not an oversight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from brent_synth.candidates.base import (
    DensityForecast,
    LocationScaleForecast,
    StudentTStd,
    check_fit_input,
    check_simulate_args,
)


class FittedIidStudentT:
    name = "iid_t"
    n_params = 3

    def __init__(self, params: dict[str, float], loglik: float) -> None:
        self.params = params
        self.loglik = loglik
        self._frozen = sps.t(
            df=params["nu"], loc=params["mu"], scale=params["scale"]
        )

    def simulate(self, horizon: int, n_paths: int, seed: int) -> np.ndarray:
        check_simulate_args(horizon, n_paths)
        rng = np.random.default_rng(seed)
        uniforms = rng.random((n_paths, horizon))
        return self._frozen.ppf(uniforms)

    def forecast_density(self, realised: pd.Series) -> DensityForecast:
        n = len(realised)
        nu, scale = self.params["nu"], self.params["scale"]
        # scipy's t has variance nu/(nu-2); the standardised z carries the
        # unit-variance version, so sigma absorbs the rest.
        sigma = scale * np.sqrt(nu / (nu - 2.0))
        return LocationScaleForecast(
            mu=np.full(n, self.params["mu"]),
            sigma=np.full(n, sigma),
            dist=StudentTStd(nu),
        )


class IidStudentT:
    """Independent Student-t returns, fitted by maximum likelihood."""

    name = "iid_t"
    step = 0

    def fit(self, returns: pd.Series) -> FittedIidStudentT:
        values = check_fit_input(returns, self.name)
        df, loc, scale = sps.t.fit(values)
        if df <= 2.0:
            raise ValueError(
                f"{self.name}: fitted df={df:.3f} <= 2, so the variance is "
                "infinite and the model is unusable here."
            )
        loglik = float(sps.t.logpdf(values, df, loc=loc, scale=scale).sum())
        return FittedIidStudentT(
            {"mu": float(loc), "scale": float(scale), "nu": float(df)}, loglik
        )
