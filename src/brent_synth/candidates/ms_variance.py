"""Step 5: two-regime Markov-switching variance.

Deliberately the "regimes without a fat-tailed shock" contender. Within
a regime the innovation is **Gaussian**; all of the unconditional excess
kurtosis comes from mixing a calm regime with a turbulent one, and all
of the clustering comes from the regimes being persistent rather than
from a GARCH recursion. It is on the ladder to test whether that
mechanism alone can stand in for conditional volatility plus a heavy
shock — a genuinely different story about where Brent's fat tails come
from, not a small variation on the rungs below.

Fitted in percent units for the same numerical reason arch works there:
raw daily variances are around 1e-4 and the optimiser handles them
poorly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from brent_synth.candidates.base import (
    DensityForecast,
    MixtureForecast,
    check_fit_input,
    check_simulate_args,
)

PERCENT = 100.0
K_REGIMES = 2

#: statsmodels uses random restarts to escape local optima. Fixed so a
#: refit of the same window returns the same model.
SEARCH_REPS = 20
SEARCH_SEED = 20240


def _build(values_percent: np.ndarray) -> MarkovRegression:
    return MarkovRegression(
        values_percent,
        k_regimes=K_REGIMES,
        trend="c",
        switching_variance=True,
    )


class FittedMarkovSwitchingVariance:
    name = "ms_variance"
    n_params = 6

    def __init__(self, result: object, train: pd.Series) -> None:
        self._sm_params = result.params
        self._train = pd.Series(np.asarray(train, dtype="float64"))
        self.loglik = float(result.llf)

        raw = np.asarray(result.params, dtype="float64")
        mu_percent = raw[2:4]
        sigma2_percent = raw[4:6]

        # Order regimes by variance so regime 0 is always the calm one;
        # statsmodels' labelling is arbitrary and would otherwise flip
        # between origins and make the parameter-drift plot meaningless.
        self._order = np.argsort(sigma2_percent)
        mu_percent = mu_percent[self._order]
        sigma2_percent = sigma2_percent[self._order]

        # regime_transition[i, j] is P(next = i | current = j): columns
        # are the "from" state. Transpose to the row-stochastic form the
        # simulation walks.
        transition = np.asarray(result.regime_transition)[:, :, 0]
        transition = transition.T[np.ix_(self._order, self._order)]
        self.transition = transition

        self.mu = mu_percent / PERCENT
        self.sigma2 = sigma2_percent / PERCENT**2
        self.sigma = np.sqrt(self.sigma2)

        filtered = np.asarray(result.filtered_marginal_probabilities)
        self.initial_probabilities = filtered[-1][self._order]

        self.params = {
            "mu_0": float(self.mu[0]),
            "mu_1": float(self.mu[1]),
            "sigma2_0": float(self.sigma2[0]),
            "sigma2_1": float(self.sigma2[1]),
            "p00": float(transition[0, 0]),
            "p11": float(transition[1, 1]),
        }

    def simulate(self, horizon: int, n_paths: int, seed: int) -> np.ndarray:
        check_simulate_args(horizon, n_paths)
        rng = np.random.default_rng(seed)

        # Start from the regime the market was filtered into on the last
        # training day, then advance the chain one step per simulated day.
        state = rng.choice(
            K_REGIMES, size=n_paths, p=self.initial_probabilities
        )
        out = np.empty((n_paths, horizon), dtype="float64")
        to_high = self.transition[:, 1]

        for t in range(horizon):
            state = (rng.random(n_paths) < to_high[state]).astype(np.int64)
            out[:, t] = self.mu[state] + self.sigma[state] * rng.normal(
                size=n_paths
            )
        return out

    def forecast_density(self, realised: pd.Series) -> DensityForecast:
        n = len(realised)
        combined = np.concatenate(
            [np.asarray(self._train), np.asarray(realised, dtype="float64")]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            filtered_result = _build(combined * PERCENT).filter(self._sm_params)

        filtered = np.asarray(
            filtered_result.filtered_marginal_probabilities
        )[:, self._order]

        # Day t's regime distribution given data through t-1: push the
        # filtered probabilities of t-1 one step through the chain. Using
        # filtered[t] here instead would leak day t's own return.
        predicted = filtered[:-1] @ self.transition
        weights = predicted[-n:]

        return MixtureForecast(
            weights=weights,
            mu=np.broadcast_to(self.mu, (n, K_REGIMES)),
            sigma=np.broadcast_to(self.sigma, (n, K_REGIMES)),
        )


class MarkovSwitchingVariance:
    """Two-regime Gaussian mixture with Markov-persistent variance."""

    name = "ms_variance"
    step = 5

    def fit(self, returns: pd.Series) -> FittedMarkovSwitchingVariance:
        values = check_fit_input(returns, self.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _build(values * PERCENT).fit(
                search_reps=SEARCH_REPS,
                rng=np.random.default_rng(SEARCH_SEED),
                disp=0,
            )
        return FittedMarkovSwitchingVariance(result, returns)
