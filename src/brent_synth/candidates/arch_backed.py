"""Candidates backed directly by an ``arch`` volatility process.

One wrapper covers every model arch can specify, so adding a rung to the
ladder is a line in the registry rather than a new file. The wrapper
owns the two things that are easy to get wrong across all of them: the
percent rescaling arch requires, and making simulation reproducible.

Scale
-----
arch fits on percent returns; everything crossing this module's boundary
is raw log returns. ``mu`` divides by 100 and ``omega`` by 100^2 because
it is a variance — including FIGARCH's omega. Every other parameter
(alpha, beta, gamma, phi, d, nu, lambda) is dimensionless and passes
through untouched.

Reproducibility
---------------
arch's simulator draws from a distribution object rather than an
explicit generator, so a seed is attached by handing it a freshly seeded
distribution per path. Seeds are spawned from one ``SeedSequence``, so
the whole set is determined by the caller's single integer and paths do
not share a stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import Normal, SkewStudent, StudentsT

from brent_synth.candidates.base import (
    DensityForecast,
    LocationScaleForecast,
    NormalStd,
    SkewStudentStd,
    StudentTStd,
    check_fit_input,
    check_simulate_args,
)

PERCENT = 100.0

#: arch distribution classes, and the standardised law each corresponds to.
DIST_CLASSES = {"normal": Normal, "t": StudentsT, "skewt": SkewStudent}

#: arch calls the skew-t degrees of freedom "eta"; model.py already
#: reports it as "nu", and the ladder needs one name across candidates.
PARAM_ALIASES = {"eta": "nu"}


def _to_raw(name: str, value: float) -> float:
    """Undo arch's percent scale for one parameter."""
    if name == "mu":
        return value / PERCENT
    if name == "omega":
        return value / PERCENT**2
    return value


def filtered_sigma(
    train: pd.Series,
    realised: pd.Series,
    params_percent: pd.Series,
    dist: str,
    spec_kwargs: dict,
) -> tuple[np.ndarray, bool]:
    """One-step-ahead conditional volatilities for ``realised``, leak-free.

    Returns ``(sigma_raw, bounds_binding)``.

    The obvious implementation — build the model on the concatenated
    sample and call ``.fix()`` — has a subtle flaw: ``fix`` derives its
    variance backcast from *all* the residuals it is handed, so the
    starting value of the recursion is computed partly from the test
    data. Its influence decays geometrically and is immaterial thousands
    of steps later, but "immaterial" is not "absent", and this is an
    out-of-time experiment.

    arch's ``fix`` takes no ``backcast`` argument (``fit`` does, but that
    re-estimates), so the recursion is driven directly instead: the
    backcast comes from the **training residuals only**, and
    ``compute_variance`` — the same call ``fix`` makes internally — walks
    the combined series from there. Day one then opens at exactly the
    one-step-ahead variance the training fit ended on, which a test
    asserts bit-for-bit.

    ``var_bounds`` are loose clipping bounds arch uses to keep the
    likelihood finite; they are computed over the combined residuals
    because the array must span them. The second return value reports
    whether they ever actually bound — if they never do, no information
    passes through them either.
    """
    train_percent = np.asarray(train, dtype="float64") * PERCENT
    combined_percent = np.concatenate(
        [train_percent, np.asarray(realised, dtype="float64") * PERCENT]
    )
    spec = arch_model(
        combined_percent, mean="Constant", dist=dist, **spec_kwargs
    )
    volatility = spec.volatility

    mu = float(params_percent["mu"])
    resid_train = train_percent - mu
    resid_all = combined_percent - mu

    backcast = volatility.backcast(resid_train)
    var_bounds = volatility.variance_bounds(resid_all)
    vol_params = np.asarray(params_percent, dtype="float64")[
        1 : 1 + volatility.num_params
    ]

    sigma2 = np.zeros(resid_all.size, dtype="float64")
    volatility.compute_variance(
        vol_params, resid_all, sigma2, backcast, var_bounds
    )

    binding = bool(
        np.any(sigma2 <= var_bounds[:, 0]) or np.any(sigma2 >= var_bounds[:, 1])
    )
    n = len(realised)
    return np.sqrt(sigma2[-n:]) / PERCENT, binding


class FittedArchCandidate:
    def __init__(
        self,
        name: str,
        result: object,
        spec_kwargs: dict,
        dist: str,
        train: pd.Series,
        simulation_note: str,
    ) -> None:
        self.name = name
        self.result = result
        self._spec_kwargs = spec_kwargs
        self._dist = dist
        self._train = pd.Series(np.asarray(train, dtype="float64"))
        self.simulation_note = simulation_note

        self.params_percent = result.params
        self.params = {
            PARAM_ALIASES.get(str(k), str(k)): float(_to_raw(str(k), float(v)))
            for k, v in result.params.items()
        }
        self.n_params = len(self.params)
        self.loglik = float(result.loglikelihood)

        # One-step-ahead conditional variance, in percent units, which is
        # where simulated paths must start from.
        forecast = result.forecast(horizon=1, reindex=False)
        self.forecast_variance_percent = float(
            np.asarray(forecast.variance)[-1, 0]
        )
        self.forecast_variance = self.forecast_variance_percent / PERCENT**2

    def _spec(self, values_percent: np.ndarray):
        return arch_model(
            values_percent,
            mean="Constant",
            dist=self._dist,
            **self._spec_kwargs,
        )

    def _standard_dist(self):
        if self._dist == "normal":
            return NormalStd()
        if self._dist == "t":
            return StudentTStd(self.params["nu"])
        return SkewStudentStd(self.params["nu"], self.params["lambda"])

    def simulate(self, horizon: int, n_paths: int, seed: int) -> np.ndarray:
        check_simulate_args(horizon, n_paths)
        spec = self._spec(np.asarray(self._train) * PERCENT)
        dist_class = DIST_CLASSES[self._dist]

        out = np.empty((n_paths, horizon), dtype="float64")
        for i, child in enumerate(np.random.SeedSequence(seed).spawn(n_paths)):
            spec.distribution = dist_class(seed=np.random.default_rng(child))
            simulated = spec.simulate(
                self.params_percent,
                nobs=horizon,
                burn=0,
                initial_value_vol=self.forecast_variance_percent,
            )
            out[i] = simulated["data"].to_numpy() / PERCENT
        return out

    def forecast_density(self, realised: pd.Series) -> DensityForecast:
        sigma, binding = filtered_sigma(
            self._train,
            realised,
            self.params_percent,
            self._dist,
            self._spec_kwargs,
        )
        self.variance_bounds_binding = binding
        return LocationScaleForecast(
            mu=np.full(len(realised), self.params["mu"]),
            sigma=sigma,
            dist=self._standard_dist(),
        )


class ArchCandidate:
    """A scenario model specified by an arch volatility process."""

    def __init__(
        self,
        name: str,
        step: int,
        vol: str,
        dist: str,
        vol_kwargs: dict | None = None,
        simulation_note: str = "paths start from the one-step-ahead variance",
    ) -> None:
        self.name = name
        self.step = step
        self.vol = vol
        self.dist = dist
        self.vol_kwargs = dict(vol_kwargs or {})
        self.simulation_note = simulation_note

    @property
    def spec_kwargs(self) -> dict:
        return {"vol": self.vol, **self.vol_kwargs}

    def fit(self, returns: pd.Series) -> FittedArchCandidate:
        values = check_fit_input(returns, self.name)
        spec = arch_model(
            values * PERCENT, mean="Constant", dist=self.dist, **self.spec_kwargs
        )
        result = spec.fit(disp="off", show_warning=False)
        return FittedArchCandidate(
            self.name,
            result,
            self.spec_kwargs,
            self.dist,
            returns,
            self.simulation_note,
        )


class GarchNormal(ArchCandidate):
    """Step 1: GARCH(1,1) with Gaussian innovations.

    Clustering but no fat-tailed shock and no leverage. The rung that
    isolates what conditional volatility alone is worth.
    """

    def __init__(self) -> None:
        super().__init__(
            "garch_normal", 1, "GARCH", "normal", {"p": 1, "q": 1}
        )


class FigarchSkewT(ArchCandidate):
    """Step 4: FIGARCH(1,d,1) with skewed-t innovations.

    Long memory in volatility. SPEC 2 found Brent's ACF of squared
    returns decays far more slowly than GARCH's geometric rate, and SPEC
    4's report showed the gap directly; the fractional differencing
    parameter d is the standard answer to it. This rung asks whether that
    slow decay is worth paying for out of sample.
    """

    #: arch's default FIGARCH truncation lag, recorded because it caps
    #: how much of the long memory the recursion actually carries.
    TRUNCATION = 1000

    def __init__(self) -> None:
        super().__init__(
            "figarch_skewt", 4, "FIGARCH", "skewt", {"p": 1, "q": 1}
        )
