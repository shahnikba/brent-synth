"""Contract tests for every rung of the ladder.

All candidates are exercised against one synthetic series with known
dynamics, so a failure points at the candidate rather than at whatever
the market happened to do. The series is GARCH(1,1)-normal with moderate
persistence — deliberately not any candidate's own law, and tame enough
that every candidate can actually fit it.
"""

import numpy as np
import pandas as pd
import pytest

from brent_synth import model as gjr
from brent_synth.candidates import CANDIDATES, CANDIDATES_BY_NAME
from brent_synth.candidates.base import (
    MIN_FIT_OBS,
    DensityForecast,
    MixtureForecast,
)

N_TRAIN = 1500
N_TEST = 252
TAU = np.array([0.005, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.995])


def _garch_normal_series(n: int, seed: int = 7) -> pd.Series:
    """GARCH(1,1)-normal: omega 4e-6, alpha .07, beta .88 (persistence .95)."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta, mu = 4.0e-6, 0.07, 0.88, 0.0002
    variance = omega / (1.0 - alpha - beta)
    out = np.empty(n)
    for t in range(n):
        shock = np.sqrt(variance) * rng.normal()
        out[t] = mu + shock
        variance = omega + alpha * shock**2 + beta * variance
    index = pd.bdate_range("2010-01-04", periods=n)
    return pd.Series(out, index=index, name="log_return")


@pytest.fixture(scope="module")
def series() -> pd.Series:
    return _garch_normal_series(N_TRAIN + N_TEST)


@pytest.fixture(scope="module")
def train(series: pd.Series) -> pd.Series:
    return series.iloc[:N_TRAIN]


@pytest.fixture(scope="module")
def test_window(series: pd.Series) -> pd.Series:
    return series.iloc[N_TRAIN : N_TRAIN + N_TEST]


@pytest.fixture(scope="module")
def fitted_all(train: pd.Series) -> dict:
    return {c.name: c.fit(train) for c in CANDIDATES}


ALL_NAMES = [c.name for c in CANDIDATES]


def test_registry_is_ordered_and_unique() -> None:
    steps = [c.step for c in CANDIDATES]
    assert steps == sorted(steps)
    assert len(set(steps)) == len(steps)
    assert len(CANDIDATES_BY_NAME) == len(CANDIDATES)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fit_returns_a_usable_model(fitted_all: dict, name: str) -> None:
    fitted = fitted_all[name]
    assert fitted.params, name
    assert all(np.isfinite(v) for v in fitted.params.values())
    assert np.isfinite(fitted.loglik)
    assert fitted.n_params > 0
    assert fitted.name == name


@pytest.mark.parametrize("name", ALL_NAMES)
def test_simulate_shape_and_determinism(fitted_all: dict, name: str) -> None:
    fitted = fitted_all[name]
    first = fitted.simulate(N_TEST, 50, seed=1)
    assert first.shape == (50, N_TEST)
    assert np.isfinite(first).all()
    assert np.array_equal(first, fitted.simulate(N_TEST, 50, seed=1))
    assert not np.array_equal(first, fitted.simulate(N_TEST, 50, seed=2))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_simulated_scale_is_sane(
    fitted_all: dict, name: str, train: pd.Series
) -> None:
    ratio = fitted_all[name].simulate(N_TEST, 200, seed=3).std() / train.std()
    assert 1 / 3 < ratio < 3, f"{name}: std ratio {ratio:.3f}"


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forecast_density_is_a_proper_distribution(
    fitted_all: dict, name: str, test_window: pd.Series
) -> None:
    forecast = fitted_all[name].forecast_density(test_window)
    assert isinstance(forecast, DensityForecast)
    assert forecast.n == len(test_window)

    quantiles = forecast.ppf(TAU)
    assert quantiles.shape == (len(test_window), TAU.size)
    # Quantiles must increase with tau, and cdf must invert ppf.
    assert (np.diff(quantiles, axis=1) > 0).all()
    for j, tau in enumerate(TAU):
        assert np.abs(forecast.cdf(quantiles[:, j]) - tau).max() < 1e-6

    values = forecast.pit_check = forecast.cdf(test_window.to_numpy())
    assert ((values > 0.0) & (values < 1.0)).all()
    assert np.isfinite(forecast.logpdf(test_window.to_numpy())).all()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_fit_rejects_bad_input(name: str) -> None:
    candidate = CANDIDATES_BY_NAME[name]
    gapped = _garch_normal_series(MIN_FIT_OBS + 100).copy()
    gapped.iloc[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        candidate.fit(gapped)
    with pytest.raises(ValueError, match="at least"):
        candidate.fit(_garch_normal_series(MIN_FIT_OBS - 1))


@pytest.mark.parametrize("name", ALL_NAMES)
def test_simulate_rejects_bad_arguments(fitted_all: dict, name: str) -> None:
    with pytest.raises(ValueError, match="horizon must be"):
        fitted_all[name].simulate(0, 10, seed=1)
    with pytest.raises(ValueError, match="n_paths must be"):
        fitted_all[name].simulate(10, 0, seed=1)


# --- the incumbent must stay the incumbent ---------------------------------


def test_gjr_wrapper_matches_model_module_exactly(
    fitted_all: dict, train: pd.Series
) -> None:
    """The wrapper must not quietly become a different model."""
    wrapped = fitted_all["gjr_skewt"]
    direct = gjr.fit(train)
    assert wrapped.params == direct.params
    assert np.array_equal(
        wrapped.simulate(N_TEST, 40, seed=11),
        gjr.simulate(direct, horizon=N_TEST, n_paths=40, seed=11, initial_var="last"),
    )


def test_gjr_forecast_sigma_matches_a_hand_recursion(
    fitted_all: dict, test_window: pd.Series
) -> None:
    """Pin the filtered volatility to the GJR recursion, written out.

    Day 1 opens at the one-step-ahead forecast variance; each later day
    applies the recursion to the shock actually realised. Agreement to
    1e-10 also shows arch's backcast on the combined series has no
    material influence this far into the sample.
    """
    fitted = fitted_all["gjr_skewt"]
    p = fitted.fit.params
    y = test_window.to_numpy()

    variance = fitted.fit.forecast_variance
    expected = np.empty(len(y))
    for t in range(len(y)):
        expected[t] = variance
        shock = y[t] - p["mu"]
        leverage = p["alpha"] + p["gamma"] * (shock < 0.0)
        variance = p["omega"] + leverage * shock**2 + p["beta"] * variance

    forecast = fitted.forecast_density(test_window)
    assert np.abs(forecast.sigma - np.sqrt(expected)).max() < 1e-10


def test_variance_target_pins_the_long_run_variance(
    fitted_all: dict, train: pd.Series
) -> None:
    """Only omega moves; the implied long-run variance hits the sample's."""
    vt = fitted_all["gjr_skewt_vt"]
    base = gjr.fit(train)
    target = float(np.var(train.to_numpy(), ddof=1))

    implied = vt.params["omega"] / (1.0 - vt.fit.persistence)
    assert implied == pytest.approx(target, rel=1e-12)
    assert vt.fit.unconditional_variance == pytest.approx(target, rel=1e-12)

    for name in ("mu", "alpha", "beta", "gamma", "nu", "lambda"):
        assert vt.params[name] == pytest.approx(base.params[name]), name
    assert vt.params["omega"] != pytest.approx(base.params["omega"])


def test_variance_target_refuses_a_non_stationary_fit(monkeypatch) -> None:
    """No finite long-run variance means nothing to target."""
    candidate = CANDIDATES_BY_NAME["gjr_skewt_vt"]
    series = _garch_normal_series(600)

    class Explosive:
        params = {"alpha": 0.2, "beta": 0.9, "gamma": 0.1, "nu": 6.0, "lambda": 0.0}
        persistence = 1.05

    monkeypatch.setattr(
        "brent_synth.candidates.gjr_skewt.gjr.fit", lambda r: Explosive()
    )
    with pytest.raises(ValueError, match="no finite unconditional variance"):
        candidate.fit(series)


# --- regime switching ------------------------------------------------------


def test_regimes_are_ordered_and_stochastic(
    fitted_all: dict, test_window: pd.Series
) -> None:
    fitted = fitted_all["ms_variance"]
    assert fitted.params["sigma2_0"] < fitted.params["sigma2_1"]
    assert np.allclose(fitted.transition.sum(axis=1), 1.0)  # rows, not columns
    assert (fitted.transition >= 0.0).all()
    assert fitted.initial_probabilities.sum() == pytest.approx(1.0)

    forecast = fitted.forecast_density(test_window)
    assert isinstance(forecast, MixtureForecast)
    assert np.allclose(forecast.weights.sum(axis=1), 1.0)


def test_mixture_ppf_inverts_its_cdf() -> None:
    """Bisection must actually solve the mixture quantile."""
    n, k = 40, 2
    rng = np.random.default_rng(0)
    weights = np.tile([0.3, 0.7], (n, 1))
    mu = np.tile([-0.001, 0.002], (n, 1))
    sigma = np.tile([0.005, 0.03], (n, 1))
    forecast = MixtureForecast(weights, mu, sigma)
    quantiles = forecast.ppf(TAU)
    for j, tau in enumerate(TAU):
        assert np.abs(forecast.cdf(quantiles[:, j]) - tau).max() < 1e-9
    assert rng is not None


def test_mixture_rejects_bad_weights() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        MixtureForecast(
            np.array([[0.3, 0.3]]), np.zeros((1, 2)), np.ones((1, 2))
        )
