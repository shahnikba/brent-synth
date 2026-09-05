"""Tests for the GJR-GARCH(1,1,1)-skewt fit and simulator.

The last test is the important one: it simulates from known parameters,
refits, and checks the parameters come back. That closes the loop
between :func:`fit` and :func:`simulate` — either one being wrong on its
own would break it, including the percent/raw rescaling.
"""

import numpy as np
import pandas as pd
import pytest

from brent_synth.data import load_returns
from brent_synth.model import ModelFit, fit, fit_and_simulate, simulate

TRUE_PARAMS = {
    "mu": 0.0005,
    "omega": 4.0e-6,
    "alpha": 0.06,
    "gamma": 0.05,
    "beta": 0.90,
    "nu": 6.0,
    "lambda": -0.10,
}


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return load_returns()


@pytest.fixture(scope="module")
def fitted(returns: pd.Series) -> ModelFit:
    return fit(returns)


# --- fit -------------------------------------------------------------------


def test_fit_returns_all_params(fitted: ModelFit) -> None:
    assert set(fitted.params) == {
        "mu",
        "omega",
        "alpha",
        "gamma",
        "beta",
        "nu",
        "lambda",
    }
    assert all(np.isfinite(v) for v in fitted.params.values())


def test_variance_params_are_admissible(fitted: ModelFit) -> None:
    p = fitted.params
    assert p["omega"] > 0.0
    assert 0.0 < p["beta"] < 1.0
    assert 0.0 < p["alpha"] + p["beta"] < 1.0


def test_innovations_have_finite_variance(fitted: ModelFit) -> None:
    """nu > 2 or the skew-t has no second moment."""
    assert fitted.params["nu"] > 2.0


def test_leverage_term_is_present(fitted: ModelFit) -> None:
    """gamma > 0 means negative shocks raise variance more than positive."""
    assert fitted.params["gamma"] > 0.0


def test_fit_is_stationary(fitted: ModelFit) -> None:
    """alpha + gamma/2 + beta < 1 is the GJR stationarity condition.

    Oil sits very close to 1; only exceeding it is a problem.
    """
    assert fitted.persistence < 1.0
    assert fitted.is_stationary


def test_fit_reports_information_criteria(fitted: ModelFit) -> None:
    assert np.isfinite(fitted.loglik)
    assert np.isfinite(fitted.aic)
    assert np.isfinite(fitted.bic)
    assert fitted.result is not None


def test_variances_are_positive(fitted: ModelFit) -> None:
    assert fitted.last_variance > 0.0
    assert fitted.forecast_variance > 0.0
    assert fitted.unconditional_variance > 0.0


def test_forecast_variance_is_one_step_past_the_sample(
    fitted: ModelFit, returns: pd.Series
) -> None:
    """sigma_{T+1}^2 must be the GJR recursion applied to the last shock."""
    p = fitted.params
    shock = float(returns.iloc[-1]) - p["mu"]
    leverage = p["alpha"] + p["gamma"] * (shock < 0.0)
    expected = p["omega"] + leverage * shock**2 + p["beta"] * fitted.last_variance
    assert fitted.forecast_variance == pytest.approx(expected)
    assert fitted.forecast_variance != fitted.last_variance


def test_fit_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="at least 100"):
        fit(pd.Series(np.zeros(50)))


# --- simulate --------------------------------------------------------------


def test_simulate_shape_and_finiteness(fitted: ModelFit) -> None:
    sims = simulate(fitted, horizon=60, n_paths=200, seed=0)
    assert sims.shape == (200, 60)
    assert np.isfinite(sims).all()


def test_simulate_is_deterministic_under_a_seed(fitted: ModelFit) -> None:
    first = simulate(fitted, horizon=30, n_paths=100, seed=123)
    second = simulate(fitted, horizon=30, n_paths=100, seed=123)
    assert np.array_equal(first, second)


def test_different_seeds_give_different_paths(fitted: ModelFit) -> None:
    first = simulate(fitted, horizon=30, n_paths=100, seed=123)
    second = simulate(fitted, horizon=30, n_paths=100, seed=124)
    assert not np.array_equal(first, second)


def test_simulated_scale_matches_reality(
    fitted: ModelFit, returns: pd.Series
) -> None:
    """Loose guard against the percent/raw rescaling bug.

    A missing division by 100 would show up here as a ratio near 100,
    not near 1.
    """
    ratio = simulate(fitted, horizon=252, n_paths=500, seed=5).std() / returns.std()
    assert 1 / 3 < ratio < 3


def test_initial_var_flag_is_wired(fitted: ModelFit) -> None:
    """'last' and 'unconditional' must give a different day-1 variance.

    Holding the seed fixed, the innovations are identical, so any
    difference on day 1 comes from the starting variance alone.
    'last' means the one-step-ahead forecast sigma_{T+1}^2, so the
    day-1 spread must track that rather than sigma_T^2.
    """
    assert fitted.forecast_variance != fitted.unconditional_variance
    day_one_last = simulate(fitted, horizon=1, n_paths=500, seed=9, initial_var="last")
    day_one_uncond = simulate(
        fitted, horizon=1, n_paths=500, seed=9, initial_var="unconditional"
    )
    assert not np.allclose(day_one_last, day_one_uncond)

    # Day-1 dispersion must scale as sqrt(forecast variance), not sqrt(sigma_T^2).
    centred_last = day_one_last - fitted.params["mu"]
    centred_uncond = day_one_uncond - fitted.params["mu"]
    observed = float(np.std(centred_last) / np.std(centred_uncond))
    expected = float(
        np.sqrt(fitted.forecast_variance / fitted.unconditional_variance)
    )
    assert observed == pytest.approx(expected, rel=1e-9)


def test_simulate_rejects_bad_arguments(fitted: ModelFit) -> None:
    with pytest.raises(ValueError, match="initial_var must be"):
        simulate(fitted, horizon=5, n_paths=5, initial_var="stationary")
    with pytest.raises(ValueError, match="horizon must be"):
        simulate(fitted, horizon=0, n_paths=5)
    with pytest.raises(ValueError, match="n_paths must be"):
        simulate(fitted, horizon=5, n_paths=0)


def test_fit_and_simulate_defaults(returns: pd.Series) -> None:
    fitted, sims = fit_and_simulate(returns)
    assert isinstance(fitted, ModelFit)
    assert sims.shape == (5000, 252)
    assert np.isfinite(sims).all()


# --- round-trip recovery ---------------------------------------------------


def test_fit_recovers_known_parameters() -> None:
    """Simulate from known parameters, refit, and check they come back.

    This is the model analogue of the t(4) Hill recovery in the
    diagnostics: it proves fit and simulate agree with each other,
    rather than merely that each one runs.
    """
    persistence = (
        TRUE_PARAMS["alpha"] + TRUE_PARAMS["gamma"] / 2.0 + TRUE_PARAMS["beta"]
    )
    unconditional = TRUE_PARAMS["omega"] / (1.0 - persistence)
    known = ModelFit(
        params=dict(TRUE_PARAMS),
        loglik=0.0,
        aic=0.0,
        bic=0.0,
        last_variance=unconditional,
        forecast_variance=unconditional,
        unconditional_variance=unconditional,
    )

    path = simulate(
        known, horizon=20_000, n_paths=1, seed=11, initial_var="unconditional"
    )[0]
    recovered = fit(pd.Series(path))

    assert recovered.params["alpha"] == pytest.approx(TRUE_PARAMS["alpha"], abs=0.02)
    assert recovered.params["gamma"] == pytest.approx(TRUE_PARAMS["gamma"], abs=0.02)
    assert recovered.params["beta"] == pytest.approx(TRUE_PARAMS["beta"], abs=0.02)
    assert recovered.params["omega"] == pytest.approx(TRUE_PARAMS["omega"], rel=0.35)
    assert recovered.params["nu"] == pytest.approx(TRUE_PARAMS["nu"], rel=0.25)
    assert recovered.params["lambda"] == pytest.approx(
        TRUE_PARAMS["lambda"], abs=0.05
    )
    assert recovered.persistence == pytest.approx(persistence, abs=0.01)
