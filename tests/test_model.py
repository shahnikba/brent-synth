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
from brent_synth.model import (
    ModelFit,
    _leverage_weight,
    fit,
    fit_and_simulate,
    simulate,
)

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
    weight = _leverage_weight(TRUE_PARAMS["nu"], TRUE_PARAMS["lambda"])
    persistence = (
        TRUE_PARAMS["alpha"] + TRUE_PARAMS["gamma"] * weight + TRUE_PARAMS["beta"]
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


# --- leverage weight on gamma ----------------------------------------------


def test_leverage_weight_is_a_half_only_when_symmetric() -> None:
    """The gamma coefficient is E[z^2 1{z<0}], not P(z<0).

    They agree at 1/2 for symmetric innovations, which is why the
    textbook rule reads gamma/2. Under negative skew the partial second
    moment rises above 1/2 while the probability of a negative shock
    falls below it, so substituting the probability moves the weight in
    the wrong direction.
    """
    assert _leverage_weight(6.0, 0.0) == pytest.approx(0.5, abs=1e-6)

    for lam in (-0.10, -0.30, -0.50, -0.70):
        weight = _leverage_weight(6.0, lam)
        assert weight > 0.5, lam

    # Monotone in the skew, and nowhere near the probability it replaced.
    weights = [_leverage_weight(6.0, lam) for lam in (0.0, -0.1, -0.3, -0.5, -0.7)]
    assert weights == sorted(weights)
    assert _leverage_weight(5.75, -0.112) == pytest.approx(0.5424, abs=5e-4)


def test_leverage_weight_matches_monte_carlo() -> None:
    """Cross-check the closed form against draws from the same law."""
    from arch.univariate import SkewStudent

    rng = np.random.default_rng(0)
    for nu, lam in ((6.0, -0.30), (5.75, -0.112)):
        z = np.asarray(
            SkewStudent().ppf(rng.random(2_000_000), parameters=np.array([nu, lam]))
        )
        empirical = float((z**2 * (z < 0)).mean())
        assert _leverage_weight(nu, lam) == pytest.approx(empirical, abs=5e-3)


@pytest.mark.parametrize("lam", [0.0, -0.30, -0.70])
def test_unconditional_variance_matches_the_simulated_process(lam: float) -> None:
    """The decisive test: does the formula predict the process it describes?

    Simulated at low persistence so E[sigma^2] converges quickly, this
    measures E[(r - mu)^2] — which equals E[sigma^2], since E[z^2] = 1 —
    and checks the reported unconditional variance against it. The old
    gamma/2 expression is wrong by up to 17% here; the parametrisation is
    chosen so that error is far outside the tolerance.
    """
    params = {
        "mu": 0.0,
        "omega": 4.0e-6,
        "alpha": 0.05,
        "gamma": 0.10,
        "beta": 0.79,
        "nu": 6.0,
        "lambda": lam,
    }
    # Build from the module's OWN persistence, so a wrong weight in
    # _persistence propagates into what is being asserted. Passing a
    # locally computed value in would test the test, not the module.
    probe = ModelFit(
        params=params,
        loglik=0.0,
        aic=0.0,
        bic=0.0,
        last_variance=1.0,
        forecast_variance=1.0,
        unconditional_variance=1.0,
    )
    claimed = params["omega"] / (1.0 - probe.persistence)
    fitted = ModelFit(
        params=params,
        loglik=0.0,
        aic=0.0,
        bic=0.0,
        last_variance=claimed,
        forecast_variance=claimed,
        unconditional_variance=claimed,
    )

    paths = simulate(
        fitted, horizon=400, n_paths=4000, seed=11, initial_var="unconditional"
    )
    measured = float(((paths - params["mu"]) ** 2)[:, 100:].mean())  # burn in
    assert fitted.unconditional_variance == pytest.approx(measured, rel=0.02)

    if lam != 0.0:
        half = params["omega"] / (
            1.0 - (params["alpha"] + params["gamma"] / 2.0 + params["beta"])
        )
        assert abs(half / measured - 1.0) > 0.03  # the old rule is clearly off


def test_persistence_uses_the_partial_moment(fitted: ModelFit) -> None:
    p = fitted.params
    expected = p["alpha"] + p["gamma"] * fitted.leverage_weight + p["beta"]
    assert fitted.persistence == pytest.approx(expected)
    assert fitted.leverage_weight > 0.5  # Brent's innovations are left-skewed

    half_rule = p["alpha"] + p["gamma"] / 2.0 + p["beta"]
    assert fitted.persistence > half_rule  # the old rule understated it


def test_fit_refuses_to_splice_gaps() -> None:
    """A GARCH likelihood is adjacency-dependent; a hole must not close."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.02, 2000)
    values[900:1000] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        fit(pd.Series(values))
