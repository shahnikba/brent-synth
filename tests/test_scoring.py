"""Tests for the day-level scores.

The important ones check a score against a closed form or an independent
calculation, not against itself.
"""

import numpy as np
import pytest
from scipy import stats as sps

from brent_synth.candidates.base import LocationScaleForecast, NormalStd
from brent_synth.scoring import (
    TAU_GRID,
    christoffersen_pvalue,
    crps,
    diebold_mariano,
    kupiec_pvalue,
    nll,
    pit,
    sign_split_nll,
    tail_crps,
    var_exceedances,
)


@pytest.fixture(scope="module")
def gaussian():
    rng = np.random.default_rng(0)
    n = 300
    mu = rng.normal(0.0, 0.01, n)
    sigma = np.abs(rng.normal(0.02, 0.005, n)) + 0.005
    forecast = LocationScaleForecast(mu, sigma, NormalStd())
    y = mu + sigma * rng.normal(size=n)
    return forecast, y, mu, sigma


def test_crps_matches_the_gaussian_closed_form(gaussian) -> None:
    """CRPS(N(mu,s), y) = s[z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi)]."""
    forecast, y, mu, sigma = gaussian
    z = (y - mu) / sigma
    closed_form = sigma * (
        z * (2 * sps.norm.cdf(z) - 1) + 2 * sps.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )
    assert np.abs(crps(forecast, y) / closed_form - 1).max() < 2e-3


def test_tail_crps_never_exceeds_crps(gaussian) -> None:
    forecast, y, _, _ = gaussian
    assert (tail_crps(forecast, y) <= crps(forecast, y) + 1e-12).all()
    assert (tail_crps(forecast, y) >= 0.0).all()


def test_tail_crps_narrows_with_tau_max(gaussian) -> None:
    forecast, y, _, _ = gaussian
    wide = tail_crps(forecast, y, tau_max=0.20).mean()
    narrow = tail_crps(forecast, y, tau_max=0.02).mean()
    assert narrow < wide


def test_tail_crps_rejects_an_empty_tail(gaussian) -> None:
    forecast, y, _, _ = gaussian
    with pytest.raises(ValueError, match="No grid points"):
        tail_crps(forecast, y, tau_max=1e-9)


def test_nll_and_pit_agree_with_the_forecast(gaussian) -> None:
    forecast, y, mu, sigma = gaussian
    assert np.allclose(nll(forecast, y), -sps.norm.logpdf(y, mu, sigma))
    assert np.allclose(pit(forecast, y), sps.norm.cdf(y, mu, sigma))


def test_kupiec_matches_an_independent_calculation() -> None:
    x, n, p = 5, 252, 0.01
    rate = x / n
    statistic = -2.0 * (
        (n - x) * np.log(1 - p) + x * np.log(p)
    ) + 2.0 * ((n - x) * np.log(1 - rate) + x * np.log(rate))
    assert kupiec_pvalue(x, n, p) == pytest.approx(
        float(sps.chi2.sf(statistic, 1))
    )


def test_kupiec_handles_the_boundaries() -> None:
    """x = 0 and x = n are the cases the naive formula cannot express."""
    assert 0.0 <= kupiec_pvalue(0, 252, 0.01) <= 1.0
    assert 0.0 <= kupiec_pvalue(252, 252, 0.01) <= 1.0
    with pytest.raises(ValueError):
        kupiec_pvalue(253, 252, 0.01)


def test_christoffersen_needs_two_exceedances() -> None:
    assert np.isnan(christoffersen_pvalue(np.array([True] + [False] * 20)))
    assert np.isnan(christoffersen_pvalue(np.zeros(20, dtype=bool)))
    clustered = np.zeros(200, dtype=bool)
    clustered[50:56] = True
    assert christoffersen_pvalue(clustered) < 0.05  # bunched breaches caught


def test_pit_of_the_true_model_is_uniform() -> None:
    """The end-to-end calibration check: right model, flat PIT."""
    rng = np.random.default_rng(3)
    n = 2000
    omega, alpha, beta = 4.0e-6, 0.07, 0.88
    variance = omega / (1 - alpha - beta)
    y = np.empty(n)
    sigma = np.empty(n)
    for t in range(n):
        sigma[t] = np.sqrt(variance)
        y[t] = sigma[t] * rng.normal()
        variance = omega + alpha * y[t] ** 2 + beta * variance

    forecast = LocationScaleForecast(np.zeros(n), sigma, NormalStd())
    values = pit(forecast, y)
    assert sps.kstest(values, "uniform").pvalue > 0.01


def test_var_exceedances_track_the_quantile(gaussian) -> None:
    forecast, y, _, _ = gaussian
    for level, expected in ((0.95, 0.05), (0.99, 0.01)):
        exceed = var_exceedances(forecast, y, level)
        threshold = forecast.ppf(np.array([1 - level]))[:, 0]
        assert np.array_equal(exceed, y < threshold)
        assert abs(exceed.mean() - expected) < 0.05
    with pytest.raises(ValueError, match="level must be"):
        var_exceedances(forecast, y, 1.5)


def test_sign_split_reports_both_halves(gaussian) -> None:
    forecast, y, _, _ = gaussian
    previous = np.concatenate([[0.0], y[:-1]])
    after_down, after_up = sign_split_nll(forecast, y, previous)
    assert np.isfinite(after_down) and np.isfinite(after_up)
    with pytest.raises(ValueError, match="y_prev must have"):
        sign_split_nll(forecast, y, previous[:-5])


def test_diebold_mariano_on_identical_losses() -> None:
    a = np.random.default_rng(0).normal(size=300)
    statistic, p_value = diebold_mariano(a, a)
    assert np.isnan(statistic) and np.isnan(p_value)


def test_diebold_mariano_detects_a_real_difference() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=800)
    worse = base + 0.4
    statistic, p_value = diebold_mariano(worse, base)
    assert statistic > 0  # positive means the first argument lost
    assert p_value < 0.01
    assert diebold_mariano(base, worse)[0] < 0


def test_tau_grid_is_the_pre_registered_one() -> None:
    assert TAU_GRID.size == 999
    assert TAU_GRID[0] == pytest.approx(0.0005)
    assert TAU_GRID[-1] == pytest.approx(0.9995)
