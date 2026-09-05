"""Tests for the statistical diagnostics.

Most tests assert properties of real Brent returns. The final test is
different in kind: it feeds the Hill estimator a sample whose tail index
is known by construction, and checks the estimator recovers it. That one
validates the code, not the data.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from brent_synth.data import load_returns
from brent_synth.diagnostics import (
    acf_returns,
    acf_squared,
    hill_estimator,
    ljung_box_squared,
    mean_excess,
    moments,
    pot_gpd_fit,
    run_all,
)


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return load_returns()


# --- moments ---------------------------------------------------------------


def test_moments_has_all_keys(returns: pd.Series) -> None:
    result = moments(returns)
    assert set(result) == {
        "mean",
        "std",
        "skew",
        "kurtosis",
        "excess_kurtosis",
        "annualised_vol",
        "n_obs",
    }
    assert all(np.isfinite(v) for v in result.values())


def test_returns_are_leptokurtic(returns: pd.Series) -> None:
    """Real returns have fatter tails than a Gaussian."""
    assert moments(returns)["excess_kurtosis"] > 0.0


def test_annualised_vol_matches_std(returns: pd.Series) -> None:
    result = moments(returns)
    assert result["annualised_vol"] == pytest.approx(result["std"] * np.sqrt(252))
    assert result["kurtosis"] == pytest.approx(result["excess_kurtosis"] + 3.0)
    assert result["n_obs"] == len(returns)


# --- autocorrelation -------------------------------------------------------


def test_acf_lengths_and_lag_zero(returns: pd.Series) -> None:
    for values in (acf_returns(returns, nlags=40), acf_squared(returns, nlags=40)):
        assert values.shape == (41,)
        assert values[0] == pytest.approx(1.0)


def test_returns_are_close_to_unpredictable(returns: pd.Series) -> None:
    assert abs(acf_returns(returns)[1]) < 0.1


def test_squared_returns_are_autocorrelated(returns: pd.Series) -> None:
    """Volatility clusters even though the returns themselves do not."""
    assert acf_squared(returns)[1] > abs(acf_returns(returns)[1])


def test_clustering_is_significant(returns: pd.Series) -> None:
    assert ljung_box_squared(returns, lags=10) < 0.05


# --- tails -----------------------------------------------------------------


@pytest.mark.parametrize("tail", ["left", "right"])
def test_hill_arrays_align(returns: pd.Series, tail: str) -> None:
    result = hill_estimator(returns, tail=tail)
    k, alpha = result["k"], result["alpha"]
    assert k.shape == alpha.shape
    assert k.size > 0
    assert np.all(np.diff(k) == 1)
    assert np.isfinite(alpha).all()
    assert np.all(alpha > 0.0)


@pytest.mark.parametrize("tail", ["left", "right"])
def test_pot_gpd_fit_reports_a_heavy_tail(returns: pd.Series, tail: str) -> None:
    result = pot_gpd_fit(returns, tail=tail)
    assert set(result) == {
        "threshold",
        "xi",
        "beta",
        "n_exceedances",
        "tail_index",
    }
    assert result["threshold"] > 0.0
    assert result["beta"] > 0.0
    assert result["n_exceedances"] > 0
    assert result["xi"] > 0.0
    assert result["tail_index"] > 0.0


def test_pot_threshold_tracks_the_quantile(returns: pd.Series) -> None:
    """A higher threshold quantile leaves fewer exceedances."""
    low = pot_gpd_fit(returns, tail="left", threshold_quantile=0.90)
    high = pot_gpd_fit(returns, tail="left", threshold_quantile=0.99)
    assert high["threshold"] > low["threshold"]
    assert high["n_exceedances"] < low["n_exceedances"]


def test_mean_excess_shape(returns: pd.Series) -> None:
    result = mean_excess(returns, tail="left")
    thresholds, excess = result["thresholds"], result["mean_excess"]
    assert thresholds.shape == excess.shape
    assert np.all(np.diff(thresholds) >= 0.0)
    assert np.all(excess >= 0.0)
    assert np.isfinite(excess).all()


def test_left_tail_is_heavier_than_a_normal(returns: pd.Series) -> None:
    """Sanity: the fitted GPD shape is positive, i.e. power-law decay."""
    assert pot_gpd_fit(returns, tail="left")["xi"] > 0.0


def test_bad_tail_argument_is_rejected(returns: pd.Series) -> None:
    with pytest.raises(ValueError, match="tail must be"):
        hill_estimator(returns, tail="middle")


# --- run_all ---------------------------------------------------------------


def test_run_all_is_populated(returns: pd.Series) -> None:
    result = run_all(returns)
    assert set(result) == {"moments", "autocorrelation", "tails"}
    assert result["moments"]["n_obs"] == len(returns)
    assert result["autocorrelation"]["ljung_box_squared_pvalue"] < 0.05
    for tail in ("left", "right"):
        block = result["tails"][tail]
        assert set(block) == {"hill", "pot_gpd", "mean_excess"}
        assert block["hill"]["alpha"].size > 0
        assert block["pot_gpd"]["tail_index"] > 0.0
        assert block["mean_excess"]["thresholds"].size > 0


# --- estimator recovery ----------------------------------------------------


def test_hill_recovers_student_t_tail_index() -> None:
    """The real check on the Hill code.

    A Student-t with df degrees of freedom has tail index alpha = df, so
    a t(4) sample must come back at alpha ~ 4. Read the plateau of the
    Hill plot (the median over a mid-range band of k) rather than any
    single k: small k is unbiased but noisy, large k drifts into the bulk.
    """
    sample = pd.Series(
        stats.t.rvs(df=4, size=200_000, random_state=np.random.default_rng(0))
    )
    result = hill_estimator(sample, tail="left")
    k, alpha = result["k"], result["alpha"]

    plateau = alpha[(k >= 50) & (k <= 500)]
    assert np.nanmedian(plateau) == pytest.approx(4.0, abs=0.5)


def test_hill_separates_heavy_from_light_tails() -> None:
    """t(4) must read as materially heavier-tailed than t(10)."""
    band = slice(None)
    estimates = {}
    for df in (4, 10):
        sample = pd.Series(
            stats.t.rvs(df=df, size=200_000, random_state=np.random.default_rng(df))
        )
        result = hill_estimator(sample, tail="left")
        k, alpha = result["k"], result["alpha"]
        estimates[df] = np.nanmedian(alpha[(k >= 50) & (k <= 500)][band])

    assert estimates[4] < estimates[10]
    assert estimates[10] == pytest.approx(10.0, rel=0.35)


def test_gpd_recovers_a_known_shape() -> None:
    """POT on an exact GPD sample recovers the shape it was drawn with."""
    true_xi = 0.25
    sample = pd.Series(
        -stats.genpareto.rvs(
            c=true_xi, size=100_000, random_state=np.random.default_rng(1)
        )
    )
    result = pot_gpd_fit(sample, tail="left", threshold_quantile=0.95)
    assert result["xi"] == pytest.approx(true_xi, abs=0.08)
    assert result["tail_index"] == pytest.approx(1.0 / true_xi, rel=0.4)
