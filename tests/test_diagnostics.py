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
    MIN_OBS_MOMENTS,
    XI_SIGNIFICANCE_Z,
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
        "xi_se",
        "beta",
        "n_exceedances",
        "tail_index",
    }
    assert result["threshold"] > 0.0
    assert result["beta"] > 0.0
    assert result["n_exceedances"] > 0
    assert result["xi"] > 0.0
    assert result["xi_se"] > 0.0


def test_only_the_left_tail_supports_a_tail_index(returns: pd.Series) -> None:
    """Both tails fit xi > 0, but only the left one significantly so.

    Left: xi/se ~ 1.87, past the one-sided 95% mark, so 1/xi is reported.
    Right: xi/se ~ 1.25, short of it, so the tail index is withheld
    rather than dressed up as a number the data cannot support.
    """
    left = pot_gpd_fit(returns, tail="left")
    right = pot_gpd_fit(returns, tail="right")

    assert left["xi"] > XI_SIGNIFICANCE_Z * left["xi_se"]
    assert left["tail_index"] == pytest.approx(1.0 / left["xi"])

    assert right["xi"] > 0.0
    assert right["xi"] <= XI_SIGNIFICANCE_Z * right["xi_se"]
    assert np.isnan(right["tail_index"])


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
        assert block["pot_gpd"]["xi"] > 0.0
        assert block["mean_excess"]["thresholds"].size > 0
    # Only the left tail clears the significance bar; see the dedicated test.
    assert result["tails"]["left"]["pot_gpd"]["tail_index"] > 0.0


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


# --- regression tests for reported defects ---------------------------------


def test_mean_excess_uses_values_not_positions() -> None:
    """Tied observations must not be counted as their own exceedances.

    Taking everything positionally after the threshold includes the ties,
    each contributing zero excess. On [1, 2, 2, 2, 5] that gave
    [1.75, 1.0, 1.5, 3.0] instead of the defined [1.75, 3, 3, 3].
    """
    losses = pd.Series([-1.0, -2.0, -2.0, -2.0, -5.0])
    result = mean_excess(losses, tail="left")
    assert np.allclose(result["thresholds"], [1.0, 2.0, 2.0, 2.0])
    assert np.allclose(result["mean_excess"], [1.75, 3.0, 3.0, 3.0])


def test_mean_excess_matches_the_definition_directly() -> None:
    """Brute-force E[X - u | X > u] against the vectorised form."""
    rng = np.random.default_rng(4)
    sample = np.round(rng.exponential(1.0, 400), 2)  # rounding forces ties
    result = mean_excess(pd.Series(-sample), tail="left")
    for threshold, value in zip(result["thresholds"], result["mean_excess"]):
        above = sample[sample > threshold]
        assert value == pytest.approx((above - threshold).mean())


def test_mean_excess_omits_the_maximum(returns: pd.Series) -> None:
    result = mean_excess(returns, tail="left")
    losses = -returns[returns < 0].to_numpy()
    assert result["thresholds"].max() < losses.max()


def test_mean_excess_rejects_a_constant_tail() -> None:
    with pytest.raises(ValueError, match="identical"):
        mean_excess(pd.Series([-2.0] * 30), tail="left")


@pytest.mark.parametrize("noise", [0.0, 1e-15, 1e-13, 1e-11])
def test_hill_returns_nan_on_degenerate_input(noise: float) -> None:
    """Near-constant losses must not yield a colossal alpha.

    The old guard tested the mean log excess against exact zero; on
    near-constant input it lands just above and slipped through,
    reporting alpha ~ 1e15 as though it were a tail index.
    """
    rng = np.random.default_rng(0)
    losses = 1.0 + rng.normal(0.0, noise, 200) if noise else np.ones(200)
    result = hill_estimator(pd.Series(-losses), tail="left")
    assert np.isnan(result["alpha"]).all()


def test_tail_index_is_withheld_when_xi_is_insignificant() -> None:
    """Exponential losses have no power-law tail; say so with NaN."""
    for seed in (4, 6):
        rng = np.random.default_rng(seed)
        sample = pd.Series(-rng.exponential(0.02, 5000))
        result = pot_gpd_fit(sample, tail="left")
        assert result["xi"] > 0.0  # noises above zero
        assert result["xi"] <= XI_SIGNIFICANCE_Z * result["xi_se"]
        assert np.isnan(result["tail_index"])


def test_xi_standard_error_matches_the_asymptotic_form(returns: pd.Series) -> None:
    result = pot_gpd_fit(returns, tail="left")
    expected = (1.0 + result["xi"]) / np.sqrt(result["n_exceedances"])
    assert result["xi_se"] == pytest.approx(expected)


def test_threshold_quantile_is_of_the_tail_not_the_returns(
    returns: pd.Series,
) -> None:
    """Document the sharp edge: 0.95 of losses is ~2.4% of all returns."""
    result = pot_gpd_fit(returns, tail="left", threshold_quantile=0.95)
    losses = -returns[returns < 0].to_numpy()
    assert result["threshold"] == pytest.approx(np.quantile(losses, 0.95))

    overall_tail_probability = float((returns < -result["threshold"]).mean())
    assert overall_tail_probability < 0.05  # not the 5% the name suggests
    assert result["n_exceedances"] == int((losses > result["threshold"]).sum())


def test_order_dependent_stats_refuse_to_splice_gaps() -> None:
    """Dropping a NaN run makes its two sides lag-1 neighbours."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=2000)
    values[900:1000] = np.nan
    gapped = pd.Series(values)

    for call in (acf_returns, acf_squared, ljung_box_squared):
        with pytest.raises(ValueError, match="non-finite"):
            call(gapped)

    # Order-independent statistics are unaffected and still work.
    assert np.isfinite(moments(gapped)["std"])


def test_acf_rejects_more_lags_than_the_sample_supports() -> None:
    """statsmodels truncates silently; the caller must be told instead."""
    short = pd.Series(np.random.default_rng(0).normal(size=10))
    for call in (acf_returns, acf_squared):
        with pytest.raises(ValueError, match="needs more than"):
            call(short, nlags=50)
    assert acf_returns(short, nlags=5).shape == (6,)


def test_ljung_box_rejects_too_many_lags() -> None:
    short = pd.Series(np.random.default_rng(0).normal(size=10))
    with pytest.raises(ValueError, match="needs more than"):
        ljung_box_squared(short, lags=50)


def test_moments_rejects_too_few_observations() -> None:
    """A 1-point series raised nothing and returned NaNs; be consistent."""
    with pytest.raises(ValueError, match="at least 4 observations"):
        moments(pd.Series([0.01]))
    with pytest.raises(ValueError, match="empty"):
        moments(pd.Series([], dtype="float64"))
    assert np.isfinite(moments(pd.Series([0.01, -0.02, 0.03, -0.01]))["skew"])
    assert MIN_OBS_MOMENTS == 4


def test_moments_returns_plain_floats(returns: pd.Series) -> None:
    result = moments(returns)
    for name, value in result.items():
        expected = int if name == "n_obs" else float
        assert type(value) is expected, name
