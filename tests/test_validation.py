"""Tests for the bootstrap validator and the HTML report.

These run offline. Rather than the real Brent series they use a long
GJR-GARCH path simulated from fixed parameters, which gives a stand-in
"real market" with genuine clustering and fat tails, at a size that
keeps the bootstrap fast and the results deterministic.
"""

import numpy as np
import pandas as pd
import pytest

from brent_synth.model import ModelFit, simulate
from brent_synth.validation import (
    STAT_NAMES,
    bootstrap_bands,
    compute_stats,
    make_report,
    run_validation,
    validate,
)

HORIZON = 252
N_BOOT = 400
BOOT_KWARGS = {"n_boot": N_BOOT, "block": 20, "seed": 7}

TRUE_PARAMS = {
    "mu": 0.0003,
    "omega": 5.0e-6,
    "alpha": 0.07,
    "gamma": 0.06,
    "beta": 0.88,
    "nu": 5.5,
    "lambda": -0.12,
}


def _known_fit() -> ModelFit:
    persistence = (
        TRUE_PARAMS["alpha"] + TRUE_PARAMS["gamma"] / 2.0 + TRUE_PARAMS["beta"]
    )
    unconditional = TRUE_PARAMS["omega"] / (1.0 - persistence)
    return ModelFit(
        params=dict(TRUE_PARAMS),
        loglik=-1234.5,
        aic=2483.0,
        bic=2520.0,
        last_variance=unconditional,
        forecast_variance=unconditional,
        unconditional_variance=unconditional,
    )


@pytest.fixture(scope="module")
def market() -> pd.Series:
    """A stand-in 'real' series: 5000 days of clustered, fat-tailed returns."""
    path = simulate(
        _known_fit(), horizon=5000, n_paths=1, seed=3, initial_var="unconditional"
    )[0]
    return pd.Series(path)


@pytest.fixture(scope="module")
def synth(market: pd.Series) -> np.ndarray:
    return simulate(_known_fit(), horizon=HORIZON, n_paths=400, seed=42)


@pytest.fixture(scope="module")
def bands(market: pd.Series) -> dict:
    return bootstrap_bands(market, horizon=HORIZON, **BOOT_KWARGS)


# --- compute_stats ---------------------------------------------------------


def test_compute_stats_on_paths(synth: np.ndarray) -> None:
    result = compute_stats(synth)
    assert set(result) == set(STAT_NAMES)
    assert all(np.isfinite(v) for v in result.values())


def test_compute_stats_on_a_flat_sample(market: pd.Series) -> None:
    result = compute_stats(market.to_numpy()[:HORIZON])
    assert set(result) == set(STAT_NAMES)
    assert all(np.isfinite(v) for v in result.values())


def test_var_and_es_are_positive_losses(synth: np.ndarray) -> None:
    """VaR/ES are loss magnitudes; the percentile rows stay signed."""
    result = compute_stats(synth)
    assert result["var_99"] > result["var_95"] > 0.0
    assert result["es_99"] > result["es_95"] > 0.0
    assert result["es_95"] > result["var_95"]  # ES is the mean beyond VaR
    assert result["left_q01"] < result["left_q05"] < 0.0
    assert 0.0 < result["right_q95"] < result["right_q99"]
    assert 0.0 < result["max_drawdown"] < 1.0


def test_max_drawdown_is_a_path_functional() -> None:
    """A monotonically rising path has no drawdown; a falling one does."""
    rising = np.full((1, 20), 0.01)
    assert compute_stats(rising)["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    falling = np.full((1, 20), -0.01)
    expected = 1.0 - np.exp(-0.20)
    assert compute_stats(falling)["max_drawdown"] == pytest.approx(expected)


def test_too_short_a_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 11 periods"):
        compute_stats(np.zeros((5, 8)))


# --- bootstrap -------------------------------------------------------------


def test_bands_are_ordered(bands: dict) -> None:
    for name in STAT_NAMES:
        lo, median, hi, draws = bands[name]
        assert lo < median < hi, name
        assert draws.shape == (N_BOOT,)
        assert np.isfinite(draws).all(), name


def test_bootstrap_is_deterministic(market: pd.Series) -> None:
    first = bootstrap_bands(market, horizon=HORIZON, **BOOT_KWARGS)
    second = bootstrap_bands(market, horizon=HORIZON, **BOOT_KWARGS)
    for name in STAT_NAMES:
        assert np.array_equal(first[name][3], second[name][3]), name


def test_bootstrap_preserves_clustering(market: pd.Series, bands: dict) -> None:
    """Block resampling must keep volatility clustering alive.

    An iid bootstrap would centre the ACF band on zero; the block
    bootstrap must put its median clearly above that.
    """
    assert bands["acf_sq_lag1"][1] > 0.05


def test_bootstrap_rejects_too_short_a_series() -> None:
    with pytest.raises(ValueError, match="at least"):
        bootstrap_bands(pd.Series(np.zeros(100)), horizon=HORIZON)


# --- validate --------------------------------------------------------------


def test_validate_shape_and_columns(synth: np.ndarray, market: pd.Series) -> None:
    results = validate(synth, market, **BOOT_KWARGS)
    assert len(results) == len(STAT_NAMES)
    assert list(results["statistic"]) == list(STAT_NAMES)
    assert results["passed"].dtype == bool
    assert results["passed"].notna().all()


def test_position_is_zero_exactly_when_passing(
    synth: np.ndarray, market: pd.Series
) -> None:
    results = validate(synth, market, **BOOT_KWARGS)
    inside = results["passed"]
    assert (results.loc[inside, "position"] == 0.0).all()
    assert (results.loc[~inside, "position"] != 0.0).all()


def test_horizon_mismatch_is_rejected(market: pd.Series) -> None:
    paths = simulate(_known_fit(), horizon=60, n_paths=20, seed=1)
    with pytest.raises(ValueError, match="must use the same horizon"):
        validate(paths, market, horizon=HORIZON, **BOOT_KWARGS)


def test_real_blocks_mostly_pass(market: pd.Series) -> None:
    """The validator must not reject the real DGP against itself.

    Contiguous 252-day blocks of the market are fed back in as if they
    were synthetic. A validator that flags these is too strict to be
    useful.
    """
    values = market.to_numpy()
    n_blocks = values.size // HORIZON
    blocks = values[: n_blocks * HORIZON].reshape(n_blocks, HORIZON)
    results = validate(blocks, market, **BOOT_KWARGS)
    assert results["passed"].mean() >= 0.75


# --- known-fail guard ------------------------------------------------------


def test_gaussian_noise_is_rejected(market: pd.Series) -> None:
    """iid Gaussian noise, mean/variance matched, must be caught.

    This is the guard that proves the validator has teeth: a model with
    neither fat tails nor clustering has to be rejected somewhere.

    It is rejected on excess kurtosis. It is NOT rejected on the ACF
    statistics, and that is a real limitation of the design rather than
    a fluke of these seeds: the 2.5% bound of ACF(r²) over 252-day
    windows is negative — a quiet year of a genuinely clustered market
    shows no measurable clustering — so a two-sided 95% band at this
    horizon cannot exclude zero. That holds at every block length
    tried (5 to 100 days). What the ACF statistics do show is direction,
    asserted below: noise sits far under the bootstrap median.
    """
    rng = np.random.default_rng(0)
    noise = rng.normal(market.mean(), market.std(), size=(400, HORIZON))
    results = validate(noise, market, **BOOT_KWARGS).set_index("statistic")

    assert not results.loc["excess_kurtosis", "passed"]
    assert results.loc["excess_kurtosis", "position"] < 0.0

    for lag in ("acf_sq_lag1", "acf_sq_lag5", "acf_sq_lag10"):
        assert results.loc[lag, "synthetic"] < results.loc[lag, "boot_median"], lag


def test_validator_separates_noise_from_the_real_process(
    market: pd.Series, synth: np.ndarray
) -> None:
    """Synthetic paths from the true model must beat Gaussian noise."""
    rng = np.random.default_rng(1)
    noise = rng.normal(market.mean(), market.std(), size=(400, HORIZON))
    good = validate(synth, market, **BOOT_KWARGS)["passed"].sum()
    bad = validate(noise, market, **BOOT_KWARGS)["passed"].sum()
    assert good >= bad


# --- report ----------------------------------------------------------------


def test_report_is_written_and_self_contained(
    market: pd.Series, synth: np.ndarray, bands: dict, tmp_path
) -> None:
    results = validate(synth, market, **BOOT_KWARGS)
    out = tmp_path / "validation.html"
    path = make_report(
        market,
        _known_fit(),
        synth,
        results,
        out_path=str(out),
        seed=42,
        n_boot=N_BOOT,
        boot_drawdowns=bands["max_drawdown"][3],
    )
    text = out.read_text(encoding="utf-8")

    assert path == str(out)
    assert text.count("data:image/png;base64,") == 5  # all five plots embedded
    assert "http://" not in text and "https://" not in text  # nothing external
    assert "Failure modes" in text and "TODO" in text
    assert "PASS" in text or "FAIL" in text
    for label in ("Excess kurtosis", "Max drawdown", "ACF of r², lag 10"):
        assert label in text


def test_report_is_byte_identical_across_runs(
    market: pd.Series, synth: np.ndarray, bands: dict, tmp_path
) -> None:
    """Fixed seeds must give a reproducible report, plots included."""
    results_first = validate(synth, market, **BOOT_KWARGS)
    results_second = validate(synth, market, **BOOT_KWARGS)
    pd.testing.assert_frame_equal(results_first, results_second)

    payloads = []
    for name in ("first.html", "second.html"):
        out = tmp_path / name
        make_report(
            market,
            _known_fit(),
            synth,
            results_first,
            out_path=str(out),
            seed=42,
            n_boot=N_BOOT,
            boot_drawdowns=bands["max_drawdown"][3],
        )
        payloads.append(out.read_bytes())
    assert payloads[0] == payloads[1]


def test_run_validation_end_to_end(tmp_path) -> None:
    """Smoke test of the whole pipeline, at reduced size."""
    out = tmp_path / "pipeline.html"
    path = run_validation(
        seed=42, horizon=60, n_paths=150, n_boot=150, out_path=str(out)
    )
    assert path == str(out)
    assert out.stat().st_size > 20_000
    assert "data:image/png;base64," in out.read_text(encoding="utf-8")
