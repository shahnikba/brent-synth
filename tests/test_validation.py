"""Tests for the bootstrap validator and the HTML report.

These run offline. Rather than the real Brent series they use a long
GJR-GARCH path simulated from fixed parameters, which gives a stand-in
"real market" with genuine clustering and fat tails, at a size that
keeps the bootstrap fast and the results deterministic.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from statsmodels.tsa.stattools import acf

from brent_synth.model import ModelFit, _leverage_weight, simulate
from brent_synth.validation import (
    DISPERSION_BOUNDS,
    FIGURE_SLUGS,
    DUPLICATE_STATS,
    INDEPENDENT_STATS,
    STAT_NAMES,
    bootstrap_bands,
    compute_path_stats,
    compute_stats,
    make_markdown_report,
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
        TRUE_PARAMS["alpha"]
        + TRUE_PARAMS["gamma"]
        * _leverage_weight(TRUE_PARAMS["nu"], TRUE_PARAMS["lambda"])
        + TRUE_PARAMS["beta"]
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


# --- regression: the two sides must compute the same statistic ------------


def test_compute_stats_is_per_path_not_pooled() -> None:
    """Pooling across paths at different vol levels is not the statistic.

    Two groups of Gaussian paths, one ten times as volatile as the
    other. Every individual path is Gaussian, so the per-path excess
    kurtosis is ~0; the pooled mixture is strongly leptokurtic. The old
    code returned the mixture value and compared it against a band built
    from single 252-day samples.
    """
    rng = np.random.default_rng(0)
    quiet = rng.normal(0.0, 0.005, (250, HORIZON))
    loud = rng.normal(0.0, 0.050, (250, HORIZON))
    paths = np.vstack([quiet, loud])

    pooled = float(sps.kurtosis(paths.ravel(), fisher=True, bias=False))
    reported = compute_stats(paths)["excess_kurtosis"]

    assert pooled > 2.0  # the mixture looks fat-tailed
    assert abs(reported) < 0.5  # each path does not
    assert reported == pytest.approx(
        float(np.median(compute_path_stats(paths)["excess_kurtosis"]))
    )


def test_acf_is_not_computed_across_path_boundaries() -> None:
    """Splicing paths manufactures adjacencies that never happened.

    Alternating quiet and loud iid paths have no within-path structure,
    so every per-path ACF is ~0. Concatenating them makes the level
    changes look like autocorrelation in squared returns.
    """
    rng = np.random.default_rng(1)
    paths = np.vstack(
        [
            rng.normal(0.0, 0.005 if i % 2 else 0.05, HORIZON)
            for i in range(200)
        ]
    )
    spliced = acf(paths.ravel() ** 2, nlags=1, fft=True)[1]
    per_path = compute_stats(paths)["acf_sq_lag1"]

    assert spliced > per_path
    assert abs(per_path) < 0.1


def test_one_path_gives_the_same_answer_either_shape(market: pd.Series) -> None:
    """A 1-D sample and a 1-row array are the same thing."""
    sample = market.to_numpy()[:HORIZON]
    flat = compute_stats(sample)
    shaped = compute_stats(sample[None, :])
    for name in STAT_NAMES:
        assert flat[name] == pytest.approx(shaped[name]), name


def test_bootstrap_and_synthetic_sides_share_one_code_path(
    market: pd.Series,
) -> None:
    """Feeding bootstrap-shaped input through both entry points agrees."""
    values = market.to_numpy()
    blocks = values[: 4 * HORIZON].reshape(4, HORIZON)
    per_path = compute_path_stats(blocks)
    for name in STAT_NAMES:
        assert per_path[name].shape == (4,)
        singles = [compute_stats(blocks[i])[name] for i in range(4)]
        assert np.allclose(per_path[name], singles), name


# --- duplicates ------------------------------------------------------------


def test_var_rows_are_aliases_of_the_tail_quantiles(synth: np.ndarray) -> None:
    """var_95 and var_99 are sign flips, not independent checks."""
    stats_ = compute_stats(synth)
    assert stats_["var_95"] == -stats_["left_q05"]
    assert stats_["var_99"] == -stats_["left_q01"]
    assert set(DUPLICATE_STATS) == {"var_95", "var_99"}
    assert len(INDEPENDENT_STATS) == len(STAT_NAMES) - 2


def test_duplicates_are_excluded_from_the_headline(
    synth: np.ndarray, market: pd.Series, bands: dict
) -> None:
    results = validate(synth, market, bands=bands)
    assert (~results["independent"]).sum() == 2
    for duplicate, original in DUPLICATE_STATS.items():
        row = results.set_index("statistic")
        assert row.loc[duplicate, "passed"] == row.loc[original, "passed"]


# --- dispersion ------------------------------------------------------------


def test_dispersion_catches_over_variable_paths(
    market: pd.Series, bands: dict
) -> None:
    """A model can sit on the median and still be far too variable."""
    rng = np.random.default_rng(3)
    target = float(market.std())

    # Vol drawn per path from a wide spread: right median, wrong spread.
    wide_sigma = target * np.exp(rng.normal(0.0, 0.7, (400, 1)))
    wide = rng.normal(0.0, 1.0, (400, HORIZON)) * wide_sigma
    row = validate(wide, market, bands=bands).set_index("statistic").loc["std"]
    assert row["dispersion_ratio"] > DISPERSION_BOUNDS[1]
    assert not row["dispersion_passed"]

    # Matched spread passes.
    tight_sigma = target * np.exp(rng.normal(0.0, 0.18, (400, 1)))
    tight = rng.normal(0.0, 1.0, (400, HORIZON)) * tight_sigma
    tight_row = (
        validate(tight, market, bands=bands).set_index("statistic").loc["std"]
    )
    assert tight_row["dispersion_passed"]


def test_synthetic_spread_columns_bracket_the_median(
    synth: np.ndarray, market: pd.Series, bands: dict
) -> None:
    results = validate(synth, market, bands=bands)
    assert (results["synth_lo"] <= results["synthetic"]).all()
    assert (results["synthetic"] <= results["synth_hi"]).all()


# --- reuse, gaps, seeds ----------------------------------------------------


def test_precomputed_bands_give_identical_results(
    synth: np.ndarray, market: pd.Series
) -> None:
    """run_validation reuses its bootstrap; that must change nothing."""
    computed = validate(synth, market, **BOOT_KWARGS)
    bands = bootstrap_bands(market, horizon=HORIZON, **BOOT_KWARGS)
    reused = validate(synth, market, bands=bands)
    pd.testing.assert_frame_equal(computed, reused)


def test_bootstrap_options_with_bands_are_rejected(
    synth: np.ndarray, market: pd.Series, bands: dict
) -> None:
    with pytest.raises(ValueError, match="ignored when `bands`"):
        validate(synth, market, bands=bands, n_boot=99)


def test_bootstrap_refuses_to_splice_gaps(market: pd.Series) -> None:
    """A block bootstrap exists to preserve adjacency."""
    gapped = market.copy()
    gapped.iloc[[10, 50, 100, 200, 300]] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        bootstrap_bands(gapped, horizon=HORIZON, **BOOT_KWARGS)


def test_report_honours_the_bootstrap_seed(
    market: pd.Series, synth: np.ndarray, bands: dict, tmp_path
) -> None:
    """The drawdown plot must use the caller's seed, not a hardcoded 7."""
    results = validate(synth, market, bands=bands)
    payloads = []
    for boot_seed in (7, 99):
        out = tmp_path / f"seed_{boot_seed}.html"
        make_report(
            market,
            _known_fit(),
            synth,
            results,
            out_path=str(out),
            seed=42,
            n_boot=N_BOOT,
            boot_seed=boot_seed,
        )
        payloads.append(out.read_bytes())
    assert payloads[0] != payloads[1]


def test_report_labels_persistence_with_the_current_formula(
    market: pd.Series, synth: np.ndarray, bands: dict, tmp_path
) -> None:
    results = validate(synth, market, bands=bands)
    out = tmp_path / "labels.html"
    make_report(
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
    assert "γ/2" not in text
    assert "leverage weight" in text
    assert "Spread" in text  # dispersion column is surfaced


# --- markdown report -------------------------------------------------------


def _markdown_kwargs(bands: dict) -> dict:
    return {
        "seed": 42,
        "n_boot": N_BOOT,
        "boot_drawdowns": bands["max_drawdown"][3],
    }


def test_markdown_report_is_written_with_its_figures(
    market: pd.Series, synth: np.ndarray, bands: dict, tmp_path
) -> None:
    results = validate(synth, market, bands=bands)
    out = tmp_path / "validation.md"
    path = make_markdown_report(
        market, _known_fit(), synth, results, out_path=str(out),
        **_markdown_kwargs(bands),
    )
    assert path == str(out)

    figure_dir = tmp_path / "validation_figures"
    written = sorted(p.name for p in figure_dir.iterdir())
    assert written == sorted(f"{slug}.png" for slug in FIGURE_SLUGS.values())
    assert all((figure_dir / name).stat().st_size > 1000 for name in written)

    text = out.read_text(encoding="utf-8")
    for name in written:
        assert f"(validation_figures/{name})" in text  # relative, resolvable
    assert "http://" not in text and "https://" not in text


def test_markdown_report_carries_the_same_content_as_the_html(
    market: pd.Series, synth: np.ndarray, bands: dict, tmp_path
) -> None:
    """The two formats must not tell different stories."""
    results = validate(synth, market, bands=bands)
    md = tmp_path / "v.md"
    html_out = tmp_path / "v.html"
    make_markdown_report(
        market, _known_fit(), synth, results, out_path=str(md),
        **_markdown_kwargs(bands),
    )
    make_report(
        market, _known_fit(), synth, results, out_path=str(html_out),
        **_markdown_kwargs(bands),
    )
    text = md.read_text(encoding="utf-8")

    n_pass = int(results.loc[results["independent"], "passed"].sum())
    n_total = int(results["independent"].sum())
    assert f"**{n_pass} of {n_total}**" in text
    assert f"<b>{n_pass} of {n_total}</b>" in html_out.read_text(encoding="utf-8")

    for label in ("Excess kurtosis", "Max drawdown (median)", "ACF of r², lag 10"):
        assert label in text
    assert "Failure modes" in text and "TODO" in text
    assert "γ/2" not in text
    assert "leverage weight" in text
    assert text.count("| PASS |") + text.count("| **FAIL** |") == len(STAT_NAMES)


def test_markdown_report_is_byte_identical_across_runs(
    market: pd.Series, synth: np.ndarray, bands: dict, tmp_path
) -> None:
    results = validate(synth, market, bands=bands)
    out = tmp_path / "validation.md"
    figure = tmp_path / "validation_figures" / "qq-plot.png"

    # Same path twice: the figure directory is named after the document,
    # so writing to two different names would differ by design.
    make_markdown_report(
        market, _known_fit(), synth, results, out_path=str(out),
        **_markdown_kwargs(bands),
    )
    first, first_png = out.read_bytes(), figure.read_bytes()

    make_markdown_report(
        market, _known_fit(), synth, results, out_path=str(out),
        **_markdown_kwargs(bands),
    )
    assert out.read_bytes() == first
    assert figure.read_bytes() == first_png  # plots reproduce too


def test_run_validation_writes_both_formats(tmp_path) -> None:
    html_out = tmp_path / "pipeline.html"
    md_out = tmp_path / "pipeline.md"
    run_validation(
        seed=42,
        horizon=60,
        n_paths=150,
        n_boot=150,
        out_path=str(html_out),
        markdown_path=str(md_out),
    )
    assert html_out.exists() and md_out.exists()
    assert (tmp_path / "pipeline_figures").is_dir()
    assert "# Brent synthetic-path validation" in md_out.read_text(encoding="utf-8")


def test_markdown_can_be_skipped(tmp_path) -> None:
    html_out = tmp_path / "only.html"
    run_validation(
        seed=42,
        horizon=60,
        n_paths=150,
        n_boot=150,
        out_path=str(html_out),
        markdown_path=None,
    )
    assert html_out.exists()
    assert not list(tmp_path.glob("*.md"))
