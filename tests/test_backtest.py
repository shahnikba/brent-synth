"""Tests for the backtest orchestration.

The leakage test is the one that matters: it is the only thing standing
between an out-of-time comparison and an in-sample one dressed up as
one.
"""

import numpy as np
import pandas as pd
import pytest

from brent_synth.backtest import (
    BACKTEST_SEED,
    ORIGIN_YEARS,
    PREREGISTRATION_PATH,
    RANK_STATS,
    RANKING_LOSSES,
    TEST_HORIZON,
    _score_one,
    check_preregistration,
    confirm,
    origin_index,
    path_coverage_loss,
    percentile_ranks,
    preregistration_hash,
    rank_table,
    regime_labels,
    run_backtest,
    score_models,
    select_champion,
    split_at_origin,
    write_preregistration,
)
from brent_synth.candidates import CANDIDATES_BY_NAME
from brent_synth.data import load_returns


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return load_returns()


# --- ranks -----------------------------------------------------------------


def test_percentile_ranks_locate_the_realised_year() -> None:
    sims = {name: np.arange(101, dtype="float64") for name in RANK_STATS}
    at_median = percentile_ranks(sims, {name: 50.0 for name in RANK_STATS})
    assert all(v == pytest.approx(0.5) for v in at_median.values())
    assert path_coverage_loss(at_median) == pytest.approx(0.0)

    above = percentile_ranks(sims, {name: 1e6 for name in RANK_STATS})
    assert all(v == pytest.approx(1.0) for v in above.values())
    assert path_coverage_loss(above) == pytest.approx(0.5)

    below = percentile_ranks(sims, {name: -1e6 for name in RANK_STATS})
    assert all(v == pytest.approx(0.0) for v in below.values())


def test_percentile_ranks_use_mid_rank_for_ties() -> None:
    sims = {name: np.zeros(100) for name in RANK_STATS}
    tied = percentile_ranks(sims, {name: 0.0 for name in RANK_STATS})
    assert all(v == pytest.approx(0.5) for v in tied.values())


# --- origins ---------------------------------------------------------------


def test_origins_are_the_last_trading_day_of_the_year(returns: pd.Series) -> None:
    dates = []
    for year in ORIGIN_YEARS:
        index = origin_index(returns, year)
        date = returns.index[index]
        assert date <= pd.Timestamp(year=year, month=12, day=31)
        assert returns.index[index + 1] > pd.Timestamp(year=year, month=12, day=31)
        dates.append(date)
    assert dates == sorted(dates)
    assert len(set(dates)) == len(dates)


def test_split_is_exactly_the_horizon(returns: pd.Series) -> None:
    for year in ORIGIN_YEARS:
        train, test = split_at_origin(returns, year)
        assert len(test) == TEST_HORIZON
        assert train.index[-1] < test.index[0]
        assert len(train) + len(test) <= len(returns)


def test_a_short_origin_raises_rather_than_shortening(returns: pd.Series) -> None:
    """The last origin must have a full year after it, or say so."""
    truncated = returns.iloc[: origin_index(returns, ORIGIN_YEARS[0]) + 100]
    with pytest.raises(ValueError, match="does not extend far enough"):
        split_at_origin(truncated, ORIGIN_YEARS[0])


def test_regime_labels_are_terciles(returns: pd.Series) -> None:
    frame = regime_labels(returns)
    assert len(frame) == len(ORIGIN_YEARS) == 14
    counts = sorted(frame["regime"].value_counts().to_list())
    assert counts == [4, 5, 5]
    assert set(frame["regime"]) == {"calm", "normal", "stress"}


# --- leakage ---------------------------------------------------------------


def test_nothing_after_the_origin_reaches_the_fit(returns: pd.Series) -> None:
    """The load-bearing test of the whole exercise.

    Multiply every post-origin return by ten and refit. If any future
    data reached the estimator, the parameters or the simulated paths
    would move. Only the density scores — which are *supposed* to read
    the realised returns — may change.
    """
    candidate = CANDIDATES_BY_NAME["garch_normal"]
    year = 2015
    train, test = split_at_origin(returns, year)

    tampered = returns.copy()
    origin = origin_index(returns, year)
    tampered.iloc[origin + 1 :] = tampered.iloc[origin + 1 :] * 10.0
    tampered_train, tampered_test = split_at_origin(tampered, year)

    assert train.equals(tampered_train)  # the split itself is clean

    honest = candidate.fit(train)
    contaminated = candidate.fit(tampered_train)
    assert honest.params == contaminated.params
    assert np.array_equal(
        honest.simulate(60, 20, seed=BACKTEST_SEED),
        contaminated.simulate(60, 20, seed=BACKTEST_SEED),
    )

    # And the scores that read the future do move, so the test has teeth.
    clean_scores, _ = _score_one(candidate, train, test, 50, BACKTEST_SEED)
    dirty_scores, _ = _score_one(candidate, train, tampered_test, 50, BACKTEST_SEED)
    assert clean_scores["nll"].mean() != pytest.approx(dirty_scores["nll"].mean())


# --- cache -----------------------------------------------------------------


def test_cache_avoids_a_second_fit(returns: pd.Series, tmp_path, monkeypatch) -> None:
    candidate = CANDIDATES_BY_NAME["iid_t"]
    calls = {"n": 0}
    original = candidate.fit

    def counting_fit(series):
        calls["n"] += 1
        return original(series)

    monkeypatch.setattr(candidate, "fit", counting_fit)
    kwargs = dict(
        candidates=(candidate,),
        origin_years=(2015,),
        n_paths=20,
        cache_dir=tmp_path,
        allow_unregistered=True,
    )
    first = run_backtest(returns, **kwargs)
    assert calls["n"] == 1
    second = run_backtest(returns, **kwargs)
    assert calls["n"] == 1  # served from cache
    pd.testing.assert_frame_equal(first.summary, second.summary)


def test_cache_is_invalidated_by_a_plan_change(
    returns: pd.Series, tmp_path, monkeypatch
) -> None:
    """A cache entry written under a different plan must not be reused."""
    candidate = CANDIDATES_BY_NAME["iid_t"]
    kwargs = dict(
        candidates=(candidate,),
        origin_years=(2015,),
        n_paths=20,
        cache_dir=tmp_path,
        allow_unregistered=True,
    )
    run_backtest(returns, **kwargs)
    cached = next(tmp_path.glob("*.parquet"))
    before = cached.stat().st_mtime_ns

    monkeypatch.setattr("brent_synth.backtest.STRESS_WEIGHT", 3.0)
    run_backtest(returns, **kwargs)
    assert cached.stat().st_mtime_ns != before


# --- pre-registration ------------------------------------------------------


def test_preregistration_round_trip(tmp_path) -> None:
    path = tmp_path / "preregistration.md"
    write_preregistration(path)
    matched, message = check_preregistration(path)
    assert matched
    assert preregistration_hash()[:16] in message


def test_a_changed_plan_is_refused(tmp_path, monkeypatch) -> None:
    path = tmp_path / "preregistration.md"
    write_preregistration(path)
    monkeypatch.setattr("brent_synth.backtest.STRESS_WEIGHT", 5.0)

    with pytest.raises(RuntimeError, match="does not match"):
        check_preregistration(path)
    matched, message = check_preregistration(path, allow_unregistered=True)
    assert not matched
    assert "UNREGISTERED" in message


def test_a_missing_plan_is_refused(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="No pre-registration"):
        check_preregistration(tmp_path / "absent.md")


def test_the_committed_preregistration_matches_the_code() -> None:
    """The file in the repo must describe the code that is running."""
    matched, _ = check_preregistration(PREREGISTRATION_PATH)
    assert matched


# --- ranking ---------------------------------------------------------------


def _summary(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["status"] = "ok"
    frame["regime"] = frame.get("regime", "normal")
    return frame


def test_ties_are_broken_by_the_lower_step() -> None:
    """Two identical models: the simpler one wins."""
    rows = []
    for year in (2011, 2012):
        for model, step in (("complex", 4), ("simple", 1)):
            rows.append(
                {
                    "model": model,
                    "step": step,
                    "origin_year": year,
                    "regime": "normal",
                    **{loss: 1.0 for loss in RANKING_LOSSES},
                }
            )
    summary = _summary(rows)
    scored = score_models(summary, (2011, 2012))
    assert scored.iloc[0]["weighted_score"] == pytest.approx(
        scored.iloc[1]["weighted_score"]
    )
    assert select_champion(summary, (2011, 2012)) == "simple"


def test_stress_origins_carry_more_weight() -> None:
    """A model that wins only in stress must beat one that loses there."""
    rows = []
    for year, regime in ((2011, "calm"), (2012, "stress")):
        for model, step in (("stress_winner", 1), ("calm_winner", 2)):
            wins = (model == "stress_winner") == (regime == "stress")
            rows.append(
                {
                    "model": model,
                    "step": step,
                    "origin_year": year,
                    "regime": regime,
                    **{loss: (0.0 if wins else 1.0) for loss in RANKING_LOSSES},
                }
            )
    scored = score_models(_summary(rows), (2011, 2012))
    assert scored.iloc[0]["model"] == "stress_winner"


def test_confirmation_reports_agreement_honestly() -> None:
    rows = []
    for year in (2019, 2020):
        for model, step, loss in (("a", 1, 1.0), ("b", 2, 2.0)):
            rows.append(
                {
                    "model": model,
                    "step": step,
                    "origin_year": year,
                    "regime": "normal",
                    **{name: loss for name in RANKING_LOSSES},
                }
            )
    summary = _summary(rows)
    table, agrees = confirm(summary, "a", (2019, 2020))
    assert agrees
    table, agrees = confirm(summary, "b", (2019, 2020))
    assert not agrees
    assert table.iloc[0]["model"] == "a"


def test_failed_runs_are_ranked_last_not_dropped() -> None:
    """Under Amendment 1 a broken model still appears, ranked worst."""
    rows = [
        {
            "model": "good",
            "step": 1,
            "origin_year": 2011,
            "regime": "normal",
            "status": "ok",
            **{loss: 1.0 for loss in RANKING_LOSSES},
        },
        {
            "model": "broken",
            "step": 2,
            "origin_year": 2011,
            "regime": "normal",
            "status": "failed",
            **{loss: np.nan for loss in RANKING_LOSSES},
        },
    ]
    summary = pd.DataFrame(rows)

    amended = score_models(summary, (2011,), failure_rule="worst_rank")
    assert list(amended["model"]) == ["good", "broken"]
    assert amended.set_index("model").loc["broken", "weighted_score"] == 2.0

    # The original rule still exists, and still hides the failure entirely.
    dropped = score_models(summary, (2011,), failure_rule="drop")
    assert list(dropped["model"]) == ["good"]


# --- Amendment 1: unavailable models take the worst rank -------------------


def _mixed_summary() -> pd.DataFrame:
    """Three models over two origins; one fails at the second origin."""
    rows = []
    for year in (2011, 2012):
        for model, step, loss in (("best", 1, 0.1), ("mid", 2, 0.2), ("flaky", 3, 0.05)):
            failed = model == "flaky" and year == 2012
            rows.append(
                {
                    "model": model,
                    "step": step,
                    "origin_year": year,
                    "regime": "normal",
                    "status": "failed" if failed else "ok",
                    **{
                        name: (np.nan if failed else loss) for name in RANKING_LOSSES
                    },
                }
            )
    return pd.DataFrame(rows)


def test_amendment_gives_failures_the_worst_rank() -> None:
    """A model that cannot produce scenarios has failed, not abstained."""
    summary = _mixed_summary()
    ranked = rank_table(summary, (2011, 2012), failure_rule="worst_rank")

    failed = ranked[(ranked["model"] == "flaky") & (ranked["origin_year"] == 2012)]
    assert len(failed) == 1
    n_models = summary["model"].nunique()
    for loss in RANKING_LOSSES:
        assert float(failed[f"rank_{loss}"].iloc[0]) == float(n_models)

    # Every model is now scored on every origin.
    scored = score_models(summary, (2011, 2012), failure_rule="worst_rank")
    assert set(scored["n_origins"]) == {2}
    assert int(scored.set_index("model").loc["flaky", "n_failed"]) == 1


def test_original_rule_drops_failures_and_flatters_them() -> None:
    """The bias the amendment removes, demonstrated."""
    summary = _mixed_summary()
    dropped = score_models(summary, (2011, 2012), failure_rule="drop")
    amended = score_models(summary, (2011, 2012), failure_rule="worst_rank")

    # 'flaky' wins on the one origin it survives, so dropping flatters it.
    assert int(dropped.set_index("model").loc["flaky", "n_origins"]) == 1
    assert int(amended.set_index("model").loc["flaky", "n_origins"]) == 2
    assert (
        amended.set_index("model").loc["flaky", "weighted_score"]
        > dropped.set_index("model").loc["flaky", "weighted_score"]
    )


def test_failure_rule_is_validated() -> None:
    with pytest.raises(ValueError, match="failure_rule must be"):
        rank_table(_mixed_summary(), (2011,), failure_rule="ignore")


def test_both_rulings_are_available_on_the_real_run(returns: pd.Series) -> None:
    """Both must be computable so the report can show them side by side."""
    summary = _mixed_summary()
    for rule in ("drop", "worst_rank"):
        champion = select_champion(summary, (2011, 2012), failure_rule=rule)
        table, agrees = confirm(summary, champion, (2011, 2012), failure_rule=rule)
        assert not table.empty
        assert agrees


# --- report headline helpers ----------------------------------------------


def test_pit_verdict_classifies_the_three_shapes() -> None:
    """Flat, U and hump must be told apart, not just flagged as non-uniform."""
    from brent_synth.comparison_report import pit_verdict

    rng = np.random.default_rng(0)

    flat = rng.uniform(size=4000)
    shape, _, p_value = pit_verdict(flat)
    assert shape == "flat" and p_value >= 0.05

    # Over-confident: reality lands in the tails more than allowed.
    u_shaped = np.concatenate(
        [rng.uniform(0, 0.06, 1400), rng.uniform(0.94, 1, 1400), rng.uniform(size=800)]
    )
    assert pit_verdict(u_shaped)[0].startswith("U")

    # Under-confident: density spread wider than needed.
    humped = np.clip(rng.normal(0.5, 0.11, 4000), 0.001, 0.999)
    assert pit_verdict(humped)[0].startswith("hump")


def test_pit_verdict_needs_enough_days() -> None:
    from brent_synth.comparison_report import pit_verdict

    assert pit_verdict(np.array([0.1, 0.5, 0.9]))[0] == "n/a"


def test_persistence_table_flags_the_garch_family(returns: pd.Series) -> None:
    """The champion's own persistence crosses 1 at some origins."""
    from brent_synth.comparison_report import persistence_table

    params = pd.DataFrame(
        [
            {"model": "gjr_skewt", "origin_year": 2014, "param": p, "value": v}
            for p, v in (
                ("alpha", 0.06),
                ("beta", 0.93),
                ("gamma", 0.02),
                ("nu", 6.0),
                ("lambda", -0.1),
            )
        ]
    )
    table = persistence_table(params)
    assert list(table["model"]) == ["gjr_skewt"]
    assert float(table["2014"].iloc[0]) > 1.0


# --- SPEC 5.2 reporting helpers -------------------------------------------


def test_pit_diagnostics_reports_effect_sizes() -> None:
    """A p-value alone cannot say whether a deviation matters."""
    from brent_synth.comparison_report import pit_diagnostics

    # A deterministic uniform grid, not a random draw: the chi-square
    # rejects 5% of genuinely uniform samples, and a seeded draw that
    # happens to land there would make this test a coin flip. (One does:
    # default_rng(0).uniform(6000) gives p = 0.040 with a KS distance of
    # 0.007 — which is the argument for reporting effect sizes at all.)
    flat = pit_diagnostics((np.arange(6000) + 0.5) / 6000)
    assert flat["pit_shape"] == "flat"
    assert flat["ks"] < 0.01
    assert flat["max_bin_dev"] < 0.01

    # A badly humped sample must show a much larger effect, not just a
    # smaller p-value.
    rng = np.random.default_rng(0)
    humped = pit_diagnostics(np.clip(rng.normal(0.5, 0.1, 6000), 1e-3, 1 - 1e-3))
    assert humped["pit_shape"].startswith("hump")
    assert humped["ks"] > 10 * flat["ks"]
    assert humped["max_bin_dev"] > 10 * flat["max_bin_dev"]


def test_a_marginal_p_value_comes_with_a_small_effect() -> None:
    """The reason effect sizes ship alongside the chi-square.

    A uniform draw that trips the 5% threshold must still show a tiny
    KS distance, so a reader can tell a real miscalibration from noise.
    """
    from brent_synth.comparison_report import pit_diagnostics

    result = pit_diagnostics(np.random.default_rng(0).uniform(size=6000))
    assert result["chi2_p"] < 0.05  # rejected
    assert result["ks"] < 0.02  # but the deviation is negligible

    missing = pit_diagnostics(np.array([0.2, 0.5]))
    assert missing["pit_shape"] == "n/a"
    assert np.isnan(missing["ks"])


def test_max_bin_deviation_is_in_density_units() -> None:
    """0.02 must mean 'a bin holding 7% where it should hold 5%'."""
    from brent_synth.comparison_report import PIT_BINS, pit_diagnostics

    # Half the mass crammed into the first bin.
    values = np.concatenate([np.full(1000, 0.01), np.linspace(0.05, 0.99, 1000)])
    result = pit_diagnostics(values)
    assert result["max_bin_dev"] == pytest.approx(0.5 - 1.0 / PIT_BINS, abs=0.02)


def test_per_loss_ranks_covers_every_loss_and_period() -> None:
    from brent_synth.comparison_report import per_loss_ranks

    summary = _mixed_summary()
    summary = pd.concat(
        [summary, summary.assign(origin_year=2019), summary.assign(origin_year=2020)]
    )
    table = per_loss_ranks(summary, ("best", "mid"))
    assert len(table) == len(RANKING_LOSSES)
    for model in ("best", "mid"):
        for period in ("selection", "confirmation"):
            assert f"{model} ({period})" in table.columns


def test_common_origins_are_those_every_model_survived() -> None:
    from brent_synth.comparison_report import common_origins

    summary = _mixed_summary()  # 'flaky' fails at 2012 only
    assert common_origins(summary) == [2011]


def test_persistence_by_origin_flags_the_boundary() -> None:
    from brent_synth.comparison_report import persistence_by_origin

    class Fake:
        params = pd.DataFrame(
            [
                {"model": "gjr_skewt", "origin_year": year, "param": p, "value": v}
                for year, alpha in ((2014, 0.08), (2015, 0.04))
                for p, v in (
                    ("alpha", alpha),
                    ("beta", 0.93),
                    ("gamma", 0.02),
                    ("nu", 6.0),
                    ("lambda", -0.1),
                )
            ]
        )

    table = persistence_by_origin(Fake()).set_index("origin_year")
    assert table.loc[2014, "non_stationary"] == "gjr_skewt"
    assert table.loc[2015, "non_stationary"] == "-"
    assert "gjr nu" in table.columns and "gjr gamma" in table.columns


def test_stress_losses_always_include_the_covid_origin(returns: pd.Series) -> None:
    """Origin 2019's test window is calendar 2020; it must never drop out."""
    from brent_synth.comparison_report import stress_origin_losses

    class Fake:
        summary = pd.DataFrame(
            [
                {
                    "origin_year": year,
                    "regime": "calm",  # deliberately not labelled stress
                    "model": "m",
                    "status": "ok",
                    "rank_max_drawdown": 0.5,
                    "rank_worst_day": 0.5,
                    **{loss: 1.0 for loss in RANKING_LOSSES},
                }
                for year in (2013, 2019)
            ]
        )

    table = stress_origin_losses(Fake())
    assert 2019 in set(table["origin_year"])
    assert 2013 not in set(table["origin_year"])
    for loss in RANKING_LOSSES:
        assert loss in table.columns
