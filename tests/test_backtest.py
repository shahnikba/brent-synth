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


def test_failed_runs_are_excluded_from_ranking() -> None:
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
    scored = score_models(pd.DataFrame(rows), (2011,))
    assert list(scored["model"]) == ["good"]
