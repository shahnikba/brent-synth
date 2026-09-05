"""Out-of-time comparison of the candidate ladder.

Every candidate is fitted only on returns up to an origin date and
scored only on the 252 trading days after it, at fourteen origins. A
champion is chosen on the first eight and confirmed on the last six,
which contain 2020.

The selection rule is fixed **before** the run and written to
``docs/preregistration.md``; :func:`run_backtest` refuses to start if
the file is missing or no longer matches the constants in this module.
That is the whole guard against picking the winner after seeing the
scoreboard — with six candidates and four losses there is more than
enough freedom to justify almost any of them after the fact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from brent_synth.candidates import CANDIDATES
from brent_synth.data import load_returns
from brent_synth.scoring import (
    christoffersen_pvalue,
    crps,
    kupiec_pvalue,
    nll,
    pit,
    sign_split_nll,
    tail_crps,
    var_exceedances,
)
from brent_synth.validation import compute_path_stats

# --- pre-registered constants ---------------------------------------------

ORIGIN_YEARS = tuple(range(2011, 2025))
SELECTION_YEARS = tuple(range(2011, 2019))
CONFIRMATION_YEARS = tuple(range(2019, 2025))
TEST_HORIZON = 252
BACKTEST_N_PATHS = 2000
BACKTEST_SEED = 2024
STRESS_WEIGHT = 2.0

#: Path statistics the percentile ranks are computed over. `mean` is
#: excluded because at 252 days it is almost pure noise, and the VaR
#: aliases because they duplicate the tail quantiles exactly.
RANK_STATS = (
    "std",
    "skew",
    "excess_kurtosis",
    "left_q01",
    "left_q05",
    "right_q95",
    "right_q99",
    "es_99",
    "acf_sq_lag1",
    "max_drawdown",
    "worst_day",
)

#: The four per-origin losses the ranking rule uses, all lower-is-better.
RANKING_LOSSES = ("mean_nll", "mean_tail_crps", "exc99_abs_error", "path_coverage_loss")

VAR_LEVELS = (0.95, 0.99)
REGIME_LABELS = ("calm", "normal", "stress")
PREREGISTRATION_PATH = Path("docs/preregistration.md")
DEFAULT_CACHE_DIR = Path("reports/backtest_cache")


# --- path statistics -------------------------------------------------------


def path_stats_with_worst_day(paths: np.ndarray) -> dict[str, np.ndarray]:
    """`validation.compute_path_stats` plus the single worst day.

    Added here rather than in ``validation`` so that module's
    pre-registered statistic list stays exactly as SPEC 4 fixed it.
    """
    stats = compute_path_stats(paths)
    stats["worst_day"] = np.asarray(paths, dtype="float64").reshape(
        -1, np.asarray(paths).shape[-1]
    ).min(axis=1)
    return stats


def percentile_ranks(
    sim_stats: dict[str, np.ndarray], real_stats: dict[str, float]
) -> dict[str, float]:
    """Where the realised year sat in each simulated distribution.

    Mid-rank for ties, so a discrete or degenerate simulated
    distribution cannot push a rank to exactly 0 or 1 by accident.
    """
    ranks = {}
    for name in RANK_STATS:
        simulated = np.asarray(sim_stats[name], dtype="float64")
        realised = float(np.asarray(real_stats[name]).ravel()[0])
        below = float(np.mean(simulated < realised))
        at_or_below = float(np.mean(simulated <= realised))
        ranks[name] = 0.5 * (below + at_or_below)
    return ranks


def path_coverage_loss(ranks: dict[str, float]) -> float:
    """Mean |rank - 0.5|: 0 means the realised year sat dead centre."""
    return float(np.mean([abs(v - 0.5) for v in ranks.values()]))


# --- origins ---------------------------------------------------------------


def origin_index(returns: pd.Series, year: int) -> int:
    """Position of the last trading day on or before 31 December ``year``."""
    cutoff = pd.Timestamp(year=year, month=12, day=31)
    positions = np.flatnonzero(np.asarray(returns.index <= cutoff))
    if positions.size == 0:
        raise ValueError(f"No returns on or before {cutoff.date()}.")
    return int(positions[-1])


def split_at_origin(
    returns: pd.Series, year: int, horizon: int = TEST_HORIZON
) -> tuple[pd.Series, pd.Series]:
    """Training and test slices for one origin.

    The split is positional off the origin index, so nothing dated after
    the origin can reach ``fit`` — the leakage test depends on exactly
    this.
    """
    index = origin_index(returns, year)
    train = returns.iloc[: index + 1]
    test = returns.iloc[index + 1 : index + 1 + horizon]
    if len(test) < horizon:
        raise ValueError(
            f"Origin {year} has only {len(test)} trading days after it, need "
            f"{horizon}. The data does not extend far enough for this origin."
        )
    return train, test


def regime_labels(
    returns: pd.Series, origin_years=ORIGIN_YEARS, horizon: int = TEST_HORIZON
) -> pd.DataFrame:
    """Label each test year calm / normal / stress by its realised vol.

    Terciles of the realised standard deviation across the test windows.
    These describe the *outcome* years and are used only for weighting
    and for grouping the report; nothing here reaches any fit.
    """
    rows = []
    for year in origin_years:
        _, test = split_at_origin(returns, year, horizon)
        rows.append(
            {
                "origin_year": year,
                "origin_date": returns.index[origin_index(returns, year)],
                "realised_std": float(test.std(ddof=1)),
            }
        )
    frame = pd.DataFrame(rows)

    # Rank-based terciles rather than pd.qcut on the values: qcut needs
    # three distinct bin edges and raises outright on a short or tied
    # set of origins, which is exactly the case a quick partial run
    # hits. Ranking splits by position, so it degrades gracefully to
    # however many origins there are and is unbothered by ties.
    n = len(frame)
    position = frame["realised_std"].rank(method="first").to_numpy() - 1.0
    tercile = np.minimum((position * len(REGIME_LABELS) // n).astype(int),
                         len(REGIME_LABELS) - 1)
    frame["regime"] = np.asarray(REGIME_LABELS)[tercile]
    return frame


# --- pre-registration ------------------------------------------------------


def preregistration_text(candidates=CANDIDATES) -> str:
    """The document whose hash locks the analysis plan."""
    ladder = "\n".join(
        f"| {c.step} | `{c.name}` | {type(c).__doc__.strip().splitlines()[0]} |"
        for c in sorted(candidates, key=lambda c: c.step)
    )
    losses = "\n".join(f"{i}. `{name}`" for i, name in enumerate(RANKING_LOSSES, 1))
    body = f"""# Pre-registration — Brent scenario-generator comparison

Fixed before any origin was scored. `run_backtest` recomputes the hash
below from the live constants and refuses to run if they have drifted.

## Candidates

| Step | Name | Description |
|---|---|---|
{ladder}

## Ranking losses

Each is computed per (model, origin), ranked across models at that
origin (1 = best), then averaged with the stress weighting below.

{losses}

## Origins and split

- Origin for year Y: last trading day on or before Y-12-31.
- Training: every return up to and including the origin.
- Test: the next {TEST_HORIZON} returns.
- Origins: {list(ORIGIN_YEARS)}
- Selection: {list(SELECTION_YEARS)}
- Confirmation: {list(CONFIRMATION_YEARS)}

## Simulation

- Paths per (model, origin): {BACKTEST_N_PATHS}
- Seed: {BACKTEST_SEED}
- Path statistics ranked: {list(RANK_STATS)}

## Weighting

- Stress-labelled origins carry weight {STRESS_WEIGHT}; others 1.
- Regimes are terciles of realised test-window volatility: {list(REGIME_LABELS)}.

## Selection rule

Champion = lowest weighted mean rank on the selection origins; ties
broken by lower ladder step. The same table is then recomputed on the
confirmation origins and reported verbatim, whether or not it agrees.
"""
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return body + f"\n## Hash\n\n`sha256:{digest}`\n"


def preregistration_hash(candidates=CANDIDATES) -> str:
    text = preregistration_text(candidates)
    body = text.split("\n## Hash\n")[0]
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_preregistration(
    path: Path = PREREGISTRATION_PATH, candidates=CANDIDATES
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(preregistration_text(candidates), encoding="utf-8")
    return str(path)


def check_preregistration(
    path: Path = PREREGISTRATION_PATH,
    candidates=CANDIDATES,
    allow_unregistered: bool = False,
) -> tuple[bool, str]:
    """Return ``(matched, message)``; raise unless overridden."""
    expected = preregistration_hash(candidates)
    path = Path(path)
    if not path.exists():
        message = f"No pre-registration at {path}."
    else:
        recorded = path.read_text(encoding="utf-8")
        matched = f"sha256:{expected}" in recorded
        if matched:
            return True, f"matched sha256:{expected[:16]}…"
        message = (
            f"Pre-registration at {path} does not match the current constants "
            f"(expected sha256:{expected[:16]}…)."
        )
    if not allow_unregistered:
        raise RuntimeError(
            message + " Re-run `preregister` and commit it, or pass "
            "--allow-unregistered (which is recorded in the report)."
        )
    return False, message + " RUN WAS UNREGISTERED."


# --- results ---------------------------------------------------------------


@dataclass
class BacktestResults:
    daily: pd.DataFrame
    summary: pd.DataFrame
    params: pd.DataFrame
    regimes: pd.DataFrame
    preregistration: str = ""
    registered: bool = True
    n_paths: int = BACKTEST_N_PATHS
    seed: int = BACKTEST_SEED
    notes: dict[str, str] = field(default_factory=dict)


def _cache_path(cache_dir: Path, model: str, year: int) -> Path:
    return Path(cache_dir) / f"{model}__{year}.parquet"


def _write_cache(path: Path, daily: pd.DataFrame, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(daily, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[b"payload"] = json.dumps(payload).encode("utf-8")
    table = table.replace_schema_metadata(metadata)
    temporary = path.with_suffix(".tmp")
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_cache(path: Path, prereg_hash: str) -> tuple[pd.DataFrame, dict] | None:
    if not path.exists():
        return None
    try:
        table = pq.read_table(path)
        payload = json.loads((table.schema.metadata or {})[b"payload"].decode())
    except Exception:
        return None
    if payload.get("prereg_hash") != prereg_hash:
        return None
    return table.to_pandas(), payload


def _score_one(
    candidate, train: pd.Series, test: pd.Series, n_paths: int, seed: int
) -> tuple[pd.DataFrame, dict]:
    """Fit, forecast, simulate and score one (candidate, origin)."""
    started = time.time()
    fitted = candidate.fit(train)
    fit_seconds = time.time() - started

    y = test.to_numpy()
    forecast = fitted.forecast_density(test)
    if forecast.n != len(test):
        raise ValueError(
            f"{candidate.name}: forecast has {forecast.n} days for a "
            f"{len(test)}-day test window."
        )

    daily = pd.DataFrame(
        {
            "date": test.index,
            "nll": nll(forecast, y),
            "crps": crps(forecast, y),
            "tail_crps": tail_crps(forecast, y),
            "pit": pit(forecast, y),
        }
    )
    for level in VAR_LEVELS:
        daily[f"exc{int(level * 100)}"] = var_exceedances(forecast, y, level)

    # The day before each test day: the last training return, then the
    # test returns shifted by one.
    previous = np.concatenate([[float(train.iloc[-1])], y[:-1]])
    nll_after_down, nll_after_up = sign_split_nll(forecast, y, previous)

    paths = fitted.simulate(TEST_HORIZON, n_paths, seed)
    ranks = percentile_ranks(
        path_stats_with_worst_day(paths), path_stats_with_worst_day(y[None, :])
    )

    summary = {
        "status": "ok",
        "message": "",
        "fit_seconds": fit_seconds,
        "loglik": float(fitted.loglik),
        "n_params": int(fitted.n_params),
        "mean_nll": float(daily["nll"].mean()),
        "mean_crps": float(daily["crps"].mean()),
        "mean_tail_crps": float(daily["tail_crps"].mean()),
        "nll_after_down": nll_after_down,
        "nll_after_up": nll_after_up,
        "path_coverage_loss": path_coverage_loss(ranks),
        "realised_std": float(test.std(ddof=1)),
        "simulation_note": getattr(fitted, "simulation_note", ""),
    }
    for level in VAR_LEVELS:
        tag = int(level * 100)
        exceed = daily[f"exc{tag}"].to_numpy()
        n_exceed = int(exceed.sum())
        expected = 1.0 - level
        summary[f"exc{tag}_count"] = n_exceed
        summary[f"exc{tag}_rate"] = n_exceed / len(test)
        summary[f"exc{tag}_abs_error"] = abs(n_exceed / len(test) - expected)
        summary[f"kupiec{tag}_p"] = kupiec_pvalue(n_exceed, len(test), expected)
        summary[f"christoffersen{tag}_p"] = christoffersen_pvalue(exceed)
    for name, value in ranks.items():
        summary[f"rank_{name}"] = value

    params = {str(k): float(v) for k, v in fitted.params.items()}
    return daily, {"summary": summary, "params": params}


def run_backtest(
    returns: pd.Series | None = None,
    candidates=CANDIDATES,
    origin_years=ORIGIN_YEARS,
    n_paths: int = BACKTEST_N_PATHS,
    seed: int = BACKTEST_SEED,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    allow_unregistered: bool = False,
    preregistration_path: Path = PREREGISTRATION_PATH,
    verbose: bool = False,
) -> BacktestResults:
    """Fit and score every candidate at every origin.

    Results are cached per (candidate, origin) and keyed by the
    pre-registration hash, so changing the plan invalidates the cache
    rather than silently mixing runs from different rules.
    """
    registered, message = check_preregistration(
        preregistration_path, candidates, allow_unregistered
    )
    prereg_hash = preregistration_hash(candidates)
    if returns is None:
        returns = load_returns()

    regimes = regime_labels(returns, origin_years)
    regime_by_year = dict(zip(regimes["origin_year"], regimes["regime"]))

    daily_frames, summary_rows, param_rows = [], [], []
    for candidate in candidates:
        for year in origin_years:
            cached = _read_cache(_cache_path(cache_dir, candidate.name, year), prereg_hash)
            if cached is not None:
                daily, payload = cached
            else:
                train, test = split_at_origin(returns, year)
                try:
                    daily, payload = _score_one(candidate, train, test, n_paths, seed)
                except Exception as error:  # noqa: BLE001 - recorded, not raised
                    daily = pd.DataFrame()
                    payload = {
                        "summary": {
                            "status": "failed",
                            "message": f"{type(error).__name__}: {error}",
                        },
                        "params": {},
                    }
                payload["prereg_hash"] = prereg_hash
                _write_cache(
                    _cache_path(cache_dir, candidate.name, year), daily, payload
                )

            if verbose:
                print(
                    f"  {candidate.name:<16} {year}  "
                    f"{payload['summary'].get('status')}",
                    file=sys.stderr,
                )

            if len(daily):
                labelled = daily.copy()
                labelled.insert(0, "model", candidate.name)
                labelled.insert(1, "origin_year", year)
                daily_frames.append(labelled)

            row = {
                "model": candidate.name,
                "step": candidate.step,
                "origin_year": year,
                "regime": regime_by_year[year],
                **payload["summary"],
            }
            summary_rows.append(row)
            for name, value in payload["params"].items():
                param_rows.append(
                    {
                        "model": candidate.name,
                        "origin_year": year,
                        "param": name,
                        "value": value,
                    }
                )

    return BacktestResults(
        daily=(
            pd.concat(daily_frames, ignore_index=True)
            if daily_frames
            else pd.DataFrame()
        ),
        summary=pd.DataFrame(summary_rows),
        params=pd.DataFrame(param_rows),
        regimes=regimes,
        preregistration=message,
        registered=registered,
        n_paths=n_paths,
        seed=seed,
    )


# --- ranking ---------------------------------------------------------------


def rank_table(summary: pd.DataFrame, origin_years=None) -> pd.DataFrame:
    """Rank each model against the others at each origin, per loss."""
    frame = summary.copy()
    if origin_years is not None:
        frame = frame[frame["origin_year"].isin(origin_years)]
    frame = frame[frame["status"] == "ok"]
    if frame.empty:
        return pd.DataFrame(columns=["model", "origin_year", *RANKING_LOSSES])

    ranked = frame[["model", "step", "origin_year", "regime", *RANKING_LOSSES]].copy()
    for loss in RANKING_LOSSES:
        ranked[f"rank_{loss}"] = ranked.groupby("origin_year")[loss].rank(
            method="average"
        )
    rank_columns = [f"rank_{loss}" for loss in RANKING_LOSSES]
    ranked["mean_rank"] = ranked[rank_columns].mean(axis=1)
    ranked["weight"] = np.where(ranked["regime"] == "stress", STRESS_WEIGHT, 1.0)
    return ranked


def score_models(summary: pd.DataFrame, origin_years=None) -> pd.DataFrame:
    """Weighted mean rank per model, plus rank stability."""
    ranked = rank_table(summary, origin_years)
    if ranked.empty:
        return pd.DataFrame()

    ranked = ranked.copy()
    top_two = ranked.groupby("origin_year")["mean_rank"].rank(method="min") <= 2
    ranked["in_top_two"] = top_two

    rows = []
    for (model, step), group in ranked.groupby(["model", "step"]):
        weights = group["weight"].to_numpy()
        rows.append(
            {
                "model": model,
                "step": step,
                "weighted_score": float(
                    np.average(group["mean_rank"].to_numpy(), weights=weights)
                ),
                "rank_stability": float(group["in_top_two"].mean()),
                "n_origins": int(len(group)),
                **{
                    f"mean_{loss}": float(group[loss].mean())
                    for loss in RANKING_LOSSES
                },
            }
        )
    scored = pd.DataFrame(rows).sort_values(["weighted_score", "step"])
    return scored.reset_index(drop=True)


def select_champion(summary: pd.DataFrame, origin_years=SELECTION_YEARS) -> str:
    """Lowest weighted score on the selection origins; ties to lower step."""
    scored = score_models(summary, origin_years)
    if scored.empty:
        raise ValueError("No successful runs on the selection origins.")
    best = scored.iloc[0]
    return str(best["model"])


def confirm(
    summary: pd.DataFrame, champion: str, origin_years=CONFIRMATION_YEARS
) -> tuple[pd.DataFrame, bool]:
    """Recompute the table on held-out origins; report whether it agrees."""
    scored = score_models(summary, origin_years)
    if scored.empty:
        return scored, False
    return scored, bool(scored.iloc[0]["model"] == champion)


# --- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m brent_synth.backtest")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preregister", help="write docs/preregistration.md")

    run_parser = sub.add_parser("run", help="fit and score every candidate")
    run_parser.add_argument("--n-paths", type=int, default=BACKTEST_N_PATHS)
    run_parser.add_argument("--allow-unregistered", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")

    report_parser = sub.add_parser("report", help="write the comparison report")
    report_parser.add_argument("--n-paths", type=int, default=BACKTEST_N_PATHS)
    report_parser.add_argument("--allow-unregistered", action="store_true")
    report_parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "preregister":
        print(write_preregistration())
        return 0

    results = run_backtest(
        n_paths=args.n_paths,
        allow_unregistered=args.allow_unregistered,
        verbose=not getattr(args, "quiet", False),
    )
    failures = (results.summary["status"] != "ok").sum()
    print(
        f"Scored {len(results.summary)} (model, origin) pairs, "
        f"{failures} unavailable.",
        file=sys.stderr,
    )

    if args.command == "report":
        from brent_synth.comparison_report import make_comparison_report

        paths = make_comparison_report(results)
        for path in paths:
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
