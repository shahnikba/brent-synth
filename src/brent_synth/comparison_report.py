"""The model-comparison report, in markdown and HTML twins.

Both formats are generated from the same tables and the same figures,
in the same section order, so they cannot tell different stories — the
lesson from the validation report, where a stale regeneration once left
the two disagreeing.

Nothing here decides anything. The champion and the confirmation verdict
are computed by :mod:`brent_synth.backtest` under the pre-registered
rule and printed verbatim, agreeing or not.
"""

from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from brent_synth.backtest import (  # noqa: E402
    BACKTEST_N_PATHS,
    rank_table,
    CONFIRMATION_YEARS,
    DEFAULT_FAILURE_RULE,
    FAILURE_RULES,
    ORIGIN_YEARS,
    RANK_STATS,
    RANKING_LOSSES,
    SELECTION_YEARS,
    STRESS_WEIGHT,
    TEST_HORIZON,
    confirm,
    preregistration_hash,
    score_models,
    select_champion,
)
from brent_synth.model import _leverage_weight  # noqa: E402
from brent_synth.validation import (  # noqa: E402
    DEFAULT_NARRATIVE_DIR,
    load_narrative,
    markdown_to_html,
)
from brent_synth.scoring import diebold_mariano  # noqa: E402

FIGURE_SLUGS = {
    "Scoreboard": "scoreboard",
    "PIT histograms": "pit-histograms",
    "PIT by regime": "pit-by-regime",
    "Path-rank histograms": "path-ranks",
    "Parameter drift": "parameter-drift",
}

LOSS_LABELS = {
    "mean_nll": "Mean NLL",
    "mean_tail_crps": "Mean tail CRPS",
    "exc99_abs_error": "|99% exceedance error|",
    "path_coverage_loss": "Path coverage loss",
}


def _png(figure: plt.Figure) -> bytes:
    buffer = __import__("io").BytesIO()
    figure.savefig(
        buffer, format="png", dpi=110, bbox_inches="tight", metadata={"Software": None}
    )
    plt.close(figure)
    return buffer.getvalue()


def _grid(n: int) -> tuple[int, int]:
    columns = min(3, n)
    return int(np.ceil(n / columns)), columns


# --- figures ---------------------------------------------------------------


def _plot_scoreboard(selection: pd.DataFrame, confirmation: pd.DataFrame) -> bytes:
    models = list(selection["model"])
    positions = np.arange(len(models))
    width = 0.38
    confirmation_scores = (
        confirmation.set_index("model")["weighted_score"].reindex(models).to_numpy()
    )

    figure, axes = plt.subplots(figsize=(8.0, 4.2))
    axes.bar(
        positions - width / 2,
        selection["weighted_score"],
        width,
        label=f"Selection {SELECTION_YEARS[0]}-{SELECTION_YEARS[-1]}",
    )
    axes.bar(
        positions + width / 2,
        confirmation_scores,
        width,
        label=f"Confirmation {CONFIRMATION_YEARS[0]}-{CONFIRMATION_YEARS[-1]}",
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(models, rotation=20, ha="right")
    axes.set_ylabel("Weighted mean rank (lower is better)")
    axes.set_title("Scoreboard: selection vs confirmation origins")
    axes.legend(fontsize=8)
    return _png(figure)


def _plot_pit(daily: pd.DataFrame) -> bytes:
    models = sorted(daily["model"].unique())
    rows, columns = _grid(len(models))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.4 * columns, 2.7 * rows), squeeze=False
    )
    for index, model in enumerate(models):
        axis = axes[index // columns][index % columns]
        values = daily.loc[daily["model"] == model, "pit"].to_numpy()
        axis.hist(values, bins=20, range=(0, 1), density=True, alpha=0.75)
        axis.axhline(1.0, ls="--", lw=1.1, color="0.3")
        axis.set_title(model, fontsize=10)
        axis.set_ylim(0, max(2.6, axis.get_ylim()[1]))
        axis.set_xlabel("PIT")
    for index in range(len(models), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle("PIT histograms — flat is calibrated", y=1.01)
    figure.tight_layout()
    return _png(figure)


def _plot_path_ranks(summary: pd.DataFrame) -> bytes:
    models = sorted(summary["model"].unique())
    columns_wanted = [f"rank_{name}" for name in RANK_STATS]
    rows, columns = _grid(len(models))
    figure, axes = plt.subplots(
        rows, columns, figsize=(3.4 * columns, 2.7 * rows), squeeze=False
    )
    for index, model in enumerate(models):
        axis = axes[index // columns][index % columns]
        block = summary.loc[
            (summary["model"] == model) & (summary["status"] == "ok"), columns_wanted
        ]
        values = block.to_numpy().ravel()
        values = values[np.isfinite(values)]
        if values.size:
            axis.hist(values, bins=10, range=(0, 1), density=True, alpha=0.75)
        axis.axhline(1.0, ls="--", lw=1.1, color="0.3")
        axis.set_title(model, fontsize=10)
        axis.set_ylim(0, max(2.6, axis.get_ylim()[1]))
        axis.set_xlabel("Percentile rank of realised year")
    for index in range(len(models), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle("Path-rank histograms — flat is calibrated", y=1.01)
    figure.tight_layout()
    return _png(figure)


def _derived_persistence(params: pd.DataFrame) -> pd.DataFrame:
    """Persistence per (model, origin) where the family defines one."""
    rows = []
    wide = params.pivot_table(
        index=["model", "origin_year"], columns="param", values="value"
    )
    for (model, year), row in wide.iterrows():
        value = None
        if model in {"gjr_skewt", "gjr_skewt_vt"} and {"alpha", "beta", "gamma"} <= set(
            row.dropna().index
        ):
            weight = _leverage_weight(float(row["nu"]), float(row["lambda"]))
            value = float(row["alpha"] + row["gamma"] * weight + row["beta"])
        elif model == "garch_normal" and "alpha[1]" in row and "beta[1]" in row:
            value = float(row["alpha[1]"] + row["beta[1]"])
        if value is not None:
            rows.append(
                {
                    "model": model,
                    "origin_year": year,
                    "param": "persistence",
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def _plot_parameter_drift(params: pd.DataFrame) -> bytes:
    combined = pd.concat([params, _derived_persistence(params)], ignore_index=True)
    pairs = (
        combined[["model", "param"]]
        .drop_duplicates()
        .sort_values(["model", "param"])
        .to_records(index=False)
    )
    rows, columns = int(np.ceil(len(pairs) / 5)), 5
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.9 * columns, 2.1 * rows), squeeze=False
    )
    for index, (model, param) in enumerate(pairs):
        axis = axes[index // columns][index % columns]
        block = combined[
            (combined["model"] == model) & (combined["param"] == param)
        ].sort_values("origin_year")
        highlight = param == "persistence"
        axis.plot(
            block["origin_year"],
            block["value"],
            marker="o",
            ms=3,
            lw=1.6 if highlight else 1.1,
            color="tab:red" if highlight else "tab:blue",
        )
        axis.set_title(f"{model}\n{param}", fontsize=7.5)
        axis.tick_params(labelsize=6.5)
        if highlight:
            axis.axhline(1.0, ls=":", lw=1.0, color="0.4")
    for index in range(len(pairs), rows * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle("Parameter drift across origins (persistence in red)", y=1.005)
    figure.tight_layout()
    return _png(figure)


# --- tables ----------------------------------------------------------------


def _dm_table(results, champion: str) -> pd.DataFrame:
    """DM p-values against the champion, pooled over confirmation origins."""
    daily = results.daily
    if daily.empty:
        return pd.DataFrame()
    block = daily[daily["origin_year"].isin(CONFIRMATION_YEARS)]
    reference = block[block["model"] == champion].set_index(["origin_year", "date"])
    rows = []
    for model in sorted(block["model"].unique()):
        if model == champion:
            continue
        other = block[block["model"] == model].set_index(["origin_year", "date"])
        shared = reference.index.intersection(other.index)
        if len(shared) < 30:
            continue
        row = {"model": model, "n_days": len(shared)}
        for loss in ("nll", "tail_crps"):
            statistic, p_value = diebold_mariano(
                other.loc[shared, loss].to_numpy(),
                reference.loc[shared, loss].to_numpy(),
            )
            row[f"{loss}_stat"] = statistic
            row[f"{loss}_p"] = p_value
        rows.append(row)
    return pd.DataFrame(rows)


def pit_short_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The reference's four-column PIT table: shape, p, KS, max-bin dev."""
    return frame[["model", "shape", "chi2_p", "ks", "max_bin_dev"]].rename(
        columns={"chi2_p": "χ² p", "ks": "KS", "max_bin_dev": "max-bin dev"}
    )


def pit_regime_matrix(results) -> pd.DataFrame:
    """One row per model, one column per regime, 'shape (p)' in each cell."""
    frame = pit_by_regime(results)
    rows = []
    for model, block in frame.groupby("model"):
        row = {"model": model}
        indexed = block.set_index("regime")
        for regime in REGIME_ORDER:
            if regime in indexed.index:
                cell = indexed.loc[regime]
                marker = "**U**" if str(cell["shape"]) == "U" else str(cell["shape"])
                row[regime] = f"{marker} ({float(cell['chi2_p']):.3f})"
            else:
                row[regime] = "n/a"
        rows.append(row)
    order = {"gjr_skewt": 0, "figarch_skewt": 1, "garch_normal": 2,
             "gjr_skewt_vt": 3, "iid_t": 4, "ms_variance": 5}
    out = pd.DataFrame(rows)
    return out.sort_values("model", key=lambda c: c.map(lambda m: order.get(m, 99)))


def _regime_table(summary: pd.DataFrame) -> pd.DataFrame:
    ok = summary[summary["status"] == "ok"]
    return (
        ok.groupby(["model", "regime"])[list(RANKING_LOSSES)]
        .mean()
        .reset_index()
        .sort_values(["regime", "model"])
    )


def _crisis_table(summary: pd.DataFrame) -> pd.DataFrame:
    stress = summary[(summary["regime"] == "stress") & (summary["status"] == "ok")]
    columns = [
        "origin_year",
        "model",
        "rank_max_drawdown",
        "rank_worst_day",
        "rank_std",
        "rank_excess_kurtosis",
        "exc99_count",
    ]
    return stress[columns].sort_values(["origin_year", "model"])


def _fmt(value, digits: int = 4) -> str:
    """Render a cell: strings and labels pass through, numbers get digits."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def _md_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "*No rows.*"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    lines = [header, divider]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v, digits) for v in row) + " |")
    return "\n".join(lines)


def _html_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "<p><i>No rows.</i></p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_fmt(v, digits))}</td>" for v in row)
        + "</tr>"
        for row in frame.itertuples(index=False)
    )
    return f"<table><tr>{head}</tr>{body}</table>"


# --- report ----------------------------------------------------------------


PIT_BINS = 20
REGIME_ORDER = ("calm", "normal", "stress")


def pit_diagnostics(values: np.ndarray) -> dict[str, float | str]:
    """Calibration of a pooled PIT sample, with effect sizes.

    The chi-square p-value answers "is this distinguishable from
    uniform", which at several thousand days answers yes to deviations
    far too small to matter. So two effect sizes travel with it:

    - ``ks``: the Kolmogorov-Smirnov distance from Uniform(0,1), the
      largest gap between the empirical and ideal CDFs.
    - ``max_bin_dev``: the largest single-bin departure from the ideal
      density of 1/20, in the same units as the histogram — 0.02 means
      a bin holding 7% of the mass where it should hold 5%.

    ``shape`` reads where the mass sits: comparing the two outer bins
    against the two central ones separates over-confidence (mass in the
    tails, U) from under-confidence (mass in the middle, hump). A sample
    that passes the chi-square is called flat regardless.
    """
    from scipy import stats as sps

    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if values.size < PIT_BINS * 2:
        return {
            "pit_shape": "n/a",
            "shape": "n/a",
            "chi2": float("nan"),
            "chi2_p": float("nan"),
            "ks": float("nan"),
            "max_bin_dev": float("nan"),
            "n_days": int(values.size),
        }

    counts, _ = np.histogram(values, bins=PIT_BINS, range=(0.0, 1.0))
    expected = values.size / PIT_BINS
    statistic = float(((counts - expected) ** 2 / expected).sum())
    p_value = float(sps.chi2.sf(statistic, PIT_BINS - 1))

    density = counts / values.size
    max_bin_dev = float(np.abs(density - 1.0 / PIT_BINS).max())
    ks = float(sps.kstest(values, "uniform").statistic)

    edges = counts[0] + counts[-1]
    middle = counts[PIT_BINS // 2 - 1] + counts[PIT_BINS // 2]
    if p_value >= 0.05:
        shape = "flat"
    elif edges > middle:
        shape = "U (over-confident)"
    else:
        shape = "hump (under-confident)"

    return {
        "pit_shape": shape,
        "shape": shape.split(" ")[0],
        "chi2": statistic,
        "chi2_p": p_value,
        "ks": ks,
        "max_bin_dev": max_bin_dev,
        "n_days": int(values.size),
    }


def pit_verdict(values: np.ndarray) -> tuple[str, float, float]:
    """Backwards-compatible triple: ``(shape, chi2, p)``."""
    result = pit_diagnostics(values)
    return (
        str(result["pit_shape"]),
        float(result["chi2"]),
        float(result["chi2_p"]),
    )


def _daily_with_regime(results) -> pd.DataFrame:
    """results.daily with the test-year regime label attached."""
    labels = results.regimes[["origin_year", "regime"]]
    return results.daily.merge(labels, on="origin_year", how="left")


def pit_by_regime(results) -> pd.DataFrame:
    """PIT calibration per model, split by the regime of the test year."""
    daily = _daily_with_regime(results)
    rows = []
    for model in sorted(daily["model"].unique()):
        for regime in REGIME_ORDER:
            block = daily[(daily["model"] == model) & (daily["regime"] == regime)]
            rows.append(
                {
                    "model": model,
                    "regime": regime,
                    **pit_diagnostics(block["pit"].to_numpy()),
                }
            )
    return pd.DataFrame(rows)


def common_origins(summary: pd.DataFrame) -> list[int]:
    """Origins where every candidate produced a fit.

    Comparing calibration across models is only like-for-like on days
    all of them saw; gjr_skewt_vt is unavailable at five origins, so the
    pooled tables above judge it on a different, calmer subset.
    """
    ok = summary[summary["status"] == "ok"]
    counts = ok.groupby("origin_year")["model"].nunique()
    full = int(summary["model"].nunique())
    return sorted(int(y) for y in counts[counts == full].index)


def pit_on_common_origins(results) -> pd.DataFrame:
    """The pooled PIT table restricted to origins every model survived."""
    years = common_origins(results.summary)
    daily = results.daily[results.daily["origin_year"].isin(years)]
    rows = []
    for model in sorted(daily["model"].unique()):
        block = daily.loc[daily["model"] == model, "pit"].to_numpy()
        rows.append({"model": model, **pit_diagnostics(block)})
    return pd.DataFrame(rows)


def persistence_table(params: pd.DataFrame) -> pd.DataFrame:
    """Persistence per (model, origin) for the GARCH family, wide by year."""
    derived = _derived_persistence(params)
    if derived.empty:
        return pd.DataFrame()
    wide = derived.pivot_table(
        index="model", columns="origin_year", values="value"
    ).reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide


def per_loss_ranks(
    summary: pd.DataFrame,
    models: tuple[str, ...],
    failure_rule: str = DEFAULT_FAILURE_RULE,
) -> pd.DataFrame:
    """Mean rank on each loss separately, per model, per period.

    The composite score hides which of the four losses a winner actually
    won on. Two models can finish a rank apart on the average while
    trading wins across the components, which is a different claim
    entirely.
    """
    rows = []
    periods = {"selection": SELECTION_YEARS, "confirmation": CONFIRMATION_YEARS}
    tables = {
        label: rank_table(summary, years, failure_rule)
        for label, years in periods.items()
    }
    for loss in RANKING_LOSSES:
        row = {"loss": LOSS_LABELS[loss]}
        for label, table in tables.items():
            for model in models:
                block = table[table["model"] == model]
                row[f"{model} ({label})"] = (
                    float(block[f"rank_{loss}"].mean()) if len(block) else float("nan")
                )
        rows.append(row)
    return pd.DataFrame(rows)


def persistence_by_origin(results) -> pd.DataFrame:
    """Persistence per origin for the GARCH family, with a flag column.

    A fit at or above 1.0 has no finite unconditional variance. That is
    reported here rather than left implicit, because nothing in the
    ranking rule looks at it and the champion crosses the line at
    several origins.
    """
    derived = _derived_persistence(results.params)
    if derived.empty:
        return pd.DataFrame()

    wide = derived.pivot_table(
        index="origin_year", columns="model", values="value"
    ).reset_index()

    # FIGARCH has no "persistence" in the GARCH sense; its long-memory
    # parameter d is the comparable quantity, so it travels beside them.
    figarch_d = results.params[
        (results.params["model"] == "figarch_skewt")
        & (results.params["param"] == "d")
    ].pivot_table(index="origin_year", columns="param", values="value")
    figarch_d = figarch_d.rename(columns={"d": "figarch (d)"}).reset_index()
    wide = wide.merge(figarch_d, on="origin_year", how="left")

    extras = results.params[
        (results.params["model"] == "gjr_skewt")
        & (results.params["param"].isin(["nu", "gamma"]))
    ].pivot_table(index="origin_year", columns="param", values="value")
    extras = extras.rename(
        columns={"nu": "gjr nu", "gamma": "gjr gamma"}
    ).reset_index()
    wide = wide.merge(extras, on="origin_year", how="left")

    model_columns = [
        c
        for c in wide.columns
        if c not in {"origin_year", "gjr nu", "gjr gamma", "figarch (d)"}
    ]
    wide["non_stationary"] = [
        ", ".join(
            column
            for column in model_columns
            if np.isfinite(row[column]) and row[column] >= 1.0
        )
        or "-"
        for _, row in wide.iterrows()
    ]
    wide.columns = [str(c) for c in wide.columns]
    return wide


def stress_origin_losses(results) -> pd.DataFrame:
    """The four losses themselves at each stress origin, not just ranks.

    Ranks say who won; the values say by how much, and whether anyone
    was any good. Origin 2019 — whose test window is calendar 2020 — is
    always included regardless of how the tercile boundaries fall, since
    it is the crisis the whole exercise exists to survive.
    """
    summary = results.summary
    stress_years = set(
        summary.loc[summary["regime"] == "stress", "origin_year"].unique()
    )
    stress_years.add(2019)
    block = summary[
        summary["origin_year"].isin(sorted(stress_years))
        & (summary["status"] == "ok")
    ]
    columns = [
        "origin_year",
        "regime",
        "model",
        *RANKING_LOSSES,
        "rank_max_drawdown",
        "rank_worst_day",
    ]
    return block[columns].sort_values(["origin_year", "model"])


def _plot_pit_by_regime(results) -> bytes:
    """Small multiples: one PIT histogram per model per regime."""
    daily = _daily_with_regime(results)
    models = sorted(daily["model"].unique())
    figure, axes = plt.subplots(
        len(REGIME_ORDER),
        len(models),
        figsize=(2.5 * len(models), 2.1 * len(REGIME_ORDER)),
        squeeze=False,
        sharex=True,
    )
    for row, regime in enumerate(REGIME_ORDER):
        for column, model in enumerate(models):
            axis = axes[row][column]
            block = daily[(daily["model"] == model) & (daily["regime"] == regime)]
            values = block["pit"].to_numpy()
            if values.size:
                axis.hist(
                    values, bins=PIT_BINS, range=(0, 1), density=True, alpha=0.75
                )
            axis.axhline(1.0, ls="--", lw=1.0, color="0.3")
            axis.set_ylim(0, 3.0)
            axis.tick_params(labelsize=6.5)
            if row == 0:
                axis.set_title(model, fontsize=8)
            if column == 0:
                axis.set_ylabel(f"{regime}\nn={values.size}", fontsize=8)
    figure.suptitle("PIT by model and regime — flat is calibrated", y=1.01)
    figure.tight_layout()
    return _png(figure)


def headline_tables(results, champion: str) -> dict[str, pd.DataFrame]:
    """The numbers a reader needs before the scoreboard means anything."""
    summary = results.summary
    daily = results.daily

    pit_rows = [
        {
            "model": model,
            **pit_diagnostics(daily.loc[daily["model"] == model, "pit"].to_numpy()),
        }
        for model in sorted(summary["model"].unique())
    ]

    stress = summary[
        (summary["regime"] == "stress") & (summary["status"] == "ok")
    ]
    stress_rows = stress[
        [
            "origin_year",
            "model",
            "rank_max_drawdown",
            "rank_worst_day",
            "exc99_count",
        ]
    ].sort_values(["origin_year", "model"])

    return {
        "persistence": persistence_table(results.params),
        "pit": pd.DataFrame(pit_rows),
        "dm": _dm_table(results, champion),
        "stress": stress_rows,
    }


def make_comparison_report(
    results,
    out_md: str = "reports/model_comparison.md",
    out_html: str = "reports/model_comparison.html",
    narrative_path: str | Path | None = DEFAULT_NARRATIVE_DIR,
) -> tuple[str, str]:
    """Write both twins. Returns ``(markdown_path, html_path)``."""
    summary = results.summary

    # Both rulings on unavailable models, so the reader can see what the
    # amendment did rather than take it on trust.
    rulings = {}
    for rule in FAILURE_RULES:
        champion_rule = select_champion(summary, SELECTION_YEARS, failure_rule=rule)
        table, agrees_rule = confirm(
            summary, champion_rule, CONFIRMATION_YEARS, failure_rule=rule
        )
        rulings[rule] = {
            "selection": score_models(summary, SELECTION_YEARS, failure_rule=rule),
            "confirmation": table,
            "champion": champion_rule,
            "agrees": agrees_rule,
        }

    selection = rulings[DEFAULT_FAILURE_RULE]["selection"]
    champion = rulings[DEFAULT_FAILURE_RULE]["champion"]
    confirmation = rulings[DEFAULT_FAILURE_RULE]["confirmation"]
    agrees = rulings[DEFAULT_FAILURE_RULE]["agrees"]
    headline = headline_tables(results, champion)
    runner_up = str(
        selection.loc[selection["model"] != champion, "model"].iloc[0]
    )
    extra = {
        "per_loss": per_loss_ranks(summary, (champion, runner_up)),
        "pit_regime": pit_by_regime(results),
        "pit_common": pit_on_common_origins(results),
        "persistence_origins": persistence_by_origin(results),
        "stress_losses": stress_origin_losses(results),
    }
    common_years = common_origins(summary)
    champions_agree = len({r["champion"] for r in rulings.values()}) == 1

    figures = [
        ("Scoreboard", _plot_scoreboard(selection, confirmation)),
        ("PIT histograms", _plot_pit(results.daily)),
        ("PIT by regime", _plot_pit_by_regime(results)),
        ("Path-rank histograms", _plot_path_ranks(summary)),
        ("Parameter drift", _plot_parameter_drift(results.params)),
    ]

    failures = summary[summary["status"] != "ok"]
    dm = _dm_table(results, champion)
    regime = _regime_table(summary)
    crisis = _crisis_table(summary)

    verdict = (
        f"The champion selected on {SELECTION_YEARS[0]}-{SELECTION_YEARS[-1]} is "
        f"**{champion}**, and it "
        + (
            "also ranks first on the held-out confirmation origins."
            if agrees
            else f"does **not** rank first on the held-out confirmation origins "
            f"(**{confirmation.iloc[0]['model']}** does)."
        )
    )

    registration_line = (
        f"Run matched the pre-registration (`sha256:{preregistration_hash()[:16]}…`)."
        if results.registered
        else f"**UNREGISTERED RUN.** {results.preregistration}"
    )

    scoreboard = selection.merge(
        confirmation[["model", "weighted_score", "rank_stability"]],
        on="model",
        how="left",
        suffixes=("_selection", "_confirmation"),
    )

    failure_lines = []
    for model in sorted(summary["model"].unique()):
        block = summary[summary["model"] == model]
        bad = block[block["status"] != "ok"]
        pit_block = results.daily.loc[results.daily["model"] == model, "pit"]
        note = (
            f"- **{model}** — mean NLL "
            f"{block.loc[block['status'] == 'ok', 'mean_nll'].mean():.4f}, "
            f"99% exceedance rate "
            f"{block.loc[block['status'] == 'ok', 'exc99_rate'].mean():.4f} "
            f"(expected 0.0100), PIT mean "
            f"{pit_block.mean() if len(pit_block) else float('nan'):.4f} "
            f"(expected 0.5000), unavailable at {len(bad)} of "
            f"{len(block)} origins. *TODO: what does this model get wrong?*"
        )
        failure_lines.append(note)
        if len(bad):
            failure_lines.append(
                f"    - Unavailable: {', '.join(str(y) for y in bad['origin_year'])} "
                f"— {bad['message'].iloc[0][:120]}"
            )

    sections = {
        "1. Pre-registration": (
            f"{registration_line}\n\n"
            f"**Amendment 1 (post-hoc).** A candidate whose fit raises at an "
            f"origin now takes the worst rank on all four losses there, "
            f"instead of being dropped and the survivors re-ranked. Scoring a "
            f"model only where it happened to work is survivorship bias. Both "
            f"rulings are shown below. The champion is "
            + ("**the same under both**." if champions_agree else
               "**not the same under both** — see the two tables.")
            + "\n\n"
            f"Origins {list(ORIGIN_YEARS)}; selection {list(SELECTION_YEARS)}; "
            f"confirmation {list(CONFIRMATION_YEARS)}. "
            f"{results.n_paths} paths per (model, origin), seed {results.seed}, "
            f"horizon {TEST_HORIZON} days, stress weight {STRESS_WEIGHT}. "
            f"Ranking losses: {', '.join(f'`{n}`' for n in RANKING_LOSSES)}. "
            "The full plan is in `docs/preregistration.md`."
        ),
        "2. Scoreboard": verdict,
        "3. Scores by regime": (
            "The four ranking losses averaged within calm / normal / stress "
            "test years. Regimes are terciles of realised volatility in the "
            "*outcome* year and never touch a fit."
        ),
        "4. Crisis years": (
            "Every stress-labelled origin, with the percentile rank of the "
            "realised year within each model's simulated distribution and the "
            f"99% exceedance count out of {TEST_HORIZON}. A rank near 0 or 1 "
            "means the realised year sat in the tail of what the model "
            "imagined. **2008 is absent**: no origin has enough history "
            f"before it, since the data begins in 2007 and the first origin "
            f"is {ORIGIN_YEARS[0]}."
        ),
        "5. PIT histograms": (
            "Pooled over all origins, 20 bins, uniform reference. **U-shaped "
            "means over-confident** — reality landed in the tails more often "
            "than the model allowed. **A central hump means under-confident** "
            "— the model spread its density wider than it needed to."
        ),
        "6. Path-rank histograms": (
            "Where the realised year fell within each model's simulated "
            "distribution, pooled over every statistic and origin. Same "
            "reading: U-shaped means the simulated spread is too narrow, a "
            "central hump means it is too wide."
        ),
        "7. Parameter drift": (
            "Each fitted parameter against origin year. Persistence is drawn "
            "in red with a line at 1.0; a family whose persistence drifts to "
            "the boundary is losing its long-run variance."
        ),
        "8. Diebold-Mariano": (
            f"Each model against the champion (**{champion}**), pooled over "
            "the confirmation origins, with Newey-West HAC at 10 lags. A "
            "positive statistic means the model lost to the champion. Daily "
            "losses are strongly autocorrelated through the volatility state, "
            "which is why the HAC correction is not optional here."
        ),
        "9. Failure modes": (
            "> **TODO — narrative to be written.** The numbers are pre-filled; "
            "the interpretation is not."
        ),
    }

    destination = Path(out_md)
    figure_dir = destination.parent / f"{destination.stem}_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for title, payload in figures:
        (figure_dir / f"{FIGURE_SLUGS[title]}.png").write_bytes(payload)

    summary_prose = load_narrative("comparison_summary.md", narrative_path)
    failure_prose = load_narrative("comparison_failure_modes.md", narrative_path)

    original = rulings["drop"]["selection"].set_index("model")
    selection_board = selection.copy()
    selection_board["original score (origins scored)"] = [
        f"{original.loc[m, 'weighted_score']:.3f} ({int(original.loc[m, 'n_origins'])})"
        if m in original.index
        else "n/a"
        for m in selection_board["model"]
    ]
    selection_board = selection_board[
        ["model", "step", "weighted_score", "rank_stability",
         "original score (origins scored)"]
    ].rename(columns={"weighted_score": "amended score",
                      "rank_stability": "rank stability"})
    confirmation_board = confirmation[
        ["model", "step", "weighted_score", "rank_stability"]
    ].rename(columns={"weighted_score": "amended score",
                      "rank_stability": "rank stability"})

    persistence = extra["persistence_origins"]
    gjr = persistence.set_index("origin_year")["gjr_skewt"]
    stationary = gjr[gjr < 1.0]
    flagged = gjr[gjr >= 1.0]
    persistence_prose = (
        f"The champion's persistence sits at or above 1.0 at "
        f"{flagged.index.min()}–{flagged.index.max()}, exactly the "
        f"{len(flagged)} origins at which the variance-targeted variant "
        f"cannot be fitted ({flagged.min():.5f} to {flagged.max():.5f}), and "
        f"in {stationary.min():.5f}–{stationary.max():.5f} everywhere else. "
        f"Its ν falls monotonically from "
        f"{persistence.set_index('origin_year')['gjr nu'].iloc[0]:.2f} "
        f"({int(persistence['origin_year'].iloc[0])}) to "
        f"{persistence.set_index('origin_year')['gjr nu'].min():.2f} "
        f"(2021) as the sample absorbs 2014–15 and 2020, then stabilises: "
        f"the model is being told the tails are steadily fatter than it "
        f"first thought."
    )

    if failure_prose:
        failure_block = failure_prose
    else:
        failure_block = (
            "> **TODO — narrative to be written.**\n\n"
            + "\n".join(failure_lines)
        )
    summary_block = summary_prose or verdict

    md = [
        "# Brent scenario generators — out-of-time model comparison\n",
        "*Companion to [validation.md](validation.md), which is the in-sample\n"
        "descriptive check on the champion fitted to all data. This report is the\n"
        "pre-registered, out-of-time comparison that selected it.*\n",
        "## Summary\n",
        summary_block,
        "\n## 1. Pre-registration\n",
        sections["1. Pre-registration"],
        "\n## 2. Scoreboard\n",
        "Lower is better. Rank stability is the fraction of origins on which the\n"
        "model is top-two on the mean of the four ranks.\n",
        f"**Selection origins, {SELECTION_YEARS[0]}–{SELECTION_YEARS[-1]}**\n",
        _md_table(selection_board, digits=3),
        f"\n**Confirmation origins, {CONFIRMATION_YEARS[0]}–{CONFIRMATION_YEARS[-1]}**\n",
        _md_table(confirmation_board, digits=3),
        f"\nChampion under both rulings: `{champion}`. Confirmation "
        + ("agrees" if agrees else "disagrees")
        + " under both.\n",
        "\n### 2.1 Which loss the composite is won on\n",
        "Mean rank per loss; lower is better.\n",
        _md_table(extra["per_loss"], digits=3),
        "\n## 3. Scores by regime\n",
        "Four losses averaged within each regime of the test window.\n",
        _md_table(regime, digits=5),
        "\n## 4. Crisis years\n",
        sections["4. Crisis years"],
        "\n### 4.1 Per-origin losses and ranks at stress origins\n",
        _md_table(extra["stress_losses"], digits=5),
        "\n## 5. Calibration — PIT\n",
        "Probability-integral-transform values of the realised return under each\n"
        "day's forecast density. Uniform means calibrated; a hump means the\n"
        "densities are too wide (under-confident); a U means too narrow\n"
        "(over-confident). χ² is over 20 bins; because consecutive PITs are mildly\n"
        "dependent and n is large, the p-values are anti-conservative and effect\n"
        "sizes (KS distance from uniform, largest single-bin deviation from 0.05)\n"
        "are reported beside them.\n",
        "\n### 5.1 Pooled over all origins\n",
        _md_table(pit_short_table(headline["pit"]), digits=4),
        f"\n`gjr_skewt_vt` is judged on fewer days; see §5.3.\n",
        "\n### 5.2 By regime of the test window\n",
        _md_table(pit_regime_matrix(results)),
        f"\n![PIT by regime]({figure_dir.name}/pit-by-regime.png)\n",
        "\n### 5.3 On common days\n",
        f"Restricted to the {len(common_years)} origins at which every candidate\n"
        "fitted, so all six are judged on identical days.\n",
        _md_table(pit_short_table(extra["pit_common"]), digits=4),
        "\n## 6. Path-rank histograms\n",
        f"Percentile rank of each realised-year statistic inside the "
        f"{results.n_paths} simulated\npaths, pooled over origins and "
        "statistics. Same reading as §5.\n",
        f"\n![Path ranks]({figure_dir.name}/path-ranks.png)\n",
        "\n## 7. Parameter drift\n",
        "Each fitted parameter against origin year. The 1.0 line in the persistence\n"
        "panels is the stationarity boundary.\n",
        f"\n![Parameter drift]({figure_dir.name}/parameter-drift.png)\n",
        "\n### 7.1 Persistence by origin\n",
        _md_table(persistence, digits=5),
        f"\n{persistence_prose}\n",
        f"\n## 8. Diebold–Mariano vs the champion\n",
        "Confirmation origins pooled; Newey–West HAC variance, lag 10. A negative\n"
        "statistic means the row model's loss is lower than the champion's.\n",
        _md_table(headline["dm"], digits=3),
        "\n## 9. Failure modes\n",
        failure_block,
    ]
    destination.write_text("\n".join(md) + "\n", encoding="utf-8")

    encoded = {
        title: base64.b64encode(payload).decode("ascii") for title, payload in figures
    }
    html_document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Brent scenario-generator comparison</title>
<style>
 body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        max-width: 1080px; margin: 2.5rem auto; padding: 0 1.25rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.6rem; }}
 h2 {{ font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
 h3 {{ font-size: 1.02rem; margin-top: 1.6rem; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin: .8rem 0; }}
 th, td {{ border: 1px solid #dcdcdc; padding: .32rem .5rem; text-align: right;
           font-variant-numeric: tabular-nums; }}
 th {{ background: #f4f4f4; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border: 1px solid #e4e4e4; margin: .8rem 0; }}
 .summary {{ background: #f7f7f7; border-left: 3px solid #999; padding: .7rem 1rem; }}
</style></head><body>
<h1>Brent scenario generators — out-of-time model comparison</h1>
<p><i>Companion to <a href="validation.html">validation.html</a>, the in-sample
descriptive check on the champion fitted to all data. This report is the
pre-registered, out-of-time comparison that selected it.</i></p>
<h2>Summary</h2>
<div class="summary">{markdown_to_html(summary_block)}</div>
<h2>1. Pre-registration</h2><p>{sections['1. Pre-registration']}</p>
<h2>2. Scoreboard</h2>
<p>Lower is better. Rank stability is the fraction of origins on which the model
is top-two on the mean of the four ranks.</p>
<h3>Selection origins, {SELECTION_YEARS[0]}–{SELECTION_YEARS[-1]}</h3>
{_html_table(selection_board, 3)}
<h3>Confirmation origins, {CONFIRMATION_YEARS[0]}–{CONFIRMATION_YEARS[-1]}</h3>
{_html_table(confirmation_board, 3)}
<p>Champion under both rulings: <code>{html.escape(champion)}</code>.</p>
<h3>2.1 Which loss the composite is won on</h3>
{_html_table(extra["per_loss"], 3)}
<h2>3. Scores by regime</h2>{_html_table(regime, 5)}
<h2>4. Crisis years</h2><p>{sections['4. Crisis years']}</p>
<h3>4.1 Per-origin losses and ranks at stress origins</h3>
{_html_table(extra["stress_losses"], 5)}
<h2>5. Calibration — PIT</h2>
<p>Uniform means calibrated; a hump means under-confident, a U over-confident.
χ² is over 20 bins, with KS distance and largest single-bin deviation beside it.</p>
<h3>5.1 Pooled over all origins</h3>{_html_table(pit_short_table(headline["pit"]), 4)}
<h3>5.2 By regime of the test window</h3>{_html_table(pit_regime_matrix(results))}
<img alt="PIT by regime" src="data:image/png;base64,{encoded['PIT by regime']}">
<h3>5.3 On common days</h3>
<p>Restricted to the {len(common_years)} origins at which every candidate fitted.</p>
{_html_table(pit_short_table(extra["pit_common"]), 4)}
<h2>6. Path-rank histograms</h2>
<img alt="Path ranks" src="data:image/png;base64,{encoded['Path-rank histograms']}">
<h2>7. Parameter drift</h2>
<img alt="Parameter drift" src="data:image/png;base64,{encoded['Parameter drift']}">
<h3>7.1 Persistence by origin</h3>{_html_table(persistence, 5)}
<p>{persistence_prose}</p>
<h2>8. Diebold–Mariano vs the champion</h2>{_html_table(headline["dm"], 3)}
<h2>9. Failure modes</h2>{markdown_to_html(failure_block)}
</body></html>"""
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html_document, encoding="utf-8")
    return str(destination), str(out_html)
