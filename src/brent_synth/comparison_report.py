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
    CONFIRMATION_YEARS,
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
from brent_synth.scoring import diebold_mariano  # noqa: E402

FIGURE_SLUGS = {
    "Scoreboard": "scoreboard",
    "PIT histograms": "pit-histograms",
    "Path-rank histograms": "path-rank-histograms",
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


def make_comparison_report(
    results,
    out_md: str = "reports/model_comparison.md",
    out_html: str = "reports/model_comparison.html",
) -> tuple[str, str]:
    """Write both twins. Returns ``(markdown_path, html_path)``."""
    summary = results.summary
    selection = score_models(summary, SELECTION_YEARS)
    champion = select_champion(summary, SELECTION_YEARS)
    confirmation, agrees = confirm(summary, champion, CONFIRMATION_YEARS)

    figures = [
        ("Scoreboard", _plot_scoreboard(selection, confirmation)),
        ("PIT histograms", _plot_pit(results.daily)),
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

    md = [
        "# Brent scenario-generator comparison\n",
        "*In-sample descriptive checks on the champion fitted to all data are "
        "in [validation.md](validation.md).*\n",
        "## 1. Pre-registration\n",
        sections["1. Pre-registration"],
    ]
    md.append(f"\n## 2. Scoreboard\n\n{sections['2. Scoreboard']}\n")
    md.append(_md_table(scoreboard))
    md.append(f"\n![Scoreboard]({figure_dir.name}/scoreboard.png)\n")
    md.append(f"\n## 3. Scores by regime\n\n{sections['3. Scores by regime']}\n")
    md.append(_md_table(regime))
    md.append(f"\n## 4. Crisis years\n\n{sections['4. Crisis years']}\n")
    md.append(_md_table(crisis))
    md.append(f"\n## 5. PIT histograms\n\n{sections['5. PIT histograms']}\n")
    md.append(f"\n![PIT histograms]({figure_dir.name}/pit-histograms.png)\n")
    md.append(
        f"\n## 6. Path-rank histograms\n\n{sections['6. Path-rank histograms']}\n"
    )
    md.append(
        f"\n![Path-rank histograms]({figure_dir.name}/path-rank-histograms.png)\n"
    )
    md.append(f"\n## 7. Parameter drift\n\n{sections['7. Parameter drift']}\n")
    md.append(f"\n![Parameter drift]({figure_dir.name}/parameter-drift.png)\n")
    md.append(f"\n## 8. Diebold-Mariano\n\n{sections['8. Diebold-Mariano']}\n")
    md.append(_md_table(dm))
    md.append(f"\n## 9. Failure modes\n\n{sections['9. Failure modes']}\n")
    md.extend(failure_lines)
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
 table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin: .8rem 0; }}
 th, td {{ border: 1px solid #dcdcdc; padding: .32rem .5rem; text-align: right;
           font-variant-numeric: tabular-nums; }}
 th {{ background: #f4f4f4; }}
 td:first-child, th:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border: 1px solid #e4e4e4; margin: .8rem 0; }}
 .summary {{ background: #f7f7f7; border-left: 3px solid #999; padding: .7rem 1rem; }}
 .todo {{ background: #fffbe6; border-left: 3px solid #d9a400; padding: .7rem 1rem; }}
</style></head><body>
<h1>Brent scenario-generator comparison</h1>
<p><i>In-sample descriptive checks on the champion fitted to all data are in
<a href="validation.html">validation.html</a>.</i></p>
<p class="summary">{sections['2. Scoreboard']}</p>
<h2>1. Pre-registration</h2><p>{sections['1. Pre-registration']}</p>
<h2>2. Scoreboard</h2>{_html_table(scoreboard)}
<img alt="Scoreboard" src="data:image/png;base64,{encoded['Scoreboard']}">
<h2>3. Scores by regime</h2><p>{sections['3. Scores by regime']}</p>{_html_table(regime)}
<h2>4. Crisis years</h2><p>{sections['4. Crisis years']}</p>{_html_table(crisis)}
<h2>5. PIT histograms</h2><p>{sections['5. PIT histograms']}</p>
<img alt="PIT histograms" src="data:image/png;base64,{encoded['PIT histograms']}">
<h2>6. Path-rank histograms</h2><p>{sections['6. Path-rank histograms']}</p>
<img alt="Path-rank histograms" src="data:image/png;base64,{encoded['Path-rank histograms']}">
<h2>7. Parameter drift</h2><p>{sections['7. Parameter drift']}</p>
<img alt="Parameter drift" src="data:image/png;base64,{encoded['Parameter drift']}">
<h2>8. Diebold-Mariano</h2><p>{sections['8. Diebold-Mariano']}</p>{_html_table(dm)}
<h2>9. Failure modes</h2><div class="todo">{sections['9. Failure modes']}
<ul>{''.join(f'<li>{html.escape(line.lstrip("- ").lstrip())}</li>' for line in failure_lines)}</ul></div>
</body></html>"""
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    Path(out_html).write_text(html_document, encoding="utf-8")
    return str(destination), str(out_html)
