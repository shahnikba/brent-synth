"""Validate synthetic paths against real Brent returns.

The question this module answers is not "are the synthetic statistics
close to the real ones" — that has no natural threshold. It is "could
the real market plausibly have produced the synthetic numbers". Real
Brent is one realisation of 4754 days, so its own sample statistics are
random; a 252-day window of it lands somewhere in a distribution. That
distribution *is* the acceptance band.

So the reference is a stationary block bootstrap of the real returns,
resampled at exactly the horizon the synthetic paths run for, and a
synthetic statistic passes when it falls inside the real 2.5-97.5%
band. Comparing a 252-day synthetic path against statistics measured
over the full 4754-day history would be a category error: the band
would be far too tight and everything would fail.

The mirror of that error is just as easy to commit, and this module
used to commit it. Pooling every synthetic day into one 1,260,000-point
sample and comparing *that* against a band built from 252-day samples
is not a like-for-like comparison either: for anything nonlinear in the
sample the two are different quantities. Pooling mixes paths whose
realised volatility spans 37x, and a mixture across vol levels is far
more leptokurtic than any single path — pooled excess kurtosis read
99.6 where the per-path median was 2.0, and between-path level
differences inflated the ACF of squared returns roughly fourfold, on
top of splicing n_paths - 1 false lag-1 adjacencies at the joins.

So **every statistic is computed per path, on exactly `horizon`
observations, on both sides**. The synthetic side is a distribution
over paths; the real side is a distribution over bootstrap samples.
That symmetry is the whole design, and it is what
:func:`compute_path_stats` enforces.

Location and dispersion are reported separately. A pooled point
estimate had no sampling noise, so the test could only ever ask whether
the model's central value landed in the band — it had no power to
detect wrong spread. Comparing the two distributions catches that:
Brent's synthetic paths sit correctly on the median but over-disperse
volatility, which only the dispersion column shows.

Sign conventions
----------------
Returns stay signed throughout. VaR and ES are reported as **positive
loss magnitudes** (VaR_95 = 0.04 means a 4% loss), while the raw tail
quantiles ``left_q01`` and ``left_q05`` stay negative, being quantiles
of the signed return. ``max_drawdown`` is a positive fraction.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: the report is written, never displayed.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402
from statsmodels.tsa.stattools import acf  # noqa: E402

from brent_synth.data import load_returns  # noqa: E402
from brent_synth.model import ModelFit, fit as fit_model, simulate  # noqa: E402

DEFAULT_HORIZON = 252
DEFAULT_N_PATHS = 5000
DEFAULT_N_BOOT = 2000

#: Shortest horizon every statistic is defined at: the lag-10 ACF of
#: squared returns needs 11 observations.
MIN_PERIODS = 11

#: Expected block length for the stationary bootstrap, in trading days.
#: Volatility clustering in oil runs over weeks, so blocks must be long
#: enough to carry a cluster through the resample: an iid bootstrap would
#: destroy exactly the dependence being tested and hand back a band that
#: wrongly rejects any clustering model. Roughly a trading month is the
#: usual choice. Band widths on the ACF statistics are mildly sensitive
#: to this — shorter blocks narrow them, longer blocks widen them — so it
#: is a reported parameter, not a hidden constant.
DEFAULT_BLOCK = 20

#: Report order. Grouped: moments, then risk measures, then raw tail
#: quantiles, then clustering, then the path functional.
STAT_NAMES = (
    "mean",
    "std",
    "skew",
    "excess_kurtosis",
    "var_95",
    "var_99",
    "es_95",
    "es_99",
    "left_q01",
    "left_q05",
    "right_q95",
    "right_q99",
    "acf_sq_lag1",
    "acf_sq_lag5",
    "acf_sq_lag10",
    "max_drawdown",
)

#: var_95 and var_99 are the left tail quantiles with the sign flipped:
#: var_95 == -left_q05 and var_99 == -left_q01, bit-for-bit, by
#: construction. Both names are reported because both are conventional,
#: but they are one check each, not two, and the headline pass count
#: must not double-count tail agreement.
DUPLICATE_STATS = {"var_95": "left_q05", "var_99": "left_q01"}

#: The statistics that are independent checks of each other.
INDEPENDENT_STATS = tuple(n for n in STAT_NAMES if n not in DUPLICATE_STATS)

#: Acceptable ratio of synthetic to real inter-quantile width. A model
#: can sit perfectly on the median and still generate paths half or
#: twice as variable as the market; this is the check for that. The
#: bounds are deliberately loose - both sides are themselves estimated
#: from finite samples - so a flag here means a gross mismatch of
#: spread, not a marginal one.
DISPERSION_BOUNDS = (0.5, 2.0)

STAT_LABELS = {
    "mean": "Mean daily return",
    "std": "Daily volatility",
    "skew": "Skewness",
    "excess_kurtosis": "Excess kurtosis",
    "var_95": "VaR 95% (loss)",
    "var_99": "VaR 99% (loss)",
    "es_95": "ES 95% (loss)",
    "es_99": "ES 99% (loss)",
    "left_q01": "1st percentile return",
    "left_q05": "5th percentile return",
    "right_q95": "95th percentile return",
    "right_q99": "99th percentile return",
    "acf_sq_lag1": "ACF of r², lag 1",
    "acf_sq_lag5": "ACF of r², lag 5",
    "acf_sq_lag10": "ACF of r², lag 10",
    "max_drawdown": "Max drawdown (median)",
}


def _as_paths(paths_or_sample: np.ndarray | pd.Series) -> np.ndarray:
    """Coerce to a 2-D (n_paths, horizon) array.

    A 1-D bootstrap sample is one path, so both sides of the comparison
    run through identical code and cannot drift apart.
    """
    arr = np.asarray(paths_or_sample, dtype="float64")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D array, got shape {arr.shape}.")
    # acf_sq_lag10 is a required statistic and needs 11 points to exist.
    if arr.shape[1] < MIN_PERIODS:
        raise ValueError(
            f"Need at least {MIN_PERIODS} periods per path to compute every "
            f"statistic (the lag-10 ACF), got {arr.shape[1]}."
        )
    if not np.isfinite(arr).all():
        raise ValueError("Input contains non-finite values.")
    return arr


def _max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Max peak-to-trough drawdown per path, as a positive fraction.

    Works on the cumulative-return curve with a zero prepended, so a
    drawdown that starts on day 1 is counted.
    """
    cumulative = np.cumsum(paths, axis=1)
    cumulative = np.concatenate(
        [np.zeros((cumulative.shape[0], 1)), cumulative], axis=1
    )
    running_peak = np.maximum.accumulate(cumulative, axis=1)
    drawdown = 1.0 - np.exp(cumulative - running_peak)
    return drawdown.max(axis=1)


def _acf_at_lags(paths: np.ndarray, lags: tuple[int, ...]) -> dict[int, np.ndarray]:
    """ACF of each row at the given lags, vectorised across rows.

    The same biased estimator statsmodels uses by default: the
    numerator sums over the overlapping span, the denominator over the
    whole series. Computed row-wise rather than on a concatenation,
    which is the point — splicing rows together would manufacture
    adjacencies that do not exist.
    """
    centred = paths - paths.mean(axis=1, keepdims=True)
    denominator = (centred**2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            lag: np.where(
                denominator > 0.0,
                (centred[:, lag:] * centred[:, :-lag]).sum(axis=1) / denominator,
                0.0,
            )
            for lag in lags
        }


def _mean_beyond(paths: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """Row-wise mean of the values at or below each row's cutoff."""
    mask = paths <= cutoff[:, None]
    counts = mask.sum(axis=1)
    totals = np.where(mask, paths, 0.0).sum(axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), cutoff)


def compute_path_stats(paths_or_sample) -> dict[str, np.ndarray]:
    """Every statistic, computed **per path**, as arrays of length n_paths.

    This is the primitive both sides of the comparison run through: the
    synthetic paths and the bootstrap samples are both (n, horizon)
    arrays of the same horizon, and neither is ever pooled. A 1-D input
    is one path and yields length-1 arrays.
    """
    paths = _as_paths(paths_or_sample)

    quantiles = np.quantile(paths, [0.01, 0.05, 0.95, 0.99], axis=1)
    q01, q05, q95, q99 = quantiles
    acf_squared = _acf_at_lags(paths**2, (1, 5, 10))

    return {
        "mean": paths.mean(axis=1),
        "std": paths.std(axis=1, ddof=1),
        "skew": sps.skew(paths, axis=1, bias=False),
        "excess_kurtosis": sps.kurtosis(paths, axis=1, fisher=True, bias=False),
        # VaR/ES flipped to positive loss magnitudes.
        "var_95": -q05,
        "var_99": -q01,
        "es_95": -_mean_beyond(paths, q05),
        "es_99": -_mean_beyond(paths, q01),
        # Raw signed quantiles, left ones negative.
        "left_q01": q01,
        "left_q05": q05,
        "right_q95": q95,
        "right_q99": q99,
        "acf_sq_lag1": acf_squared[1],
        "acf_sq_lag5": acf_squared[5],
        "acf_sq_lag10": acf_squared[10],
        "max_drawdown": _max_drawdown(paths),
    }


def compute_stats(paths_or_sample) -> dict[str, float]:
    """Works on a (n, horizon) synthetic array OR a 1-D bootstrap sample.

    Returns the **median across paths** of each per-path statistic — the
    typical path, on the same footing as a single bootstrap sample. On a
    1-D input the median of one value is that value, so the two sides of
    the comparison genuinely share one code path and cannot drift apart.

    Nothing here pools days across paths. See the module docstring for
    why that mattered.
    """
    return {
        name: float(np.median(values))
        for name, values in compute_path_stats(paths_or_sample).items()
    }


def _stationary_bootstrap_indices(
    n: int, horizon: int, n_boot: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices, shape (n_boot, horizon).

    Each step either continues the current block (wrapping at the end of
    the series) or jumps to a fresh uniform start, with jump probability
    1/block. Block lengths are therefore geometric with mean ``block``,
    which is what keeps the resample stationary while carrying runs of
    high volatility through intact.
    """
    jump_probability = 1.0 / block
    indices = np.empty((n_boot, horizon), dtype=np.int64)
    indices[:, 0] = rng.integers(0, n, n_boot)
    jumps = rng.random((n_boot, horizon)) < jump_probability
    fresh_starts = rng.integers(0, n, (n_boot, horizon))

    for t in range(1, horizon):
        indices[:, t] = np.where(
            jumps[:, t], fresh_starts[:, t], (indices[:, t - 1] + 1) % n
        )
    return indices


def bootstrap_bands(
    returns: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    n_boot: int = DEFAULT_N_BOOT,
    block: int = DEFAULT_BLOCK,
    seed: int = 7,
) -> dict[str, tuple[float, float, float, np.ndarray]]:
    """Stationary block bootstrap.

    Draw ``n_boot`` samples of length ``horizon`` from real returns,
    blocks of expected length ``block`` to preserve clustering. For each
    sample compute every statistic. Return
    ``{stat_name: (lo2.5, median, hi97.5, full_array)}``.
    """
    values = np.asarray(returns, dtype="float64").ravel()
    n_missing = int((~np.isfinite(values)).sum())
    if n_missing:
        # A block bootstrap exists to preserve adjacency; dropping a gap
        # closes it and glues non-neighbours together in every resample
        # that spans the hole. Refused here as in diagnostics and model.
        raise ValueError(
            f"Return series contains {n_missing} non-finite value(s). "
            "The block bootstrap resamples runs of adjacent days, so the "
            "gaps cannot be dropped silently — handle them explicitly first."
        )
    if values.size < horizon:
        raise ValueError(
            f"Need at least {horizon} returns to bootstrap at that horizon, "
            f"got {values.size}."
        )
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}.")

    rng = np.random.default_rng(seed)
    indices = _stationary_bootstrap_indices(
        values.size, horizon, n_boot, block, rng
    )
    # Every replicate is a row of one (n_boot, horizon) array, so the
    # whole bootstrap distribution comes from a single vectorised pass
    # rather than n_boot separate calls.
    drawn = compute_path_stats(values[indices])

    return {
        name: (
            float(np.quantile(values_, 0.025)),
            float(np.quantile(values_, 0.5)),
            float(np.quantile(values_, 0.975)),
            values_,
        )
        for name, values_ in drawn.items()
    }


def validate(
    synth_paths: np.ndarray,
    returns: pd.Series,
    bands: dict[str, tuple[float, float, float, np.ndarray]] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """For each statistic: synthetic value, real bootstrap (lo, median,
    hi), PASS if synthetic in [lo, hi] else FAIL, and the z-like
    position (how many band-widths outside).

    ``synthetic`` is the median across synthetic paths, each measured on
    ``horizon`` days — the same footing as one bootstrap replicate.
    ``synth_lo``/``synth_hi`` give the 2.5-97.5% spread across paths, and
    ``dispersion_ratio`` compares that width to the real band's.
    A model can sit dead on the median and still be far too variable,
    which ``passed`` alone cannot see; ``dispersion_passed`` can.

    ``position`` is 0 inside the band, negative below it and positive
    above, measured in band widths — so -0.5 means the synthetic value
    sits half a band below the 2.5% bound.

    Pass ``bands`` to reuse an already-computed bootstrap; otherwise one
    is run here. Both sides are computed at the same horizon; a mismatch
    raises, since it would silently invalidate every comparison.
    """
    paths = _as_paths(synth_paths)
    horizon = kwargs.pop("horizon", paths.shape[1])
    if paths.shape[1] != horizon:
        raise ValueError(
            f"Synthetic horizon {paths.shape[1]} does not match the bootstrap "
            f"horizon {horizon}; both sides must use the same horizon."
        )

    if bands is None:
        bands = bootstrap_bands(returns, horizon=horizon, **kwargs)
    elif kwargs:
        raise ValueError(
            f"Bootstrap options {sorted(kwargs)} are ignored when `bands` is "
            "supplied; pass them to bootstrap_bands instead."
        )

    per_path = compute_path_stats(paths)

    rows = []
    for name in STAT_NAMES:
        lo, median, hi, _ = bands[name]
        draws = per_path[name]
        value = float(np.median(draws))
        synth_lo, synth_hi = (float(x) for x in np.quantile(draws, [0.025, 0.975]))

        width = hi - lo
        if lo <= value <= hi:
            position = 0.0
        elif width > 0.0:
            position = (value - lo) / width if value < lo else (value - hi) / width
        else:
            position = float("nan")

        if width > 0.0:
            ratio = (synth_hi - synth_lo) / width
            low, high = DISPERSION_BOUNDS
            dispersion_passed = bool(low <= ratio <= high)
        else:
            ratio = float("nan")
            dispersion_passed = False

        rows.append(
            {
                "statistic": name,
                "label": STAT_LABELS[name],
                "synthetic": value,
                "synth_lo": synth_lo,
                "synth_hi": synth_hi,
                "boot_lo": lo,
                "boot_median": median,
                "boot_hi": hi,
                "passed": bool(lo <= value <= hi),
                "position": position,
                "dispersion_ratio": ratio,
                "dispersion_passed": dispersion_passed,
                "independent": name not in DUPLICATE_STATS,
            }
        )

    return pd.DataFrame(rows)


# --- report ----------------------------------------------------------------


def _figure_to_png(figure: plt.Figure) -> bytes:
    """Render to PNG bytes. Metadata is stripped so runs are byte-identical."""
    buffer = io.BytesIO()
    figure.savefig(
        buffer, format="png", dpi=110, bbox_inches="tight", metadata={"Software": None}
    )
    plt.close(figure)
    return buffer.getvalue()


def _plot_histogram(real: np.ndarray, synthetic: np.ndarray) -> bytes:
    figure, axes = plt.subplots(figsize=(7.5, 4.2))
    edges = np.linspace(
        min(real.min(), np.quantile(synthetic, 0.0005)),
        max(real.max(), np.quantile(synthetic, 0.9995)),
        121,
    )
    axes.hist(real, bins=edges, density=True, alpha=0.55, label="Real Brent")
    axes.hist(
        synthetic, bins=edges, density=True, histtype="step", lw=1.6, label="Synthetic"
    )
    axes.set_yscale("log")
    axes.set_xlabel("Daily log return")
    axes.set_ylabel("Density (log scale)")
    axes.set_title("Return distribution — log density exposes the tails")
    axes.legend()
    return _figure_to_png(figure)


def _plot_qq(real: np.ndarray, synthetic: np.ndarray) -> bytes:
    probabilities = np.linspace(0.001, 0.999, 400)
    real_q = np.quantile(real, probabilities)
    synth_q = np.quantile(synthetic, probabilities)
    figure, axes = plt.subplots(figsize=(5.4, 5.2))
    axes.scatter(real_q, synth_q, s=7)
    limits = [min(real_q.min(), synth_q.min()), max(real_q.max(), synth_q.max())]
    axes.plot(limits, limits, ls="--", lw=1.2, color="0.35")
    axes.set_xlabel("Real quantile")
    axes.set_ylabel("Synthetic quantile")
    axes.set_title("QQ: synthetic vs real")
    axes.set_aspect("equal", adjustable="box")
    return _figure_to_png(figure)


def _plot_acf(real: np.ndarray, paths: np.ndarray, nlags: int = 40) -> bytes:
    real_acf = acf(real**2, nlags=nlags, fft=True)
    per_path = np.array([acf(p**2, nlags=nlags, fft=True) for p in paths[:400]])
    mean_acf = per_path.mean(axis=0)
    lags = np.arange(1, nlags + 1)

    figure, axes = plt.subplots(figsize=(7.5, 4.0))
    axes.bar(lags, real_acf[1:], width=0.75, alpha=0.55, label="Real Brent")
    axes.plot(lags, mean_acf[1:], marker="o", ms=3.2, lw=1.4, label="Synthetic mean")
    axes.axhline(0.0, lw=0.8, color="0.4")
    axes.set_xlabel("Lag (trading days)")
    axes.set_ylabel("ACF of squared returns")
    axes.set_title("Volatility clustering: decay of ACF(r²)")
    axes.legend()
    return _figure_to_png(figure)


def _plot_fan(real: np.ndarray, paths: np.ndarray) -> bytes:
    """Recent real path spliced to synthetic cones, both indexed to 100.

    The level is normalised rather than taken from actual Brent prices
    so the report needs no price data of its own.
    """
    lookback = min(252, real.size)
    recent = 100.0 * np.exp(np.cumsum(real[-lookback:]))
    recent = np.concatenate([[100.0], recent])
    recent *= 100.0 / recent[-1]  # end the history at 100, where paths start

    horizon = paths.shape[1]
    cones = 100.0 * np.exp(np.cumsum(paths, axis=1))
    future_x = np.arange(1, horizon + 1)
    past_x = np.arange(-lookback, 1)

    figure, axes = plt.subplots(figsize=(8.0, 4.4))
    axes.plot(past_x, recent, lw=1.4, color="0.2", label="Real (last year, indexed)")
    for lo, hi, alpha in ((5, 95, 0.16), (25, 75, 0.28)):
        axes.fill_between(
            future_x,
            np.percentile(cones, lo, axis=0),
            np.percentile(cones, hi, axis=0),
            alpha=alpha,
            color="tab:blue",
            lw=0,
            label=f"Synthetic {lo}-{hi}%",
        )
    axes.plot(
        future_x, np.median(cones, axis=0), lw=1.4, color="tab:blue", label="Median"
    )
    axes.axvline(0.0, ls=":", lw=1.0, color="0.5")
    axes.set_xlabel("Trading days from today")
    axes.set_ylabel("Index (today = 100)")
    axes.set_title("Fan chart: simulated paths continuing from today's vol state")
    axes.legend(fontsize=8)
    return _figure_to_png(figure)


def _plot_drawdown(boot_drawdowns: np.ndarray, paths: np.ndarray) -> bytes:
    synthetic = _max_drawdown(paths)
    figure, axes = plt.subplots(figsize=(7.5, 4.0))
    edges = np.linspace(
        0.0, max(np.quantile(boot_drawdowns, 0.999), np.quantile(synthetic, 0.999)), 60
    )
    axes.hist(
        boot_drawdowns, bins=edges, density=True, alpha=0.55, label="Real (bootstrap)"
    )
    axes.hist(
        synthetic, bins=edges, density=True, histtype="step", lw=1.6, label="Synthetic"
    )
    axes.axvline(
        np.median(boot_drawdowns), color="tab:blue", ls="--", lw=1.2, label="Real median"
    )
    axes.axvline(
        np.median(synthetic), color="tab:orange", ls="--", lw=1.2, label="Synth median"
    )
    axes.set_xlabel(f"Max drawdown over {paths.shape[1]} days")
    axes.set_ylabel("Density")
    axes.set_title("Drawdown distribution")
    axes.legend(fontsize=8)
    return _figure_to_png(figure)


FIGURE_SLUGS = {
    "Return distribution": "return-distribution",
    "QQ plot": "qq-plot",
    "ACF of squared returns": "acf-squared-returns",
    "Fan chart": "fan-chart",
    "Drawdown distribution": "drawdown-distribution",
}


def _prepare(
    returns: pd.Series,
    synth_paths: np.ndarray,
    boot_drawdowns: np.ndarray | None,
    n_boot: int,
    block: int,
    boot_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared setup for both report formats."""
    real = np.asarray(returns, dtype="float64").ravel()
    real = real[np.isfinite(real)]
    paths = _as_paths(synth_paths)

    if boot_drawdowns is None:
        bands = bootstrap_bands(
            returns,
            horizon=paths.shape[1],
            n_boot=n_boot,
            block=block,
            seed=boot_seed,
        )
        boot_drawdowns = bands["max_drawdown"][3]
    return real, paths, boot_drawdowns


def _build_figures(
    real: np.ndarray, paths: np.ndarray, boot_drawdowns: np.ndarray
) -> list[tuple[str, bytes]]:
    """The five required plots, as (title, PNG bytes)."""
    pooled = paths.ravel()
    return [
        ("Return distribution", _plot_histogram(real, pooled)),
        ("QQ plot", _plot_qq(real, pooled)),
        ("ACF of squared returns", _plot_acf(real, paths)),
        ("Fan chart", _plot_fan(real, paths)),
        ("Drawdown distribution", _plot_drawdown(boot_drawdowns, paths)),
    ]


def _summary_counts(results_df: pd.DataFrame) -> tuple[int, int, int]:
    independent = results_df.loc[results_df["independent"]]
    return (
        int(independent["passed"].sum()),
        len(independent),
        int((~independent["dispersion_passed"]).sum()),
    )


def _results_table_html(results: pd.DataFrame) -> str:
    header = (
        "<tr><th>Statistic</th><th>Synthetic median</th><th>Synthetic 2.5-97.5%</th>"
        "<th>Real 2.5%</th><th>Real median</th><th>Real 97.5%</th>"
        "<th>Position</th><th>Spread</th><th>Result</th></tr>"
    )
    rows = []
    for row in results.itertuples():
        verdict = "PASS" if row.passed else "FAIL"
        spread = "n/a" if not np.isfinite(row.dispersion_ratio) else (
            f"{row.dispersion_ratio:.2f}x"
        )
        spread_class = "ok" if row.dispersion_passed else "wide"
        label = html.escape(row.label)
        if not row.independent:
            label += " <span class='alias'>= -%s</span>" % html.escape(
                STAT_LABELS[DUPLICATE_STATS[row.statistic]]
            )
        rows.append(
            f'<tr class="{verdict.lower()}">'
            f"<td class='stat'>{label}</td>"
            f"<td>{row.synthetic:.5f}</td>"
            f"<td>{row.synth_lo:.5f} to {row.synth_hi:.5f}</td>"
            f"<td>{row.boot_lo:.5f}</td>"
            f"<td>{row.boot_median:.5f}</td>"
            f"<td>{row.boot_hi:.5f}</td>"
            f"<td>{row.position:+.2f}</td>"
            f"<td class='{spread_class}'>{spread}</td>"
            f"<td class='verdict'>{verdict}</td></tr>"
        )
    return f"<table class='results'>{header}{''.join(rows)}</table>"


def make_report(
    returns: pd.Series,
    fit: ModelFit,
    synth_paths: np.ndarray,
    results_df: pd.DataFrame,
    out_path: str = "reports/validation.html",
    seed: int | None = None,
    block: int = DEFAULT_BLOCK,
    n_boot: int = DEFAULT_N_BOOT,
    boot_seed: int = 7,
    boot_drawdowns: np.ndarray | None = None,
) -> str:
    """Self-contained HTML. Embeds plots as base64 PNGs. Returns the path.

    Deliberately carries no generation timestamp: the report must be
    byte-identical across runs at fixed seeds, and a clock would break
    that.
    """
    real, paths, boot_drawdowns = _prepare(
        returns, synth_paths, boot_drawdowns, n_boot, block, boot_seed
    )
    figures = _build_figures(real, paths, boot_drawdowns)

    independent = results_df.loc[results_df["independent"]]
    n_pass, n_total, n_wide = _summary_counts(results_df)
    failures = results_df.loc[~results_df["passed"] & results_df["independent"]]
    wide = independent.loc[~independent["dispersion_passed"]]

    params_rows = "".join(
        f"<tr><td class='stat'>{html.escape(name)}</td><td>{value:.6g}</td></tr>"
        for name, value in fit.params.items()
    )

    wide_lines = "".join(
        f"<li><b>{html.escape(row.label)}</b> — synthetic paths span "
        f"{row.synth_lo:.5f} to {row.synth_hi:.5f}, "
        f"{row.dispersion_ratio:.2f}x the real band "
        f"[{row.boot_lo:.5f}, {row.boot_hi:.5f}]."
        " <i>TODO: is this over- or under-dispersion the model's doing?</i></li>"
        for row in wide.itertuples()
    )

    failure_lines = (
        "".join(
            f"<li><b>{html.escape(row.label)}</b> — synthetic "
            f"{row.synthetic:.5f} vs real band "
            f"[{row.boot_lo:.5f}, {row.boot_hi:.5f}], "
            f"{abs(row.position):.2f} band-widths "
            f"{'below' if row.position < 0 else 'above'}."
            " <i>TODO: explain why the model does this.</i></li>"
            for row in failures.itertuples()
        )
        or "<li>No statistic fell outside its band. <i>TODO: say whether "
        "that is genuine agreement or a band wide enough to hide a "
        "problem.</i></li>"
    )

    figure_blocks = "".join(
        f"<figure><img alt='{html.escape(title)}' "
        f"src='data:image/png;base64,{base64.b64encode(payload).decode('ascii')}'>"
        f"<figcaption>{html.escape(title)}</figcaption></figure>"
        for title, payload in figures
    )

    seed_text = "not recorded" if seed is None else str(seed)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Brent synthetic-path validation</title>
<style>
 body {{ font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        max-width: 980px; margin: 2.5rem auto; padding: 0 1.25rem; color: #1a1a1a; }}
 h1 {{ font-size: 1.6rem; margin-bottom: .2rem; }}
 h2 {{ font-size: 1.15rem; margin-top: 2.2rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: .8rem; }}
 th, td {{ border: 1px solid #dcdcdc; padding: .38rem .55rem; text-align: right;
           font-variant-numeric: tabular-nums; }}
 th {{ background: #f4f4f4; text-align: right; font-weight: 600; }}
 td.stat, th:first-child {{ text-align: left; }}
 td.verdict {{ font-weight: 700; }}
 tr.pass td.verdict {{ color: #14691f; }}
 tr.fail td.verdict {{ color: #a11414; }}
 tr.fail {{ background: #fdf3f3; }}
 figure {{ margin: 1.6rem 0; }}
 img {{ max-width: 100%; border: 1px solid #e4e4e4; }}
 figcaption {{ font-size: 12.5px; color: #666; margin-top: .35rem; }}
 .summary {{ background: #f7f7f7; border-left: 3px solid #999; padding: .7rem 1rem; }}
 .alias {{ color: #888; font-weight: 400; font-size: 11.5px; }}
 td.wide {{ color: #a11414; font-weight: 600; }}
 .todo {{ background: #fffbe6; border-left: 3px solid #d9a400; padding: .7rem 1rem; }}
 code {{ background: #f2f2f2; padding: .1rem .3rem; }}
</style></head><body>
<h1>Brent synthetic-path validation</h1>
<p><i>The in-sample descriptive check on the champion fitted to all data. The
out-of-time comparison that selected it is in
<a href="model_comparison.html">model_comparison.html</a>.</i></p>
<p class="summary"><b>{n_pass} of {n_total}</b> independent statistics fall inside
the real bootstrap band, and <b>{n_wide}</b> show a spread outside
{DISPERSION_BOUNDS[0]:g}-{DISPERSION_BOUNDS[1]:g}x the real one. Synthetic paths come
from GJR-GARCH(1,1,1) with skewed-t innovations; the acceptance band is a
stationary block bootstrap of the real returns at the same horizon. Every
statistic is measured per path on {paths.shape[1]} days, on both sides — VaR
rows are the signed tail quantiles under another name and are excluded from
the count above.</p>

<h2>1. Run configuration</h2>
<table>
<tr><td class="stat">Real observations</td><td>{real.size}</td></tr>
<tr><td class="stat">Horizon (days)</td><td>{paths.shape[1]}</td></tr>
<tr><td class="stat">Synthetic paths</td><td>{paths.shape[0]}</td></tr>
<tr><td class="stat">Bootstrap replicates</td><td>{n_boot}</td></tr>
<tr><td class="stat">Bootstrap block (expected days)</td><td>{block}</td></tr>
<tr><td class="stat">Simulation seed</td><td>{html.escape(seed_text)}</td></tr>
<tr><td class="stat">Bootstrap seed</td><td>{boot_seed}</td></tr>
{params_rows}
<tr><td class="stat">persistence (α + γ·E[z²1{{z&lt;0}}] + β)</td><td>{fit.persistence:.6f}</td></tr>
<tr><td class="stat">leverage weight E[z²1{{z&lt;0}}]</td><td>{fit.leverage_weight:.6f}</td></tr>
<tr><td class="stat">forecast variance σ²(T+1)</td><td>{fit.forecast_variance:.6g}</td></tr>
<tr><td class="stat">unconditional variance</td><td>{fit.unconditional_variance:.6g}</td></tr>
</table>

<h2>2. Acceptance table</h2>
<p>A statistic PASSES when the synthetic median lies inside the real
2.5–97.5% bootstrap band. <code>Position</code> is 0 inside the band, and
otherwise counts band-widths beyond the breached bound (negative below,
positive above). <code>Spread</code> is the synthetic 2.5–97.5% width over the
real one: a model can sit on the median and still be far too variable, and
only this column sees that. VaR and ES are positive loss magnitudes; the
percentile rows are signed returns. Rows marked <span class="alias">= -…</span>
are the same number as another row, reported under both conventional names and
counted once.</p>
{_results_table_html(results_df)}

<h2>3. Plots</h2>
{figure_blocks}

<h2>4. Failure modes</h2>
<div class="todo">
<p><b>TODO — narrative to be written.</b> The numbers below are pre-filled;
the interpretation is not.</p>
<ul>{failure_lines}</ul>
<p><b>Spread mismatches ({n_wide}):</b></p>
<ul>{wide_lines or "<li>Every statistic's spread sits within the bounds.</li>"}</ul>
<p><i>TODO: state whether the ACF decay mismatch is acceptable for the
intended use, and whether the block length of {block} days is doing
material work in the width of these bands.</i></p>
</div>
</body></html>"""

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return str(destination)



def _markdown_table(results: pd.DataFrame) -> str:
    header = (
        "| Statistic | Synthetic median | Synthetic 2.5-97.5% | Real 2.5% | "
        "Real median | Real 97.5% | Position | Spread | Result |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|:--|"
    )
    lines = [header]
    for row in results.itertuples():
        verdict = "PASS" if row.passed else "**FAIL**"
        spread = (
            "n/a"
            if not np.isfinite(row.dispersion_ratio)
            else f"{row.dispersion_ratio:.2f}x"
        )
        if not row.dispersion_passed and np.isfinite(row.dispersion_ratio):
            spread = f"**{spread}**"
        label = row.label
        if not row.independent:
            label += f" *(= -{STAT_LABELS[DUPLICATE_STATS[row.statistic]]})*"
        lines.append(
            f"| {label} | {row.synthetic:.5f} | "
            f"{row.synth_lo:.5f} to {row.synth_hi:.5f} | {row.boot_lo:.5f} | "
            f"{row.boot_median:.5f} | {row.boot_hi:.5f} | {row.position:+.2f} | "
            f"{spread} | {verdict} |"
        )
    return "\n".join(lines)


def make_markdown_report(
    returns: pd.Series,
    fit: ModelFit,
    synth_paths: np.ndarray,
    results_df: pd.DataFrame,
    out_path: str = "reports/validation.md",
    seed: int | None = None,
    block: int = DEFAULT_BLOCK,
    n_boot: int = DEFAULT_N_BOOT,
    boot_seed: int = 7,
    boot_drawdowns: np.ndarray | None = None,
) -> str:
    """Markdown twin of :func:`make_report`. Returns the path.

    Same content and same section order as the HTML, so the two cannot
    tell different stories. Markdown has no portable way to inline an
    image, so the five plots are written as PNG files beside the
    document — into ``<stem>_figures/`` — and linked relatively. That
    keeps the report readable anywhere markdown renders, at the cost of
    it being a directory rather than one file.

    Carries no timestamp, for the same reason the HTML does not: fixed
    seeds must give a reproducible document.
    """
    real, paths, boot_drawdowns = _prepare(
        returns, synth_paths, boot_drawdowns, n_boot, block, boot_seed
    )
    figures = _build_figures(real, paths, boot_drawdowns)

    independent = results_df.loc[results_df["independent"]]
    n_pass, n_total, n_wide = _summary_counts(results_df)
    failures = results_df.loc[~results_df["passed"] & results_df["independent"]]
    wide = independent.loc[~independent["dispersion_passed"]]

    destination = Path(out_path)
    figure_dir = destination.parent / f"{destination.stem}_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_lines = []
    for title, payload in figures:
        filename = f"{FIGURE_SLUGS[title]}.png"
        (figure_dir / filename).write_bytes(payload)
        figure_lines.append(
            f"### {title}\n\n![{title}]({figure_dir.name}/{filename})"
        )

    params_rows = "\n".join(
        f"| {name} | {value:.6g} |" for name, value in fit.params.items()
    )

    if len(failures):
        failure_lines = "\n".join(
            f"- **{row.label}** — synthetic {row.synthetic:.5f} vs real band "
            f"[{row.boot_lo:.5f}, {row.boot_hi:.5f}], "
            f"{abs(row.position):.2f} band-widths "
            f"{'below' if row.position < 0 else 'above'}. "
            "*TODO: explain why the model does this.*"
            for row in failures.itertuples()
        )
    else:
        failure_lines = (
            "- No statistic fell outside its band. *TODO: say whether that is "
            "genuine agreement or a band wide enough to hide a problem.*"
        )

    if len(wide):
        wide_lines = "\n".join(
            f"- **{row.label}** — synthetic paths span {row.synth_lo:.5f} to "
            f"{row.synth_hi:.5f}, {row.dispersion_ratio:.2f}x the real band "
            f"[{row.boot_lo:.5f}, {row.boot_hi:.5f}]. "
            "*TODO: is this over- or under-dispersion the model's doing?*"
            for row in wide.itertuples()
        )
    else:
        wide_lines = "- Every statistic's spread sits within the bounds."

    seed_text = "not recorded" if seed is None else str(seed)
    low, high = DISPERSION_BOUNDS
    document = f"""# Brent synthetic-path validation

*The in-sample descriptive check on the champion fitted to all data. The
out-of-time comparison that selected it is in
[model_comparison.md](model_comparison.md).*

**{n_pass} of {n_total}** independent statistics fall inside the real bootstrap
band, and **{n_wide}** show a spread outside {low:g}-{high:g}x the real one.

Synthetic paths come from GJR-GARCH(1,1,1) with skewed-t innovations; the
acceptance band is a stationary block bootstrap of the real returns at the same
horizon. Every statistic is measured per path on {paths.shape[1]} days, on both
sides. VaR rows are the signed tail quantiles under another name and are
excluded from the count above.

## 1. Run configuration

| Setting | Value |
|---|---:|
| Real observations | {real.size} |
| Horizon (days) | {paths.shape[1]} |
| Synthetic paths | {paths.shape[0]} |
| Bootstrap replicates | {n_boot} |
| Bootstrap block (expected days) | {block} |
| Simulation seed | {seed_text} |
| Bootstrap seed | {boot_seed} |
{params_rows}
| persistence (a + g*E[z^2 1{{z<0}}] + b) | {fit.persistence:.6f} |
| leverage weight E[z^2 1{{z<0}}] | {fit.leverage_weight:.6f} |
| forecast variance sigma^2(T+1) | {fit.forecast_variance:.6g} |
| unconditional variance | {fit.unconditional_variance:.6g} |

## 2. Acceptance table

A statistic PASSES when the synthetic median lies inside the real 2.5-97.5%
bootstrap band. `Position` is 0 inside the band, and otherwise counts
band-widths beyond the breached bound (negative below, positive above).
`Spread` is the synthetic 2.5-97.5% width over the real one: a model can sit on
the median and still be far too variable, and only this column sees that. VaR
and ES are positive loss magnitudes; the percentile rows are signed returns.
Rows marked *(= -...)* are the same number as another row, reported under both
conventional names and counted once.

{_markdown_table(results_df)}

## 3. Plots

{chr(10).join(chr(10).join([line, ""]) for line in figure_lines)}

## 4. Failure modes

> **TODO — narrative to be written.** The numbers below are pre-filled; the
> interpretation is not.

**Outside the band ({len(failures)}):**

{failure_lines}

**Spread mismatches ({n_wide}):**

{wide_lines}

*TODO: state whether the ACF decay mismatch is acceptable for the intended use,
and whether the block length of {block} days is doing material work in the width
of these bands.*
"""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return str(destination)


def run_validation(
    seed: int = 42,
    horizon: int = DEFAULT_HORIZON,
    n_paths: int = DEFAULT_N_PATHS,
    n_boot: int = DEFAULT_N_BOOT,
    block: int = DEFAULT_BLOCK,
    boot_seed: int = 7,
    out_path: str = "reports/validation.html",
    markdown_path: str | None = "reports/validation.md",
) -> str:
    """load returns → fit → simulate → bootstrap → validate → report.
    One call. Returns report path.

    Writes the HTML report and, unless ``markdown_path`` is None, a
    markdown twin of it beside the plots.
    """
    returns = load_returns()
    fitted = fit_model(returns)
    paths = simulate(
        fitted, horizon=horizon, n_paths=n_paths, seed=seed, initial_var="last"
    )
    bands = bootstrap_bands(
        returns, horizon=horizon, n_boot=n_boot, block=block, seed=boot_seed
    )
    # Reuse the bootstrap just computed rather than running an identical
    # one inside validate(); it is the dominant cost of this pipeline.
    results = validate(paths, returns, bands=bands)
    shared = {
        "seed": seed,
        "block": block,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
        "boot_drawdowns": bands["max_drawdown"][3],
    }
    report_path = make_report(
        returns, fitted, paths, results, out_path=out_path, **shared
    )
    if markdown_path is not None:
        make_markdown_report(
            returns, fitted, paths, results, out_path=markdown_path, **shared
        )
    return report_path
