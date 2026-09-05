# SPEC 1 — Repo Setup + Data Ingestion

## Goal
Set up the project skeleton and a data layer that fetches Brent crude
daily prices and returns clean log returns. Model/validation come later.

## Repo
- Private repo (already created under `shahram`). Clone it.
- Python 3.11, managed with `uv` + `pyproject.toml`.
- Layout:
  brent-synth/
    README.md
    pyproject.toml
    .gitignore
    src/brent_synth/
      __init__.py
      data.py
    tests/
      test_data.py
- `.gitignore` excludes: data/, *.csv, *.parquet, __pycache__/, .venv/, reports/
- Never commit data.

## Data layer (src/brent_synth/data.py)
Fetch Brent close prices, compute log returns, cache locally.

Rules:
- Source: Yahoo Finance ticker `BZ=F` via `yfinance` (pin the version).
- Fetch >=10 years up to today (use period="max"). No hardcoded end date.
- Use daily `Close`. Drop NaN closes.
- Log returns: r_t = ln(P_t) - ln(P_{t-1}). Drop first NaN.
- Cache raw prices to data/brent_raw.parquet on first fetch.
  Reload from cache unless refresh=True.

Public interface (keep stable):
    def load_prices(refresh: bool = False) -> pd.Series:
        """Datetime-indexed daily Brent close prices, named 'close'."""

    def load_returns(refresh: bool = False) -> pd.Series:
        """Datetime-indexed daily log returns, named 'log_return', NaNs dropped."""

Both return a pandas Series with a sorted, unique DatetimeIndex.

## Tests (tests/test_data.py)
- Returns series non-empty, >=2400 observations.
- No NaNs or infs in returns.
- Index monotonic increasing, no duplicates.
- Prices reconstruct from returns within tolerance.
- Sanity: abs(daily log return) < 1.0 for all rows.

## Notes
- yfinance may return multi-index columns for one ticker; select `Close`
  defensively (works whether column is `Close` or `('Close','BZ=F')`).
- No modelling logic in this module — fetch, clean, cache, return only.

## Done when
From a clean clone + install:
`from brent_synth.data import load_returns; load_returns()`
returns a clean series, tests pass, no data tracked by git.
