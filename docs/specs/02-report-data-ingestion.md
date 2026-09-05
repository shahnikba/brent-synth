# Report — SPEC 1, data ingestion

## Status
Complete. 8 tests passing.

## Environment notes
- `uv` was not installed; installed into pyenv 3.11.8 (`uv 0.12.10`).
- `gh` CLI not installed, so the GitHub repo could not be created from here.
- Resolved deps: `yfinance==1.7.0` (pinned per spec), pandas 3.0.5,
  numpy 2.4.6, pyarrow 25.0.1, pytest 9.1.1. `uv.lock` committed.

## Data layer
`src/brent_synth/data.py` — `load_prices()` / `load_returns()`, interface
as specified.

- Yahoo `BZ=F`, `period="max"`, no hardcoded end date.
- Close extraction handles both flat `Close` and MultiIndex `('Close','BZ=F')`.
- Cache: `data/brent_raw.parquet` (65 KB), reloaded unless `refresh=True`.
- `.gitignore` covers `data/`, `*.csv`, `*.parquet`, `__pycache__/`,
  `.venv/`, `reports/`. `git ls-files` confirms no data tracked.

Data retrieved: 4755 closes, 2007-07-30 to 2026-09-04 (19.1 years), 4754
log returns. Zero NaNs/infs. $75.74 to $96.28.

Verified from a clean clone into a temp dir with no cache: `uv sync` ->
`load_returns()` -> 4754 obs -> 8 tests pass.

## Note
The observation count later fell to 4753 when the provisional final bar
was found and dropped (see 09/10).
