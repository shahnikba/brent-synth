# brent-synth

Brent crude research project. This stage covers the data layer only:
fetch daily Brent close prices, clean them, cache them, and return log
returns. Modelling and validation come later.

## Requirements

- Python 3.11
- [`uv`](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync --extra dev
```

## Usage

```python
from brent_synth.data import load_prices, load_returns

prices = load_prices()    # pd.Series, name="close", DatetimeIndex
returns = load_returns()  # pd.Series, name="log_return", DatetimeIndex
```

Both return a pandas Series with a sorted, unique `DatetimeIndex`.
Returns are `r_t = ln(P_t) - ln(P_{t-1})` with the leading NaN dropped.

Pass `refresh=True` to bypass the cache and re-download.

## Data

Source is Yahoo Finance ticker `BZ=F` (ICE Brent front-month future),
fetched with `period="max"`. The raw closes are cached to
`data/brent_raw.parquet` on first use and reloaded from there
afterwards.

`data/` is gitignored. No data is ever committed.

## Tests

```bash
uv run pytest
```

The tests hit the cache if it is warm, and the network otherwise.
