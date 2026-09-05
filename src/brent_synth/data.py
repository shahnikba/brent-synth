"""Brent crude daily prices and log returns.

Fetch, clean, cache, return. No modelling logic belongs in this module.

Source is Yahoo Finance ticker ``BZ=F`` (ICE Brent front-month future).
Raw closes are cached to ``data/brent_raw.parquet`` so that repeat calls
do not hit the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "BZ=F"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_PATH = DATA_DIR / "brent_raw.parquet"


def _download_prices() -> pd.Series:
    """Download the full available daily close history for ``TICKER``."""
    raw = yf.download(
        TICKER,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {TICKER!r}.")
    return _extract_close(raw)


def _extract_close(frame: pd.DataFrame) -> pd.Series:
    """Pull the close column out of a yfinance frame.

    yfinance may return either flat columns (``Close``) or a MultiIndex
    keyed by field and ticker (``('Close', 'BZ=F')``), depending on
    version and on how many tickers were requested. Handle both.
    """
    columns = frame.columns
    if isinstance(columns, pd.MultiIndex):
        level = 0 if "Close" in columns.get_level_values(0) else 1
        close = frame.xs("Close", axis=1, level=level)
        if isinstance(close, pd.DataFrame):
            if close.shape[1] != 1:
                raise ValueError(
                    f"Expected exactly one close column, got {list(close.columns)}."
                )
            close = close.iloc[:, 0]
    else:
        if "Close" not in columns:
            raise ValueError(f"No 'Close' column in downloaded data: {list(columns)}.")
        close = frame["Close"]

    return _clean_prices(close)


def _clean_prices(close: pd.Series) -> pd.Series:
    """Normalise a raw close series: datetime index, sorted, unique, no NaN."""
    close = pd.Series(close).astype("float64")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close = close.dropna()
    close = close[close > 0]
    close = close[~close.index.duplicated(keep="last")]
    close = close.sort_index()
    close.name = "close"
    close.index.name = "date"
    return close


def _read_cache() -> pd.Series | None:
    """Return the cached close series, or ``None`` if there is no usable cache."""
    if not CACHE_PATH.exists():
        return None
    frame = pd.read_parquet(CACHE_PATH)
    if "close" not in frame.columns or frame.empty:
        return None
    return _clean_prices(frame["close"])


def _write_cache(prices: pd.Series) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_frame(name="close").to_parquet(CACHE_PATH)


def load_prices(refresh: bool = False) -> pd.Series:
    """Datetime-indexed daily Brent close prices, named 'close'.

    Reads from the local parquet cache when available; downloads from
    Yahoo Finance (and populates the cache) otherwise, or whenever
    ``refresh`` is True.
    """
    if not refresh:
        cached = _read_cache()
        if cached is not None:
            return cached

    prices = _download_prices()
    _write_cache(prices)
    return prices


def load_returns(refresh: bool = False) -> pd.Series:
    """Datetime-indexed daily log returns, named 'log_return', NaNs dropped.

    ``r_t = ln(P_t) - ln(P_{t-1})``. The first observation has no
    predecessor and is dropped.
    """
    prices = load_prices(refresh=refresh)
    returns = np.log(prices).diff().dropna()
    returns.name = "log_return"
    returns.index.name = "date"
    return returns
