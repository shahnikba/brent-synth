"""Brent crude daily prices and log returns.

Fetch, clean, cache, return. No modelling logic belongs in this module.

Source is Yahoo Finance ticker ``BZ=F`` (ICE Brent front-month future).
Raw closes are cached to ``data/brent_raw.parquet`` so that repeat calls
do not hit the network.
"""

from __future__ import annotations

import datetime as dt
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yfinance as yf

TICKER = "BZ=F"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_PATH = DATA_DIR / "brent_raw.parquet"

#: Parquet key holding the UTC date the cache was fetched on. A cache
#: written before this key existed reads back as unknown age, which
#: counts as stale, so old caches heal themselves on first use.
FETCHED_ON_KEY = b"brent_synth_fetched_on"


def _download_prices() -> pd.Series:
    """Download the full available daily close history for ``TICKER``.

    The provisional final bar, if present, is dropped before the series
    is returned — see :func:`_drop_provisional_bar`.
    """
    raw = yf.download(
        TICKER,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {TICKER!r}.")

    close = _extract_column(raw, "Close")
    try:
        volume = _extract_column(raw, "Volume")
    except ValueError:
        # Older or trimmed responses may omit volume; without it the
        # provisional bar cannot be detected, so keep every row.
        volume = None

    if volume is not None:
        close = _drop_provisional_bar(close, volume)
    return _clean_prices(close)


def _extract_column(frame: pd.DataFrame, field: str) -> pd.Series:
    """Pull one field out of a yfinance frame.

    yfinance may return either flat columns (``Close``) or a MultiIndex
    keyed by field and ticker (``('Close', 'BZ=F')``), depending on
    version and on how many tickers were requested. Handle both.
    """
    columns = frame.columns
    if isinstance(columns, pd.MultiIndex):
        if field not in columns.get_level_values(0):
            if field not in columns.get_level_values(1):
                raise ValueError(f"No {field!r} column in downloaded data.")
            level = 1
        else:
            level = 0
        series = frame.xs(field, axis=1, level=level)
        if isinstance(series, pd.DataFrame):
            if series.shape[1] != 1:
                raise ValueError(
                    f"Expected exactly one {field!r} column, got "
                    f"{list(series.columns)}."
                )
            series = series.iloc[:, 0]
    else:
        if field not in columns:
            raise ValueError(
                f"No {field!r} column in downloaded data: {list(columns)}."
            )
        series = frame[field]
    return series


def _drop_provisional_bar(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Drop the final bar when it is Yahoo's live quote rather than a settle.

    Yahoo publishes the in-progress session as a complete daily row: its
    ``Close`` is the last traded price and its ``Volume`` is copied
    verbatim from the previous day. Caching that row would freeze an
    intraday tick in as a settlement price.

    Detection is the duplicated volume, not the bar's date — the stale
    row can carry the *previous* session's date, so a "is this bar dated
    today" test misses it entirely (observed on a Saturday, with the
    final bar dated the preceding Friday).

    Only the final bar is ever tested. Volume does legitimately repeat
    day-to-day about 1.5% of the time, so this occasionally discards one
    genuine settle — which costs nothing lasting: on the next fetch that
    row has a successor, is no longer final, and is kept.
    """
    if close.size < 2 or volume.size < 2:
        return close

    aligned = volume.reindex(close.index)
    last, previous = aligned.iloc[-1], aligned.iloc[-2]
    if pd.notna(last) and pd.notna(previous) and last == previous:
        return close.iloc[:-1]
    return close


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


def _read_cache() -> tuple[pd.Series, dt.date | None] | None:
    """Return ``(prices, fetched_on)``, or ``None`` if there is no usable cache.

    Every failure mode collapses to ``None`` so the caller re-downloads:
    a missing file, an empty or wrong-shaped table, and — the case that
    used to be fatal — a corrupt one. A parquet file truncated by an
    interrupted write raises out of the arrow reader, which previously
    escaped straight through :func:`load_prices` and bricked the module
    until somebody deleted the file by hand.

    ``fetched_on`` is ``None`` for a cache written without the metadata
    key, which the caller treats as stale.
    """
    if not CACHE_PATH.exists():
        return None

    try:
        table = pq.read_table(CACHE_PATH)
        frame = table.to_pandas()
    except Exception as error:  # noqa: BLE001 - any unreadable cache is refetched
        warnings.warn(
            f"Ignoring unreadable price cache at {CACHE_PATH} ({error}); "
            "re-downloading.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    if frame.empty or "close" not in frame.columns:
        return None

    fetched_on = None
    metadata = table.schema.metadata or {}
    stamp = metadata.get(FETCHED_ON_KEY)
    if stamp is not None:
        try:
            fetched_on = dt.date.fromisoformat(stamp.decode())
        except (ValueError, UnicodeDecodeError):
            fetched_on = None

    try:
        prices = _clean_prices(frame["close"])
    except Exception as error:  # noqa: BLE001 - unusable contents are refetched
        warnings.warn(
            f"Ignoring malformed price cache at {CACHE_PATH} ({error}); "
            "re-downloading.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None

    if prices.empty:
        return None
    return prices, fetched_on


def _write_cache(prices: pd.Series, fetched_on: dt.date | None = None) -> None:
    """Write the cache atomically, stamped with the fetch date.

    The write goes to a temporary file in the same directory and is then
    renamed over the target. ``os.replace`` is atomic on POSIX, so an
    interrupted write leaves the previous cache intact rather than a
    half-written file that cannot be read back.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stamp = fetched_on or dt.datetime.now(dt.timezone.utc).date()

    table = pa.Table.from_pandas(prices.to_frame(name="close"))
    metadata = dict(table.schema.metadata or {})
    metadata[FETCHED_ON_KEY] = stamp.isoformat().encode()
    table = table.replace_schema_metadata(metadata)

    temporary = CACHE_PATH.with_name(CACHE_PATH.name + ".tmp")
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, CACHE_PATH)
    finally:
        temporary.unlink(missing_ok=True)


def load_prices(refresh: bool = False) -> pd.Series:
    """Datetime-indexed daily Brent close prices, named 'close'.

    Reads from the local parquet cache when it was fetched today;
    otherwise re-downloads from Yahoo Finance and rewrites the cache.
    Passing ``refresh=True`` always re-downloads.

    A cache is only trusted for the UTC day it was written on, so this
    makes at most one network call per day and cannot silently serve a
    series that stops months ago. If that refresh fails — no network,
    Yahoo down — the stale cache is returned with a warning rather than
    raising, so offline work still runs on last known data and is told
    that is what it is.
    """
    today = dt.datetime.now(dt.timezone.utc).date()
    cached: pd.Series | None = None

    if not refresh:
        entry = _read_cache()
        if entry is not None:
            cached, fetched_on = entry
            if fetched_on is not None and fetched_on >= today:
                return cached

    try:
        prices = _download_prices()
    except Exception as error:  # noqa: BLE001 - fall back to a stale cache
        if cached is None:
            raise
        warnings.warn(
            f"Could not refresh Brent prices ({error}); serving cached data "
            f"ending {cached.index[-1].date()}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return cached

    _write_cache(prices, fetched_on=today)
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
