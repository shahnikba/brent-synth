"""Contract tests for the Brent data layer."""

import numpy as np
import pandas as pd
import pytest

from brent_synth.data import load_prices, load_returns

MIN_OBSERVATIONS = 2400


@pytest.fixture(scope="module")
def prices() -> pd.Series:
    return load_prices()


@pytest.fixture(scope="module")
def returns() -> pd.Series:
    return load_returns()


def test_returns_non_empty_and_long_enough(returns: pd.Series) -> None:
    assert len(returns) >= MIN_OBSERVATIONS


def test_returns_are_finite(returns: pd.Series) -> None:
    assert not returns.isna().any()
    assert np.isfinite(returns.to_numpy()).all()


def test_index_is_sorted_and_unique(returns: pd.Series) -> None:
    assert isinstance(returns.index, pd.DatetimeIndex)
    assert returns.index.is_monotonic_increasing
    assert not returns.index.has_duplicates


def test_prices_reconstruct_from_returns(prices: pd.Series, returns: pd.Series) -> None:
    """Cumulating the log returns off the first close recovers the price path."""
    rebuilt = prices.iloc[0] * np.exp(returns.cumsum())
    expected = prices.loc[returns.index]
    assert np.allclose(rebuilt.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9)


def test_daily_moves_are_sane(returns: pd.Series) -> None:
    assert returns.abs().max() < 1.0


def test_series_names(prices: pd.Series, returns: pd.Series) -> None:
    assert prices.name == "close"
    assert returns.name == "log_return"


def test_prices_are_positive(prices: pd.Series) -> None:
    assert (prices > 0).all()


def test_history_spans_at_least_ten_years(prices: pd.Series) -> None:
    span_days = (prices.index[-1] - prices.index[0]).days
    assert span_days >= 10 * 365


# --- cache lifecycle -------------------------------------------------------
#
# These run offline: the downloader is replaced and the cache is pointed at
# a temp directory, so nothing here touches Yahoo.

import datetime as dt  # noqa: E402
import warnings  # noqa: E402

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from brent_synth import data as data_module  # noqa: E402


def _fake_prices(n: int = 300, start: str = "2020-01-01") -> pd.Series:
    index = pd.bdate_range(start, periods=n)
    values = 50.0 + np.arange(n, dtype="float64") * 0.1
    series = pd.Series(values, index=index, name="close")
    series.index.name = "date"
    return series


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the module's cache at a temp dir and count downloads."""
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_module, "CACHE_PATH", tmp_path / "brent_raw.parquet")

    calls = {"n": 0}

    def fake_download() -> pd.Series:
        calls["n"] += 1
        return _fake_prices()

    monkeypatch.setattr(data_module, "_download_prices", fake_download)
    return calls


def test_fresh_cache_is_served_without_downloading(isolated_cache) -> None:
    data_module.load_prices()
    assert isolated_cache["n"] == 1
    data_module.load_prices()
    assert isolated_cache["n"] == 1  # same UTC day, no second call


def test_stale_cache_triggers_a_refresh(isolated_cache) -> None:
    """The bug: a cache from an earlier day must not be served silently."""
    yesterday = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    data_module._write_cache(_fake_prices(n=100), fetched_on=yesterday)

    data_module.load_prices()
    assert isolated_cache["n"] == 1  # refetched rather than trusted


def test_truncated_cache_is_served_only_until_the_next_day(
    isolated_cache,
) -> None:
    """A short cache written long ago must not survive as the answer."""
    old = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=400)
    data_module._write_cache(_fake_prices(n=50), fetched_on=old)

    prices = data_module.load_prices()
    assert isolated_cache["n"] == 1
    assert len(prices) == 300  # the refetched series, not the 50-row cache


def test_cache_without_a_fetch_stamp_counts_as_stale(isolated_cache) -> None:
    """Caches predating the metadata key heal themselves."""
    frame = _fake_prices(n=80).to_frame(name="close")
    pq.write_table(pa.Table.from_pandas(frame), data_module.CACHE_PATH)

    prices = data_module.load_prices()
    assert isolated_cache["n"] == 1
    assert len(prices) == 300


def test_corrupt_cache_falls_back_to_download(isolated_cache) -> None:
    """Garbage in the cache file must not escape as ArrowInvalid."""
    data_module.CACHE_PATH.write_bytes(b"not a parquet file at all")

    with pytest.warns(RuntimeWarning, match="unreadable price cache"):
        prices = data_module.load_prices()
    assert isolated_cache["n"] == 1
    assert len(prices) == 300


def test_half_written_cache_falls_back_to_download(isolated_cache) -> None:
    """An interrupted write leaves a truncated file; recover from it."""
    data_module._write_cache(_fake_prices(), fetched_on=dt.date(2020, 1, 1))
    payload = data_module.CACHE_PATH.read_bytes()
    data_module.CACHE_PATH.write_bytes(payload[: len(payload) // 2])

    with pytest.warns(RuntimeWarning):
        prices = data_module.load_prices()
    assert isolated_cache["n"] == 1
    assert len(prices) == 300


def test_cache_write_is_atomic(isolated_cache) -> None:
    """A failed write must leave the previous cache readable."""
    data_module._write_cache(_fake_prices(n=120), fetched_on=dt.date(2024, 5, 1))
    before = data_module._read_cache()
    assert before is not None and len(before[0]) == 120

    def explode(*args, **kwargs):
        raise OSError("disk full")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(data_module.pq, "write_table", explode)
        with pytest.raises(OSError):
            data_module._write_cache(_fake_prices(n=300))

    after = data_module._read_cache()
    assert after is not None and len(after[0]) == 120  # untouched
    assert not data_module.CACHE_PATH.with_name(
        data_module.CACHE_PATH.name + ".tmp"
    ).exists()


def test_refresh_flag_always_downloads(isolated_cache) -> None:
    data_module.load_prices()
    data_module.load_prices(refresh=True)
    assert isolated_cache["n"] == 2


def test_offline_refresh_serves_stale_cache_with_a_warning(
    tmp_path, monkeypatch
) -> None:
    """Losing the network degrades to last known data, loudly."""
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_module, "CACHE_PATH", tmp_path / "brent_raw.parquet")
    old = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=30)
    data_module._write_cache(_fake_prices(n=90), fetched_on=old)

    def offline() -> pd.Series:
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(data_module, "_download_prices", offline)

    with pytest.warns(RuntimeWarning, match="Could not refresh"):
        prices = data_module.load_prices()
    assert len(prices) == 90


def test_offline_without_a_cache_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(data_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(data_module, "CACHE_PATH", tmp_path / "brent_raw.parquet")

    def offline() -> pd.Series:
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(data_module, "_download_prices", offline)
    with pytest.raises(RuntimeError, match="network unreachable"):
        data_module.load_prices()


# --- provisional bar -------------------------------------------------------


def test_provisional_bar_is_dropped() -> None:
    """Yahoo's live row copies the previous day's volume; drop it."""
    close = _fake_prices(n=10)
    volume = pd.Series(np.arange(1, 11) * 1000.0, index=close.index)
    volume.iloc[-1] = volume.iloc[-2]  # the tell

    kept = data_module._drop_provisional_bar(close, volume)
    assert len(kept) == 9
    assert kept.index[-1] == close.index[-2]


def test_settled_final_bar_is_kept() -> None:
    close = _fake_prices(n=10)
    volume = pd.Series(np.arange(1, 11) * 1000.0, index=close.index)

    kept = data_module._drop_provisional_bar(close, volume)
    assert len(kept) == 10
    assert kept.index[-1] == close.index[-1]


def test_provisional_check_survives_missing_volume() -> None:
    """A NaN volume must not silently drop a real bar."""
    close = _fake_prices(n=10)
    volume = pd.Series(np.arange(1, 11) * 1000.0, index=close.index)
    volume.iloc[-1] = np.nan

    assert len(data_module._drop_provisional_bar(close, volume)) == 10


def test_provisional_check_handles_tiny_series() -> None:
    close = _fake_prices(n=1)
    volume = pd.Series([100.0], index=close.index)
    assert len(data_module._drop_provisional_bar(close, volume)) == 1


def test_extract_column_handles_both_column_shapes() -> None:
    index = pd.bdate_range("2024-01-01", periods=3)
    flat = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Volume": [10, 20, 30]}, index=index)
    assert list(data_module._extract_column(flat, "Close")) == [1.0, 2.0, 3.0]
    assert list(data_module._extract_column(flat, "Volume")) == [10, 20, 30]

    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_product([["Close", "Volume"], ["BZ=F"]])
    assert list(data_module._extract_column(multi, "Close")) == [1.0, 2.0, 3.0]
    assert list(data_module._extract_column(multi, "Volume")) == [10, 20, 30]

    with pytest.raises(ValueError, match="No 'Close' column"):
        data_module._extract_column(flat[["Volume"]], "Close")
