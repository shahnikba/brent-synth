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
