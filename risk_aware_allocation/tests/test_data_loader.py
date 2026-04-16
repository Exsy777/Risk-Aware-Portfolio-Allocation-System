"""Tests for data loading and validation behavior."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DataConfig, ProjectConfig
from src.data_loader import (
    PROCESSED_PRICES_FILENAME,
    PROCESSED_RETURNS_FILENAME,
    build_return_frame,
    get_aligned_market_data,
    save_raw_data,
)
from src.validation import (
    validate_expected_tickers,
    validate_no_duplicate_dates,
    validate_sorted_index,
)


def test_build_return_frame_computes_expected_values() -> None:
    """Return frame should match pct_change output with first row dropped."""
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 103.0, 103.0],
            "SHY": [50.0, 50.5, 50.5, 51.51],
        },
        index=dates,
    )

    returns = build_return_frame(prices)

    expected = pd.DataFrame(
        {
            "SPY": [0.01, (103.0 / 101.0) - 1.0, 0.0],
            "SHY": [0.01, 0.0, (51.51 / 50.5) - 1.0],
        },
        index=dates[1:],
    )
    pd.testing.assert_frame_equal(returns, expected)


def test_validation_rejects_duplicate_dates() -> None:
    """Validation should fail on duplicate date index entries."""
    duplicated_idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    frame = pd.DataFrame({"SPY": [1.0, 2.0]}, index=duplicated_idx)

    with pytest.raises(ValueError, match="duplicate"):
        validate_no_duplicate_dates(frame, frame_name="prices")


def test_validation_rejects_unsorted_index() -> None:
    """Validation should fail on descending/unsorted date index."""
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-01"])
    frame = pd.DataFrame({"SPY": [1.0, 2.0]}, index=idx)

    with pytest.raises(ValueError, match="sorted"):
        validate_sorted_index(frame, frame_name="prices")


def test_validation_rejects_missing_expected_tickers() -> None:
    """Validation should fail when expected ticker columns are missing."""
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    frame = pd.DataFrame({"SPY": [1.0, 1.1, 1.2]}, index=idx)

    with pytest.raises(ValueError, match="Ticker columns mismatch"):
        validate_expected_tickers(frame, expected_tickers=("SPY", "SHY"))


def test_get_aligned_market_data_uses_cached_raw_and_saves_processed(tmp_path: Path) -> None:
    """Aligned market data should be built from cache and saved to processed outputs."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    columns = pd.MultiIndex.from_product(
        [["SPY", "SHY"], ["Adj Close", "Close"]], names=["Ticker", "Field"]
    )
    raw = pd.DataFrame(
        [
            [100.0, 100.0, 80.0, 80.0],
            [101.0, 101.0, 80.2, 80.2],
            [102.0, 102.0, 80.4, 80.4],
            [101.5, 101.5, 80.5, 80.5],
            [103.0, 103.0, 80.6, 80.6],
            [104.0, 104.0, 80.8, 80.8],
        ],
        index=dates,
        columns=columns,
    )
    save_raw_data(raw_data=raw, raw_dir=raw_dir)

    config = ProjectConfig(
        data=DataConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            tickers=("SPY", "SHY"),
            cache_raw_dir=raw_dir,
            cache_processed_dir=processed_dir,
            price_field="Adj Close",
        )
    )

    market_data = get_aligned_market_data(config=config, use_cache=True, save_processed=True)

    assert market_data.metadata["source"] == "cache"
    assert list(market_data.prices.columns) == ["SPY", "SHY"]
    assert list(market_data.returns.columns) == ["SPY", "SHY"]
    assert market_data.returns.index.equals(market_data.prices.index[1:])
    assert (processed_dir / PROCESSED_PRICES_FILENAME).exists()
    assert (processed_dir / PROCESSED_RETURNS_FILENAME).exists()
