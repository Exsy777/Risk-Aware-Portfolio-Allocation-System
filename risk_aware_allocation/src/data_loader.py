"""Data ingestion and caching utilities for market prices and returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from .config import ProjectConfig
from .validation import run_all_data_checks

RAW_DOWNLOAD_FILENAME = "raw_download.csv"
PROCESSED_PRICES_FILENAME = "processed_prices.csv"
PROCESSED_RETURNS_FILENAME = "processed_returns.csv"


@dataclass(slots=True)
class MarketData:
    """Container for aligned price/return frames and loader metadata."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def download_price_data(
    tickers: tuple[str, ...],
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    """Download raw daily data from Yahoo Finance for the requested tickers."""
    raw_data = yf.download(
        tickers=list(tickers),
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        actions=False,
        threads=True,
    )

    if raw_data.empty:
        raise ValueError("Downloaded data is empty for the requested date range/tickers.")

    return raw_data


def save_raw_data(raw_data: pd.DataFrame, raw_dir: Path, filename: str = RAW_DOWNLOAD_FILENAME) -> Path:
    """Persist raw downloaded data to disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / filename
    raw_data.to_csv(output_path)
    return output_path


def load_cached_raw_data(raw_dir: Path, filename: str = RAW_DOWNLOAD_FILENAME) -> pd.DataFrame:
    """Load previously cached raw download from disk."""
    cached_path = raw_dir / filename
    if not cached_path.exists():
        raise FileNotFoundError(f"Cached raw file not found at {cached_path}.")

    raw_data = pd.read_csv(cached_path, header=[0, 1], index_col=0, parse_dates=True)
    return raw_data


def _extract_price_series(raw_data: pd.DataFrame, ticker: str, price_field: str) -> tuple[pd.Series, str]:
    """Extract a ticker price series from raw data with adjusted-close fallback."""
    if isinstance(raw_data.columns, pd.MultiIndex):
        if ticker not in raw_data.columns.get_level_values(0):
            raise ValueError(f"Ticker {ticker!r} is missing from raw download.")

        ticker_frame = raw_data[ticker]
    else:
        ticker_frame = raw_data.copy()

    if price_field in ticker_frame.columns:
        return ticker_frame[price_field], price_field

    if "Close" in ticker_frame.columns:
        return ticker_frame["Close"], "Close"

    raise ValueError(
        f"No {price_field!r} or 'Close' column available for ticker {ticker!r}."
    )


def build_price_frame(raw_data: pd.DataFrame, tickers: tuple[str, ...], price_field: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build a standardized price frame indexed by date with one column per ticker."""
    series_by_ticker: dict[str, pd.Series] = {}
    price_field_used: dict[str, str] = {}

    for ticker in tickers:
        series, used_field = _extract_price_series(raw_data=raw_data, ticker=ticker, price_field=price_field)
        series_by_ticker[ticker] = series
        price_field_used[ticker] = used_field

    prices = pd.DataFrame(series_by_ticker)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.sort_index()
    prices = prices.loc[:, list(tickers)]
    prices = prices.dropna(how="all")

    return prices, price_field_used


def build_return_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Build simple daily returns from prices using pct_change."""
    returns = prices.pct_change(fill_method=None)
    returns = returns.dropna(how="any")
    return returns


def get_aligned_market_data(
    config: ProjectConfig,
    use_cache: bool = True,
    save_processed: bool = True,
    max_missing_fraction: float | None = None,
) -> MarketData:
    """Load, standardize, validate, and optionally persist aligned market data."""
    raw_dir = config.data.cache_raw_dir
    processed_dir = config.data.cache_processed_dir

    raw_path = raw_dir / RAW_DOWNLOAD_FILENAME
    if use_cache and raw_path.exists():
        raw_data = load_cached_raw_data(raw_dir=raw_dir)
        data_source = "cache"
    else:
        raw_data = download_price_data(
            tickers=config.data.tickers,
            start_date=config.data.start_date,
            end_date=config.data.end_date,
        )
        save_raw_data(raw_data=raw_data, raw_dir=raw_dir)
        data_source = "download"

    prices, field_map = build_price_frame(
        raw_data=raw_data,
        tickers=config.data.tickers,
        price_field=config.data.price_field,
    )
    returns = build_return_frame(prices)

    run_all_data_checks(
        prices=prices,
        returns=returns,
        config=config,
        max_missing_fraction=max_missing_fraction,
    )

    if save_processed:
        processed_dir.mkdir(parents=True, exist_ok=True)
        prices.to_csv(processed_dir / PROCESSED_PRICES_FILENAME)
        returns.to_csv(processed_dir / PROCESSED_RETURNS_FILENAME)

    metadata = {
        "source": data_source,
        "price_field_requested": config.data.price_field,
        "price_field_used": field_map,
        "raw_file": str(raw_path),
        "processed_prices_file": str(processed_dir / PROCESSED_PRICES_FILENAME),
        "processed_returns_file": str(processed_dir / PROCESSED_RETURNS_FILENAME),
    }
    return MarketData(prices=prices, returns=returns, metadata=metadata)
