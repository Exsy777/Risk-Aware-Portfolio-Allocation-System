"""Validation helpers for market price and return data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .config import ProjectConfig


def validate_nonempty_frame(df: pd.DataFrame, frame_name: str) -> None:
    """Ensure a DataFrame is not empty."""
    if df.empty:
        raise ValueError(f"{frame_name} is empty.")


def validate_datetime_index(df: pd.DataFrame, frame_name: str) -> None:
    """Ensure a DataFrame uses a DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"{frame_name} must use a pandas DatetimeIndex.")


def validate_sorted_index(df: pd.DataFrame, frame_name: str) -> None:
    """Ensure index ordering is monotonic increasing."""
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{frame_name} index must be sorted ascending.")


def validate_no_duplicate_dates(df: pd.DataFrame, frame_name: str) -> None:
    """Ensure index dates are unique."""
    if df.index.has_duplicates:
        raise ValueError(f"{frame_name} contains duplicate dates in index.")


def validate_expected_tickers(df: pd.DataFrame, expected_tickers: Iterable[str]) -> None:
    """Ensure DataFrame contains exactly the expected ticker columns."""
    expected = list(expected_tickers)
    actual = list(df.columns)
    missing = [ticker for ticker in expected if ticker not in df.columns]
    extras = [ticker for ticker in df.columns if ticker not in expected]
    if missing or extras or actual != expected:
        raise ValueError(
            "Ticker columns mismatch. "
            f"Expected order={expected}, actual order={actual}, missing={missing}, extras={extras}."
        )


def validate_missingness(
    df: pd.DataFrame,
    frame_name: str,
    max_missing_fraction: float | None = None,
) -> None:
    """Optionally enforce a maximum missing-value fraction."""
    if max_missing_fraction is None:
        return

    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1 inclusive.")

    missing_fraction = float(df.isna().mean().mean())
    if missing_fraction > max_missing_fraction:
        raise ValueError(
            f"{frame_name} missing fraction {missing_fraction:.4f} exceeds "
            f"allowed {max_missing_fraction:.4f}."
        )


def validate_aligned_frames(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Validate alignment and return construction consistency."""
    if list(prices.columns) != list(returns.columns):
        raise ValueError("prices and returns must have identical ticker columns in same order.")

    expected_returns = prices.pct_change(fill_method=None).dropna(how="any")
    if not returns.index.equals(expected_returns.index):
        raise ValueError("returns index is inconsistent with prices pct_change/dropna index.")

    if returns.shape != expected_returns.shape:
        raise ValueError("returns shape is inconsistent with expected pct_change/dropna shape.")


def run_all_data_checks(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: ProjectConfig,
    max_missing_fraction: float | None = None,
) -> None:
    """Run all standard sanity checks for price/return datasets."""
    for frame_name, frame in (("prices", prices), ("returns", returns)):
        validate_nonempty_frame(frame, frame_name)
        validate_datetime_index(frame, frame_name)
        validate_sorted_index(frame, frame_name)
        validate_no_duplicate_dates(frame, frame_name)
        validate_expected_tickers(frame, config.data.tickers)
        validate_missingness(frame, frame_name, max_missing_fraction=max_missing_fraction)

    validate_aligned_frames(prices=prices, returns=returns)
