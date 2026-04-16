"""Tests for forward label engineering behavior and alignment."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FeatureConfig, ProjectConfig
from src.labels import build_label_frame, compute_forward_realized_volatility


def _synthetic_returns() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    values = [0.01, 0.03, -0.02, 0.04, -0.01, 0.02, -0.03, 0.01, 0.00, 0.05]
    return pd.DataFrame({"SPY": values, "SHY": [0.001] * 10}, index=dates)


def test_forward_vol_exact_offset_alignment() -> None:
    returns = _synthetic_returns()
    horizon = 5

    labels = compute_forward_realized_volatility(returns=returns, horizon=horizon)

    t = returns.index[0]
    expected = returns["SPY"].iloc[1:6].std(ddof=1)
    assert np.isclose(labels.loc[t], expected)


def test_last_horizon_rows_are_nan() -> None:
    returns = _synthetic_returns()
    horizon = 5

    labels = compute_forward_realized_volatility(returns=returns, horizon=horizon)

    assert labels.iloc[-horizon:].isna().all()


def test_label_column_name_is_correct() -> None:
    returns = _synthetic_returns()

    labels = compute_forward_realized_volatility(returns=returns, horizon=5)

    assert labels.name == "spy_fwd_vol_5d"


def test_build_label_frame_preserves_sorted_index() -> None:
    returns = _synthetic_returns()
    config = ProjectConfig(features=FeatureConfig(forward_vol_horizon=5))

    label_frame = build_label_frame(returns=returns, config=config)

    assert label_frame.index.equals(returns.index)
    assert label_frame.index.is_monotonic_increasing
    assert list(label_frame.columns) == ["spy_fwd_vol_5d"]


def test_label_excludes_current_day_return() -> None:
    returns = _synthetic_returns()
    horizon = 5

    baseline = compute_forward_realized_volatility(returns=returns, horizon=horizon)

    mutated = returns.copy()
    mutated.iloc[0, mutated.columns.get_loc("SPY")] = 10.0
    recomputed = compute_forward_realized_volatility(returns=mutated, horizon=horizon)

    assert baseline.iloc[0] == recomputed.iloc[0]


def test_label_is_not_trailing_window() -> None:
    returns = _synthetic_returns()
    horizon = 5

    forward = compute_forward_realized_volatility(returns=returns, horizon=horizon)
    trailing_like = returns["SPY"].rolling(window=horizon).std(ddof=1)

    # At index 4, trailing std exists but forward label should still exist at index 4
    # using returns 5..9; they should not match trailing computation in general.
    idx = returns.index[4]
    assert not np.isclose(forward.loc[idx], trailing_like.loc[idx])
