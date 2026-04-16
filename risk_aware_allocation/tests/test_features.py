"""Tests for leakage-safe feature engineering behavior."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FeatureConfig, ProjectConfig
from src.features import (
    build_feature_frame,
    compute_downside_volatility,
    compute_drawdown_features,
    compute_lagged_returns,
    compute_moving_averages,
    compute_rolling_volatility,
)


def _build_synthetic_inputs(periods: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    spy = pd.Series(np.linspace(100.0, 140.0, periods), index=dates)
    shy = pd.Series(np.linspace(80.0, 84.0, periods), index=dates)
    prices = pd.DataFrame({"SPY": spy, "SHY": shy})
    returns = prices.pct_change(fill_method=None).dropna(how="any")
    return prices, returns


def test_lagged_return_math_is_correct() -> None:
    prices, _ = _build_synthetic_inputs(periods=30)
    lagged = compute_lagged_returns(prices=prices, short_window=5, medium_window=20)

    t = prices.index[20]
    expected_5d = (prices.loc[t, "SPY"] / prices.loc[prices.index[15], "SPY"]) - 1.0
    expected_20d = (prices.loc[t, "SPY"] / prices.loc[prices.index[0], "SPY"]) - 1.0

    assert lagged.loc[t, "spy_ret_5d"] == expected_5d
    assert lagged.loc[t, "spy_ret_20d"] == expected_20d


def test_rolling_volatility_has_expected_nan_behavior() -> None:
    _, returns = _build_synthetic_inputs(periods=90)
    vol = compute_rolling_volatility(returns=returns, short_window=20, long_window=60)

    assert vol["spy_vol_20d"].iloc[:19].isna().all()
    assert not np.isnan(vol["spy_vol_20d"].iloc[19])
    assert vol["spy_vol_60d"].iloc[:59].isna().all()


def test_moving_average_spread_ratio_is_correct() -> None:
    prices, _ = _build_synthetic_inputs(periods=70)
    ma = compute_moving_averages(prices=prices, short_window=20, long_window=50)

    t = prices.index[60]
    expected = (prices["SPY"].iloc[41:61].mean() / prices["SPY"].iloc[11:61].mean()) - 1.0
    assert np.isclose(ma.loc[t, "spy_ma_spread_ratio"], expected)


def test_drawdown_is_zero_at_high_and_negative_below_high() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"SPY": [100, 105, 103, 106, 104, 107], "SHY": [80, 80, 80, 80, 80, 80]}, index=dates)
    drawdown = compute_drawdown_features(prices=prices, window=3)

    assert drawdown.loc[dates[3], "spy_drawdown_3d"] == 0.0
    assert drawdown.loc[dates[4], "spy_drawdown_3d"] < 0.0


def test_downside_volatility_behavior() -> None:
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    returns = pd.DataFrame({"SPY": [0.01, -0.02, 0.03, -0.04, -0.01, 0.02, -0.03]}, index=dates)

    downside = compute_downside_volatility(returns=returns, window=3)
    manual_window = pd.Series([0.0, -0.04, -0.01])
    expected = manual_window.std()

    assert np.isclose(downside.loc[dates[4], "spy_downside_vol_3d"], expected)


def test_build_feature_frame_expected_columns_and_sorted_index() -> None:
    prices, returns = _build_synthetic_inputs(periods=90)
    cfg = ProjectConfig(
        features=FeatureConfig(
            short_return_window=5,
            medium_return_window=20,
            short_vol_window=20,
            long_vol_window=60,
            moving_average_short=20,
            moving_average_long=50,
            downside_vol_window=20,
            forward_vol_horizon=5,
        )
    )
    feature_frame = build_feature_frame(prices=prices, returns=returns, config=cfg, include_shy_context=True)

    expected_cols = {
        "spy_ret_5d",
        "spy_ret_20d",
        "spy_vol_20d",
        "spy_vol_60d",
        "spy_ma_20",
        "spy_ma_50",
        "spy_ma_spread_ratio",
        "spy_drawdown_20d",
        "spy_downside_vol_20d",
        "shy_ret_5d",
    }
    assert expected_cols.issubset(set(feature_frame.columns))
    assert feature_frame.index.is_monotonic_increasing


def test_features_do_not_use_forward_shifted_information() -> None:
    prices, returns = _build_synthetic_inputs(periods=70)
    cfg = ProjectConfig()

    baseline = build_feature_frame(prices=prices, returns=returns, config=cfg)

    mutated_prices = prices.copy()
    mutated_prices.loc[prices.index[-1], "SPY"] = prices.loc[prices.index[-1], "SPY"] * 5.0
    mutated_returns = mutated_prices.pct_change(fill_method=None).dropna(how="any")

    mutated = build_feature_frame(prices=mutated_prices, returns=mutated_returns, config=cfg)

    cutoff = prices.index[-2]
    baseline_slice = baseline.loc[:cutoff]
    mutated_slice = mutated.loc[:cutoff]
    pd.testing.assert_frame_equal(baseline_slice, mutated_slice)
