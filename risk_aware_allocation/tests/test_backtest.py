"""Tests for backtest engine behavior and alignment."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.backtest import (
    build_static_benchmark_weights,
    compute_equity_curve,
    compute_portfolio_returns,
    compute_transaction_cost_series,
    expand_weights_to_daily_returns,
    run_backtest,
)
from src.config import ProjectConfig, StrategyConfig
from src.strategy import StrategyResult


def _returns_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "SPY": [0.01, 0.02, -0.01, 0.005, 0.01, -0.005],
            "SHY": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        },
        index=idx,
    )


def _strategy_result() -> StrategyResult:
    rebalance_idx = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"])
    weights = pd.DataFrame(
        {
            "SPY": [0.8, 0.3, 0.8],
            "SHY": [0.2, 0.7, 0.2],
        },
        index=rebalance_idx,
    )
    turnover = pd.Series([0.0, 1.0, 1.0], index=rebalance_idx, name="turnover")
    regimes = pd.Series(["low", "high", "low"], index=rebalance_idx)
    return StrategyResult(weights=weights, turnover=turnover, regime_labels=regimes)


def test_compute_portfolio_returns_synthetic() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    returns = pd.DataFrame({"SPY": [0.01, -0.02], "SHY": [0.0, 0.01]}, index=idx)
    weights = pd.DataFrame({"SPY": [0.8, 0.2], "SHY": [0.2, 0.8]}, index=idx)

    output = compute_portfolio_returns(returns, weights)

    assert np.isclose(output.iloc[0], 0.008)
    assert np.isclose(output.iloc[1], 0.004)


def test_compute_equity_curve_compounding() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.10, -0.05, 0.02], index=idx)

    equity = compute_equity_curve(returns, initial_value=1.0)

    assert np.isclose(equity.iloc[0], 1.10)
    assert np.isclose(equity.iloc[1], 1.045)
    assert np.isclose(equity.iloc[2], 1.0659)


def test_transaction_cost_series_applied_on_rebalance_dates_only() -> None:
    returns = _returns_frame()
    strategy = _strategy_result()

    cost = compute_transaction_cost_series(
        weight_frame=strategy.weights,
        returns_index=returns.index,
        transaction_cost_bps=5.0,
    )

    # next-day application => rebalance 1/3 affects 1/4, rebalance 1/5 affects 1/6
    assert np.isclose(cost.loc["2024-01-04"], 0.0005)
    assert np.isclose(cost.loc["2024-01-06"], 0.0005)
    assert np.isclose(cost.sum(), 0.001)


def test_expand_weights_to_daily_returns_next_day_convention() -> None:
    returns = _returns_frame()
    strategy = _strategy_result()

    expanded = expand_weights_to_daily_returns(strategy.weights, returns.index)

    # weights from 1/1 apply starting 1/2
    assert np.isnan(expanded.loc["2024-01-01", "SPY"])
    assert np.isclose(expanded.loc["2024-01-02", "SPY"], 0.8)
    # rebalance at 1/3 applies on 1/4
    assert np.isclose(expanded.loc["2024-01-04", "SPY"], 0.3)


def test_run_backtest_returns_aligned_strategy_series() -> None:
    returns = _returns_frame()
    strategy = _strategy_result()
    cfg = ProjectConfig(strategy=StrategyConfig(transaction_cost_bps=5.0))

    result = run_backtest(returns=returns, strategy_result=strategy, config=cfg)

    assert result.strategy_daily_returns.index.equals(result.strategy_equity_curve.index)
    assert result.strategy_daily_returns.index.equals(result.strategy_daily_weights.index)
    assert result.strategy_daily_returns.index.is_monotonic_increasing


def test_benchmark_weights_and_returns_constructed_correctly() -> None:
    returns = _returns_frame()
    idx = returns.index[1:]

    bmk = build_static_benchmark_weights(index=idx, weights_dict={"SPY": 0.8, "SHY": 0.2})
    assert (bmk["SPY"] == 0.8).all()
    assert (bmk["SHY"] == 0.2).all()


def test_no_future_leakage_through_weight_timing() -> None:
    returns = _returns_frame()
    strategy = _strategy_result()
    expanded = expand_weights_to_daily_returns(strategy.weights, returns.index)

    # first day has no active weight, so rebalance-date decision is not applied same-day
    assert expanded.loc[returns.index[0]].isna().all()


def test_missing_assets_or_unsorted_indexes_raise_clear_errors() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-01"])
    returns_unsorted = pd.DataFrame({"SPY": [0.01, 0.02], "SHY": [0.0, 0.0]}, index=idx)
    strategy = _strategy_result()
    cfg = ProjectConfig()

    with pytest.raises(ValueError, match="sorted ascending"):
        run_backtest(returns=returns_unsorted, strategy_result=strategy, config=cfg)

    returns = _returns_frame().drop(columns=["SHY"])
    with pytest.raises(ValueError, match="missing"):
        run_backtest(returns=returns, strategy_result=strategy, config=cfg)


def test_compute_portfolio_returns_raises_on_nan_inputs() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="D")
    returns = pd.DataFrame({"SPY": [0.01, float("nan")], "SHY": [0.0, 0.0]}, index=idx)
    weights = pd.DataFrame({"SPY": [0.8, 0.8], "SHY": [0.2, 0.2]}, index=idx)

    with pytest.raises(ValueError, match="must not contain NaN"):
        compute_portfolio_returns(returns, weights)
