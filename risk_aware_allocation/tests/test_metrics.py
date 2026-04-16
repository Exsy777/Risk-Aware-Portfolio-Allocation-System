"""Tests for performance metrics utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    annualized_return,
    annualized_volatility,
    calculate_performance_metrics,
    compute_max_drawdown,
    sharpe_ratio,
)


def test_annualized_return_deterministic() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.01, 0.02, 0.00], index=idx)

    out = annualized_return(returns, periods_per_year=252)

    assert np.isclose(out, (0.01 + 0.02 + 0.00) / 3 * 252)


def test_zero_volatility_sharpe_nan() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.Series([0.0] * 5, index=idx)

    vol = annualized_volatility(returns)
    sr = sharpe_ratio(returns)

    assert np.isclose(vol, 0.0)
    assert np.isnan(sr)


def test_compute_max_drawdown_exact_value() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    equity = pd.Series([1.0, 1.2, 1.1, 1.3, 1.0], index=idx)

    dd = compute_max_drawdown(equity)

    # 1.0 / 1.3 - 1 = -0.230769...
    assert np.isclose(dd, (1.0 / 1.3) - 1.0)


def test_calculate_performance_metrics_columns_consistent() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = pd.Series([0.01, -0.01, 0.02, 0.00, 0.01], index=idx)
    turnover = pd.Series([0.0, 1.0, 0.0, 0.5, 0.0], index=idx)

    metrics = calculate_performance_metrics(
        daily_returns=returns,
        turnover_series=turnover,
        total_transaction_cost=0.001,
    )

    expected_keys = {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "turnover",
        "total_transaction_cost",
        "num_rebalances",
    }
    assert set(metrics.keys()) == expected_keys
