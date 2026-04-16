"""Threshold sensitivity analysis utilities for risk-aware allocation."""

from __future__ import annotations

import copy

import pandas as pd

from .backtest import run_backtest
from .config import ProjectConfig
from .metrics import calculate_performance_metrics
from .strategy import run_strategy_from_predictions


REQUIRED_RESULT_COLUMNS = [
    "threshold",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "calmar_ratio",
    "turnover",
    "total_transaction_cost",
    "num_rebalances",
]


def run_threshold_sensitivity(
    predictions: pd.Series | pd.DataFrame,
    returns: pd.DataFrame,
    config: ProjectConfig,
    threshold_values: list[float] | pd.Series,
) -> pd.DataFrame:
    """Run threshold sensitivity by reusing strategy and backtest pipeline per threshold."""
    thresholds = sorted(float(value) for value in threshold_values)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        scenario_config = copy.deepcopy(config)
        scenario_config.strategy.volatility_threshold = threshold

        strategy_result = run_strategy_from_predictions(predictions=predictions, config=scenario_config)
        backtest_result = run_backtest(
            returns=returns,
            strategy_result=strategy_result,
            config=scenario_config,
        )

        metrics = calculate_performance_metrics(
            daily_returns=backtest_result.strategy_daily_returns,
            turnover_series=backtest_result.turnover,
            total_transaction_cost=backtest_result.transaction_cost_total,
            num_rebalances=int((backtest_result.turnover > 0).sum()),
        )

        rows.append({"threshold": threshold, **metrics})

    result = pd.DataFrame(rows)
    return result.loc[:, REQUIRED_RESULT_COLUMNS]


def summarize_threshold_results(results: pd.DataFrame) -> dict[str, float]:
    """Summarize threshold sensitivity outcomes using simple diagnostics."""
    if results.empty:
        raise ValueError("results cannot be empty.")

    best_sharpe_row = results.sort_values("sharpe_ratio", ascending=False).iloc[0]
    lowest_drawdown_row = results.sort_values("max_drawdown", ascending=True).iloc[0]

    return {
        "best_sharpe_threshold": float(best_sharpe_row["threshold"]),
        "best_sharpe_ratio": float(best_sharpe_row["sharpe_ratio"]),
        "lowest_drawdown_threshold": float(lowest_drawdown_row["threshold"]),
        "lowest_max_drawdown": float(lowest_drawdown_row["max_drawdown"]),
        "threshold_count": int(len(results)),
    }
