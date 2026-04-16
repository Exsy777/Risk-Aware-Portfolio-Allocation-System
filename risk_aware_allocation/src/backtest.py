"""Backtesting engine for portfolio simulation with transaction costs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import ProjectConfig
from .strategy import StrategyResult, compute_turnover


TIMING_CONVENTION = "next_day_apply"


@dataclass(slots=True)
class BacktestResult:
    """Container for strategy and benchmark backtest outputs."""

    strategy_daily_returns: pd.Series
    strategy_equity_curve: pd.Series
    strategy_daily_weights: pd.DataFrame
    turnover: pd.Series
    transaction_cost_series: pd.Series
    transaction_cost_total: float
    benchmark_returns: dict[str, pd.Series] = field(default_factory=dict)
    benchmark_equity_curves: dict[str, pd.Series] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_sorted_datetime_index(index: pd.DatetimeIndex, name: str) -> None:
    """Validate that index is DatetimeIndex and sorted ascending."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a DatetimeIndex.")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be sorted ascending.")
    if index.has_duplicates:
        raise ValueError(f"{name} must not contain duplicate dates.")


def expand_weights_to_daily_returns(
    weight_frame: pd.DataFrame,
    returns_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Expand rebalance-date weights to daily index using next-day application.

    Timing convention: weights decided at date ``t`` apply starting from the next
    available return date strictly after ``t``.
    """
    _validate_sorted_datetime_index(weight_frame.index, "weight_frame.index")
    _validate_sorted_datetime_index(returns_index, "returns_index")

    if weight_frame.empty:
        raise ValueError("weight_frame cannot be empty.")

    daily_weights = pd.DataFrame(index=returns_index, columns=weight_frame.columns, dtype=float)

    for rebalance_date, row in weight_frame.iterrows():
        apply_pos = returns_index.searchsorted(rebalance_date, side="right")
        if apply_pos < len(returns_index):
            apply_date = returns_index[apply_pos]
            daily_weights.loc[apply_date, :] = row.values

    daily_weights = daily_weights.ffill()
    return daily_weights


def compute_portfolio_returns(
    daily_returns: pd.DataFrame,
    daily_weights: pd.DataFrame,
) -> pd.Series:
    """Compute daily portfolio returns as weighted sum of asset returns."""
    _validate_sorted_datetime_index(daily_returns.index, "daily_returns.index")
    _validate_sorted_datetime_index(daily_weights.index, "daily_weights.index")

    if daily_returns.empty or daily_weights.empty:
        raise ValueError("daily_returns and daily_weights cannot be empty.")

    if daily_returns.isna().any().any() or daily_weights.isna().any().any():
        raise ValueError("daily_returns and daily_weights must not contain NaN values.")

    if not daily_returns.index.equals(daily_weights.index):
        raise ValueError("daily_returns and daily_weights must share identical indexes.")

    if list(daily_returns.columns) != list(daily_weights.columns):
        raise ValueError("daily_returns and daily_weights must have identical columns in the same order.")

    return (daily_returns * daily_weights).sum(axis=1).rename("portfolio_return")


def compute_transaction_cost_series(
    weight_frame: pd.DataFrame,
    returns_index: pd.DatetimeIndex,
    transaction_cost_bps: float,
) -> pd.Series:
    """Compute transaction costs mapped to daily index under next-day application.

    Cost on applied rebalance date: ``turnover * (transaction_cost_bps / 10000)``.
    """
    if transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps must be non-negative.")

    _validate_sorted_datetime_index(weight_frame.index, "weight_frame.index")
    _validate_sorted_datetime_index(returns_index, "returns_index")

    turnover = compute_turnover(weight_frame)
    costs = pd.Series(0.0, index=returns_index, name="transaction_cost")

    for rebalance_date, turn in turnover.items():
        apply_pos = returns_index.searchsorted(rebalance_date, side="right")
        if apply_pos < len(returns_index):
            apply_date = returns_index[apply_pos]
            costs.loc[apply_date] += float(turn) * (transaction_cost_bps / 10000.0)

    return costs


def compute_equity_curve(return_series: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Compound return series into an equity curve."""
    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")

    _validate_sorted_datetime_index(return_series.index, "return_series.index")
    return ((1.0 + return_series).cumprod() * initial_value).rename("equity_curve")


def build_static_benchmark_weights(
    index: pd.DatetimeIndex,
    weights_dict: dict[str, float],
) -> pd.DataFrame:
    """Build constant benchmark weights across a date index."""
    _validate_sorted_datetime_index(index, "index")

    total = sum(weights_dict.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Benchmark weights must sum to 1.0.")

    return pd.DataFrame([weights_dict] * len(index), index=index)


def run_backtest(
    returns: pd.DataFrame,
    strategy_result: StrategyResult,
    config: ProjectConfig,
) -> BacktestResult:
    """Run strategy and benchmark backtests from returns and strategy weights."""
    _validate_sorted_datetime_index(returns.index, "returns.index")

    if returns.empty:
        raise ValueError("returns cannot be empty.")
    _validate_sorted_datetime_index(strategy_result.weights.index, "strategy_result.weights.index")

    if not set(strategy_result.weights.columns).issubset(set(returns.columns)):
        raise ValueError("Strategy weights contain assets missing from returns columns.")

    asset_columns = list(strategy_result.weights.columns)
    aligned_returns = returns.loc[:, asset_columns]

    daily_weights_full = expand_weights_to_daily_returns(
        weight_frame=strategy_result.weights,
        returns_index=aligned_returns.index,
    )

    valid_mask = ~daily_weights_full.isna().any(axis=1)
    daily_weights = daily_weights_full.loc[valid_mask]
    daily_returns = aligned_returns.loc[valid_mask]

    gross_strategy_returns = compute_portfolio_returns(
        daily_returns=daily_returns,
        daily_weights=daily_weights,
    )

    transaction_costs_full = compute_transaction_cost_series(
        weight_frame=strategy_result.weights,
        returns_index=aligned_returns.index,
        transaction_cost_bps=config.strategy.transaction_cost_bps,
    )
    transaction_costs = transaction_costs_full.loc[valid_mask]

    net_strategy_returns = (gross_strategy_returns - transaction_costs).rename("strategy_return")
    strategy_equity = compute_equity_curve(net_strategy_returns)

    benchmark_defs = {
        "benchmark_80_20": {"SPY": 0.80, "SHY": 0.20},
        "benchmark_60_40": {"SPY": 0.60, "SHY": 0.40},
    }

    benchmark_returns: dict[str, pd.Series] = {}
    benchmark_equity: dict[str, pd.Series] = {}

    for name, weights_dict in benchmark_defs.items():
        benchmark_weights = build_static_benchmark_weights(index=daily_returns.index, weights_dict=weights_dict)
        benchmark_daily_returns = compute_portfolio_returns(daily_returns=daily_returns, daily_weights=benchmark_weights)
        benchmark_returns[name] = benchmark_daily_returns.rename(name)
        benchmark_equity[name] = compute_equity_curve(benchmark_daily_returns).rename(name)

    metadata = {
        "timing_convention": TIMING_CONVENTION,
        "transaction_cost_bps": config.strategy.transaction_cost_bps,
        "benchmark_transaction_costs_applied": False,
        "n_strategy_days": int(len(daily_returns)),
    }

    return BacktestResult(
        strategy_daily_returns=net_strategy_returns,
        strategy_equity_curve=strategy_equity,
        strategy_daily_weights=daily_weights,
        turnover=compute_turnover(strategy_result.weights),
        transaction_cost_series=transaction_costs,
        transaction_cost_total=float(transaction_costs.sum()),
        benchmark_returns=benchmark_returns,
        benchmark_equity_curves=benchmark_equity,
        metadata=metadata,
    )
