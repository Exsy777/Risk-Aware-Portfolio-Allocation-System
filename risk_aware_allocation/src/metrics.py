"""Performance metric calculations for strategy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(return_series: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compute annualized arithmetic return from periodic returns."""
    if return_series.empty:
        raise ValueError("return_series cannot be empty.")
    return float(return_series.mean() * periods_per_year)


def annualized_volatility(return_series: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Compute annualized volatility from periodic returns."""
    if return_series.empty:
        raise ValueError("return_series cannot be empty.")
    return float(return_series.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    return_series: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute annualized Sharpe ratio from periodic returns."""
    ann_return = annualized_return(return_series, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(return_series, periods_per_year=periods_per_year)
    if ann_vol == 0:
        return float("nan")
    return float((ann_return - risk_free_rate) / ann_vol)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    if equity_curve.empty:
        raise ValueError("equity_curve cannot be empty.")
    running_max = equity_curve.cummax()
    drawdown = (equity_curve / running_max) - 1.0
    return float(drawdown.min())


def calmar_ratio(
    return_series: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute Calmar ratio as annualized return divided by absolute max drawdown."""
    ann_return = annualized_return(return_series, periods_per_year=periods_per_year)
    max_dd = compute_max_drawdown(equity_curve)
    if max_dd >= 0:
        return float("nan")
    return float(ann_return / abs(max_dd))


def calculate_performance_metrics(
    daily_returns: pd.Series,
    turnover_series: pd.Series | None = None,
    total_transaction_cost: float = 0.0,
    num_rebalances: int | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict[str, float]:
    """Calculate coherent performance metrics for one return stream."""
    equity_curve = (1.0 + daily_returns).cumprod()

    turnover = float(turnover_series.sum()) if turnover_series is not None else np.nan
    if num_rebalances is None:
        num_rebalances = int((turnover_series > 0).sum()) if turnover_series is not None else 0

    return {
        "annualized_return": annualized_return(daily_returns, periods_per_year=periods_per_year),
        "annualized_volatility": annualized_volatility(daily_returns, periods_per_year=periods_per_year),
        "sharpe_ratio": sharpe_ratio(daily_returns, periods_per_year=periods_per_year),
        "max_drawdown": compute_max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(daily_returns, equity_curve, periods_per_year=periods_per_year),
        "turnover": float(turnover),
        "total_transaction_cost": float(total_transaction_cost),
        "num_rebalances": int(num_rebalances),
    }


def summarize_backtest_result(backtest_result: object) -> pd.DataFrame:
    """Create a comparable metrics table for strategy and benchmarks."""
    strategy_metrics = calculate_performance_metrics(
        daily_returns=backtest_result.strategy_daily_returns,
        turnover_series=backtest_result.turnover,
        total_transaction_cost=backtest_result.transaction_cost_total,
        num_rebalances=int((backtest_result.turnover > 0).sum()),
    )

    rows: list[dict[str, float | str]] = [{"portfolio": "dynamic_strategy", **strategy_metrics}]

    for name, returns in backtest_result.benchmark_returns.items():
        benchmark_metrics = calculate_performance_metrics(
            daily_returns=returns,
            turnover_series=None,
            total_transaction_cost=0.0,
            num_rebalances=0,
        )
        rows.append({"portfolio": name, **benchmark_metrics})

    return pd.DataFrame(rows).set_index("portfolio")
