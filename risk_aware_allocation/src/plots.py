"""Plotting utilities for visualizing backtest and sensitivity outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curves(
    strategy_equity: pd.Series,
    benchmark_equity_curves: dict[str, pd.Series],
):
    """Plot strategy and benchmark equity curves on one chart."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(strategy_equity.index, strategy_equity.values, label="dynamic_strategy", linewidth=2)
    for name, series in benchmark_equity_curves.items():
        ax.plot(series.index, series.values, label=name, alpha=0.85)
    ax.set_title("Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_drawdowns(
    strategy_equity: pd.Series,
    benchmark_equity_curves: dict[str, pd.Series],
):
    """Plot drawdown curves for strategy and benchmarks."""

    def _drawdown(series: pd.Series) -> pd.Series:
        return (series / series.cummax()) - 1.0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(strategy_equity.index, _drawdown(strategy_equity), label="dynamic_strategy", linewidth=2)
    for name, series in benchmark_equity_curves.items():
        ax.plot(series.index, _drawdown(series), label=name, alpha=0.85)
    ax.set_title("Drawdown Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_predictions_vs_actuals(predictions: pd.Series, actuals: pd.Series):
    """Plot predicted versus realized forward volatility."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(predictions.index, predictions.values, label="predicted_vol", linewidth=2)
    ax.plot(actuals.index, actuals.values, label="realized_forward_vol", alpha=0.8)
    ax.set_title("Predicted vs Realized Forward Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def plot_weights(weight_frame: pd.DataFrame):
    """Plot allocation weights over time."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for column in weight_frame.columns:
        ax.plot(weight_frame.index, weight_frame[column], label=column)
    ax.set_title("Strategy Weights Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    return fig


def _plot_metric_vs_threshold(results: pd.DataFrame, metric: str, title: str):
    """Generic helper to plot a metric against threshold values."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(results["threshold"], results[metric], marker="o")
    ax.set_xlabel("Volatility Threshold")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return fig


def plot_threshold_sensitivity_sharpe(results: pd.DataFrame):
    """Plot Sharpe ratio versus threshold."""
    return _plot_metric_vs_threshold(results, metric="sharpe_ratio", title="Sharpe Ratio vs Threshold")


def plot_threshold_sensitivity_drawdown(results: pd.DataFrame):
    """Plot max drawdown versus threshold."""
    return _plot_metric_vs_threshold(results, metric="max_drawdown", title="Max Drawdown vs Threshold")


def plot_threshold_sensitivity_turnover(results: pd.DataFrame):
    """Plot turnover versus threshold."""
    return _plot_metric_vs_threshold(results, metric="turnover", title="Turnover vs Threshold")


# Backward-compatible aliases used in earlier phases.
plot_sharpe_vs_threshold = plot_threshold_sensitivity_sharpe
plot_max_drawdown_vs_threshold = plot_threshold_sensitivity_drawdown
plot_turnover_vs_threshold = plot_threshold_sensitivity_turnover
