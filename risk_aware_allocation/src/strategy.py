"""Decision rules that convert risk forecasts into portfolio allocations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .config import ProjectConfig


HIGH_RISK_LABEL = "high_risk_forecast"
LOW_RISK_LABEL = "low_risk_forecast"


@dataclass(slots=True)
class StrategyResult:
    """Container for generated strategy weights and turnover diagnostics."""

    weights: pd.DataFrame
    turnover: pd.Series
    regime_labels: pd.Series
    regime_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def map_prediction_to_weights(
    prediction_value: float,
    config: ProjectConfig,
) -> tuple[dict[str, float], str]:
    """Map one volatility prediction to a configured weight regime."""
    threshold = config.strategy.volatility_threshold
    if prediction_value >= threshold:
        return dict(config.strategy.high_risk_weights), HIGH_RISK_LABEL
    return dict(config.strategy.low_risk_weights), LOW_RISK_LABEL


def build_weight_frame(
    predictions: pd.Series | pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a dated weight frame from predicted volatility values."""
    if isinstance(predictions, pd.DataFrame):
        if predictions.shape[1] != 1:
            raise ValueError("predictions DataFrame must have exactly one column.")
        pred_series = predictions.iloc[:, 0]
    else:
        pred_series = predictions.copy()

    if pred_series.empty:
        raise ValueError("predictions cannot be empty.")

    if not isinstance(pred_series.index, pd.DatetimeIndex):
        raise ValueError("predictions must be indexed by DatetimeIndex.")

    if pred_series.index.has_duplicates:
        raise ValueError("predictions index must not contain duplicate dates.")

    pred_series = pred_series.sort_index()

    if pred_series.isna().any():
        raise ValueError("predictions contain NaN values; clean inputs before strategy mapping.")

    tickers = list(config.data.tickers)
    high_keys = set(config.strategy.high_risk_weights.keys())
    low_keys = set(config.strategy.low_risk_weights.keys())
    if high_keys != low_keys:
        raise ValueError("high_risk_weights and low_risk_weights must share identical ticker keys.")

    if set(tickers) != high_keys:
        raise ValueError("Ticker set in config.data.tickers must match strategy weight keys.")

    weight_rows: list[dict[str, float]] = []
    regime_labels: list[str] = []

    for value in pred_series.astype(float):
        mapped_weights, label = map_prediction_to_weights(value, config=config)
        weight_rows.append({ticker: mapped_weights[ticker] for ticker in tickers})
        regime_labels.append(label)

    weights = pd.DataFrame(weight_rows, index=pred_series.index)
    regimes = pd.Series(regime_labels, index=pred_series.index, name="regime")

    return weights, regimes


def compute_turnover(weight_frame: pd.DataFrame) -> pd.Series:
    """Compute turnover as sum(abs(w_t - w_{t-1})) with first row set to 0."""
    if weight_frame.empty:
        raise ValueError("weight_frame cannot be empty.")

    diffs = weight_frame.diff().abs().sum(axis=1)
    diffs.iloc[0] = 0.0
    return diffs.rename("turnover")


def summarize_weight_regimes(
    weight_frame: pd.DataFrame,
    regime_labels: pd.Series,
) -> dict[str, int]:
    """Summarize count of each regime label."""
    if not weight_frame.index.equals(regime_labels.index):
        raise ValueError("weight_frame and regime_labels must share identical index.")

    return {
        HIGH_RISK_LABEL: int((regime_labels == HIGH_RISK_LABEL).sum()),
        LOW_RISK_LABEL: int((regime_labels == LOW_RISK_LABEL).sum()),
    }


def run_strategy_from_predictions(
    predictions: pd.Series | pd.DataFrame,
    config: ProjectConfig,
) -> StrategyResult:
    """Generate weights, turnover, and regime summaries from model predictions."""
    weights, regimes = build_weight_frame(predictions=predictions, config=config)
    turnover = compute_turnover(weights)
    regime_counts = summarize_weight_regimes(weight_frame=weights, regime_labels=regimes)

    metadata = {
        "threshold": config.strategy.volatility_threshold,
        "tickers": list(config.data.tickers),
        "turnover_definition": "sum(abs(w_t - w_{t-1})), first row set to 0",
    }

    return StrategyResult(
        weights=weights,
        turnover=turnover,
        regime_labels=regimes,
        regime_counts=regime_counts,
        metadata=metadata,
    )
