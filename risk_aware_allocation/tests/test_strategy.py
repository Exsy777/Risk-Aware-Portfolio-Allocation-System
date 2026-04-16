"""Tests for prediction-to-weight strategy mapping logic."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ProjectConfig, StrategyConfig
from src.strategy import (
    HIGH_RISK_LABEL,
    LOW_RISK_LABEL,
    build_weight_frame,
    compute_turnover,
    map_prediction_to_weights,
    run_strategy_from_predictions,
    summarize_weight_regimes,
)


def _config() -> ProjectConfig:
    return ProjectConfig(
        strategy=StrategyConfig(
            volatility_threshold=0.20,
            high_risk_weights={"SPY": 0.30, "SHY": 0.70},
            low_risk_weights={"SPY": 0.80, "SHY": 0.20},
        )
    )


def test_prediction_above_threshold_maps_to_high_risk_weights() -> None:
    cfg = _config()
    weights, regime = map_prediction_to_weights(0.25, cfg)

    assert regime == HIGH_RISK_LABEL
    assert weights == {"SPY": 0.30, "SHY": 0.70}


def test_prediction_below_threshold_maps_to_low_risk_weights() -> None:
    cfg = _config()
    weights, regime = map_prediction_to_weights(0.10, cfg)

    assert regime == LOW_RISK_LABEL
    assert weights == {"SPY": 0.80, "SHY": 0.20}


def test_weight_rows_sum_to_one() -> None:
    cfg = _config()
    idx = pd.date_range("2024-01-01", periods=4, freq="W-FRI")
    preds = pd.Series([0.10, 0.25, 0.21, 0.15], index=idx)

    weights, _ = build_weight_frame(predictions=preds, config=cfg)

    assert (weights.sum(axis=1) == 1.0).all()


def test_weight_frame_preserves_prediction_index_order() -> None:
    cfg = _config()
    idx = pd.to_datetime(["2024-01-12", "2024-01-05", "2024-01-19"])
    preds = pd.Series([0.10, 0.25, 0.15], index=idx)

    weights, _ = build_weight_frame(predictions=preds, config=cfg)

    assert list(weights.index) == sorted(list(idx))


def test_ticker_columns_in_expected_order() -> None:
    cfg = _config()
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    preds = pd.Series([0.10, 0.21, 0.22], index=idx)

    weights, _ = build_weight_frame(predictions=preds, config=cfg)

    assert list(weights.columns) == ["SPY", "SHY"]


def test_turnover_computation_on_regime_switches() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    weights = pd.DataFrame(
        {
            "SPY": [0.80, 0.30, 0.30, 0.80],
            "SHY": [0.20, 0.70, 0.70, 0.20],
        },
        index=idx,
    )

    turnover = compute_turnover(weights)

    assert turnover.iloc[0] == 0.0
    assert turnover.iloc[1] == 1.0
    assert turnover.iloc[2] == 0.0
    assert turnover.iloc[3] == 1.0


def test_constant_predictions_have_zero_turnover_after_first_row() -> None:
    cfg = _config()
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    preds = pd.Series([0.1] * 5, index=idx)

    weights, _ = build_weight_frame(predictions=preds, config=cfg)
    turnover = compute_turnover(weights)

    assert turnover.iloc[0] == 0.0
    assert (turnover.iloc[1:] == 0.0).all()


def test_regime_summary_counts_work() -> None:
    cfg = _config()
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    preds = pd.Series([0.1, 0.25, 0.3, 0.05], index=idx)

    result = run_strategy_from_predictions(predictions=preds, config=cfg)
    summary = summarize_weight_regimes(weight_frame=result.weights, regime_labels=result.regime_labels)

    assert summary[HIGH_RISK_LABEL] == 2
    assert summary[LOW_RISK_LABEL] == 2
