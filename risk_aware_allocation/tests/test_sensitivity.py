"""Tests for threshold sensitivity analysis layer."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ProjectConfig, StrategyConfig
from src.sensitivity import REQUIRED_RESULT_COLUMNS, run_threshold_sensitivity


def _inputs() -> tuple[pd.Series, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    predictions = pd.Series([0.10, 0.15, 0.20, 0.25, 0.18, 0.22, 0.12, 0.30], index=idx)
    returns = pd.DataFrame(
        {
            "SPY": [0.01, -0.02, 0.005, 0.01, -0.005, 0.007, 0.002, -0.003],
            "SHY": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
        },
        index=idx,
    )
    return predictions, returns


def test_one_row_per_threshold_and_sorted_order() -> None:
    predictions, returns = _inputs()
    config = ProjectConfig(strategy=StrategyConfig(volatility_threshold=0.2))

    result = run_threshold_sensitivity(
        predictions=predictions,
        returns=returns,
        config=config,
        threshold_values=[0.25, 0.10, 0.20],
    )

    assert len(result) == 3
    assert list(result["threshold"]) == [0.10, 0.20, 0.25]


def test_output_includes_required_columns() -> None:
    predictions, returns = _inputs()
    config = ProjectConfig()

    result = run_threshold_sensitivity(
        predictions=predictions,
        returns=returns,
        config=config,
        threshold_values=[0.15, 0.25],
    )

    assert list(result.columns) == REQUIRED_RESULT_COLUMNS


def test_different_thresholds_can_change_turnover_or_costs() -> None:
    predictions, returns = _inputs()
    config = ProjectConfig()

    result = run_threshold_sensitivity(
        predictions=predictions,
        returns=returns,
        config=config,
        threshold_values=[0.12, 0.28],
    )

    assert result["turnover"].nunique() > 1 or result["total_transaction_cost"].nunique() > 1


def test_original_config_not_mutated() -> None:
    predictions, returns = _inputs()
    config = ProjectConfig(strategy=StrategyConfig(volatility_threshold=0.21))

    _ = run_threshold_sensitivity(
        predictions=predictions,
        returns=returns,
        config=config,
        threshold_values=[0.10, 0.20, 0.30],
    )

    assert config.strategy.volatility_threshold == 0.21


def test_empty_threshold_values_raises() -> None:
    predictions, returns = _inputs()
    config = ProjectConfig()

    with pytest.raises(ValueError, match="threshold_values cannot be empty"):
        run_threshold_sensitivity(
            predictions=predictions,
            returns=returns,
            config=config,
            threshold_values=[],
        )
