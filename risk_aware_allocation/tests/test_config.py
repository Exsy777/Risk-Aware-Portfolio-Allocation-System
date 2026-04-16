"""Tests for configuration defaults and sanity validation."""

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DataConfig, FeatureConfig, ProjectConfig, StrategyConfig, get_default_config


def test_default_config_builds_and_validates() -> None:
    """Default config should initialize and pass validation checks."""
    config = get_default_config()

    assert isinstance(config, ProjectConfig)
    assert config.data.tickers == ("SPY", "SHY")
    assert config.features.forward_vol_horizon == 5


def test_default_weight_dicts_sum_to_one() -> None:
    """High/low risk weights should each sum to 1.0."""
    strategy = StrategyConfig()

    assert pytest.approx(sum(strategy.high_risk_weights.values()), rel=0.0, abs=1e-9) == 1.0
    assert pytest.approx(sum(strategy.low_risk_weights.values()), rel=0.0, abs=1e-9) == 1.0


def test_invalid_moving_average_order_raises() -> None:
    """Short moving average window cannot be greater than or equal to long window."""
    config = ProjectConfig(features=FeatureConfig(moving_average_short=50, moving_average_long=20))

    with pytest.raises(ValueError, match="moving_average_short"):
        config.validate()


def test_empty_ticker_list_raises() -> None:
    """Ticker list must be non-empty."""
    config = ProjectConfig(data=DataConfig(tickers=()))

    with pytest.raises(ValueError, match="tickers"):
        config.validate()
