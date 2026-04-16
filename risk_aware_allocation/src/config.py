"""Typed configuration models for the risk-aware allocation project."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Mapping

DATE_FORMAT = "%Y-%m-%d"
WEIGHT_TOLERANCE = 1e-6


@dataclass(slots=True)
class DataConfig:
    """Configuration for data sourcing and local caching."""

    start_date: str = "2003-01-01"
    end_date: str | None = None
    tickers: tuple[str, ...] = ("SPY", "SHY")
    benchmark_tickers: tuple[str, ...] = ("SPY", "SHY")
    cache_raw_dir: Path = Path("risk_aware_allocation/data/raw")
    cache_processed_dir: Path = Path("risk_aware_allocation/data/processed")
    price_field: str = "Adj Close"

    def validate(self) -> None:
        """Validate data settings for basic format and completeness."""
        self._validate_date(self.start_date, field_name="start_date")
        if self.end_date is not None:
            self._validate_date(self.end_date, field_name="end_date")

        if not self.tickers:
            raise ValueError("tickers must contain at least one symbol.")

    @staticmethod
    def _validate_date(date_value: str, field_name: str) -> None:
        """Validate that a date string is parseable with YYYY-MM-DD format."""
        try:
            datetime.strptime(date_value, DATE_FORMAT)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must use YYYY-MM-DD format: received {date_value!r}."
            ) from exc


@dataclass(slots=True)
class FeatureConfig:
    """Configuration for leakage-safe feature and label windows."""

    short_return_window: int = 5
    medium_return_window: int = 20
    short_vol_window: int = 20
    long_vol_window: int = 60
    moving_average_short: int = 20
    moving_average_long: int = 50
    downside_vol_window: int = 20
    forward_vol_horizon: int = 5

    def validate(self) -> None:
        """Validate feature window settings."""
        if self.moving_average_short >= self.moving_average_long:
            raise ValueError(
                "moving_average_short must be smaller than moving_average_long."
            )


@dataclass(slots=True)
class SplitConfig:
    """Configuration for walk-forward split behavior."""

    rebalance_frequency: str = "W-FRI"
    train_window_days: int = 252 * 5
    test_window_days: int = 63
    walk_forward_mode: Literal["rolling", "expanding"] = "expanding"
    minimum_history_required: int = 252


@dataclass(slots=True)
class StrategyConfig:
    """Configuration for risk-threshold allocation decisions."""

    volatility_threshold: float = 0.18
    high_risk_weights: dict[str, float] = field(
        default_factory=lambda: {"SPY": 0.30, "SHY": 0.70}
    )
    low_risk_weights: dict[str, float] = field(
        default_factory=lambda: {"SPY": 0.80, "SHY": 0.20}
    )
    transaction_cost_bps: float = 5.0

    def validate(self) -> None:
        """Validate strategy weight sanity and transaction cost assumptions."""
        self._validate_weights(self.high_risk_weights, label="high_risk_weights")
        self._validate_weights(self.low_risk_weights, label="low_risk_weights")

        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative.")

    @staticmethod
    def _validate_weights(weights: Mapping[str, float], label: str) -> None:
        """Validate non-empty long-only weights that sum to one within tolerance."""
        if not weights:
            raise ValueError(f"{label} cannot be empty.")

        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > WEIGHT_TOLERANCE:
            raise ValueError(f"{label} must sum to 1.0 within tolerance.")


@dataclass(slots=True)
class PlotConfig:
    """Configuration for project plotting conventions."""

    figure_size: tuple[int, int] = (12, 6)
    rolling_vol_plot_window: int = 63
    date_format: str = "%Y-%m"


@dataclass(slots=True)
class ProjectConfig:
    """Top-level project configuration combining all sub-config groups."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    def validate(self) -> None:
        """Run lightweight sanity checks across all configuration groups."""
        self.data.validate()
        self.features.validate()
        self.strategy.validate()

    def with_base_dir(self, base_dir: Path) -> ProjectConfig:
        """Return a copy with data cache paths anchored to a base directory."""
        resolved_base = base_dir.resolve()
        return ProjectConfig(
            data=DataConfig(
                start_date=self.data.start_date,
                end_date=self.data.end_date,
                tickers=self.data.tickers,
                benchmark_tickers=self.data.benchmark_tickers,
                cache_raw_dir=resolved_base / "risk_aware_allocation/data/raw",
                cache_processed_dir=resolved_base / "risk_aware_allocation/data/processed",
                price_field=self.data.price_field,
            ),
            features=self.features,
            split=self.split,
            strategy=self.strategy,
            plot=self.plot,
        )


def get_default_config(base_dir: Path | None = None) -> ProjectConfig:
    """Build and validate a default project configuration instance.

    Args:
        base_dir: Optional repository root path for anchoring cache directories.

    Returns:
        A validated ``ProjectConfig`` for V1 defaults.
    """
    config = ProjectConfig()
    if base_dir is not None:
        config = config.with_base_dir(base_dir)

    config.validate()
    return config
