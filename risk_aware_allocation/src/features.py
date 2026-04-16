"""Leakage-safe feature engineering for risk forecasting."""

from __future__ import annotations

import pandas as pd

from .config import ProjectConfig


SPY_TICKER = "SPY"
SHY_TICKER = "SHY"


def _validate_inputs(prices: pd.DataFrame, returns: pd.DataFrame) -> None:
    """Validate basic index/column prerequisites for feature construction."""
    if not prices.index.is_monotonic_increasing or not returns.index.is_monotonic_increasing:
        raise ValueError("prices and returns must use ascending DatetimeIndex order.")

    if SPY_TICKER not in prices.columns:
        raise ValueError("prices must include SPY column for V1 feature construction.")

    if SPY_TICKER not in returns.columns:
        raise ValueError("returns must include SPY column for V1 feature construction.")


def compute_lagged_returns(
    prices: pd.DataFrame,
    short_window: int,
    medium_window: int,
) -> pd.DataFrame:
    """Compute trailing multi-day returns from SPY prices."""
    spy_prices = prices[SPY_TICKER]
    features = pd.DataFrame(index=prices.index)
    features[f"spy_ret_{short_window}d"] = spy_prices.pct_change(short_window)
    features[f"spy_ret_{medium_window}d"] = spy_prices.pct_change(medium_window)
    return features


def compute_rolling_volatility(
    returns: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """Compute trailing rolling volatility from SPY daily returns."""
    spy_returns = returns[SPY_TICKER]
    features = pd.DataFrame(index=returns.index)
    features[f"spy_vol_{short_window}d"] = spy_returns.rolling(window=short_window).std()
    features[f"spy_vol_{long_window}d"] = spy_returns.rolling(window=long_window).std()
    return features


def compute_moving_averages(
    prices: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """Compute trailing moving averages and MA spread ratio for SPY."""
    spy_prices = prices[SPY_TICKER]
    features = pd.DataFrame(index=prices.index)
    short_col = f"spy_ma_{short_window}"
    long_col = f"spy_ma_{long_window}"

    features[short_col] = spy_prices.rolling(window=short_window).mean()
    features[long_col] = spy_prices.rolling(window=long_window).mean()
    features["spy_ma_spread_ratio"] = (features[short_col] / features[long_col]) - 1.0
    return features


def compute_drawdown_features(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute trailing drawdown proxy as price/rolling_max - 1 for SPY."""
    spy_prices = prices[SPY_TICKER]
    rolling_max = spy_prices.rolling(window=window).max()

    features = pd.DataFrame(index=prices.index)
    features[f"spy_drawdown_{window}d"] = (spy_prices / rolling_max) - 1.0
    return features


def compute_downside_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute trailing downside volatility using negative returns with positives clipped to zero."""
    spy_returns = returns[SPY_TICKER]
    downside_returns = spy_returns.clip(upper=0.0)

    features = pd.DataFrame(index=returns.index)
    features[f"spy_downside_vol_{window}d"] = downside_returns.rolling(window=window).std()
    return features


def build_feature_frame(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: ProjectConfig,
    include_shy_context: bool = True,
) -> pd.DataFrame:
    """Build full leakage-safe predictor matrix from aligned price and return inputs."""
    _validate_inputs(prices=prices, returns=returns)

    feature_config = config.features

    lagged = compute_lagged_returns(
        prices=prices,
        short_window=feature_config.short_return_window,
        medium_window=feature_config.medium_return_window,
    )
    rolling_vol = compute_rolling_volatility(
        returns=returns,
        short_window=feature_config.short_vol_window,
        long_window=feature_config.long_vol_window,
    )
    moving_averages = compute_moving_averages(
        prices=prices,
        short_window=feature_config.moving_average_short,
        long_window=feature_config.moving_average_long,
    )
    drawdown = compute_drawdown_features(
        prices=prices,
        window=feature_config.medium_return_window,
    )
    downside_vol = compute_downside_volatility(
        returns=returns,
        window=feature_config.downside_vol_window,
    )

    feature_frame = pd.concat(
        [lagged, rolling_vol, moving_averages, drawdown, downside_vol],
        axis=1,
    ).sort_index()

    if include_shy_context and SHY_TICKER in prices.columns:
        feature_frame[f"shy_ret_{feature_config.short_return_window}d"] = prices[SHY_TICKER].pct_change(
            feature_config.short_return_window
        )

    return feature_frame
