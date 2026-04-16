"""Forward label engineering for realized volatility targets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig


SPY_TICKER = "SPY"


def compute_forward_realized_volatility(
    returns: pd.DataFrame,
    horizon: int,
    column: str = SPY_TICKER,
) -> pd.Series:
    """Compute forward realized volatility from t+1 through t+horizon.

    The label at timestamp ``t`` is the standard deviation of the *future* return
    window ``[t+1, ..., t+horizon]``. Rows with insufficient future observations
    are left as NaN.
    """
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    if column not in returns.columns:
        raise ValueError(f"returns must include column {column!r}.")

    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted ascending.")

    series = returns[column]
    output = np.full(shape=len(series), fill_value=np.nan, dtype=float)

    for i in range(len(series)):
        future_window = series.iloc[i + 1 : i + 1 + horizon]
        if len(future_window) == horizon and future_window.notna().all():
            output[i] = float(future_window.std(ddof=1))

    return pd.Series(output, index=returns.index, name=f"spy_fwd_vol_{horizon}d")


def build_label_frame(returns: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Build a single-column forward volatility label DataFrame."""
    horizon = config.features.forward_vol_horizon
    label_series = compute_forward_realized_volatility(
        returns=returns,
        horizon=horizon,
        column=SPY_TICKER,
    )
    return label_series.to_frame()
