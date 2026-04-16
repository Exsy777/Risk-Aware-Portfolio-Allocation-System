"""Walk-forward time-series splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import ProjectConfig


@dataclass(slots=True)
class WalkForwardSplit:
    """Container describing one walk-forward train/test split."""

    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex


def validate_split_boundaries(train_idx: pd.DatetimeIndex, test_idx: pd.DatetimeIndex) -> None:
    """Validate temporal ordering and non-overlap for a split."""
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError("train and test index must both be non-empty.")

    if train_idx.max() >= test_idx.min():
        raise ValueError("train window must end strictly before test window starts.")

    if not train_idx.is_monotonic_increasing or not test_idx.is_monotonic_increasing:
        raise ValueError("train and test indexes must be sorted ascending.")

    overlap = train_idx.intersection(test_idx)
    if len(overlap) > 0:
        raise ValueError("train and test indexes must not overlap.")


def generate_walk_forward_splits(
    index: pd.DatetimeIndex,
    config: ProjectConfig,
) -> list[WalkForwardSplit]:
    """Generate deterministic walk-forward splits from a shared sorted index.

    Splits are generated using positional boundaries:
    - test windows have fixed length ``test_window_days``
    - each next split advances by ``test_window_days`` positions
    - training window is either expanding or rolling
    """
    split_cfg = config.split
    mode = split_cfg.walk_forward_mode
    train_window = split_cfg.train_window_days
    test_window = split_cfg.test_window_days
    min_history = split_cfg.minimum_history_required

    if split_cfg.rebalance_frequency == "":
        raise ValueError("rebalance_frequency cannot be empty.")

    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("index must be a pandas DatetimeIndex.")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted ascending.")
    if index.has_duplicates:
        raise ValueError("index must not contain duplicate timestamps.")

    if mode not in {"expanding", "rolling"}:
        raise ValueError("walk_forward_mode must be either 'expanding' or 'rolling'.")

    if test_window <= 0 or train_window <= 0 or min_history <= 0:
        raise ValueError("train_window_days, test_window_days, and minimum_history_required must be > 0.")

    split_start = max(train_window, min_history)
    splits: list[WalkForwardSplit] = []
    split_id = 0

    while split_start + test_window <= len(index):
        if mode == "expanding":
            train_start_pos = 0
        else:
            train_start_pos = split_start - train_window

        train_end_pos = split_start
        test_end_pos = split_start + test_window

        train_idx = index[train_start_pos:train_end_pos]
        test_idx = index[split_start:test_end_pos]

        validate_split_boundaries(train_idx=train_idx, test_idx=test_idx)

        splits.append(
            WalkForwardSplit(
                split_id=split_id,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                train_index=train_idx,
                test_index=test_idx,
            )
        )

        split_start += test_window
        split_id += 1

    return splits


def split_features_and_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[WalkForwardSplit]]:
    """Align feature/label indexes and generate walk-forward splits."""
    if not isinstance(features.index, pd.DatetimeIndex) or not isinstance(labels.index, pd.DatetimeIndex):
        raise ValueError("features and labels must both use DatetimeIndex.")

    if not features.index.is_monotonic_increasing or not labels.index.is_monotonic_increasing:
        raise ValueError("features and labels indexes must be sorted ascending.")

    common_index = features.index.intersection(labels.index)
    if len(common_index) == 0:
        raise ValueError("features and labels do not share any common dates.")

    if not common_index.is_monotonic_increasing:
        raise ValueError("aligned common index is not monotonic increasing.")

    features_aligned = features.loc[common_index]
    labels_aligned = labels.loc[common_index]

    if not features_aligned.index.equals(labels_aligned.index):
        raise ValueError("aligned features and labels indexes do not match exactly.")

    splits = generate_walk_forward_splits(index=common_index, config=config)
    return features_aligned, labels_aligned, splits
