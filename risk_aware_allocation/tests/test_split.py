"""Tests for walk-forward time-series splitting."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ProjectConfig, SplitConfig
from src.split import (
    generate_walk_forward_splits,
    split_features_and_labels,
    validate_split_boundaries,
)


def _index(n: int = 30) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="D")


def test_generate_expanding_splits() -> None:
    idx = _index(30)
    cfg = ProjectConfig(
        split=SplitConfig(
            train_window_days=10,
            test_window_days=5,
            walk_forward_mode="expanding",
            minimum_history_required=8,
        )
    )

    splits = generate_walk_forward_splits(index=idx, config=cfg)

    assert len(splits) == 4
    assert len(splits[0].train_index) == 10
    assert len(splits[1].train_index) == 15
    assert splits[0].test_start == idx[10]


def test_generate_rolling_splits() -> None:
    idx = _index(30)
    cfg = ProjectConfig(
        split=SplitConfig(
            train_window_days=10,
            test_window_days=5,
            walk_forward_mode="rolling",
            minimum_history_required=8,
        )
    )

    splits = generate_walk_forward_splits(index=idx, config=cfg)

    assert len(splits) == 4
    assert all(len(split.train_index) == 10 for split in splits)
    assert splits[1].train_start == idx[5]


def test_train_precedes_test_and_no_overlap() -> None:
    idx = _index(25)
    cfg = ProjectConfig(split=SplitConfig(train_window_days=10, test_window_days=5, minimum_history_required=10))

    splits = generate_walk_forward_splits(index=idx, config=cfg)

    for split in splits:
        validate_split_boundaries(split.train_index, split.test_index)
        assert split.train_end < split.test_start
        assert len(split.train_index.intersection(split.test_index)) == 0


def test_splits_respect_minimum_history() -> None:
    idx = _index(40)
    cfg = ProjectConfig(
        split=SplitConfig(
            train_window_days=12,
            test_window_days=4,
            walk_forward_mode="expanding",
            minimum_history_required=15,
        )
    )

    splits = generate_walk_forward_splits(index=idx, config=cfg)

    assert len(splits[0].train_index) >= 15
    assert splits[0].test_start == idx[15]


def test_split_features_and_labels_aligns_common_dates() -> None:
    idx = _index(40)
    features = pd.DataFrame({"f1": range(40)}, index=idx)
    labels = pd.DataFrame({"y": range(35)}, index=idx[5:])
    cfg = ProjectConfig(split=SplitConfig(train_window_days=10, test_window_days=5, minimum_history_required=10))

    features_aligned, labels_aligned, splits = split_features_and_labels(
        features=features,
        labels=labels,
        config=cfg,
    )

    assert features_aligned.index.equals(labels_aligned.index)
    assert features_aligned.index[0] == idx[5]
    assert len(splits) > 0


def test_split_features_and_labels_raises_on_no_common_dates() -> None:
    idx = _index(20)
    features = pd.DataFrame({"f1": range(10)}, index=idx[:10])
    labels = pd.DataFrame({"y": range(10)}, index=idx[10:])
    cfg = ProjectConfig()

    with pytest.raises(ValueError, match="do not share any common dates"):
        split_features_and_labels(features=features, labels=labels, config=cfg)


def test_split_order_is_deterministic_and_ascending() -> None:
    idx = _index(35)
    cfg = ProjectConfig(split=SplitConfig(train_window_days=10, test_window_days=5, minimum_history_required=10))

    splits = generate_walk_forward_splits(index=idx, config=cfg)

    split_ids = [split.split_id for split in splits]
    test_starts = [split.test_start for split in splits]

    assert split_ids == list(range(len(splits)))
    assert test_starts == sorted(test_starts)
