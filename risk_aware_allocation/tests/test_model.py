"""Tests for walk-forward modeling workflow."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ProjectConfig, SplitConfig
from src.model import (
    fit_and_predict_single_split,
    get_model,
    run_walk_forward_model,
)
from src.split import generate_walk_forward_splits


def _synthetic_dataset(n: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    f1 = np.linspace(0.0, 10.0, n)
    f2 = np.sin(np.linspace(0.0, 4.0, n))
    y = 0.5 * f1 + 0.2 * f2

    features = pd.DataFrame({"f1": f1, "f2": f2}, index=idx)
    labels = pd.DataFrame({"spy_fwd_vol_5d": y}, index=idx)
    return features, labels


def test_get_model_supported_estimators() -> None:
    assert isinstance(get_model("linear_regression"), LinearRegression)
    assert isinstance(get_model("random_forest"), RandomForestRegressor)


def test_get_model_unsupported_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported model_name"):
        get_model("unsupported_model")


def test_fit_and_predict_single_split_preserves_test_index() -> None:
    features, labels = _synthetic_dataset(20)
    X_train = features.iloc[:15]
    y_train = labels.iloc[:15, 0]
    X_test = features.iloc[15:]

    model = get_model("linear_regression")
    preds = fit_and_predict_single_split(model=model, X_train=X_train, y_train=y_train, X_test=X_test)

    assert preds.index.equals(X_test.index)


def test_run_walk_forward_model_outputs_oos_predictions_only() -> None:
    features, labels = _synthetic_dataset(90)
    cfg = ProjectConfig(
        split=SplitConfig(
            train_window_days=30,
            test_window_days=10,
            minimum_history_required=30,
            walk_forward_mode="expanding",
        )
    )

    result = run_walk_forward_model(features=features, labels=labels, config=cfg, model_name="linear_regression")

    X_clean = features.loc[result.aligned_features_index]
    expected_splits = generate_walk_forward_splits(index=X_clean.index, config=cfg)
    expected_oos_index = expected_splits[0].test_index
    for split in expected_splits[1:]:
        expected_oos_index = expected_oos_index.append(split.test_index)

    assert result.predictions.index.equals(expected_oos_index)
    assert result.predictions.index.equals(result.actuals.index)
    assert result.predictions.index.is_monotonic_increasing


def test_nan_rows_dropped_only_at_model_assembly_stage() -> None:
    features, labels = _synthetic_dataset(60)
    features.loc[features.index[0], "f1"] = np.nan
    labels.loc[labels.index[1], "spy_fwd_vol_5d"] = np.nan

    cfg = ProjectConfig(split=SplitConfig(train_window_days=20, test_window_days=10, minimum_history_required=20))
    result = run_walk_forward_model(features=features, labels=labels, config=cfg)

    drop_report = result.metadata["drop_report"]
    assert drop_report["feature_nan_rows"] >= 1
    assert drop_report["label_nan_rows"] >= 1
    assert drop_report["rows_dropped"] >= 2


def test_no_train_test_overlap_in_split_summaries() -> None:
    features, labels = _synthetic_dataset(70)
    cfg = ProjectConfig(split=SplitConfig(train_window_days=25, test_window_days=10, minimum_history_required=25))

    result = run_walk_forward_model(features=features, labels=labels, config=cfg)

    assert all(summary["train_test_overlap"] is False for summary in result.split_summaries)


def test_linear_regression_path_runs_on_deterministic_data() -> None:
    features, labels = _synthetic_dataset(85)
    cfg = ProjectConfig(split=SplitConfig(train_window_days=30, test_window_days=10, minimum_history_required=30))

    result = run_walk_forward_model(features=features, labels=labels, config=cfg, model_name="linear_regression")

    assert len(result.predictions) > 0
    assert result.model_name == "linear_regression"
    assert set(result.feature_names) == {"f1", "f2"}
