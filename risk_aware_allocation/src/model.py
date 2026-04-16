"""Model training interfaces for walk-forward risk prediction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import ProjectConfig
from .split import generate_walk_forward_splits


@dataclass(slots=True)
class ModelResult:
    """Container for out-of-sample predictions and modeling metadata."""

    model_name: str
    predictions: pd.Series
    actuals: pd.Series
    aligned_features_index: pd.DatetimeIndex
    split_summaries: list[dict[str, Any]] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def get_model(model_name: str, random_state: int = 42) -> Any:
    """Return a baseline sklearn regressor by name."""
    normalized_name = model_name.lower().strip()

    if normalized_name == "linear_regression":
        return LinearRegression()

    if normalized_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported model_name={model_name!r}. Supported: 'linear_regression', 'random_forest'."
    )


def fit_and_predict_single_split(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Fit one model on train data and return test predictions with test index."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return pd.Series(preds, index=X_test.index, name="prediction")


def _assemble_model_data(
    features: pd.DataFrame,
    labels: pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Align features/labels and drop rows with missing values for model readiness."""
    if isinstance(labels, pd.Series):
        labels_df = labels.to_frame(name=labels.name or "label")
    else:
        labels_df = labels.copy()

    if labels_df.shape[1] != 1:
        raise ValueError("labels must be a Series or single-column DataFrame.")

    common_index = features.index.intersection(labels_df.index)
    if len(common_index) == 0:
        raise ValueError("features and labels do not share any common dates.")

    X = features.loc[common_index].copy()
    y = labels_df.loc[common_index].iloc[:, 0].copy()

    initial_rows = len(common_index)
    feature_nan_rows = int(X.isna().any(axis=1).sum())
    label_nan_rows = int(y.isna().sum())

    valid_mask = (~X.isna().any(axis=1)) & (~y.isna())
    X_clean = X.loc[valid_mask]
    y_clean = y.loc[valid_mask]

    drop_report = {
        "initial_rows": initial_rows,
        "feature_nan_rows": feature_nan_rows,
        "label_nan_rows": label_nan_rows,
        "rows_after_drop": int(len(X_clean)),
        "rows_dropped": int(initial_rows - len(X_clean)),
    }

    return X_clean, y_clean, drop_report


def run_walk_forward_model(
    features: pd.DataFrame,
    labels: pd.DataFrame | pd.Series,
    config: ProjectConfig,
    model_name: str = "linear_regression",
    random_state: int = 42,
) -> ModelResult:
    """Train baseline model across walk-forward splits and collect OOS predictions."""
    X, y, drop_report = _assemble_model_data(features=features, labels=labels)

    splits = generate_walk_forward_splits(index=X.index, config=config)
    if not splits:
        raise ValueError("No walk-forward splits available for the provided data/config.")

    prediction_chunks: list[pd.Series] = []
    actual_chunks: list[pd.Series] = []
    split_summaries: list[dict[str, Any]] = []

    for split in splits:
        X_train = X.loc[split.train_index]
        y_train = y.loc[split.train_index]
        X_test = X.loc[split.test_index]
        y_test = y.loc[split.test_index]

        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = get_model(model_name=model_name, random_state=random_state)
        preds = fit_and_predict_single_split(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
        )

        prediction_chunks.append(preds)
        actual_chunks.append(y_test.rename("actual"))
        split_summaries.append(
            {
                "split_id": split.split_id,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_start": split.train_start,
                "train_end": split.train_end,
                "test_start": split.test_start,
                "test_end": split.test_end,
                "train_test_overlap": len(split.train_index.intersection(split.test_index)) > 0,
            }
        )

    if not prediction_chunks:
        raise ValueError("No predictions were produced across splits.")

    predictions = pd.concat(prediction_chunks).sort_index()
    actuals = pd.concat(actual_chunks).sort_index()

    if not predictions.index.equals(actuals.index):
        raise ValueError("Prediction and actual indexes must match exactly after concatenation.")

    metadata: dict[str, Any] = {
        "drop_report": drop_report,
        "n_splits_generated": len(splits),
        "n_splits_used": len(split_summaries),
    }

    return ModelResult(
        model_name=model_name,
        predictions=predictions,
        actuals=actuals,
        aligned_features_index=X.index,
        split_summaries=split_summaries,
        feature_names=list(X.columns),
        metadata=metadata,
    )


def evaluate_prediction_quality(predictions: pd.Series, actuals: pd.Series) -> dict[str, float]:
    """Compute lightweight prediction quality diagnostics."""
    if not predictions.index.equals(actuals.index):
        raise ValueError("predictions and actuals must share identical indexes.")

    return {
        "mse": float(mean_squared_error(actuals, predictions)),
        "mae": float(mean_absolute_error(actuals, predictions)),
        "correlation": float(np.corrcoef(actuals.values, predictions.values)[0, 1]),
    }
