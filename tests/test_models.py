"""Tests for baseline forecaster, XGBoost model, evaluation metrics, SHAP, and persistence."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandiq.config import ModelConfig
from demandiq.models import (
    NaiveBaselineForecaster,
    XGBoostDemandForecaster,
    compute_metrics,
)


def test_compute_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    metrics = compute_metrics(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "WMAPE" in metrics
    assert "RMSLE" in metrics
    assert metrics["MAE"] > 0
    assert metrics["WMAPE"] > 0


def test_naive_baseline_forecaster():
    X = pd.DataFrame(
        {
            "center_id": [1, 1, 1, 2, 2],
            "meal_id": [10, 10, 10, 20, 20],
        }
    )
    y = pd.Series([100, 120, 110, 50, 70])
    model = NaiveBaselineForecaster()
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == 5
    assert preds[0] == 110.0
    assert preds[3] == 60.0

    # Unseen key fallback
    unseen_X = pd.DataFrame({"center_id": [99], "meal_id": [99]})
    unseen_pred = model.predict(unseen_X)
    assert unseen_pred[0] == float(y.median())

    eval_metrics = model.evaluate(X, y)
    assert "WMAPE" in eval_metrics


def test_xgboost_demand_forecaster_training_and_eval(tmp_path: Path):
    np.random.seed(42)
    n_samples = 100
    X = pd.DataFrame(
        {
            "feature_1": np.random.rand(n_samples),
            "feature_2": np.random.rand(n_samples),
            "feature_3": np.random.rand(n_samples),
        }
    )
    y = pd.Series(
        50 + 20 * X["feature_1"] + 10 * X["feature_2"] + np.random.normal(0, 2, n_samples)
    )

    X_train, y_train = X.iloc[:70], y.iloc[:70]
    X_val, y_val = X.iloc[70:85], y.iloc[70:85]
    X_test, y_test = X.iloc[85:], y.iloc[85:]

    config = ModelConfig(n_estimators=50, max_depth=3, learning_rate=0.1, early_stopping_rounds=10)
    forecaster = XGBoostDemandForecaster(model_config=config, use_log_target=True)

    # Predict before fit raises error
    with pytest.raises(RuntimeError, match="unfitted model"):
        forecaster.predict(X_test)

    forecaster.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert forecaster.is_fitted is True

    preds = forecaster.predict(X_test)
    assert len(preds) == len(X_test)
    assert np.all(preds >= 0.0)

    # Evaluate
    metrics = forecaster.evaluate(X_test, y_test)
    assert "RMSLE" in metrics
    assert "WMAPE" in metrics
    assert metrics["WMAPE"] < 1.0

    # Feature importance
    importances = forecaster.get_feature_importances(top_n=3)
    assert len(importances) == 3
    assert "feature_1" in importances

    # SHAP explanations
    shap_vals, X_matrix, feature_names = forecaster.compute_shap_values(X_test, max_samples=15)
    assert shap_vals.shape == (len(X_test), 3)
    assert feature_names == ["feature_1", "feature_2", "feature_3"]

    # Model save & load persistence
    model_path = tmp_path / "model.joblib"
    forecaster.save(model_path)
    assert model_path.exists()

    loaded_forecaster = XGBoostDemandForecaster.load(model_path)
    loaded_preds = loaded_forecaster.predict(X_test)
    np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5)
