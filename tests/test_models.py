"""Tests for baseline forecaster and model interfaces."""

import numpy as np
import pandas as pd

from demandiq.models import NaiveBaselineForecaster, XGBoostDemandForecaster


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

    # Predict known groups
    preds = model.predict(X)
    assert len(preds) == 5
    assert preds[0] == 110.0  # Median of [100, 120, 110]
    assert preds[3] == 60.0  # Median of [50, 70]

    # Predict unseen group falls back to global median
    unseen_X = pd.DataFrame({"center_id": [99], "meal_id": [99]})
    unseen_pred = model.predict(unseen_X)
    assert unseen_pred[0] == float(y.median())


def test_xgboost_placeholder_initialization():
    model = XGBoostDemandForecaster()
    X = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    y = pd.Series([10, 20])
    model.fit(X, y)
    assert model.feature_names == ["feat1", "feat2"]
    preds = model.predict(X)
    assert np.all(preds == 0.0)
