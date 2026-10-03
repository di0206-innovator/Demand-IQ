"""Tests for hyper-personalized profiling, feedback store, and self-improving loop."""

import numpy as np
import pandas as pd
import pytest

from demandiq.config import ModelConfig
from demandiq.loop.feedback import FeedbackStore
from demandiq.loop.personalization import HyperPersonalizedProfiler
from demandiq.loop.self_improving import AdaptiveSelfImprovingLoop
from demandiq.models import XGBoostDemandForecaster


@pytest.fixture
def feedback_batch_df() -> pd.DataFrame:
    """Generate realistic batch of predictions vs actuals."""
    rows = []
    for week in range(1, 15):
        for center in [10, 11]:
            for meal in [101, 102]:
                rows.append(
                    {
                        "week": week,
                        "center_id": center,
                        "meal_id": meal,
                        "checkout_price": 120.0,
                        "any_promo": 1 if week % 3 == 0 else 0,
                        "forecast_orders": 80.0 + (center % 10) * 10,
                        "actual_orders": 90.0 + (center % 10) * 10,
                        "num_orders": 90.0 + (center % 10) * 10,
                    }
                )
    return pd.DataFrame(rows)


def test_hyper_personalized_profiler(feedback_batch_df):
    profiler = HyperPersonalizedProfiler()
    profiles = profiler.fit_profiles(
        feedback_batch_df,
        forecast_col="forecast_orders",
        actual_col="actual_orders",
    )

    assert len(profiles) == 4
    prof = profiler.get_profile(10, 101)
    assert prof is not None
    assert prof.center_id == 10
    assert prof.meal_id == 101
    assert prof.recommended_safety_stock >= 5.0
    assert prof.volatility_tier in ["LOW", "MEDIUM", "HIGH"]


def test_feedback_store(feedback_batch_df):
    store = FeedbackStore(max_history=100)
    store.record_batch_feedback(feedback_batch_df)

    assert len(store.records) == len(feedback_batch_df)
    metrics = store.get_summary_metrics()
    assert metrics["sample_count"] == len(feedback_batch_df)
    assert "wmape" in metrics
    assert "mean_bias_units" in metrics

    df_out = store.to_dataframe()
    assert not df_out.empty
    assert "error" in df_out.columns


def test_empty_feedback_store():
    store = FeedbackStore()
    metrics = store.get_summary_metrics()
    assert metrics["status"] == "EMPTY"
    assert store.to_dataframe().empty


def test_adaptive_self_improving_loop(feedback_batch_df):
    feat_cols = ["checkout_price", "center_id", "meal_id", "any_promo"]
    forecaster = XGBoostDemandForecaster(model_config=ModelConfig(n_estimators=20, max_depth=2))
    forecaster.fit(feedback_batch_df[feat_cols], feedback_batch_df["actual_orders"])

    loop = AdaptiveSelfImprovingLoop(champion_model=forecaster, feature_cols=feat_cols)

    # Process first batch
    res = loop.process_incoming_batch(feedback_batch_df, actual_col="actual_orders")
    assert res.champion_wmape is not None
    assert res.adaptive_wmape is not None
    assert loop.iteration_count == 1

    # Predict adaptive with custom safety stocks
    preds, safety_stocks = loop.predict_adaptive(feedback_batch_df)
    assert len(preds) == len(feedback_batch_df)
    assert len(safety_stocks) == len(feedback_batch_df)
    assert np.all(preds >= 0.0)
    assert np.all(safety_stocks >= 5.0)


def test_self_improving_loop_force_retrain(feedback_batch_df):
    feat_cols = ["checkout_price", "center_id", "meal_id", "any_promo"]
    forecaster = XGBoostDemandForecaster(model_config=ModelConfig(n_estimators=10, max_depth=2))
    forecaster.fit(feedback_batch_df[feat_cols], feedback_batch_df["actual_orders"])

    loop = AdaptiveSelfImprovingLoop(champion_model=forecaster, feature_cols=feat_cols)
    res = loop.process_incoming_batch(
        feedback_batch_df, actual_col="actual_orders", force_retrain=True
    )

    assert res.retrain_trigger_fired is True
    assert res.challenger_wmape is not None
