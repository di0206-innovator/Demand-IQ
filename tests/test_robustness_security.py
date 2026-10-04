"""Comprehensive tests for Efficiency, Robustness, and Security enhancements across DemandIQ."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandiq.config import ModelConfig
from demandiq.data import time_based_split
from demandiq.harness.drift import DataDriftMonitor, calculate_psi
from demandiq.harness.inventory_simulation import InventorySimulator
from demandiq.inventory import compute_inventory_summary_kpis, generate_inventory_recommendations_df
from demandiq.loop.feedback import FeedbackStore
from demandiq.loop.personalization import HyperPersonalizedProfiler
from demandiq.loop.self_improving import AdaptiveSelfImprovingLoop
from demandiq.models import NaiveBaselineForecaster, XGBoostDemandForecaster, compute_metrics
from demandiq.utils import calculate_rmsle, calculate_wmape, validate_safe_path

# =====================================================================
# 1. SECURITY TESTS
# =====================================================================


def test_validate_safe_path_valid(tmp_path: Path):
    safe_file = tmp_path / "valid.txt"
    safe_file.write_text("secure content")

    resolved = validate_safe_path(safe_file, allowed_base=tmp_path, must_exist=True)
    assert resolved == safe_file.resolve()


def test_validate_safe_path_traversal_detection(tmp_path: Path):
    allowed_dir = tmp_path / "sandbox"
    allowed_dir.mkdir()
    outside_file = tmp_path / "secret.txt"
    outside_file.write_text("secret")

    traversal_path = allowed_dir / ".." / "secret.txt"
    with pytest.raises(ValueError, match="Security violation: path traversal detected"):
        validate_safe_path(traversal_path, allowed_base=allowed_dir)


def test_validate_safe_path_size_limit(tmp_path: Path):
    large_file = tmp_path / "large.bin"
    large_file.write_bytes(b"0" * 1024)

    with pytest.raises(ValueError, match="exceeds maximum allowed threshold"):
        validate_safe_path(large_file, max_size_bytes=500)


def test_model_save_and_load_security(tmp_path: Path):
    model_config = ModelConfig(n_estimators=5, max_depth=2)
    forecaster = XGBoostDemandForecaster(model_config=model_config)

    X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0]})
    y = pd.Series([10.0, 20.0, 30.0])
    forecaster.fit(X, y)

    sandbox = tmp_path / "models_dir"
    sandbox.mkdir()

    # Traversal in save
    with pytest.raises(ValueError, match="Security violation: path traversal detected"):
        forecaster.save(sandbox / ".." / "evil.joblib", allowed_base=sandbox)

    # Valid save
    valid_file = sandbox / "model.joblib"
    forecaster.save(valid_file, allowed_base=sandbox)
    assert valid_file.exists()

    # Traversal in load
    with pytest.raises(ValueError, match="Security violation: path traversal detected"):
        XGBoostDemandForecaster.load(sandbox / ".." / "evil.joblib", allowed_base=sandbox)

    # Valid load
    loaded = XGBoostDemandForecaster.load(valid_file, allowed_base=sandbox)
    assert loaded.is_fitted
    assert loaded.feature_names == ["feat1", "feat2"]


def test_model_native_json_serialization(tmp_path: Path):
    """Test safe native XGBoost JSON serialization (no pickle)."""
    model_config = ModelConfig(n_estimators=5, max_depth=2)
    forecaster = XGBoostDemandForecaster(model_config=model_config)

    X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0]})
    y = pd.Series([10.0, 20.0, 30.0])
    forecaster.fit(X, y)

    native_dir = tmp_path / "native_export"
    forecaster.save_native(native_dir)

    assert (native_dir / "xgboost_model.json").exists()
    assert (native_dir / "metadata.json").exists()

    loaded = XGBoostDemandForecaster.load_native(native_dir)
    assert loaded.is_fitted
    preds_orig = forecaster.predict(X)
    preds_loaded = loaded.predict(X)
    np.testing.assert_allclose(preds_orig, preds_loaded, rtol=1e-5)


# =====================================================================
# 2. EFFICIENCY & VECTORIZATION TESTS
# =====================================================================


def test_vectorized_simulation_equivalence():
    """Verify that vectorized simulation matches operational expectations."""
    df = pd.DataFrame(
        {
            "forecast_orders": [100.0, 150.0, 200.0, 80.0],
            "actual_orders": [110.0, 140.0, 250.0, 70.0],
        }
    )
    sim = InventorySimulator(unit_price=100.0, holding_cost_per_unit_week=5.0)
    res = sim.run_simulation(df, service_levels=[0.95])

    assert len(res.policy_comparisons) == 1
    pol = res.policy_comparisons[0]
    assert pol.total_demand_units == 570.0
    assert pol.total_prepared_units > 570.0
    assert 0 <= pol.fill_rate_percentage <= 100.0


def test_vectorized_baseline_predictions():
    forecaster = NaiveBaselineForecaster(group_cols=["c", "m"])
    X_train = pd.DataFrame({"c": [1, 1, 2, 2], "m": [10, 20, 10, 20]})
    y_train = pd.Series([50.0, 80.0, 60.0, 90.0])
    forecaster.fit(X_train, y_train)

    X_test = pd.DataFrame({"c": [1, 2, 1, 99], "m": [10, 10, 20, 99]})
    preds = forecaster.predict(X_test)
    assert preds[0] == 50.0
    assert preds[1] == 60.0
    assert preds[2] == 80.0
    assert preds[3] == forecaster.global_median


def test_vectorized_feedback_store_and_thread_safety():
    store = FeedbackStore(max_history=500)

    df_chunk = pd.DataFrame(
        {
            "week": [1, 1, 1],
            "center_id": [10, 10, 11],
            "meal_id": [100, 101, 100],
            "forecast_orders": [50.0, 75.0, 60.0],
            "actual_orders": [55.0, 70.0, 65.0],
        }
    )

    # Test concurrent batch additions
    def ingest_task():
        store.record_batch_feedback(df_chunk)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(ingest_task) for _ in range(10)]
        for f in futures:
            f.result()

    assert len(store.records) == 30
    summary = store.get_summary_metrics()
    assert summary["sample_count"] == 30
    assert "wmape" in summary


# =====================================================================
# 3. ROBUSTNESS & EDGE CASE TESTS
# =====================================================================


def test_compute_metrics_empty_and_nan():
    # Empty inputs
    assert compute_metrics(np.array([]), np.array([])) == {
        "MAE": 0.0,
        "RMSE": 0.0,
        "WMAPE": 0.0,
        "RMSLE": 0.0,
    }

    # NaNs and Infs
    y_t = np.array([10.0, np.nan, 30.0, np.inf])
    y_p = np.array([12.0, 20.0, np.nan, 40.0])
    m = compute_metrics(y_t, y_p)
    assert m["MAE"] == 2.0  # Only (10.0, 12.0) is mutually finite
    assert m["WMAPE"] == 0.2


def test_calculate_wmape_edge_cases():
    assert calculate_wmape([], []) == 0.0
    assert calculate_wmape([0.0, 0.0], [10.0, 20.0]) == 0.0
    assert calculate_wmape([np.nan], [np.nan]) == 0.0


def test_calculate_rmsle_edge_cases():
    assert calculate_rmsle([], []) == 0.0
    assert calculate_rmsle([-10.0], [-5.0]) == 0.0  # Clipped to zero -> log1p(0) - log1p(0) = 0


def test_calculate_psi_edge_cases():
    # Empty
    assert calculate_psi(np.array([]), np.array([])) == 0.0
    # Degenerate single constant value
    assert calculate_psi(np.zeros(100), np.zeros(100)) == 0.0
    # Array with Infinities and NaNs
    arr_inf = np.array([1.0, np.inf, -np.inf, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    psi = calculate_psi(arr_inf, arr_inf)
    assert psi == 0.0


def test_empty_dataframe_inventory_and_kpis():
    empty_df = pd.DataFrame(columns=["forecast_orders"])
    recs = generate_inventory_recommendations_df(empty_df)
    assert recs.empty
    assert "safety_stock" in recs.columns

    kpis = compute_inventory_summary_kpis(recs)
    assert kpis["total_forecasted_units"] == 0.0
    assert kpis["buffer_overhead_percentage"] == 0.0


def test_empty_simulation_handling():
    sim = InventorySimulator()
    res = sim.run_simulation(pd.DataFrame())
    assert res.policy_comparisons == []
    assert res.simulation_timeline_df.empty


def test_hyper_personalized_profiler_degenerate_data():
    profiler = HyperPersonalizedProfiler()
    # Empty df
    assert profiler.fit_profiles(pd.DataFrame()) == {}

    # Degenerate zero variance prices
    df_constant = pd.DataFrame(
        {
            "center_id": [1] * 10,
            "meal_id": [10] * 10,
            "checkout_price": [100.0] * 10,
            "actual_orders": [50.0] * 10,
        }
    )
    profiles = profiler.fit_profiles(df_constant)
    assert (1, 10) in profiles
    prof = profiles[(1, 10)]
    assert prof.price_elasticity_estimate == -1.2  # Safe fallback


def test_data_drift_monitor_constant_columns():
    monitor = DataDriftMonitor()
    df1 = pd.DataFrame({"constant_feat": [5.0] * 20})
    df2 = pd.DataFrame({"constant_feat": [5.0] * 20})

    report = monitor.evaluate_drift(df1, df2, feature_cols=["constant_feat"])
    assert report.drifted_feature_count == 0
    assert report.overall_status == "HEALTHY"


def test_time_based_split_invalid_boundaries():
    df = pd.DataFrame({"week": range(1, 150), "val": range(1, 150)})
    from demandiq.config import AppConfig, SplitConfig

    bad_config = AppConfig(split=SplitConfig(train_end_week=140, val_end_week=130))
    with pytest.raises(ValueError, match="must be strictly less than val_end_week"):
        time_based_split(df, time_col="week", config=bad_config)


def test_self_improving_loop_empty_batch():
    forecaster = XGBoostDemandForecaster(model_config=ModelConfig(n_estimators=5, max_depth=2))
    loop = AdaptiveSelfImprovingLoop(champion_model=forecaster, feature_cols=["feat1"])
    res = loop.process_incoming_batch(pd.DataFrame())
    assert res.champion_wmape == 0.0
    assert not res.promoted_new_champion
