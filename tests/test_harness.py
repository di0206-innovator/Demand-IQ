"""Tests for evaluation, drift, stress-testing, and simulation harnesses."""

import numpy as np
import pandas as pd
import pytest

from demandiq.config import ModelConfig
from demandiq.harness.backtesting import TimeSeriesBacktester
from demandiq.harness.drift import DataDriftMonitor, calculate_psi
from demandiq.harness.inventory_simulation import InventorySimulator
from demandiq.harness.stress_testing import ScenarioStressTester
from demandiq.models import XGBoostDemandForecaster


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Generate synthetic time series DataFrame for harness tests."""
    np.random.seed(42)
    rows = []
    for week in range(1, 30):
        for center in [10, 11]:
            for meal in [101, 102]:
                price = 100.0 + np.random.normal(0, 5)
                orders = int(
                    max(10, 50 + 20 * (center == 10) + 10 * (meal == 101) + np.random.normal(0, 5))
                )
                rows.append(
                    {
                        "week": week,
                        "center_id": center,
                        "meal_id": meal,
                        "center_type": "TYPE_A" if center == 10 else "TYPE_B",
                        "category": "Beverages",
                        "cuisine": "Indian",
                        "checkout_price": price,
                        "base_price": 100.0,
                        "discount_ratio": 0.0,
                        "price_ratio": 1.0,
                        "emailer_for_promotion": 0,
                        "homepage_featured": 0,
                        "both_promos": 0,
                        "any_promo": 0,
                        "promo_discount_synergy": 0.0,
                        "num_orders": orders,
                    }
                )
    return pd.DataFrame(rows)


def test_time_series_backtester(sample_timeseries_df):
    feature_cols = ["checkout_price", "center_id", "meal_id"]
    backtester = TimeSeriesBacktester(n_folds=2, test_window_weeks=4, min_train_weeks=10)

    def model_factory():
        return XGBoostDemandForecaster(model_config=ModelConfig(n_estimators=10, max_depth=2))

    res = backtester.run_backtest(sample_timeseries_df, model_factory, feature_cols)
    assert len(res.fold_evaluations) == 2
    assert "WMAPE" in res.overall_metrics
    assert len(res.sliced_performance) > 0
    assert not res.predictions_df.empty


def test_calculate_psi():
    expected = np.random.normal(50, 10, 1000)
    actual_similar = np.random.normal(50, 10, 1000)
    actual_shifted = np.random.normal(70, 10, 1000)

    psi_low = calculate_psi(expected, actual_similar)
    psi_high = calculate_psi(expected, actual_shifted)

    assert psi_low < 0.10
    assert psi_high > 0.20


def test_data_drift_monitor(sample_timeseries_df):
    monitor = DataDriftMonitor()
    feat_cols = ["checkout_price"]

    df1 = sample_timeseries_df.iloc[:50]
    df2 = sample_timeseries_df.iloc[50:].copy()
    df2["checkout_price"] += 100.0  # Force drift

    report = monitor.evaluate_drift(df1, df2, feature_cols=feat_cols)
    assert report.total_features_evaluated == 1
    assert report.drifted_feature_count == 1
    assert report.feature_drift_scores[0].is_drifted is True


def test_scenario_stress_tester(sample_timeseries_df):
    feat_cols = ["checkout_price", "center_id", "meal_id", "any_promo"]
    forecaster = XGBoostDemandForecaster(model_config=ModelConfig(n_estimators=15, max_depth=2))
    forecaster.fit(sample_timeseries_df[feat_cols], sample_timeseries_df["num_orders"])

    stress_tester = ScenarioStressTester(forecaster, feat_cols)
    result = stress_tester.run_all_stress_tests(sample_timeseries_df)

    assert len(result.scenario_outcomes) == 4
    assert any("Price Surge" in o.scenario_name for o in result.scenario_outcomes)
    assert any("Flash Sale" in o.scenario_name for o in result.scenario_outcomes)


def test_inventory_simulator(sample_timeseries_df):
    eval_df = sample_timeseries_df.copy()
    eval_df["forecast_orders"] = eval_df["num_orders"] * 1.05
    eval_df["actual_orders"] = eval_df["num_orders"]

    sim = InventorySimulator()
    res = sim.run_simulation(eval_df, service_levels=[0.80, 0.95, 0.99])

    assert len(res.policy_comparisons) == 3
    assert 0.80 <= res.optimal_service_level <= 0.99
    assert not res.simulation_timeline_df.empty
