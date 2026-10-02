"""Tests for inventory and safety stock calculations."""

import pandas as pd
import pytest

from demandiq.config import InventoryConfig
from demandiq.inventory import (
    calculate_safety_stock,
    compute_inventory_summary_kpis,
    generate_inventory_recommendations_df,
    get_z_score,
    recommend_inventory,
)


def test_get_z_score():
    assert get_z_score(0.95) == 1.6449
    assert get_z_score(0.99) == 2.3263
    assert get_z_score(0.80) == 0.8416


def test_calculate_safety_stock_scaling():
    ss_low = calculate_safety_stock(demand_std=10.0, service_level=0.95, lead_time_weeks=1)
    ss_high = calculate_safety_stock(demand_std=20.0, service_level=0.95, lead_time_weeks=1)
    assert ss_high > ss_low
    assert ss_low >= 5.0  # Floor check


def test_recommend_inventory_net_quantity():
    rec = recommend_inventory(
        forecast_demand=100.0,
        demand_std=15.0,
        current_inventory=30.0,
        service_level=0.95,
        lead_time_weeks=1,
    )
    assert rec.forecast_demand == 100.0
    assert rec.safety_stock > 0
    assert rec.recommended_preparation == pytest.approx(
        rec.forecast_demand + rec.safety_stock - 30.0, rel=1e-2
    )
    assert rec.reorder_point > rec.safety_stock


def test_recommend_inventory_excess_stock():
    rec = recommend_inventory(
        forecast_demand=50.0,
        demand_std=10.0,
        current_inventory=200.0,  # Surplus inventory
        service_level=0.95,
    )
    assert rec.recommended_preparation == 0.0


def test_generate_inventory_recommendations_df_and_kpis():
    df = pd.DataFrame(
        {
            "center_id": [10, 10],
            "meal_id": [101, 102],
            "forecast_orders": [150.0, 80.0],
            "current_stock": [20.0, 100.0],
        }
    )
    config = InventoryConfig(service_level=0.95, min_safety_stock=10)
    recs_df = generate_inventory_recommendations_df(
        df,
        forecast_col="forecast_orders",
        demand_std_default=15.0,
        current_inventory_col="current_stock",
        config=config,
    )

    assert "safety_stock" in recs_df.columns
    assert "reorder_point" in recs_df.columns
    assert "recommended_preparation" in recs_df.columns
    assert recs_df.loc[0, "recommended_preparation"] > 0
    assert recs_df.loc[1, "recommended_preparation"] == 5.0  # 80 + 25 - 100 = 5

    kpis = compute_inventory_summary_kpis(recs_df)
    assert kpis["total_forecasted_units"] == 230.0
    assert kpis["total_recommended_preparation_units"] > 0
    assert "buffer_overhead_percentage" in kpis
