"""Tests for inventory and safety stock calculations."""

import pytest

from demandiq.inventory import calculate_safety_stock, recommend_inventory


def test_calculate_safety_stock_scaling():
    # 95% service level has z ~ 1.6449
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
    # Gross requirement = forecast (100) + SS (~25) = ~125
    # Net recommended = gross requirement - current_inventory (30) = ~95
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
    # Net prep recommended should never be negative
    assert rec.recommended_preparation == 0.0
