"""Inventory and preparation recommendation engine for DemandIQ.

Converts ML demand forecasts into actionable inventory and preparation targets
incorporating uncertainty buffers, service levels, and lead-time variability.
"""

from dataclasses import dataclass

import numpy as np

# Standard normal inverse CDF z-scores for common cycle service levels
SERVICE_LEVEL_Z: dict[float, float] = {
    0.80: 0.8416,
    0.85: 1.0364,
    0.90: 1.2816,
    0.95: 1.6449,
    0.98: 2.0537,
    0.99: 2.3263,
}


@dataclass(frozen=True)
class InventoryRecommendation:
    """Structured output for kitchen preparation and inventory decisions."""

    forecast_demand: float
    safety_stock: float
    recommended_preparation: float
    reorder_point: float
    service_level: float
    lead_time_weeks: int


def calculate_safety_stock(
    demand_std: float | np.ndarray,
    service_level: float = 0.95,
    lead_time_weeks: int = 1,
    min_safety_stock: float = 5.0,
) -> float:
    """Calculate statistical safety stock buffer.

    Formula:
        Safety Stock = Z * sqrt(Lead Time) * std(Demand)

    Args:
        demand_std: Standard deviation of demand forecast errors (RMSE) or residual demand.
        service_level: Target cycle service level (e.g. 0.95).
        lead_time_weeks: Supplier or preparation lead time in weeks.
        min_safety_stock: Floor buffer to protect against low-volume stockouts.
    """
    z_score = SERVICE_LEVEL_Z.get(round(service_level, 2), 1.6449)
    lead_time_factor = np.sqrt(max(1, lead_time_weeks))
    raw_ss = float(z_score * lead_time_factor * demand_std)
    return float(max(min_safety_stock, np.ceil(raw_ss)))


def recommend_inventory(
    forecast_demand: float,
    demand_std: float,
    current_inventory: float = 0.0,
    service_level: float = 0.95,
    lead_time_weeks: int = 1,
    min_safety_stock: float = 5.0,
) -> InventoryRecommendation:
    """Calculate recommended order/prep volume and reorder point."""
    ss = calculate_safety_stock(
        demand_std=demand_std,
        service_level=service_level,
        lead_time_weeks=lead_time_weeks,
        min_safety_stock=min_safety_stock,
    )
    # Reorder point = expected lead-time demand + safety stock
    reorder_point = (forecast_demand * lead_time_weeks) + ss

    # Recommended preparation quantity for next cycle
    gross_requirement = forecast_demand + ss
    net_recommended = max(0.0, gross_requirement - current_inventory)

    return InventoryRecommendation(
        forecast_demand=round(forecast_demand, 2),
        safety_stock=round(ss, 2),
        recommended_preparation=round(net_recommended, 2),
        reorder_point=round(reorder_point, 2),
        service_level=service_level,
        lead_time_weeks=lead_time_weeks,
    )
