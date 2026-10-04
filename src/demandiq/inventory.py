"""Inventory and preparation recommendation engine for DemandIQ.

Converts ML demand forecasts into actionable inventory and preparation targets
incorporating uncertainty buffers, service levels, and lead-time variability.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from demandiq.config import DEFAULT_CONFIG, InventoryConfig

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


def get_z_score(service_level: float) -> float:
    """Look up or estimate standard normal inverse CDF z-score."""
    rounded_level = round(service_level, 2)
    if rounded_level in SERVICE_LEVEL_Z:
        return SERVICE_LEVEL_Z[rounded_level]
    # Simple clamped interpolation fallback
    if service_level >= 0.99:
        return 2.3263
    if service_level <= 0.80:
        return 0.8416
    return 1.6449


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
    z_score = get_z_score(service_level)
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
        forecast_demand=round(float(forecast_demand), 2),
        safety_stock=round(ss, 2),
        recommended_preparation=round(net_recommended, 2),
        reorder_point=round(reorder_point, 2),
        service_level=service_level,
        lead_time_weeks=lead_time_weeks,
    )


def generate_inventory_recommendations_df(
    df: pd.DataFrame,
    forecast_col: str = "forecast_orders",
    demand_std_default: float = 20.0,
    current_inventory_col: str | None = None,
    config: InventoryConfig = DEFAULT_CONFIG.inventory,
) -> pd.DataFrame:
    """Generate batch inventory and kitchen preparation targets across all items."""
    df_out = df.copy()
    if forecast_col not in df_out.columns:
        raise KeyError(f"Forecast column '{forecast_col}' not found in dataframe.")

    if df_out.empty:
        df_out["safety_stock"] = pd.Series(dtype=float)
        df_out["reorder_point"] = pd.Series(dtype=float)
        df_out["recommended_preparation"] = pd.Series(dtype=float)
        return df_out

    z_score = get_z_score(config.service_level)
    lead_factor = np.sqrt(max(1, config.lead_time_weeks))

    # Calculate safety stock vectorized
    ss_values = np.maximum(
        config.min_safety_stock,
        np.ceil(z_score * lead_factor * demand_std_default),
    )
    df_out["safety_stock"] = ss_values
    df_out["reorder_point"] = (df_out[forecast_col] * config.lead_time_weeks) + ss_values

    current_inv = (
        df_out[current_inventory_col]
        if (current_inventory_col and current_inventory_col in df_out.columns)
        else 0.0
    )
    gross_needed = df_out[forecast_col] + ss_values
    df_out["recommended_preparation"] = np.maximum(0.0, gross_needed - current_inv).round(1)

    return df_out


def compute_inventory_summary_kpis(df_recommendations: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate business KPIs for operational reporting."""
    if df_recommendations.empty:
        return {
            "total_forecasted_units": 0.0,
            "total_recommended_preparation_units": 0.0,
            "total_safety_stock_buffer_units": 0.0,
            "buffer_overhead_percentage": 0.0,
        }

    total_forecast = float(df_recommendations["forecast_orders"].sum())
    total_prep = float(df_recommendations["recommended_preparation"].sum())
    total_ss = float(df_recommendations["safety_stock"].sum())
    buffer_pct = (
        round((total_ss / max(total_forecast, 1.0)) * 100, 2) if total_forecast > 0 else 0.0
    )

    return {
        "total_forecasted_units": round(total_forecast, 1),
        "total_recommended_preparation_units": round(total_prep, 1),
        "total_safety_stock_buffer_units": round(total_ss, 1),
        "buffer_overhead_percentage": buffer_pct,
    }
