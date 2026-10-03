"""Inventory simulation harness: Stockout, holding cost, and waste trade-off modeling."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demandiq.inventory import get_z_score
from demandiq.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ServiceLevelSimulation:
    """Simulation results for a specific service level policy."""

    service_level: float
    z_score: float
    total_demand_units: float
    total_prepared_units: float
    total_fulfilled_units: float
    total_stockout_units: float
    total_wasted_units: float
    stockout_event_rate: float
    fill_rate_percentage: float
    estimated_holding_cost: float
    estimated_stockout_loss: float
    total_operational_cost: float


@dataclass
class SimulationResult:
    """Full inventory simulation output across policies."""

    optimal_service_level: float
    policy_comparisons: list[ServiceLevelSimulation]
    simulation_timeline_df: pd.DataFrame


class InventorySimulator:
    """Simulates dynamic inventory replenishment, stockout losses, and waste trade-offs."""

    def __init__(
        self,
        unit_price: float = 120.0,
        holding_cost_per_unit_week: float = 5.0,
        stockout_penalty_per_unit: float = 40.0,
        spoilage_rate_per_week: float = 0.30,
    ) -> None:
        self.unit_price = unit_price
        self.holding_cost_per_unit_week = holding_cost_per_unit_week
        self.stockout_penalty_per_unit = stockout_penalty_per_unit
        self.spoilage_rate_per_week = spoilage_rate_per_week

    def run_simulation(
        self,
        df_forecast_actual: pd.DataFrame,
        forecast_col: str = "forecast_orders",
        actual_col: str = "actual_orders",
        demand_std: float = 25.0,
        service_levels: list[float] | None = None,
    ) -> SimulationResult:
        """Run multi-policy inventory replay simulation."""
        if service_levels is None:
            service_levels = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]

        policies: list[ServiceLevelSimulation] = []
        best_sl = 0.95
        lowest_cost = float("inf")

        for sl in service_levels:
            z = get_z_score(sl)
            # Replay replenishment
            total_actual = float(df_forecast_actual[actual_col].sum())
            total_prep = 0.0
            total_fulfilled = 0.0
            total_stockout = 0.0
            total_wasted = 0.0
            stockout_events = 0

            for _, row in df_forecast_actual.iterrows():
                f_demand = row[forecast_col]
                a_demand = row[actual_col]

                # Safety stock buffer
                ss = max(5.0, np.ceil(z * demand_std))
                prep = f_demand + ss
                total_prep += prep

                if a_demand > prep:
                    # Stockout occurs
                    stockout = a_demand - prep
                    fulfilled = prep
                    total_stockout += stockout
                    stockout_events += 1
                else:
                    # Demand satisfied, potential waste/carryover
                    fulfilled = a_demand
                    leftover = prep - a_demand
                    waste = leftover * self.spoilage_rate_per_week
                    total_wasted += waste

                total_fulfilled += fulfilled

            fill_rate = (total_fulfilled / max(1.0, total_actual)) * 100.0
            stockout_rate = (stockout_events / max(1, len(df_forecast_actual))) * 100.0

            holding_cost = total_wasted * self.holding_cost_per_unit_week
            stockout_loss = total_stockout * (self.unit_price + self.stockout_penalty_per_unit)
            total_cost = holding_cost + stockout_loss

            if total_cost < lowest_cost:
                lowest_cost = total_cost
                best_sl = sl

            policies.append(
                ServiceLevelSimulation(
                    service_level=sl,
                    z_score=round(z, 3),
                    total_demand_units=round(total_actual, 1),
                    total_prepared_units=round(total_prep, 1),
                    total_fulfilled_units=round(total_fulfilled, 1),
                    total_stockout_units=round(total_stockout, 1),
                    total_wasted_units=round(total_wasted, 1),
                    stockout_event_rate=round(stockout_rate, 2),
                    fill_rate_percentage=round(fill_rate, 2),
                    estimated_holding_cost=round(holding_cost, 2),
                    estimated_stockout_loss=round(stockout_loss, 2),
                    total_operational_cost=round(total_cost, 2),
                )
            )

        timeline_df = df_forecast_actual[[forecast_col, actual_col]].copy()
        timeline_df["recommended_95"] = timeline_df[forecast_col] + max(
            5.0, np.ceil(1.6449 * demand_std)
        )

        return SimulationResult(
            optimal_service_level=best_sl,
            policy_comparisons=policies,
            simulation_timeline_df=timeline_df,
        )
