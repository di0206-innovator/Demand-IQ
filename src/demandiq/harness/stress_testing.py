"""Operational scenario simulation and model stress-testing harness."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demandiq.inventory import generate_inventory_recommendations_df
from demandiq.models import BaseForecaster
from demandiq.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ScenarioOutcome:
    """Outcome of a single stress scenario simulation."""

    scenario_name: str
    description: str
    baseline_total_demand: float
    stressed_total_demand: float
    percentage_change: float
    recommended_prep_units: float
    buffer_overhead_units: float
    elasticity_direction_valid: bool


@dataclass
class StressTestResult:
    """Aggregated stress testing outcomes across all operational disruption scenarios."""

    scenario_outcomes: list[ScenarioOutcome]
    summary_verdict: str


class ScenarioStressTester:
    """Stress-testing engine for validating forecasting model stability under extreme scenarios."""

    def __init__(self, forecaster: BaseForecaster, feature_cols: list[str]) -> None:
        self.forecaster = forecaster
        self.feature_cols = feature_cols

    def run_all_stress_tests(
        self,
        base_df: pd.DataFrame,
    ) -> StressTestResult:
        """Run standard battery of operational shock scenarios."""
        outcomes: list[ScenarioOutcome] = []

        # Baseline reference
        X_base = base_df[self.feature_cols].copy()
        baseline_preds = self.forecaster.predict(X_base)
        base_total = float(np.sum(baseline_preds))

        # 1. Price Hike Shock (+30% price)
        df_hike = base_df.copy()
        if "checkout_price" in df_hike.columns:
            df_hike["checkout_price"] *= 1.30
            if "price_ratio" in df_hike.columns:
                df_hike["price_ratio"] *= 1.30
        hike_preds = self.forecaster.predict(df_hike[self.feature_cols])
        hike_total = float(np.sum(hike_preds))
        ((hike_total - base_total) / max(1.0, base_total)) * 100.0

        outcomes.append(
            self._build_outcome(
                name="Price Surge (+30%)",
                description="Simulates sudden price increase across all items.",
                base_total=base_total,
                stressed_preds=hike_preds,
                stressed_df=df_hike,
                expected_direction="DOWN",
            )
        )

        # 2. Flash Discount Shock (-35% price discount)
        df_disc = base_df.copy()
        if "checkout_price" in df_disc.columns:
            df_disc["checkout_price"] *= 0.65
            if "discount_ratio" in df_disc.columns:
                df_disc["discount_ratio"] = np.clip(df_disc["discount_ratio"] + 0.35, 0, 0.9)
        disc_preds = self.forecaster.predict(df_disc[self.feature_cols])
        outcomes.append(
            self._build_outcome(
                name="Flash Sale (-35% Price)",
                description="Simulates deep price discount to test demand expansion.",
                base_total=base_total,
                stressed_preds=disc_preds,
                stressed_df=df_disc,
                expected_direction="UP",
            )
        )

        # 3. Mega Promo Blitz (Both Email + Homepage Active)
        df_blitz = base_df.copy()
        if "emailer_for_promotion" in df_blitz.columns:
            df_blitz["emailer_for_promotion"] = 1
        if "homepage_featured" in df_blitz.columns:
            df_blitz["homepage_featured"] = 1
        if "both_promos" in df_blitz.columns:
            df_blitz["both_promos"] = 1
        if "any_promo" in df_blitz.columns:
            df_blitz["any_promo"] = 1
        blitz_preds = self.forecaster.predict(df_blitz[self.feature_cols])
        outcomes.append(
            self._build_outcome(
                name="Promotional Blitz (Email + Homepage)",
                description="Simulates coordinated marketing campaign across all fulfillment centers.",
                base_total=base_total,
                stressed_preds=blitz_preds,
                stressed_df=df_blitz,
                expected_direction="UP",
            )
        )

        # 4. Total Marketing Blackout (Zero Promotions)
        df_blackout = base_df.copy()
        for col in [
            "emailer_for_promotion",
            "homepage_featured",
            "both_promos",
            "any_promo",
            "promo_discount_synergy",
        ]:
            if col in df_blackout.columns:
                df_blackout[col] = 0
        blackout_preds = self.forecaster.predict(df_blackout[self.feature_cols])
        outcomes.append(
            self._build_outcome(
                name="Promotion Blackout",
                description="Simulates organic baseline demand with zero marketing push.",
                base_total=base_total,
                stressed_preds=blackout_preds,
                stressed_df=df_blackout,
                expected_direction="DOWN",
            )
        )

        all_valid = all(o.elasticity_direction_valid for o in outcomes)
        verdict = (
            "PASSED: Model behaves consistently with economic and operational dynamics."
            if all_valid
            else "CAUTION: Some scenario responses violated theoretical directions."
        )

        return StressTestResult(scenario_outcomes=outcomes, summary_verdict=verdict)

    def _build_outcome(
        self,
        name: str,
        description: str,
        base_total: float,
        stressed_preds: np.ndarray,
        stressed_df: pd.DataFrame,
        expected_direction: str,
    ) -> ScenarioOutcome:
        stressed_total = float(np.sum(stressed_preds))
        pct_change = ((stressed_total - base_total) / max(1.0, base_total)) * 100.0

        is_valid = True
        if expected_direction == "DOWN" and pct_change > 5.0:
            is_valid = False
        elif expected_direction == "UP" and pct_change < -5.0:
            is_valid = False

        eval_df = stressed_df.copy()
        eval_df["forecast_orders"] = stressed_preds
        recs = generate_inventory_recommendations_df(eval_df, forecast_col="forecast_orders")
        prep_units = float(recs["recommended_preparation"].sum())
        buffer_units = float(recs["safety_stock"].sum())

        return ScenarioOutcome(
            scenario_name=name,
            description=description,
            baseline_total_demand=round(base_total, 1),
            stressed_total_demand=round(stressed_total, 1),
            percentage_change=round(pct_change, 2),
            recommended_prep_units=round(prep_units, 1),
            buffer_overhead_units=round(buffer_units, 1),
            elasticity_direction_valid=is_valid,
        )
