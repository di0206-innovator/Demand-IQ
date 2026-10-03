"""Hyper-personalized SKU-Center behavioral profiling and parameter calibration."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from demandiq.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class SkuCenterProfile:
    """Hyper-personalized operational and behavioral profile for a single center-meal combination."""

    center_id: int
    meal_id: int
    average_weekly_demand: float
    price_elasticity_estimate: float
    promo_uplift_ratio: float
    recent_residual_bias: float
    residual_std: float
    recommended_safety_stock: float
    volatility_tier: Literal["LOW", "MEDIUM", "HIGH"]
    volume_tier: Literal["LOW", "MEDIUM", "HIGH"]


class HyperPersonalizedProfiler:
    """Calculates granular behavioral profiles per (center_id, meal_id) pair to power adaptive decisioning."""

    def __init__(
        self,
        default_residual_std: float = 20.0,
        service_level_z: float = 1.6449,
    ) -> None:
        self.default_residual_std = default_residual_std
        self.service_level_z = service_level_z
        self.profiles: dict[tuple[int, int], SkuCenterProfile] = {}

    def fit_profiles(
        self,
        historical_df: pd.DataFrame,
        forecast_col: str = "forecast_orders",
        actual_col: str = "num_orders",
    ) -> dict[tuple[int, int], SkuCenterProfile]:
        """Compute micro-level behavioral characteristics across all SKU-center combinations."""
        logger.info("Calibrating hyper-personalized profiles across historical demand data...")
        profiles: dict[tuple[int, int], SkuCenterProfile] = {}

        if "actual_orders" in historical_df.columns:
            actual_col = "actual_orders"

        grouped = historical_df.groupby(["center_id", "meal_id"])
        for (c_id, m_id), grp in grouped:
            avg_demand = float(grp[actual_col].mean()) if actual_col in grp.columns else 50.0

            # 1. Residual Bias and Personalized Uncertainty
            if forecast_col in grp.columns and actual_col in grp.columns and len(grp) >= 3:
                residuals = grp[actual_col].to_numpy() - grp[forecast_col].to_numpy()
                recent_bias = float(np.mean(residuals[-10:]))  # trailing bias
                res_std = float(max(3.0, np.std(residuals)))
            else:
                recent_bias = 0.0
                res_std = self.default_residual_std

            # 2. Promo Uplift Ratio
            promo_ratio = 1.25
            if "any_promo" in grp.columns and actual_col in grp.columns:
                promo_sub = grp[grp["any_promo"] == 1]
                non_promo_sub = grp[grp["any_promo"] == 0]
                if len(promo_sub) >= 2 and len(non_promo_sub) >= 2:
                    p_mean = promo_sub[actual_col].mean()
                    np_mean = non_promo_sub[actual_col].mean()
                    if np_mean > 0:
                        promo_ratio = float(np.clip(p_mean / np_mean, 0.8, 3.5))

            # 3. Price Elasticity Estimate
            elasticity = -1.2
            if "checkout_price" in grp.columns and actual_col in grp.columns and len(grp) >= 5:
                prices = grp["checkout_price"].to_numpy()
                demands = grp[actual_col].to_numpy()
                if np.std(prices) > 1e-3 and np.std(demands) > 1e-3:
                    log_p = np.log(np.maximum(prices, 1.0))
                    log_d = np.log1p(np.maximum(demands, 0.0))
                    slope, _ = np.polyfit(log_p, log_d, 1)
                    elasticity = float(np.clip(slope, -4.0, 0.5))

            # 4. Tiers and Custom Safety Stock
            vol_tier: Literal["LOW", "MEDIUM", "HIGH"] = (
                "HIGH" if res_std > 35.0 else ("MEDIUM" if res_std > 15.0 else "LOW")
            )
            vol_vol_tier: Literal["LOW", "MEDIUM", "HIGH"] = (
                "HIGH" if avg_demand > 250.0 else ("MEDIUM" if avg_demand > 75.0 else "LOW")
            )

            rec_ss = float(max(5.0, np.ceil(self.service_level_z * res_std)))

            profiles[(int(c_id), int(m_id))] = SkuCenterProfile(
                center_id=int(c_id),
                meal_id=int(m_id),
                average_weekly_demand=round(avg_demand, 1),
                price_elasticity_estimate=round(elasticity, 2),
                promo_uplift_ratio=round(promo_ratio, 2),
                recent_residual_bias=round(recent_bias, 2),
                residual_std=round(res_std, 2),
                recommended_safety_stock=round(rec_ss, 1),
                volatility_tier=vol_tier,
                volume_tier=vol_vol_tier,
            )

        self.profiles = profiles
        logger.info("Successfully created %d hyper-personalized profiles.", len(self.profiles))
        return self.profiles

    def get_profile(self, center_id: int, meal_id: int) -> SkuCenterProfile | None:
        """Lookup profile for specific SKU-center combination."""
        return self.profiles.get((center_id, meal_id))
