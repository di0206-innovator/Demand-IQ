"""Feature engineering specifications and transformers for DemandIQ."""

import numpy as np
import pandas as pd

from demandiq.utils import setup_logger

logger = setup_logger(__name__)


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate pricing, discount percentage, and ratio features."""
    df_out = df.copy()
    if "checkout_price" in df_out.columns and "base_price" in df_out.columns:
        # Discount amount and discount ratio
        df_out["discount_amount"] = np.maximum(0.0, df_out["base_price"] - df_out["checkout_price"])
        df_out["discount_ratio"] = df_out["discount_amount"] / np.maximum(
            df_out["base_price"], 1e-4
        )
        df_out["price_ratio"] = df_out["checkout_price"] / np.maximum(df_out["base_price"], 1e-4)
    return df_out


def create_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate promotional interaction features."""
    df_out = df.copy()
    if "emailer_for_promotion" in df_out.columns and "homepage_featured" in df_out.columns:
        df_out["both_promos"] = (
            (df_out["emailer_for_promotion"] == 1) & (df_out["homepage_featured"] == 1)
        ).astype(int)
        df_out["any_promo"] = (
            (df_out["emailer_for_promotion"] == 1) | (df_out["homepage_featured"] == 1)
        ).astype(int)
    return df_out


def create_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "num_orders",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Generate lag features for time-series demand.

    Note: Lags must strictly be >= 1 to prevent target leakage.
    """
    if lags is None:
        lags = [1, 2, 3, 4]

    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1 to prevent data leakage, got {lag}")

    df_out = df.copy()
    df_out = df_out.sort_values(group_cols + ["week"])

    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df_out[col_name] = df_out.groupby(group_cols)[target_col].shift(lag)

    return df_out
