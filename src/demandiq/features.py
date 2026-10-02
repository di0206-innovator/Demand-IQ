"""Feature engineering pipeline for DemandIQ.

Extracts price elasticity features, promotional interactions, temporal cyclical encodings,
leakage-safe time-series lags, and rolling demand statistics.
"""

from typing import Any

import numpy as np
import pandas as pd

from demandiq.config import DEFAULT_CONFIG, FeatureConfig
from demandiq.utils import setup_logger

logger = setup_logger(__name__)


def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate pricing, discount percentage, and ratio features."""
    df_out = df.copy()
    if "checkout_price" in df_out.columns and "base_price" in df_out.columns:
        base_clean = np.maximum(df_out["base_price"].fillna(1.0), 1e-4)
        df_out["discount_amount"] = np.maximum(0.0, df_out["base_price"] - df_out["checkout_price"])
        df_out["discount_ratio"] = df_out["discount_amount"] / base_clean
        df_out["price_ratio"] = df_out["checkout_price"] / base_clean

        # Meal-level relative pricing
        if "meal_id" in df_out.columns:
            meal_mean_price = df_out.groupby("meal_id")["checkout_price"].transform("mean")
            df_out["meal_rel_price"] = df_out["checkout_price"] / np.maximum(meal_mean_price, 1e-4)
    return df_out


def create_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate promotional interaction and synergy features."""
    df_out = df.copy()
    if "emailer_for_promotion" in df_out.columns and "homepage_featured" in df_out.columns:
        p_email = df_out["emailer_for_promotion"].fillna(0).astype(int)
        p_home = df_out["homepage_featured"].fillna(0).astype(int)

        df_out["both_promos"] = (p_email == 1) & (p_home == 1)
        df_out["both_promos"] = df_out["both_promos"].astype(int)

        df_out["any_promo"] = (p_email == 1) | (p_home == 1)
        df_out["any_promo"] = df_out["any_promo"].astype(int)

        if "discount_ratio" in df_out.columns:
            df_out["promo_discount_synergy"] = df_out["any_promo"] * df_out["discount_ratio"]
    return df_out


def create_temporal_features(df: pd.DataFrame, time_col: str = "week") -> pd.DataFrame:
    """Generate cyclical calendar proxies and trend indicators."""
    df_out = df.copy()
    if time_col in df_out.columns:
        weeks = df_out[time_col]
        week_of_year = ((weeks - 1) % 52) + 1
        df_out["week_of_year"] = week_of_year
        df_out["month_proxy"] = ((weeks - 1) // 4) % 12 + 1
        df_out["quarter_proxy"] = ((weeks - 1) // 13) % 4 + 1

        # Cyclical sinusoidal encoding for annual seasonality
        df_out["sin_week_of_year"] = np.sin(2 * np.pi * week_of_year / 52.0)
        df_out["cos_week_of_year"] = np.cos(2 * np.pi * week_of_year / 52.0)
    return df_out


def create_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "num_orders",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Generate lag features for time-series demand.

    Enforces lag >= 1 strictly to prevent target leakage.
    """
    if lags is None:
        lags = [1, 2, 3, 4, 5, 10]

    for lag in lags:
        if lag < 1:
            raise ValueError(f"Lag must be >= 1 to prevent data leakage, got {lag}")

    df_out = df.copy()
    df_out = df_out.sort_values(group_cols + ["week"]).reset_index(drop=True)

    grouped = df_out.groupby(group_cols)[target_col]
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df_out[col_name] = grouped.shift(lag)

    return df_out


def create_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str = "num_orders",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Generate rolling mean and std features strictly shifted by 1 week to avoid leakage."""
    if windows is None:
        windows = [3, 5, 10]

    df_out = df.copy()
    df_out = df_out.sort_values(group_cols + ["week"]).reset_index(drop=True)

    # Base shifted series (shift(1) ensures no current week leakage)
    shifted_target = df_out.groupby(group_cols)[target_col].shift(1)

    for window in windows:
        roll_mean_col = f"{target_col}_roll_mean_{window}"
        roll_std_col = f"{target_col}_roll_std_{window}"

        # Rolling calculation over shifted target
        rolling_obj = shifted_target.groupby([df_out[col] for col in group_cols]).rolling(
            window=window, min_periods=1
        )

        df_out[roll_mean_col] = rolling_obj.mean().reset_index(drop=True)
        df_out[roll_std_col] = rolling_obj.std().reset_index(drop=True).fillna(0.0)

    return df_out


def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: list[str],
    category_mappings: dict[str, dict[Any, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, int]]]:
    """Encode categorical columns into deterministic integer codes."""
    df_out = df.copy()
    mappings: dict[str, dict[Any, int]] = category_mappings or {}

    for col in categorical_cols:
        if col in df_out.columns:
            if col not in mappings:
                unique_vals = sorted(df_out[col].dropna().unique())
                mappings[col] = {val: idx for idx, val in enumerate(unique_vals)}

            mapping = mappings[col]
            df_out[col] = df_out[col].map(mapping).fillna(-1).astype(int)

    return df_out, mappings


def build_feature_pipeline(
    raw_df: pd.DataFrame,
    config: FeatureConfig = DEFAULT_CONFIG.feature,
    category_mappings: dict[str, dict[Any, int]] | None = None,
    impute_missing_lags: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[Any, int]]]:
    """Execute end-to-end feature extraction on merged demand dataset.

    Returns:
        Transformed DataFrame, list of engineered feature names, and category mappings.
    """
    logger.info("Building feature pipeline for dataset with %d rows...", len(raw_df))
    df = raw_df.copy()

    # 1. Price features
    df = create_price_features(df)

    # 2. Promo synergy features
    df = create_promo_features(df)

    # 3. Cyclical temporal calendar features
    df = create_temporal_features(df, time_col="week")

    # 4. Lag features on target demand
    if config.target_col in df.columns:
        df = create_lag_features(
            df,
            group_cols=config.group_cols,
            target_col=config.target_col,
            lags=config.lags,
        )

    # 5. Rolling statistics
    if config.target_col in df.columns:
        df = create_rolling_features(
            df,
            group_cols=config.group_cols,
            target_col=config.target_col,
            windows=config.rolling_windows,
        )

    # 6. Categorical encoding
    df, mappings = encode_categorical_features(
        df,
        categorical_cols=config.categorical_cols,
        category_mappings=category_mappings,
    )

    # 7. Identify feature column subset (exclude target, IDs, and raw text)
    excluded_cols = {"id", config.target_col, "_target"}
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    # 8. Impute missing lag values (from initial time periods) if enabled
    if impute_missing_lags:
        for col in feature_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(0.0)

    logger.info("Feature pipeline finished. Extracted %d feature columns.", len(feature_cols))
    return df, feature_cols, mappings
