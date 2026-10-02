"""Tests for feature engineering logic, lag features, rolling statistics, and encodings."""

import numpy as np
import pandas as pd
import pytest

from demandiq.config import FeatureConfig
from demandiq.features import (
    build_feature_pipeline,
    create_lag_features,
    create_price_features,
    create_promo_features,
    create_rolling_features,
    create_temporal_features,
    encode_categorical_features,
)


def test_create_price_features():
    df = pd.DataFrame(
        {
            "meal_id": [1, 1],
            "checkout_price": [100.0, 150.0],
            "base_price": [120.0, 150.0],
        }
    )
    res = create_price_features(df)
    assert "discount_amount" in res.columns
    assert "discount_ratio" in res.columns
    assert "price_ratio" in res.columns
    assert "meal_rel_price" in res.columns
    assert res.loc[0, "discount_amount"] == 20.0
    assert res.loc[1, "discount_amount"] == 0.0


def test_create_promo_features():
    df = pd.DataFrame(
        {
            "emailer_for_promotion": [1, 0, 1],
            "homepage_featured": [1, 1, 0],
            "checkout_price": [90.0, 100.0, 80.0],
            "base_price": [100.0, 100.0, 100.0],
        }
    )
    res = create_price_features(df)
    res = create_promo_features(res)
    assert res["both_promos"].tolist() == [1, 0, 0]
    assert res["any_promo"].tolist() == [1, 1, 1]
    assert "promo_discount_synergy" in res.columns


def test_create_temporal_features():
    df = pd.DataFrame({"week": [1, 13, 26, 52, 53]})
    res = create_temporal_features(df, time_col="week")
    assert "week_of_year" in res.columns
    assert "sin_week_of_year" in res.columns
    assert "cos_week_of_year" in res.columns
    assert res.loc[0, "week_of_year"] == 1
    assert res.loc[4, "week_of_year"] == 1


def test_create_lag_features_leakage_protection():
    df = pd.DataFrame(
        {
            "center_id": [1, 1, 1],
            "meal_id": [10, 10, 10],
            "week": [1, 2, 3],
            "num_orders": [50, 60, 70],
        }
    )
    with pytest.raises(ValueError, match="Lag must be >= 1"):
        create_lag_features(df, group_cols=["center_id", "meal_id"], lags=[0, 1])

    res = create_lag_features(df, group_cols=["center_id", "meal_id"], lags=[1, 2])
    assert "num_orders_lag_1" in res.columns
    assert "num_orders_lag_2" in res.columns
    assert pd.isna(res.loc[0, "num_orders_lag_1"])
    assert res.loc[1, "num_orders_lag_1"] == 50
    assert res.loc[2, "num_orders_lag_1"] == 60


def test_create_rolling_features():
    df = pd.DataFrame(
        {
            "center_id": [1, 1, 1, 1],
            "meal_id": [10, 10, 10, 10],
            "week": [1, 2, 3, 4],
            "num_orders": [10, 20, 30, 40],
        }
    )
    res = create_rolling_features(df, group_cols=["center_id", "meal_id"], windows=[2])
    assert "num_orders_roll_mean_2" in res.columns
    assert "num_orders_roll_std_2" in res.columns
    # Row 0 has no past data -> roll_mean is NaN or imputed
    # Row 1 has past data [10] -> roll_mean = 10
    # Row 2 has past data [10, 20] -> roll_mean = 15
    assert res.loc[1, "num_orders_roll_mean_2"] == 10.0
    assert res.loc[2, "num_orders_roll_mean_2"] == 15.0


def test_encode_categorical_features():
    df = pd.DataFrame(
        {
            "center_type": ["TYPE_A", "TYPE_B", "TYPE_A"],
            "cuisine": ["Indian", "Italian", "Indian"],
        }
    )
    df_enc, mappings = encode_categorical_features(df, ["center_type", "cuisine"])
    assert "center_type" in mappings
    assert "cuisine" in mappings
    assert np.issubdtype(df_enc["center_type"].dtype, np.integer)
    assert np.issubdtype(df_enc["cuisine"].dtype, np.integer)


def test_build_feature_pipeline_end_to_end():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "week": [1, 2, 3, 4],
            "center_id": [10, 10, 10, 10],
            "meal_id": [100, 100, 100, 100],
            "checkout_price": [100.0, 95.0, 90.0, 85.0],
            "base_price": [100.0, 100.0, 100.0, 100.0],
            "emailer_for_promotion": [0, 1, 0, 1],
            "homepage_featured": [0, 0, 1, 1],
            "center_type": ["TYPE_A", "TYPE_A", "TYPE_A", "TYPE_A"],
            "cuisine": ["Indian", "Indian", "Indian", "Indian"],
            "num_orders": [50, 75, 90, 120],
        }
    )
    config = FeatureConfig(
        lags=[1], rolling_windows=[2], categorical_cols=["center_type", "cuisine"]
    )
    df_out, feature_cols, mappings = build_feature_pipeline(df, config=config)

    assert "num_orders" not in feature_cols
    assert "id" not in feature_cols
    assert "discount_ratio" in feature_cols
    assert "num_orders_lag_1" in feature_cols
    assert len(df_out) == 4
    # All features should be non-null after imputation
    assert df_out[feature_cols].isna().sum().sum() == 0
