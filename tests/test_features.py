"""Tests for feature engineering logic and leakage safety."""

import pandas as pd
import pytest

from demandiq.features import (
    create_lag_features,
    create_price_features,
    create_promo_features,
)


def test_create_price_features():
    df = pd.DataFrame(
        {
            "checkout_price": [100.0, 150.0],
            "base_price": [120.0, 150.0],
        }
    )
    res = create_price_features(df)
    assert "discount_amount" in res.columns
    assert "discount_ratio" in res.columns
    assert res.loc[0, "discount_amount"] == 20.0
    assert res.loc[1, "discount_amount"] == 0.0


def test_create_promo_features():
    df = pd.DataFrame(
        {
            "emailer_for_promotion": [1, 0, 1],
            "homepage_featured": [1, 1, 0],
        }
    )
    res = create_promo_features(df)
    assert res["both_promos"].tolist() == [1, 0, 0]
    assert res["any_promo"].tolist() == [1, 1, 1]


def test_create_lag_features_leakage_protection():
    df = pd.DataFrame(
        {
            "center_id": [1, 1, 1],
            "meal_id": [10, 10, 10],
            "week": [1, 2, 3],
            "num_orders": [50, 60, 70],
        }
    )
    # Lags < 1 should be rejected to avoid target leakage
    with pytest.raises(ValueError, match="Lag must be >= 1"):
        create_lag_features(df, group_cols=["center_id", "meal_id"], lags=[0, 1])

    # Valid lags
    res = create_lag_features(df, group_cols=["center_id", "meal_id"], lags=[1])
    assert "num_orders_lag_1" in res.columns
    assert pd.isna(res.loc[0, "num_orders_lag_1"])
    assert res.loc[1, "num_orders_lag_1"] == 50
    assert res.loc[2, "num_orders_lag_1"] == 60
