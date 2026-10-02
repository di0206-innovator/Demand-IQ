"""Tests for robust dataset schema validation, referential integrity, and splitting."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandiq.config import AppConfig, SplitConfig
from demandiq.data import (
    EXPECTED_CENTERS_COLUMNS,
    EXPECTED_MEALS_COLUMNS,
    EXPECTED_TEST_COLUMNS,
    EXPECTED_TRAIN_COLUMNS,
    RAW_SCHEMAS,
    RelationshipValidationError,
    SchemaValidationError,
    load_raw_datasets,
    time_based_split,
    validate_raw_schema,
    validate_table_relationships,
    validate_table_schema,
)


@pytest.fixture
def valid_raw_datasets() -> dict[str, pd.DataFrame]:
    """Provide a consistent set of minimal valid raw dataframes."""
    centers_df = pd.DataFrame(
        {
            "center_id": [10, 11],
            "city_code": [501, 502],
            "region_code": [30, 31],
            "center_type": ["TYPE_A", "TYPE_B"],
            "op_area": [3.5, 4.0],
        }
    )
    meals_df = pd.DataFrame(
        {
            "meal_id": [101, 102],
            "category": ["Beverages", "Rice Bowl"],
            "cuisine": ["Indian", "Italian"],
        }
    )
    train_df = pd.DataFrame(
        {
            "id": [1001, 1002],
            "week": [1, 2],
            "center_id": [10, 11],
            "meal_id": [101, 102],
            "checkout_price": [120.5, 150.0],
            "base_price": [140.0, 150.0],
            "emailer_for_promotion": [0, 1],
            "homepage_featured": [1, 0],
            "num_orders": [250, 300],
        }
    )
    test_df = pd.DataFrame(
        {
            "id": [2001, 2002],
            "week": [146, 147],
            "center_id": [10, 11],
            "meal_id": [101, 102],
            "checkout_price": [125.0, 155.0],
            "base_price": [145.0, 155.0],
            "emailer_for_promotion": [1, 0],
            "homepage_featured": [0, 0],
        }
    )
    return {
        "train": train_df,
        "test": test_df,
        "centers": centers_df,
        "meals": meals_df,
    }


def test_schema_validation_all_valid(valid_raw_datasets):
    for key, df in valid_raw_datasets.items():
        assert validate_table_schema(df, RAW_SCHEMAS[key]) is True


def test_schema_validation_missing_columns(valid_raw_datasets):
    df_missing = valid_raw_datasets["train"].drop(columns=["checkout_price"])
    with pytest.raises(SchemaValidationError, match="Missing required column"):
        validate_table_schema(df_missing, RAW_SCHEMAS["train"])


def test_schema_validation_strict_mode_extra_columns(valid_raw_datasets):
    df_extra = valid_raw_datasets["centers"].copy()
    df_extra["extra_column"] = "unexpected"
    with pytest.raises(SchemaValidationError, match="Unexpected column"):
        validate_table_schema(df_extra, RAW_SCHEMAS["centers"], strict_columns=True)


def test_schema_validation_null_primary_key(valid_raw_datasets):
    df_null_pk = valid_raw_datasets["meals"].copy()
    df_null_pk.loc[0, "meal_id"] = np.nan
    with pytest.raises(SchemaValidationError, match="contains null values"):
        validate_table_schema(df_null_pk, RAW_SCHEMAS["meals"])


def test_schema_validation_duplicate_primary_key(valid_raw_datasets):
    df_dup_pk = valid_raw_datasets["train"].copy()
    df_dup_pk.loc[1, "id"] = df_dup_pk.loc[0, "id"]
    with pytest.raises(SchemaValidationError, match="contains 1 duplicate entries"):
        validate_table_schema(df_dup_pk, RAW_SCHEMAS["train"])


def test_schema_validation_non_binary_flags(valid_raw_datasets):
    df_invalid_flag = valid_raw_datasets["train"].copy()
    df_invalid_flag.loc[0, "emailer_for_promotion"] = 5
    with pytest.raises(SchemaValidationError, match="contains non-binary values"):
        validate_table_schema(df_invalid_flag, RAW_SCHEMAS["train"])


def test_schema_validation_non_positive_price(valid_raw_datasets):
    df_neg_price = valid_raw_datasets["train"].copy()
    df_neg_price.loc[0, "checkout_price"] = 0.0
    with pytest.raises(SchemaValidationError, match="non-positive"):
        validate_table_schema(df_neg_price, RAW_SCHEMAS["train"])


def test_schema_validation_negative_num_orders(valid_raw_datasets):
    df_neg_orders = valid_raw_datasets["train"].copy()
    df_neg_orders.loc[0, "num_orders"] = -10
    with pytest.raises(SchemaValidationError, match="negative"):
        validate_table_schema(df_neg_orders, RAW_SCHEMAS["train"])


def test_validate_table_relationships_valid(valid_raw_datasets):
    assert validate_table_relationships(valid_raw_datasets, RAW_SCHEMAS) is True


def test_validate_table_relationships_unmapped_center(valid_raw_datasets):
    invalid_datasets = dict(valid_raw_datasets)
    invalid_train = invalid_datasets["train"].copy()
    invalid_train.loc[0, "center_id"] = 99999  # Unmapped center
    invalid_datasets["train"] = invalid_train

    with pytest.raises(
        RelationshipValidationError, match="Referential integrity violation on 'center_id'"
    ):
        validate_table_relationships(invalid_datasets, RAW_SCHEMAS)


def test_validate_table_relationships_unmapped_meal(valid_raw_datasets):
    invalid_datasets = dict(valid_raw_datasets)
    invalid_test = invalid_datasets["test"].copy()
    invalid_test.loc[0, "meal_id"] = 88888  # Unmapped meal
    invalid_datasets["test"] = invalid_test

    with pytest.raises(
        RelationshipValidationError, match="Referential integrity violation on 'meal_id'"
    ):
        validate_table_relationships(invalid_datasets, RAW_SCHEMAS)


def test_load_raw_datasets_missing_files_error(tmp_path: Path):
    empty_dir = tmp_path / "raw_empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Required dataset 'train.csv' not found"):
        load_raw_datasets(data_dir=empty_dir)


def test_load_raw_datasets_success(tmp_path: Path, valid_raw_datasets):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    for key, df in valid_raw_datasets.items():
        filename = (
            "train.csv"
            if key == "train"
            else "test.csv"
            if key == "test"
            else "fulfilment_center_info.csv"
            if key == "centers"
            else "meal_info.csv"
        )
        df.to_csv(data_dir / filename, index=False)

    loaded = load_raw_datasets(data_dir=data_dir, include_test=True)
    assert set(loaded.keys()) == {"train", "test", "centers", "meals"}
    assert len(loaded["train"]) == 2


def test_validate_raw_schema_backward_compatibility():
    df = pd.DataFrame(columns=list(EXPECTED_TRAIN_COLUMNS))
    assert validate_raw_schema(df, EXPECTED_TRAIN_COLUMNS) is True
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_schema(pd.DataFrame(columns=["id"]), EXPECTED_TRAIN_COLUMNS)


def test_expected_column_sets():
    assert "num_orders" in EXPECTED_TRAIN_COLUMNS
    assert "num_orders" not in EXPECTED_TEST_COLUMNS
    assert "op_area" in EXPECTED_CENTERS_COLUMNS
    assert "cuisine" in EXPECTED_MEALS_COLUMNS


def test_time_based_split_disjoint_and_temporal_order():
    df = pd.DataFrame(
        {
            "week": list(range(1, 146)),
            "val": list(range(1, 146)),
        }
    )
    config = AppConfig(split=SplitConfig(train_end_week=100, val_end_week=120, total_weeks=145))
    train, val, test = time_based_split(df, time_col="week", config=config)

    assert train["week"].max() == 100
    assert val["week"].min() == 101
    assert val["week"].max() == 120
    assert test["week"].min() == 121
    assert test["week"].max() == 145
    assert len(train) + len(val) + len(test) == 145


def test_time_based_split_missing_column():
    df = pd.DataFrame({"value": [1, 2, 3]})
    with pytest.raises(KeyError, match="Time column 'week' not present"):
        time_based_split(df, time_col="week")
