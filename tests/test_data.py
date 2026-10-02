"""Tests for data validation and time-based splitting."""

import pandas as pd
import pytest

from demandiq.config import AppConfig, SplitConfig
from demandiq.data import EXPECTED_TRAIN_COLUMNS, time_based_split, validate_raw_schema


def test_validate_raw_schema_valid():
    df = pd.DataFrame(columns=list(EXPECTED_TRAIN_COLUMNS))
    assert validate_raw_schema(df, EXPECTED_TRAIN_COLUMNS) is True


def test_validate_raw_schema_missing_column():
    df = pd.DataFrame(columns=["id", "week"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_schema(df, EXPECTED_TRAIN_COLUMNS)


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
