"""Data loading, validation, and splitting interfaces for DemandIQ."""

from pathlib import Path

import pandas as pd

from demandiq.config import DEFAULT_CONFIG, AppConfig
from demandiq.utils import setup_logger

logger = setup_logger(__name__)

RAW_DATA_FILES = {
    "train": "train.csv",
    "centers": "fulfilment_center_info.csv",
    "meals": "meal_info.csv",
}

EXPECTED_TRAIN_COLUMNS = {
    "id",
    "week",
    "center_id",
    "meal_id",
    "checkout_price",
    "base_price",
    "emailer_for_promotion",
    "homepage_featured",
    "num_orders",
}


def validate_raw_schema(df: pd.DataFrame, expected_columns: set[str]) -> bool:
    """Verify that dataframe contains all required columns.

    Raises:
        ValueError: If any expected columns are missing.
    """
    missing = expected_columns - set(df.columns)
    if missing:
        msg = f"Missing required columns in dataset: {sorted(missing)}"
        logger.error(msg)
        raise ValueError(msg)
    return True


def load_raw_datasets(
    data_dir: Path | None = None,
    config: AppConfig = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Load raw dataset CSVs from the configured directory.

    Raises:
        FileNotFoundError: If any mandatory raw data file is missing.
    """
    base_dir = data_dir or config.paths.data_raw
    datasets: dict[str, pd.DataFrame] = {}

    for key, filename in RAW_DATA_FILES.items():
        filepath = base_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required dataset '{filename}' not found at {filepath}. "
                f"Please follow instructions in data/raw/README.md to populate raw data."
            )
        logger.info("Loading %s from %s", key, filepath)
        datasets[key] = pd.read_csv(filepath)

    validate_raw_schema(datasets["train"], EXPECTED_TRAIN_COLUMNS)
    return datasets


def time_based_split(
    df: pd.DataFrame,
    time_col: str = "week",
    config: AppConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform temporal split into train, validation, and test partitions.

    Ensures no future data leaks into earlier sets.
    """
    if time_col not in df.columns:
        raise KeyError(f"Time column '{time_col}' not present in dataframe.")

    train_df = df[df[time_col] <= config.split.train_end_week].copy()
    val_df = df[
        (df[time_col] > config.split.train_end_week) & (df[time_col] <= config.split.val_end_week)
    ].copy()
    test_df = df[df[time_col] > config.split.val_end_week].copy()

    logger.info(
        "Temporal split completed: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df
