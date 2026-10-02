"""Data loading, robust schema validation, and splitting interfaces for DemandIQ."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from demandiq.config import DEFAULT_CONFIG, AppConfig
from demandiq.utils import setup_logger

logger = setup_logger(__name__)


class DataValidationError(ValueError):
    """Raised when dataset fails structural, schema, or relational validation."""


class SchemaValidationError(DataValidationError):
    """Raised when columns, types, or nullability constraints are violated."""


class RelationshipValidationError(DataValidationError):
    """Raised when referential integrity between tables is broken."""


@dataclass(frozen=True)
class TableSchema:
    """Schema definition and integrity constraints for a tabular dataset."""

    name: str
    primary_key: list[str]
    required_columns: dict[str, type | tuple[type, ...]]
    foreign_keys: dict[str, tuple[str, str]] = field(default_factory=dict)
    binary_columns: list[str] = field(default_factory=list)
    positive_numeric_columns: list[str] = field(default_factory=list)
    non_negative_numeric_columns: list[str] = field(default_factory=list)


# Schema definitions for the 4 raw Kaggle Food Demand Forecasting datasets
RAW_SCHEMAS: dict[str, TableSchema] = {
    "train": TableSchema(
        name="train.csv",
        primary_key=["id"],
        required_columns={
            "id": (int, np.integer),
            "week": (int, np.integer),
            "center_id": (int, np.integer),
            "meal_id": (int, np.integer),
            "checkout_price": (float, np.floating, int, np.integer),
            "base_price": (float, np.floating, int, np.integer),
            "emailer_for_promotion": (int, np.integer),
            "homepage_featured": (int, np.integer),
            "num_orders": (int, np.integer, float, np.floating),
        },
        foreign_keys={
            "center_id": ("centers", "center_id"),
            "meal_id": ("meals", "meal_id"),
        },
        binary_columns=["emailer_for_promotion", "homepage_featured"],
        positive_numeric_columns=["checkout_price", "base_price"],
        non_negative_numeric_columns=["num_orders"],
    ),
    "test": TableSchema(
        name="test.csv",
        primary_key=["id"],
        required_columns={
            "id": (int, np.integer),
            "week": (int, np.integer),
            "center_id": (int, np.integer),
            "meal_id": (int, np.integer),
            "checkout_price": (float, np.floating, int, np.integer),
            "base_price": (float, np.floating, int, np.integer),
            "emailer_for_promotion": (int, np.integer),
            "homepage_featured": (int, np.integer),
        },
        foreign_keys={
            "center_id": ("centers", "center_id"),
            "meal_id": ("meals", "meal_id"),
        },
        binary_columns=["emailer_for_promotion", "homepage_featured"],
        positive_numeric_columns=["checkout_price", "base_price"],
    ),
    "centers": TableSchema(
        name="fulfilment_center_info.csv",
        primary_key=["center_id"],
        required_columns={
            "center_id": (int, np.integer),
            "city_code": (int, np.integer),
            "region_code": (int, np.integer),
            "center_type": (str, object),
            "op_area": (float, np.floating, int, np.integer),
        },
        positive_numeric_columns=["op_area"],
    ),
    "meals": TableSchema(
        name="meal_info.csv",
        primary_key=["meal_id"],
        required_columns={
            "meal_id": (int, np.integer),
            "category": (str, object),
            "cuisine": (str, object),
        },
    ),
}

RAW_DATA_FILES: dict[str, str] = {
    "train": "train.csv",
    "test": "test.csv",
    "centers": "fulfilment_center_info.csv",
    "meals": "meal_info.csv",
}

EXPECTED_TRAIN_COLUMNS = set(RAW_SCHEMAS["train"].required_columns.keys())
EXPECTED_TEST_COLUMNS = set(RAW_SCHEMAS["test"].required_columns.keys())
EXPECTED_CENTERS_COLUMNS = set(RAW_SCHEMAS["centers"].required_columns.keys())
EXPECTED_MEALS_COLUMNS = set(RAW_SCHEMAS["meals"].required_columns.keys())


def validate_table_schema(
    df: pd.DataFrame,
    schema: TableSchema,
    strict_columns: bool = False,
) -> bool:
    """Validate dataframe against TableSchema rules.

    Checks:
    1. Required columns presence.
    2. Unexpected columns (if strict_columns=True).
    3. Primary key uniqueness and non-nullness.
    4. Binary flags domain ([0, 1]).
    5. Positive & non-negative numeric constraints.

    Raises:
        SchemaValidationError: If any structural constraint is violated.
    """
    df_cols = set(df.columns)
    expected_cols = set(schema.required_columns.keys())

    # 1. Missing columns
    missing_cols = expected_cols - df_cols
    if missing_cols:
        msg = f"[{schema.name}] Missing required column(s): {sorted(missing_cols)}"
        logger.error(msg)
        raise SchemaValidationError(msg)

    # 2. Unexpected columns (strict mode)
    if strict_columns:
        extra_cols = df_cols - expected_cols
        if extra_cols:
            msg = f"[{schema.name}] Unexpected column(s) found in strict mode: {sorted(extra_cols)}"
            logger.error(msg)
            raise SchemaValidationError(msg)

    # 3. Primary key uniqueness and non-nullness
    for pk_col in schema.primary_key:
        if pk_col in df.columns:
            if df[pk_col].isna().any():
                msg = f"[{schema.name}] Primary key '{pk_col}' contains null values."
                logger.error(msg)
                raise SchemaValidationError(msg)
            if df[pk_col].duplicated().any():
                dup_count = int(df[pk_col].duplicated().sum())
                msg = f"[{schema.name}] Primary key '{pk_col}' contains {dup_count} duplicate entries."
                logger.error(msg)
                raise SchemaValidationError(msg)

    # 4. Binary columns validation
    for bin_col in schema.binary_columns:
        if bin_col in df.columns and not df.empty:
            invalid_vals = set(df[bin_col].dropna().unique()) - {0, 1}
            if invalid_vals:
                msg = (
                    f"[{schema.name}] Column '{bin_col}' contains non-binary values: {invalid_vals}"
                )
                logger.error(msg)
                raise SchemaValidationError(msg)

    # 5. Positive numeric columns (> 0)
    for pos_col in schema.positive_numeric_columns:
        if pos_col in df.columns and not df.empty:
            if (df[pos_col] <= 0).any():
                non_pos_count = int((df[pos_col] <= 0).sum())
                msg = f"[{schema.name}] Column '{pos_col}' contains {non_pos_count} non-positive (<=0) values."
                logger.error(msg)
                raise SchemaValidationError(msg)

    # 6. Non-negative numeric columns (>= 0)
    for non_neg_col in schema.non_negative_numeric_columns:
        if non_neg_col in df.columns and not df.empty:
            if (df[non_neg_col] < 0).any():
                neg_count = int((df[non_neg_col] < 0).sum())
                msg = f"[{schema.name}] Column '{non_neg_col}' contains {neg_count} negative (<0) values."
                logger.error(msg)
                raise SchemaValidationError(msg)

    logger.debug("[%s] Schema validation succeeded.", schema.name)
    return True


def validate_raw_schema(df: pd.DataFrame, expected_columns: set[str]) -> bool:
    """Backward-compatible wrapper for raw schema validation.

    Raises:
        ValueError: If any expected columns are missing.
    """
    missing = expected_columns - set(df.columns)
    if missing:
        msg = f"Missing required columns in dataset: {sorted(missing)}"
        logger.error(msg)
        raise ValueError(msg)
    return True


def validate_table_relationships(
    datasets: dict[str, pd.DataFrame],
    schemas: dict[str, TableSchema] = RAW_SCHEMAS,
) -> bool:
    """Verify referential integrity across related datasets.

    Checks:
    - All center_id values in train/test exist in fulfilment_center_info.csv.
    - All meal_id values in train/test exist in meal_info.csv.

    Raises:
        RelationshipValidationError: If foreign key relationships fail.
    """
    for table_key, df in datasets.items():
        if table_key not in schemas:
            continue
        schema = schemas[table_key]
        for fk_col, (ref_table_key, ref_pk_col) in schema.foreign_keys.items():
            if ref_table_key not in datasets:
                continue
            ref_df = datasets[ref_table_key]
            if fk_col not in df.columns or ref_pk_col not in ref_df.columns:
                continue

            child_ids = set(df[fk_col].dropna().unique())
            parent_ids = set(ref_df[ref_pk_col].dropna().unique())
            unmapped = child_ids - parent_ids

            if unmapped:
                msg = (
                    f"[{schema.name}] Referential integrity violation on '{fk_col}': "
                    f"{len(unmapped)} ID(s) not found in referenced table '{schemas[ref_table_key].name}' "
                    f"(sample unmapped IDs: {sorted(unmapped)[:5]})."
                )
                logger.error(msg)
                raise RelationshipValidationError(msg)

    logger.info("All dataset referential integrity checks passed successfully.")
    return True


def load_raw_datasets(
    data_dir: Path | None = None,
    config: AppConfig = DEFAULT_CONFIG,
    include_test: bool = True,
    strict_validation: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load and validate raw dataset CSVs from the configured directory.

    Args:
        data_dir: Directory containing raw CSV files. Defaults to config.paths.data_raw.
        config: Application configuration.
        include_test: If True, attempts to load test.csv if present.
        strict_validation: If True, executes schema and relationship validation.

    Raises:
        FileNotFoundError: If mandatory raw data files (train, centers, meals) are missing.
        DataValidationError: If dataset fails schema or relationship rules.
    """
    base_dir = data_dir or config.paths.data_raw
    datasets: dict[str, pd.DataFrame] = {}

    mandatory_keys = ["train", "centers", "meals"]
    keys_to_load = mandatory_keys + (["test"] if include_test else [])

    for key in keys_to_load:
        filename = RAW_DATA_FILES[key]
        filepath = base_dir / filename
        if not filepath.exists():
            if key in mandatory_keys:
                raise FileNotFoundError(
                    f"Required dataset '{filename}' not found at {filepath}.\n"
                    f"Please follow instructions in data/raw/README.md to download "
                    f"and place raw CSV files in {base_dir}."
                )
            logger.info("Optional dataset '%s' not present at %s (skipped).", filename, filepath)
            continue

        logger.info("Loading %s from %s", key, filepath)
        datasets[key] = pd.read_csv(filepath)

    if strict_validation:
        for key, df in datasets.items():
            if key in RAW_SCHEMAS:
                validate_table_schema(df, RAW_SCHEMAS[key])
        validate_table_relationships(datasets, RAW_SCHEMAS)

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


if __name__ == "__main__":
    import sys

    print("Running DemandIQ raw dataset verification...")
    try:
        loaded = load_raw_datasets()
        print(f"Successfully loaded and validated {len(loaded)} dataset(s): {list(loaded.keys())}")
    except FileNotFoundError as e:
        print(f"[AWAITING RAW DATA] {e}", file=sys.stderr)
    except DataValidationError as e:
        print(f"[DATA VALIDATION ERROR] {e}", file=sys.stderr)
        sys.exit(1)
