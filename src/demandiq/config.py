"""Configuration module for DemandIQ.

Centralizes paths, dataset schemas, split parameters, feature definitions, and model hyperparameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathConfig:
    """Project filesystem paths."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_raw: Path = field(init=False)
    data_processed: Path = field(init=False)
    models: Path = field(init=False)
    reports_figures: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_raw", self.project_root / "data" / "raw")
        object.__setattr__(self, "data_processed", self.project_root / "data" / "processed")
        object.__setattr__(self, "models", self.project_root / "models")
        object.__setattr__(self, "reports_figures", self.project_root / "reports" / "figures")


@dataclass(frozen=True)
class InventoryConfig:
    """Inventory recommendation parameters."""

    service_level: float = 0.95  # Target cycle service level (e.g. 95%)
    lead_time_weeks: int = 1  # Lead time in weeks
    holding_cost_rate: float = 0.15  # Annual holding cost rate percentage
    min_safety_stock: int = 5  # Minimum buffer units


@dataclass(frozen=True)
class SplitConfig:
    """Time-aware train/validation/test split configuration."""

    train_end_week: int = 125
    val_end_week: int = 135
    total_weeks: int = 145


@dataclass(frozen=True)
class FeatureConfig:
    """Feature engineering configuration."""

    group_cols: list[str] = field(default_factory=lambda: ["center_id", "meal_id"])
    target_col: str = "num_orders"
    lags: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 10])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 5, 10])
    use_log_target: bool = True
    categorical_cols: list[str] = field(
        default_factory=lambda: ["center_type", "category", "cuisine", "city_code", "region_code"]
    )


@dataclass(frozen=True)
class ModelConfig:
    """XGBoost demand forecasting hyperparameters."""

    n_estimators: int = 600
    max_depth: int = 7
    learning_rate: float = 0.03
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.1
    early_stopping_rounds: int = 40
    random_state: int = 42

    def to_dict(self) -> dict[str, Any]:
        """Convert hyperparameters to dictionary for XGBoost."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "random_state": self.random_state,
        }


@dataclass(frozen=True)
class AppConfig:
    """Master application configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    random_seed: int = 42


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()
