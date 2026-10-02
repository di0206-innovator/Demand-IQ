"""Configuration module for DemandIQ.

Centralizes paths, dataset schemas, split parameters, and default settings.
"""

from dataclasses import dataclass, field
from pathlib import Path


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
class AppConfig:
    """Master application configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)
    random_seed: int = 42


# Global default configuration instance
DEFAULT_CONFIG = AppConfig()
