"""DemandIQ: AI Demand Forecasting & Inventory Intelligence package."""

from demandiq.config import DEFAULT_CONFIG, AppConfig
from demandiq.inventory import recommend_inventory
from demandiq.models import NaiveBaselineForecaster, XGBoostDemandForecaster
from demandiq.pipeline import run_training_pipeline

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "AppConfig",
    "DEFAULT_CONFIG",
    "run_training_pipeline",
    "XGBoostDemandForecaster",
    "NaiveBaselineForecaster",
    "recommend_inventory",
]
