"""Model interfaces, baseline forecaster, and XGBoost pipeline placeholders."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from demandiq.utils import setup_logger

logger = setup_logger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all DemandIQ forecasting models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Fit model to training features and target."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts."""
        pass


class NaiveBaselineForecaster(BaseForecaster):
    """Simple historical baseline using recent group-level median or overall median."""

    def __init__(self, group_cols: list[str] | None = None) -> None:
        self.group_cols = group_cols or ["center_id", "meal_id"]
        self.group_medians: dict[tuple, float] = {}
        self.global_median: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBaselineForecaster":
        df = X.copy()
        df["_target"] = y
        self.global_median = float(y.median())

        grouped = df.groupby(self.group_cols)["_target"].median()
        self.group_medians = grouped.to_dict()
        logger.info(
            "Fitted NaiveBaselineForecaster with %d group medians; global median=%.2f",
            len(self.group_medians),
            self.global_median,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        for _, row in X.iterrows():
            key = tuple(row[col] for col in self.group_cols)
            val = self.group_medians.get(key, self.global_median)
            preds.append(val)
        return np.array(preds)


class XGBoostDemandForecaster(BaseForecaster):
    """Placeholder for the primary XGBoost demand forecasting model."""

    def __init__(self, model_params: dict[str, Any] | None = None) -> None:
        self.model_params = model_params or {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.model: Any = None
        self.feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostDemandForecaster":
        logger.info(
            "Initializing XGBoost forecaster placeholder. Model training deferred to Phase 2."
        )
        self.feature_names = list(X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logger.warning("predict() called on untrained placeholder model.")
        return np.zeros(len(X))
