"""Model forecasting interfaces, baseline model, and XGBoost forecasting engine."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from demandiq.config import DEFAULT_CONFIG, ModelConfig
from demandiq.utils import calculate_rmsle, calculate_wmape, setup_logger

logger = setup_logger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute standard forecasting evaluation metrics."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(root_mean_squared_error(y_t, y_p))
    wmape = float(calculate_wmape(y_t, y_p))
    rmsle = float(calculate_rmsle(y_t, y_p))

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WMAPE": round(wmape, 4),
        "RMSLE": round(rmsle, 4),
    }


class BaseForecaster(ABC):
    """Abstract base class for all DemandIQ forecasting models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "BaseForecaster":
        """Fit model to training data."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts."""
        pass

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Evaluate model against ground truth target."""
        preds = self.predict(X)
        return compute_metrics(y.to_numpy(), preds)


class NaiveBaselineForecaster(BaseForecaster):
    """Historical baseline forecaster using group-level medians."""

    def __init__(self, group_cols: list[str] | None = None) -> None:
        self.group_cols = group_cols or ["center_id", "meal_id"]
        self.group_medians: dict[tuple, float] = {}
        self.global_median: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "NaiveBaselineForecaster":
        df = X.copy()
        df["_target"] = y.to_numpy()
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
        return np.clip(np.array(preds, dtype=float), 0, None)


class XGBoostDemandForecaster(BaseForecaster):
    """Production-grade XGBoost demand forecasting model with log-target optimization."""

    def __init__(
        self,
        model_config: ModelConfig = DEFAULT_CONFIG.model,
        use_log_target: bool = True,
    ) -> None:
        self.config = model_config
        self.use_log_target = use_log_target
        self.model: xgb.XGBRegressor | None = None
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "XGBoostDemandForecaster":
        """Train XGBoost Regressor with early stopping if validation set is supplied."""
        self.feature_names = list(X.columns)
        X_train = X[self.feature_names].copy()
        y_train = np.log1p(y.to_numpy()) if self.use_log_target else y.to_numpy()

        params = self.config.to_dict()
        early_stopping = None
        eval_set = None

        if X_val is not None and y_val is not None:
            X_v = X_val[self.feature_names].copy()
            y_v = np.log1p(y_val.to_numpy()) if self.use_log_target else y_val.to_numpy()
            eval_set = [(X_train, y_train), (X_v, y_v)]
            early_stopping = self.config.early_stopping_rounds

        self.model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            eval_metric="rmse",
            early_stopping_rounds=early_stopping,
            n_jobs=-1,
        )

        logger.info(
            "Fitting XGBoost model on %d samples (features=%d)...",
            len(X_train),
            len(self.feature_names),
        )

        if eval_set is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )
            best_iter = getattr(self.model, "best_iteration", None)
            logger.info("XGBoost training completed. Best iteration: %s", best_iter)
        else:
            self.model.fit(X_train, y_train, verbose=False)
            logger.info("XGBoost training completed without validation set.")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate non-negative demand predictions."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot predict with unfitted model. Call fit() first.")

        X_eval = X[self.feature_names].copy()
        raw_preds = self.model.predict(X_eval)

        if self.use_log_target:
            preds = np.expm1(raw_preds)
        else:
            preds = raw_preds

        return np.clip(preds, 0, None)

    def get_feature_importances(self, top_n: int = 20) -> dict[str, float]:
        """Return top feature importances by gain."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")

        importances = self.model.feature_importances_
        sorted_pairs = sorted(
            zip(self.feature_names, importances, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return {feat: float(imp) for feat, imp in sorted_pairs[:top_n]}

    def compute_shap_values(
        self,
        X_sample: pd.DataFrame,
        max_samples: int = 300,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Compute TreeExplainer SHAP values for model explainability."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted.")

        X_eval = X_sample[self.feature_names].copy()
        if len(X_eval) > max_samples:
            X_eval = X_eval.sample(n=max_samples, random_state=self.config.random_state)

        explainer = shap.TreeExplainer(self.model)
        shap_vals = explainer.shap_values(X_eval)
        return shap_vals, X_eval.to_numpy(), self.feature_names

    def save(self, filepath: Path | str) -> None:
        """Serialize trained model and feature metadata to disk."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted model.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "use_log_target": self.use_log_target,
            "config": self.config,
        }
        joblib.dump(payload, path)
        logger.info("Saved model artifact to %s", path)

    @classmethod
    def load(cls, filepath: Path | str) -> "XGBoostDemandForecaster":
        """Load serialized model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}")

        payload: dict[str, Any] = joblib.load(path)
        forecaster = cls(
            model_config=payload["config"],
            use_log_target=payload["use_log_target"],
        )
        forecaster.model = payload["model"]
        forecaster.feature_names = payload["feature_names"]
        forecaster.is_fitted = True
        logger.info(
            "Loaded model artifact from %s with %d features", path, len(forecaster.feature_names)
        )
        return forecaster
