"""Active feedback memory and performance tracking store."""

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from demandiq.models import compute_metrics
from demandiq.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ForecastFeedback:
    """Individual record of predicted vs realized demand."""

    week: int
    center_id: int
    meal_id: int
    forecast_orders: float
    actual_orders: float
    error: float
    percentage_error: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class FeedbackStore:
    """Thread-safe storage for online feedback events and rolling forecast reliability."""

    def __init__(self, max_history: int = 100000) -> None:
        if max_history <= 0:
            raise ValueError(f"max_history must be a positive integer, got {max_history}")
        self.max_history = max_history
        self.records: list[ForecastFeedback] = []
        self._lock = threading.Lock()

    def record_batch_feedback(
        self,
        df: pd.DataFrame,
        week_col: str = "week",
        center_col: str = "center_id",
        meal_col: str = "meal_id",
        forecast_col: str = "forecast_orders",
        actual_col: str = "actual_orders",
    ) -> None:
        """Append a batch of realized weekly outcomes into feedback memory using vectorized extraction."""
        if df.empty:
            return

        for col in [week_col, center_col, meal_col, forecast_col, actual_col]:
            if col not in df.columns:
                raise KeyError(f"Required feedback column '{col}' missing from DataFrame.")

        weeks = df[week_col].to_numpy(dtype=int)
        centers = df[center_col].to_numpy(dtype=int)
        meals = df[meal_col].to_numpy(dtype=int)
        f_vals = df[forecast_col].to_numpy(dtype=float)
        a_vals = df[actual_col].to_numpy(dtype=float)

        errors = a_vals - f_vals
        denoms = np.maximum(1.0, a_vals)
        pct_errors = (errors / denoms) * 100.0
        now_ts = datetime.now(UTC).isoformat()

        new_records = [
            ForecastFeedback(
                week=int(w),
                center_id=int(c),
                meal_id=int(m),
                forecast_orders=round(float(f), 2),
                actual_orders=round(float(a), 2),
                error=round(float(err), 2),
                percentage_error=round(float(pct_err), 2),
                timestamp=now_ts,
            )
            for w, c, m, f, a, err, pct_err in zip(
                weeks, centers, meals, f_vals, a_vals, errors, pct_errors, strict=False
            )
        ]

        with self._lock:
            self.records.extend(new_records)
            if len(self.records) > self.max_history:
                self.records = self.records[-self.max_history :]

        logger.info("Recorded feedback batch. Total memory records: %d", len(self.records))

    def get_summary_metrics(self) -> dict[str, Any]:
        """Compute holistic tracking metrics over historical feedback buffer."""
        with self._lock:
            records_snapshot = list(self.records)

        if not records_snapshot:
            return {"status": "EMPTY", "sample_count": 0}

        y_true = np.array([r.actual_orders for r in records_snapshot])
        y_pred = np.array([r.forecast_orders for r in records_snapshot])
        errors = np.array([r.error for r in records_snapshot])

        m = compute_metrics(y_true, y_pred)
        under_forecast_rate = float(np.mean(errors > 0)) * 100.0
        over_forecast_rate = float(np.mean(errors < 0)) * 100.0
        mean_bias = float(np.mean(errors))

        return {
            "sample_count": len(records_snapshot),
            "wmape": m["WMAPE"],
            "rmsle": m["RMSLE"],
            "mae": m["MAE"],
            "rmse": m["RMSE"],
            "mean_bias_units": round(mean_bias, 2),
            "under_forecast_rate_pct": round(under_forecast_rate, 2),
            "over_forecast_rate_pct": round(over_forecast_rate, 2),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert feedback log into DataFrame for visual analytics."""
        with self._lock:
            records_snapshot = list(self.records)
        if not records_snapshot:
            return pd.DataFrame(
                columns=[
                    "week",
                    "center_id",
                    "meal_id",
                    "forecast_orders",
                    "actual_orders",
                    "error",
                    "percentage_error",
                    "timestamp",
                ]
            )
        return pd.DataFrame([r.__dict__ for r in records_snapshot])
