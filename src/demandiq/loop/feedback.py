"""Active feedback memory and performance tracking store."""

from dataclasses import dataclass, field
from datetime import datetime
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
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class FeedbackStore:
    """Stores online feedback events and tracks rolling forecast reliability."""

    def __init__(self, max_history: int = 100000) -> None:
        self.max_history = max_history
        self.records: list[ForecastFeedback] = []

    def record_batch_feedback(
        self,
        df: pd.DataFrame,
        week_col: str = "week",
        center_col: str = "center_id",
        meal_col: str = "meal_id",
        forecast_col: str = "forecast_orders",
        actual_col: str = "actual_orders",
    ) -> None:
        """Append a batch of realized weekly outcomes into feedback memory."""
        for _, row in df.iterrows():
            f_val = float(row[forecast_col])
            a_val = float(row[actual_col])
            err = a_val - f_val
            pct_err = (err / max(1.0, a_val)) * 100.0

            fb = ForecastFeedback(
                week=int(row[week_col]),
                center_id=int(row[center_col]),
                meal_id=int(row[meal_col]),
                forecast_orders=round(f_val, 2),
                actual_orders=round(a_val, 2),
                error=round(err, 2),
                percentage_error=round(pct_err, 2),
            )
            self.records.append(fb)

        # Cap memory size
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history :]
        logger.info("Recorded feedback batch. Total memory records: %d", len(self.records))

    def get_summary_metrics(self) -> dict[str, Any]:
        """Compute holistic tracking metrics over historical feedback buffer."""
        if not self.records:
            return {"status": "EMPTY", "sample_count": 0}

        y_true = np.array([r.actual_orders for r in self.records])
        y_pred = np.array([r.forecast_orders for r in self.records])
        errors = np.array([r.error for r in self.records])

        m = compute_metrics(y_true, y_pred)
        under_forecast_rate = float(np.mean(errors > 0)) * 100.0
        over_forecast_rate = float(np.mean(errors < 0)) * 100.0
        mean_bias = float(np.mean(errors))

        return {
            "sample_count": len(self.records),
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
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.__dict__ for r in self.records])
