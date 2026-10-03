"""Time-series cross-validation and sliced error evaluation harness."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from demandiq.models import BaseForecaster, compute_metrics
from demandiq.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class FoldEvaluation:
    """Evaluation metrics and metadata for a single backtest fold."""

    fold_index: int
    train_start_week: int
    train_end_week: int
    test_start_week: int
    test_end_week: int
    train_samples: int
    test_samples: int
    metrics: dict[str, float]


@dataclass
class SlicedPerformance:
    """Performance metrics segmented across operational dimensions."""

    dimension: str
    slice_name: str
    sample_count: int
    wmape: float
    rmsle: float
    mae: float
    rmse: float


@dataclass
class BacktestResult:
    """Comprehensive output of time-series cross-validation harness."""

    overall_metrics: dict[str, float]
    fold_evaluations: list[FoldEvaluation]
    sliced_performance: list[SlicedPerformance]
    predictions_df: pd.DataFrame = field(repr=False)


class TimeSeriesBacktester:
    """Expanding and sliding window backtesting engine for demand forecasting models."""

    def __init__(
        self,
        n_folds: int = 3,
        test_window_weeks: int = 10,
        min_train_weeks: int = 50,
        expanding_window: bool = True,
        time_col: str = "week",
        target_col: str = "num_orders",
    ) -> None:
        self.n_folds = n_folds
        self.test_window_weeks = test_window_weeks
        self.min_train_weeks = min_train_weeks
        self.expanding_window = expanding_window
        self.time_col = time_col
        self.target_col = target_col

    def run_backtest(
        self,
        df: pd.DataFrame,
        model_factory: Any,
        feature_cols: list[str],
    ) -> BacktestResult:
        """Execute time-series cross validation across historical data."""
        weeks = sorted(df[self.time_col].unique())
        total_weeks = len(weeks)

        required_weeks = self.min_train_weeks + (self.n_folds * self.test_window_weeks)
        if total_weeks < required_weeks:
            # Fall back to adaptive test window size if data is compact
            adaptive_window = max(2, (total_weeks - self.min_train_weeks) // self.n_folds)
            self.test_window_weeks = adaptive_window
            logger.warning(
                "Total weeks (%d) < required (%d). Adjusted test window to %d weeks.",
                total_weeks,
                required_weeks,
                self.test_window_weeks,
            )

        fold_evaluations: list[FoldEvaluation] = []
        all_preds_list: list[pd.DataFrame] = []

        max_week = max(weeks)
        for fold_i in range(self.n_folds):
            test_end_week = max_week - ((self.n_folds - 1 - fold_i) * self.test_window_weeks)
            test_start_week = test_end_week - self.test_window_weeks + 1
            train_end_week = test_start_week - 1
            train_start_week = (
                min(weeks)
                if self.expanding_window
                else max(min(weeks), train_end_week - self.min_train_weeks + 1)
            )

            train_mask = (df[self.time_col] >= train_start_week) & (
                df[self.time_col] <= train_end_week
            )
            test_mask = (df[self.time_col] >= test_start_week) & (
                df[self.time_col] <= test_end_week
            )

            train_df = df[train_mask].copy()
            test_df = df[test_mask].copy()

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            X_train, y_train = train_df[feature_cols], train_df[self.target_col]
            X_test, y_test = test_df[feature_cols], test_df[self.target_col]

            # Instantiate and fit fresh model per fold
            model: BaseForecaster = model_factory()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = compute_metrics(y_test.to_numpy(), preds)
            fold_eval = FoldEvaluation(
                fold_index=fold_i + 1,
                train_start_week=int(train_start_week),
                train_end_week=int(train_end_week),
                test_start_week=int(test_start_week),
                test_end_week=int(test_end_week),
                train_samples=len(train_df),
                test_samples=len(test_df),
                metrics=metrics,
            )
            fold_evaluations.append(fold_eval)

            pred_slice = test_df.copy()
            pred_slice["forecast_orders"] = preds
            pred_slice["fold_index"] = fold_i + 1
            all_preds_list.append(pred_slice)

        if not all_preds_list:
            raise ValueError("Backtest failed to produce valid evaluation folds.")

        combined_preds = pd.concat(all_preds_list, ignore_index=True)
        overall_metrics = compute_metrics(
            combined_preds[self.target_col].to_numpy(),
            combined_preds["forecast_orders"].to_numpy(),
        )

        sliced_perf = self._compute_sliced_performance(combined_preds)

        return BacktestResult(
            overall_metrics=overall_metrics,
            fold_evaluations=fold_evaluations,
            sliced_performance=sliced_perf,
            predictions_df=combined_preds,
        )

    def _compute_sliced_performance(self, preds_df: pd.DataFrame) -> list[SlicedPerformance]:
        """Compute performance breakdown across slices like cuisine, promo, and center type."""
        slices: list[SlicedPerformance] = []
        slice_candidates = [
            "center_type",
            "category",
            "cuisine",
            "emailer_for_promotion",
            "homepage_featured",
        ]

        for dim in slice_candidates:
            if dim in preds_df.columns:
                for val, group in preds_df.groupby(dim):
                    if len(group) >= 5:
                        m = compute_metrics(
                            group[self.target_col].to_numpy(),
                            group["forecast_orders"].to_numpy(),
                        )
                        slices.append(
                            SlicedPerformance(
                                dimension=dim,
                                slice_name=str(val),
                                sample_count=len(group),
                                wmape=m["WMAPE"],
                                rmsle=m["RMSLE"],
                                mae=m["MAE"],
                                rmse=m["RMSE"],
                            )
                        )
        return slices
