"""Adaptive self-improving loop, dynamic bias correction, and Champion-Challenger orchestration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from demandiq.config import DEFAULT_CONFIG, AppConfig, ModelConfig
from demandiq.loop.feedback import FeedbackStore
from demandiq.loop.personalization import HyperPersonalizedProfiler
from demandiq.models import XGBoostDemandForecaster, compute_metrics
from demandiq.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class LoopUpdateResult:
    """Outcome of a self-improving loop iteration."""

    champion_wmape: float
    adaptive_wmape: float
    challenger_wmape: float | None
    promoted_new_champion: bool
    bias_correction_improvement_pct: float
    retrain_trigger_fired: bool
    trigger_reasons: list[str]


class AdaptiveSelfImprovingLoop:
    """Orchestrates continuous self-learning, adaptive residual calibration, and model promotion."""

    def __init__(
        self,
        champion_model: XGBoostDemandForecaster,
        feature_cols: list[str],
        config: AppConfig = DEFAULT_CONFIG,
        adaptation_rate: float = 0.65,
    ) -> None:
        self.champion_model = champion_model
        self.feature_cols = feature_cols
        self.config = config
        self.adaptation_rate = adaptation_rate
        self.profiler = HyperPersonalizedProfiler()
        self.feedback_store = FeedbackStore()
        self.iteration_count = 0

    def predict_adaptive(
        self,
        X_df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate point forecasts with hyper-personalized residual bias corrections and custom safety stocks."""
        if X_df.empty:
            return np.array([], dtype=float), np.array([], dtype=float)

        base_preds = self.champion_model.predict(X_df[self.feature_cols])
        n_rows = len(X_df)

        center_ids = (
            X_df["center_id"].to_numpy(dtype=int)
            if "center_id" in X_df.columns
            else np.zeros(n_rows, dtype=int)
        )
        meal_ids = (
            X_df["meal_id"].to_numpy(dtype=int)
            if "meal_id" in X_df.columns
            else np.zeros(n_rows, dtype=int)
        )

        adaptive_preds = np.empty(n_rows, dtype=float)
        custom_safety_stocks = np.empty(n_rows, dtype=float)

        for idx, (c_id, m_id, raw_pred) in enumerate(
            zip(center_ids, meal_ids, base_preds, strict=False)
        ):
            profile = self.profiler.get_profile(int(c_id), int(m_id))
            if profile is not None:
                corrected = raw_pred + (self.adaptation_rate * profile.recent_residual_bias)
                adaptive_preds[idx] = max(0.0, corrected)
                custom_safety_stocks[idx] = profile.recommended_safety_stock
            else:
                adaptive_preds[idx] = float(raw_pred)
                custom_safety_stocks[idx] = 15.0

        return adaptive_preds, custom_safety_stocks

    def process_incoming_batch(
        self,
        batch_df: pd.DataFrame,
        actual_col: str = "num_orders",
        force_retrain: bool = False,
    ) -> LoopUpdateResult:
        """Process newly realized weekly demand batch, update profiles, and evaluate Champion vs Challenger."""
        if batch_df.empty:
            return LoopUpdateResult(
                champion_wmape=0.0,
                adaptive_wmape=0.0,
                challenger_wmape=None,
                promoted_new_champion=False,
                bias_correction_improvement_pct=0.0,
                retrain_trigger_fired=False,
                trigger_reasons=["Empty batch provided."],
            )

        self.iteration_count += 1
        logger.info("Executing Self-Improving Loop (Iteration #%d)...", self.iteration_count)

        if actual_col not in batch_df.columns and "actual_orders" in batch_df.columns:
            actual_col = "actual_orders"

        y_actual = batch_df[actual_col].to_numpy()

        # 1. Evaluate Champion Point Forecasts
        base_preds = self.champion_model.predict(batch_df[self.feature_cols])
        champ_metrics = compute_metrics(y_actual, base_preds)

        # 2. Evaluate Adaptive Corrected Forecasts
        eval_batch = batch_df.copy()
        eval_batch["forecast_orders"] = base_preds
        self.profiler.fit_profiles(
            eval_batch, forecast_col="forecast_orders", actual_col=actual_col
        )

        adaptive_preds, _ = self.predict_adaptive(batch_df)
        adapt_metrics = compute_metrics(y_actual, adaptive_preds)

        # 3. Record Feedback
        eval_batch["forecast_orders"] = adaptive_preds
        eval_batch["actual_orders"] = y_actual
        self.feedback_store.record_batch_feedback(eval_batch)

        # Calculate improvement percentage
        imp_pct = 0.0
        if champ_metrics["WMAPE"] > 0:
            imp_pct = (
                (champ_metrics["WMAPE"] - adapt_metrics["WMAPE"]) / champ_metrics["WMAPE"]
            ) * 100.0

        # 4. Trigger Assessment
        trigger_reasons: list[str] = []
        if champ_metrics["WMAPE"] > 0.25:
            trigger_reasons.append(f"WMAPE exceeded tolerance: {champ_metrics['WMAPE']:.4f} > 0.25")
        if force_retrain:
            trigger_reasons.append("Manual force retrain requested.")

        promoted = False
        challenger_wmape = None

        # 5. Challenger Exploration & Auto-Tuning
        if trigger_reasons or force_retrain:
            logger.info(
                "Retrain trigger fired (%s). Training candidate Challenger model...",
                trigger_reasons,
            )
            challenger_config = ModelConfig(
                n_estimators=min(800, self.config.model.n_estimators + 100),
                learning_rate=max(0.01, self.config.model.learning_rate * 0.9),
                max_depth=self.config.model.max_depth,
                random_state=42 + self.iteration_count,
            )
            challenger = XGBoostDemandForecaster(model_config=challenger_config)

            # Fit challenger on the updated batch
            X_b = batch_df[self.feature_cols]
            y_b = batch_df[actual_col]
            challenger.fit(X_b, y_b)

            chal_preds = challenger.predict(X_b)
            chal_metrics = compute_metrics(y_actual, chal_preds)
            challenger_wmape = chal_metrics["WMAPE"]

            if challenger_wmape < champ_metrics["WMAPE"]:
                logger.info(
                    "Challenger outperformed Champion (WMAPE %.4f vs %.4f). Promoting Challenger to Champion!",
                    challenger_wmape,
                    champ_metrics["WMAPE"],
                )
                self.champion_model = challenger
                promoted = True

        return LoopUpdateResult(
            champion_wmape=champ_metrics["WMAPE"],
            adaptive_wmape=adapt_metrics["WMAPE"],
            challenger_wmape=challenger_wmape,
            promoted_new_champion=promoted,
            bias_correction_improvement_pct=round(imp_pct, 2),
            retrain_trigger_fired=len(trigger_reasons) > 0,
            trigger_reasons=trigger_reasons,
        )
