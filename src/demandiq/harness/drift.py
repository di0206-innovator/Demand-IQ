"""Data drift, concept drift, and residual variance monitoring harness."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from demandiq.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class FeatureDriftScore:
    """Drift metrics for an individual feature."""

    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    is_drifted: bool
    drift_severity: Literal["NONE", "LOW", "HIGH"]


@dataclass
class DriftReport:
    """Overall dataset drift assessment report."""

    overall_status: Literal["HEALTHY", "MODERATE_DRIFT", "CRITICAL_DRIFT"]
    drifted_feature_count: int
    total_features_evaluated: int
    target_drift_p_value: float | None
    residual_variance_ratio: float | None
    feature_drift_scores: list[FeatureDriftScore]

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "drifted_feature_count": self.drifted_feature_count,
            "total_features_evaluated": self.total_features_evaluated,
            "target_drift_p_value": self.target_drift_p_value,
            "residual_variance_ratio": self.residual_variance_ratio,
            "features": [
                {
                    "name": f.feature_name,
                    "psi": round(f.psi_score, 4),
                    "ks_stat": round(f.ks_statistic, 4),
                    "p_value": round(f.ks_p_value, 4),
                    "is_drifted": f.is_drifted,
                    "severity": f.drift_severity,
                }
                for f in self.feature_drift_scores
            ],
        }


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Calculate Population Stability Index (PSI) between baseline and production arrays.

    Robust against non-finite values (NaN, Inf, -Inf) and degenerate single-value distributions.
    """
    exp_arr = np.asarray(expected, dtype=float)
    act_arr = np.asarray(actual, dtype=float)

    exp_clean = exp_arr[np.isfinite(exp_arr)]
    act_clean = act_arr[np.isfinite(act_arr)]

    if len(exp_clean) == 0 or len(act_clean) == 0:
        return 0.0

    # Determine quantile bins from baseline distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    try:
        bin_edges = np.percentile(exp_clean, percentiles)
    except Exception:
        return 0.0

    # Ensure strictly increasing bin edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(exp_clean, bins=bin_edges)
    act_counts, _ = np.histogram(act_clean, bins=bin_edges)

    exp_pct = np.maximum(exp_counts / len(exp_clean), 1e-4)
    act_pct = np.maximum(act_counts / len(act_clean), 1e-4)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(np.clip(psi_val, 0.0, 10.0))


class DataDriftMonitor:
    """Continuous statistical monitoring harness for feature and concept drift."""

    def __init__(
        self,
        psi_threshold: float = 0.20,
        ks_alpha: float = 0.01,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha

    def evaluate_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str | None = "num_orders",
        baseline_residuals: np.ndarray | None = None,
        current_residuals: np.ndarray | None = None,
    ) -> DriftReport:
        """Evaluate distribution shifts across features, target, and prediction residuals."""
        feature_scores: list[FeatureDriftScore] = []
        drifted_count = 0

        for feat in feature_cols:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue

            exp_vals = baseline_df[feat].to_numpy(dtype=float, na_value=np.nan)
            act_vals = current_df[feat].to_numpy(dtype=float, na_value=np.nan)

            psi = calculate_psi(exp_vals, act_vals)

            # Two-sample Kolmogorov-Smirnov test with non-finite filtering
            exp_clean = exp_vals[np.isfinite(exp_vals)]
            act_clean = act_vals[np.isfinite(act_vals)]
            if len(exp_clean) > 5 and len(act_clean) > 5:
                # Handle zero-variance constant distributions gracefully
                if np.std(exp_clean) == 0.0 and np.std(act_clean) == 0.0:
                    ks_stat = 0.0 if exp_clean[0] == act_clean[0] else 1.0
                    ks_p = 1.0 if exp_clean[0] == act_clean[0] else 0.0
                else:
                    ks_res = stats.ks_2samp(exp_clean, act_clean)
                    ks_stat, ks_p = float(ks_res.statistic), float(ks_res.pvalue)
            else:
                ks_stat, ks_p = 0.0, 1.0

            is_drifted = (psi >= self.psi_threshold) or (ks_p < self.ks_alpha and ks_stat > 0.15)
            if is_drifted:
                drifted_count += 1
                severity = "HIGH" if psi >= 0.25 else "LOW"
            else:
                severity = "NONE"

            feature_scores.append(
                FeatureDriftScore(
                    feature_name=feat,
                    psi_score=psi,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p,
                    is_drifted=is_drifted,
                    drift_severity=severity,
                )
            )

        # Target distribution drift test
        target_p = None
        if target_col and target_col in baseline_df.columns and target_col in current_df.columns:
            b_target = baseline_df[target_col].dropna().to_numpy()
            c_target = current_df[target_col].dropna().to_numpy()
            if len(b_target) > 5 and len(c_target) > 5:
                target_p = float(stats.ks_2samp(b_target, c_target).pvalue)

        # Residual variance ratio (variance of current / variance of baseline)
        var_ratio = None
        if baseline_residuals is not None and current_residuals is not None:
            b_var = np.var(baseline_residuals)
            c_var = np.var(current_residuals)
            if b_var > 0:
                var_ratio = float(c_var / b_var)

        # Determine overall dataset status
        drift_ratio = drifted_count / max(1, len(feature_scores))
        if drift_ratio > 0.35 or (var_ratio and var_ratio > 1.8):
            status = "CRITICAL_DRIFT"
        elif drift_ratio > 0.15 or (target_p is not None and target_p < 0.01):
            status = "MODERATE_DRIFT"
        else:
            status = "HEALTHY"

        return DriftReport(
            overall_status=status,
            drifted_feature_count=drifted_count,
            total_features_evaluated=len(feature_scores),
            target_drift_p_value=target_p,
            residual_variance_ratio=var_ratio,
            feature_drift_scores=feature_scores,
        )
