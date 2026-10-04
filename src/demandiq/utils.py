"""Utility functions for DemandIQ: logging, seed management, and metrics."""

import logging
import random
import sys
from pathlib import Path

import numpy as np


def setup_logger(
    name: str = "demandiq",
    level: int = logging.INFO,
    log_format: str | None = None,
) -> logging.Logger:
    """Configure and return a standardized logger."""
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    """Set random seeds across random and numpy for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def calculate_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Weighted Mean Absolute Percentage Error (WMAPE).

    WMAPE = sum(|y_true - y_pred|) / sum(|y_true|)
    Robust against empty arrays, NaNs, and zero denominators.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if y_t.size == 0 or y_p.size == 0:
        return 0.0

    valid_mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.all(valid_mask):
        y_t = y_t[valid_mask]
        y_p = y_p[valid_mask]
        if y_t.size == 0:
            return 0.0

    denominator = float(np.sum(np.abs(y_t)))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(np.sum(np.abs(y_t - y_p)) / denominator)


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Logarithmic Error (RMSLE).

    RMSLE = sqrt(mean((log1p(y_pred) - log1p(y_true))^2))
    Robust against empty arrays, negative values, and non-finite numbers.
    """
    y_t = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_p = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    if y_t.size == 0 or y_p.size == 0:
        return 0.0

    valid_mask = np.isfinite(y_t) & np.isfinite(y_p)
    if not np.all(valid_mask):
        y_t = y_t[valid_mask]
        y_p = y_p[valid_mask]
        if y_t.size == 0:
            return 0.0

    diff = np.log1p(y_p) - np.log1p(y_t)
    mean_sq = float(np.mean(diff**2))
    if not np.isfinite(mean_sq) or mean_sq < 0.0:
        return 0.0
    return float(np.sqrt(mean_sq))


def validate_safe_path(
    filepath: Path | str,
    allowed_base: Path | str | None = None,
    must_exist: bool = False,
    max_size_bytes: int | None = 500 * 1024 * 1024,  # Default 500 MB limit
) -> Path:
    """Validate filesystem path against directory traversal and size limits for security.

    Args:
        filepath: Input path to validate.
        allowed_base: Optional base directory that filepath must resolve within.
        must_exist: If True, raises FileNotFoundError if path does not exist.
        max_size_bytes: Maximum permitted file size in bytes if file exists (guards against DoS).

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If path traversal or size limit is violated.
        FileNotFoundError: If must_exist is True and file is missing.
    """
    resolved = Path(filepath).resolve()

    if allowed_base is not None:
        base_resolved = Path(allowed_base).resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(
                f"Security violation: path traversal detected for '{filepath}'. "
                f"Resolved target '{resolved}' is outside allowed directory '{base_resolved}'."
            ) from None

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    if resolved.exists() and resolved.is_file() and max_size_bytes is not None:
        file_size = resolved.stat().st_size
        if file_size > max_size_bytes:
            raise ValueError(
                f"Security violation: file '{resolved}' size ({file_size} bytes) "
                f"exceeds maximum allowed threshold ({max_size_bytes} bytes)."
            )

    return resolved
