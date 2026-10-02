"""Utility functions for DemandIQ: logging, seed management, and metrics."""

import logging
import random
import sys

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
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / denominator)


def calculate_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Logarithmic Error (RMSLE).

    RMSLE = sqrt(mean((log1p(y_pred) - log1p(y_true))^2))
    """
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))
