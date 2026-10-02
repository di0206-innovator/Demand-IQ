"""Tests for utility helpers and evaluation metrics."""

import numpy as np
import pytest

from demandiq.utils import calculate_rmsle, calculate_wmape, set_seed, setup_logger


def test_setup_logger():
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert len(logger.handlers) >= 1


def test_set_seed():
    set_seed(123)
    val1 = np.random.rand()
    set_seed(123)
    val2 = np.random.rand()
    assert val1 == val2


def test_calculate_wmape():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    # Absolute errors: 10 + 10 + 30 = 50. Total true: 600. WMAPE = 50/600 ~ 0.0833
    wmape = calculate_wmape(y_true, y_pred)
    assert pytest.approx(wmape, rel=1e-3) == 50.0 / 600.0


def test_calculate_wmape_zero_denominator():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([10.0, 10.0])
    assert calculate_wmape(y_true, y_pred) == 0.0


def test_calculate_rmsle():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert pytest.approx(calculate_rmsle(y_true, y_pred), abs=1e-6) == 0.0
