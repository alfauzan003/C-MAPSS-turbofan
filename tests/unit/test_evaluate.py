"""Tests for pdm.models.evaluate — RMSE, MAE, C-MAPSS scoring."""

import math

import numpy as np

from pdm.models.evaluate import cmapss_score, mae, rmse


def test_rmse_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert math.isclose(rmse(y_true, y_pred), math.sqrt(1.0 / 3))


def test_mae_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 2.0, 5.0])
    assert mae(y_true, y_pred) == (1 + 0 + 2) / 3


def test_cmapss_score_zero_when_perfect():
    y_true = np.array([10.0, 20.0])
    assert cmapss_score(y_true, y_true) == 0.0


def test_cmapss_score_late_predictions_penalized_more_than_early():
    """C-MAPSS asymmetric penalty: predicting LATE (overshoot) is worse than early."""
    y_true = np.array([10.0])
    early = cmapss_score(y_true, np.array([5.0]))   # predicted -5
    late = cmapss_score(y_true, np.array([15.0]))   # predicted +5
    assert late > early > 0
