"""Regression metrics + the C-MAPSS asymmetric scoring function.

C-MAPSS scoring (Saxena & Goebel 2008): asymmetric penalty that punishes late
predictions (overshoot, predicting more remaining life than there is) more
heavily than early predictions (undershoot). This matches maintenance
operations: late predictions risk in-service failures; early predictions
just waste a part.

    d = y_pred - y_true
    s = sum(  exp(-d/13) - 1   for d < 0     # early
            + exp( d/10) - 1   for d >= 0)   # late
"""

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    score = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0,
    )
    return float(np.sum(score))
