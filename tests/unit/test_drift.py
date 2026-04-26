"""Tests for pdm.monitoring.drift — Population Stability Index."""

import numpy as np
import pandas as pd

from pdm.monitoring.drift import compute_psi, compute_psi_per_column


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=10_000)
    b = a.copy()
    psi = compute_psi(a, b, bins=10)
    assert psi == 0.0


def test_psi_small_for_similar_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=10_000)
    b = rng.normal(0, 1, size=10_000)
    psi = compute_psi(a, b, bins=10)
    assert 0.0 <= psi < 0.1  # < 0.1 = "no significant change"


def test_psi_large_for_very_different_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=10_000)
    b = rng.normal(5, 1, size=10_000)  # mean shifted by 5 sigmas
    psi = compute_psi(a, b, bins=10)
    assert psi > 0.25  # > 0.25 = "significant change"


def test_psi_handles_empty_compare_returns_nan():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([])
    psi = compute_psi(a, b, bins=5)
    assert np.isnan(psi)


def test_psi_handles_constant_baseline():
    a = np.full(100, 5.0)
    b = np.full(100, 5.0)
    # All-same baseline → bin edges collapse; defined as 0 by convention
    psi = compute_psi(a, b, bins=10)
    assert psi == 0.0


def test_compute_psi_per_column():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({
        "x": rng.normal(0, 1, 1000),
        "y": rng.normal(0, 1, 1000),
    })
    cmp = pd.DataFrame({
        "x": rng.normal(0, 1, 1000),       # similar
        "y": rng.normal(5, 1, 1000),       # shifted
    })
    psi = compute_psi_per_column(base, cmp, columns=["x", "y"])
    assert "x" in psi and "y" in psi
    assert psi["x"] < 0.1
    assert psi["y"] > 0.25
