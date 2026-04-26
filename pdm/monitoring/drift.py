"""Population Stability Index (PSI) — distribution-shift metric.

PSI is a per-feature scalar:
    PSI = sum( (p_compare - p_baseline) * ln(p_compare / p_baseline) )
where p_* are the bin-fractions of compare/baseline distributions over the
same bin edges (computed from baseline quantiles).

Conventional thresholds:
    PSI < 0.1   no significant change
    0.1 - 0.25  moderate shift
    PSI > 0.25  significant shift — investigate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-10  # avoid log(0) and divide-by-zero in empty bins


def compute_psi(baseline: np.ndarray, compare: np.ndarray, bins: int = 10) -> float:
    """Per-column PSI between two 1-D arrays.

    Returns 0.0 if baseline is constant (no bins to construct).
    Returns NaN if compare is empty.
    """
    baseline = np.asarray(baseline, dtype=float)
    compare = np.asarray(compare, dtype=float)
    baseline = baseline[~np.isnan(baseline)]
    compare = compare[~np.isnan(compare)]

    if len(compare) == 0:
        return float("nan")
    if len(baseline) == 0:
        return float("nan")
    if baseline.min() == baseline.max():
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 2:
        return 0.0
    # Make outer edges infinite so all of `compare` lands in some bin
    edges[0] = -np.inf
    edges[-1] = np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    cmp_counts, _ = np.histogram(compare, bins=edges)

    p_base = base_counts / max(base_counts.sum(), 1)
    p_cmp = cmp_counts / max(cmp_counts.sum(), 1)
    p_base = np.where(p_base == 0, _EPS, p_base)
    p_cmp = np.where(p_cmp == 0, _EPS, p_cmp)

    psi = float(np.sum((p_cmp - p_base) * np.log(p_cmp / p_base)))
    return psi


def compute_psi_per_column(
    baseline: pd.DataFrame,
    compare: pd.DataFrame,
    columns: list[str],
    bins: int = 10,
) -> dict[str, float]:
    """Map of column → PSI for the named columns. Skips columns missing in either side."""
    out: dict[str, float] = {}
    for col in columns:
        if col not in baseline.columns or col not in compare.columns:
            continue
        out[col] = compute_psi(baseline[col].to_numpy(), compare[col].to_numpy(), bins=bins)
    return out
