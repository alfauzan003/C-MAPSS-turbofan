"""Feature engineering — pure functions, no I/O.

Imported by both training (offline batch) and serving (online inference)
to guarantee the same transforms apply on both sides.
"""

from pdm.features.rul import compute_rul

try:
    from pdm.features.windows import compute_windows
except ImportError:
    pass

__all__ = ["compute_rul", "compute_windows"]
