"""Feature engineering — pure functions, no I/O.

Imported by both training (offline batch) and serving (online inference)
to guarantee the same transforms apply on both sides.
"""

from pdm.features.rul import compute_rul

__all__ = ["compute_rul"]
