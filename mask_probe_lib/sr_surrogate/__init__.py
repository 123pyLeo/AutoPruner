"""
Layer 3 -- SR surrogate.

Predict closed-loop SR for a candidate mask from cheap forward-pass features, so
expensive closed-loop evals can be skipped during search.

Verified method (see ``METHODS.md``):
    drift_predictor -- DriftPredictor (full_l2_rc + out_var_pr; LOO R²=0.71)
"""

from .base import SRSurrogate
from .drift_predictor import DriftPredictor, DEFAULT_FEATURES, TRAINING_FREE_FEATURES

__all__ = ["SRSurrogate", "DriftPredictor", "DEFAULT_FEATURES", "TRAINING_FREE_FEATURES"]
