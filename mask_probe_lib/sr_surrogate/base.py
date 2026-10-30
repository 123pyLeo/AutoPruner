"""
Layer 3 -- SR surrogate (base interface).

An SR surrogate predicts the *closed-loop success rate* a candidate mask would
reach after recovery, from cheap forward-pass features -- so a 30-minute closed
loop eval can be skipped during search.

Contract::

    surrogate.fit(feature_table, sr_vector)     # calibrate on known-SR masks
    surrogate.predict(features)                  # SR for a new mask
    surrogate.cross_validate(...)                # honest LOO R² / Spearman

The honesty check (leave-one-out) is mandatory: the source repo's structural-OLS
surrogate looked fine in-sample (R²=0.84) but scored LOO R²=-0.84 -- worse than
predicting the mean.  The verified two-feature drift predictor scored LOO R²=0.71.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Sequence

import numpy as np

from ..core.backend import ModelBackend


class SRSurrogate(ABC):
    name: str = "base"

    @abstractmethod
    def features(self, backend: ModelBackend, mask: Sequence[int]) -> Dict[str, float]:
        """Cheap forward-pass features for one candidate mask."""

    @abstractmethod
    def fit(self, X: np.ndarray, sr: np.ndarray) -> "SRSurrogate":
        """Calibrate the surrogate on a table of known-SR masks."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicted SR for new feature rows."""
