"""
Layer 2 -- Mask-recoverability probes (base interface).

A recoverability probe answers a different question from layer 1: given a *whole
candidate mask*, **can a small adapter restore the deleted block** so the pruned
model recovers to near-dense SR?  This reframes pruning as "recoverability, not
redundancy" -- the central verified insight of openvla-oft.

Contract::

    score = probe.score_mask(backend, mask)
    score.value        # higher == more recoverable
    score.recoverable  # bool decision under the probe's threshold

IMPORTANT shared caveat (verified in both repos): the recoverability->SR map is a
non-linear **cliff**, so these scores *rank* candidate masks and gate obviously
unrecoverable ones -- they do not predict absolute SR.  Final arbiter is always
``backend.closed_loop_sr`` (warning W5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..core.backend import ModelBackend
from ..core.types import MaskScore


class MaskRecoverabilityProbe(ABC):
    name: str = "base"

    @abstractmethod
    def score_mask(self, backend: ModelBackend, mask: Sequence[int]) -> MaskScore:
        """Score how recoverable a candidate pruning ``mask`` is."""
