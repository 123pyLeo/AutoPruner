"""
QualityDiversityArchive -- elitist champion mask per behavioural cell.
======================================================================

Source
------
* openvla : ``agentic_pruning/modules/mask_search.py`` + ``modules/grid.py``.

Idea
----
During mask search, keep the highest-SR mask in each cell of a 2-D behavioural
grid -- axes = ``(k_bin, centroid_bin)`` where ``k = len(mask)`` and
``centroid = mean(removed layer indices)``.  This preserves a *diverse* set of
good masks (small/large, early/mid/late) instead of collapsing to one, which is
what makes the agentic search explore the trade-off frontier rather than over-fit
a single region.

A mask enters its cell only if the cell is empty or its SR beats the incumbent
(elitism).  ``get_map()`` returns the per-cell champions = the trade-off map.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..core.registry import register


def mask_descriptors(mask: Sequence[int]) -> Tuple[int, float]:
    """Behavioural descriptors of a remove-mask: (k, centroid)."""
    m = list(mask)
    k = len(m)
    centroid = (sum(m) / k) if k else 0.0
    return k, centroid


def _bin(value: float, lo: float, hi: float, n_bins: int) -> int:
    if hi <= lo:
        return 0
    frac = (value - lo) / (hi - lo)
    return max(0, min(n_bins - 1, int(frac * n_bins)))


@register("mask_selection", "qd_archive")
class QualityDiversityArchive:
    """Per-(k_bin, centroid_bin) elitist archive of champion masks."""

    name = "qd_archive"

    def __init__(self, n_layers: int, k_bins: int = 5, centroid_bins: int = 5):
        self.n_layers = n_layers
        self.k_bins = k_bins
        self.centroid_bins = centroid_bins
        self._archive: Dict[Tuple[int, int], dict] = {}

    def cell_of(self, mask: Sequence[int]) -> Tuple[int, int]:
        k, centroid = mask_descriptors(mask)
        kb = _bin(k, 1, self.n_layers, self.k_bins)
        cb = _bin(centroid, 0, self.n_layers - 1, self.centroid_bins)
        return kb, cb

    def add(self, mask: Sequence[int], sr: Optional[float], iter_idx: int = -1) -> bool:
        """Insert (mask, sr); returns True if it became its cell's champion."""
        if sr is None or len(mask) == 0:
            return False
        cell = self.cell_of(mask)
        cur = self._archive.get(cell)
        if cur is None or sr > cur["sr"]:
            self._archive[cell] = {"mask": tuple(sorted(mask)), "sr": float(sr),
                                   "cell": cell, "iter": iter_idx}
            return True
        return False

    def get_map(self) -> List[dict]:
        return sorted(self._archive.values(), key=lambda r: r["sr"], reverse=True)

    def best(self) -> Optional[dict]:
        m = self.get_map()
        return m[0] if m else None
