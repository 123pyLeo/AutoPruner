"""Mask behavioral descriptors + grid binning (shared by exploration_bias + QD archive).

A pruning mask is summarized by two cheap, intuitive axes:
  - k        = number of removed layers (len(mask))
  - centroid = mean index of removed layers (early vs late in the stack)

These two axes tile the mask space into a coarse grid of cells. Two masks landing
in the SAME cell are "in the same region". Consumers:
  - exploration_bias diagnosis  → detect over-clustering of recent proposals in one cell
  - QD archive (mask_search)     → keep the best mask seen PER cell, not just one global champion

NOTE (current task = OFT/libero_object): masks are contiguous blocks [a..b], so
(k, centroid) ↔ (a, b) is a faithful, lossless description. When jump-layer
(non-contiguous) masks are introduced later (see ledger related_work_mining_2026-05-30
§9 — StarVLA future direction), centroid alone loses info; add a dispersion/spread
axis then. Kept deliberately small (5×5) because closed-loop SR is far costlier
than the CIFAR-scale searches these ideas came from.
"""
from collections import Counter

N_LAYERS = 32
K_BINS = 5
CENTROID_BINS = 5

_K_TAGS = ["k:tiny", "k:small", "k:mid", "k:large", "k:huge"]
_C_TAGS = ["pos:early", "pos:early-mid", "pos:mid", "pos:mid-late", "pos:late"]


def mask_descriptors(mask, n_layers: int = N_LAYERS):
    """Return (k, centroid) for a mask. centroid=0.0 for empty mask."""
    m = sorted(int(x) for x in mask)
    k = len(m)
    centroid = (sum(m) / k) if k else 0.0
    return k, centroid


def _bin(value: float, lo: float, hi: float, n_bins: int) -> int:
    """Map value∈[lo,hi] to a bin index in [0, n_bins). Clamps out-of-range."""
    if hi <= lo:
        return 0
    if value <= lo:
        return 0
    if value >= hi:
        return n_bins - 1
    return min(int((value - lo) / (hi - lo) * n_bins), n_bins - 1)


def cell_of(mask, n_layers: int = N_LAYERS):
    """Map a mask to its (k_bin, centroid_bin) grid cell."""
    k, centroid = mask_descriptors(mask, n_layers)
    if k == 0:
        return (0, 0)
    k_bin = _bin(k, 1, n_layers, K_BINS)
    c_bin = _bin(centroid, 0, n_layers - 1, CENTROID_BINS)
    return (k_bin, c_bin)


def cell_label(cell) -> str:
    """Human-readable region tag, e.g. 'k:large/pos:mid'."""
    k_bin, c_bin = cell
    return f"{_K_TAGS[k_bin]}/{_C_TAGS[c_bin]}"


def concentration(masks, window: int):
    """Over the last `window` masks, the max fraction landing in one cell.

    Returns (frac, top_cell, n_recent). frac∈[0,1]; top_cell is the most-sampled
    cell among the recent window; n_recent = how many masks the window actually had.
    Used by exploration_bias: high frac = the agent is orbiting one region.
    """
    recent = [m for m in list(masks)[-window:] if m]
    if not recent:
        return 0.0, None, 0
    cells = [cell_of(m) for m in recent]
    top_cell, top_n = Counter(cells).most_common(1)[0]
    return top_n / len(recent), top_cell, len(recent)
