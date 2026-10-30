"""
decide_pruning_mode -- contiguous vs skip, from the knockout anchor geometry.
=============================================================================

Implements the rule the framework locked in (ARCHITECTURE.md §4), distilled from
the OpenVLA-OFT + StarVLA experiments:

    1. Run the anchor-knockout probe -> per-layer dAll (= mean |Δ action| when the
       layer is identity-skipped). High dAll = load-bearing ANCHOR.
    2. Look at WHERE the anchors sit:
         * anchors clustered at the two ENDS, interior one tolerant band -> CONTIGUOUS
         * anchors SCATTERED through the interior                        -> SKIP
    3. Why: when anchors are scattered, any contiguous block spans an anchor and the
       model dies. Verified on StarVLA: any contiguous 10-block -> SR 0; the same 10
       layers scattered -> SR 78.8%. It's the GEOMETRY, not the prune ratio.

Hybrid use (locked): the plugin declares a default mode; the Agent calls this on the
live knockout scores to CONFIRM or OVERRIDE it. With no scores we fall back to the
plugin default (or "contiguous" as the safe baseline).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _edge_width(n_layers: int) -> int:
    """How many layers at each end count as 'the ends' (anchors there don't split
    the deletable region). ~12% of depth, at least 2."""
    return max(2, round(0.12 * n_layers))


def decide_pruning_mode(
    knockout_scores: Optional[Dict[int, float]],
    n_layers: int,
    *,
    default: Optional[str] = None,
    z_threshold: float = 1.0,
) -> dict:
    """Decide "contiguous" vs "skip" from per-layer knockout scores.

    Parameters
    ----------
    knockout_scores : {layer -> dAll} or None
        Per-layer identity-skip action perturbation. None/empty -> fall back to
        ``default`` (or "contiguous").
    n_layers : int
    default : the plugin's declared default mode (used as fallback / disagreement flag).
    z_threshold : a layer is an ANCHOR if its score's z (vs all layers) exceeds this.

    Returns
    -------
    dict with: mode, anchors, interior_anchors, agreed_with_default, reason.
    """
    if not knockout_scores:
        mode = default or "contiguous"
        return {"mode": mode, "anchors": [], "interior_anchors": [],
                "agreed_with_default": True,
                "reason": f"no knockout scores; fell back to default '{mode}'."}

    layers = sorted(knockout_scores)
    vals = np.array([float(knockout_scores[l]) for l in layers], dtype=float)
    mu, sd = float(vals.mean()), float(vals.std()) or 1e-9
    z = (vals - mu) / sd

    anchors = [layers[i] for i in range(len(layers)) if z[i] > z_threshold]
    edge = _edge_width(n_layers)
    interior_anchors = [a for a in anchors if edge <= a < (n_layers - edge)]

    if interior_anchors:
        mode = "skip"
        reason = (f"{len(interior_anchors)} interior anchor(s) {interior_anchors} "
                  f"(z>{z_threshold}) scattered inside [{edge},{n_layers - edge}) — a "
                  f"contiguous block would span them and collapse SR → SKIP.")
    else:
        mode = "contiguous"
        reason = (f"anchors {anchors} cluster at the ends (none in "
                  f"[{edge},{n_layers - edge})) → interior is one removable band → CONTIGUOUS.")

    agreed = (default is None) or (default == mode)
    if not agreed:
        reason += f"  [NOTE] overrides plugin default '{default}' based on live geometry."
    return {"mode": mode, "anchors": anchors, "interior_anchors": interior_anchors,
            "agreed_with_default": agreed, "reason": reason}


def split_into_runs(layers: List[int]) -> List[List[int]]:
    """Split a sorted layer list into maximal contiguous runs. A contiguous mask is
    ONE run; a skip mask is several. (Used by the recovery side to find断口 boundaries.)"""
    runs: List[List[int]] = []
    for l in sorted(layers):
        if runs and l == runs[-1][-1] + 1:
            runs[-1].append(l)
        else:
            runs.append([l])
    return runs
