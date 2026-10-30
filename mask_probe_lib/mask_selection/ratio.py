"""
PruneRatioSweep -- choose HOW MANY layers to remove (the dominant single lever).
================================================================================

The budget (prune ratio / k) is the single most load-bearing decision for the
recovery ceiling -- but it is **per-model and must be DISCOVERED by real SR, not
assumed from a cross-model prior**.

Cross-model seed (a STARTING point, NOT a truth)
------------------------------------------------
* starVLA prune-ratio sweep 14 % / 28 % / 44 % / 75 %:
    14 %  -> high SR but little compression
    28 %  -> sweet spot there (k10 of 36 layers; recovered 0.788 -> 0.828)
    44 %  -> 0.236 .. 0.33  (over the recoverability cliff)
    75 %  -> ~0.000         (collapse)

**Counter-example that broke the 28 % prior (FastWAM, 30-layer coupled MoT):**
    20 % (k=6, [2-7])   -> 91 %     <- the real sweet spot
    27 % (k=8, [12-19]) -> 68.5 %   <- the cross-model 28 % prior lands HERE, ~22 pts worse
    53 % (k=16)         -> 38 %
So FastWAM's sweet spot is **20 %, not 28 %**. A pipeline that seeds at 28 % and
only searches UPWARD (28->44->75) never finds it -- it must search **DOWNWARD too**.

Rule (locked)
-------------
1. ``SWEET_SPOT_SEED`` (0.28) is only the first probe point; treat it as a HYPOTHESIS.
2. Sweep BIDIRECTIONALLY -- include ratios *below* the seed (down to ~0.10).
3. ``discover_sweet_spot(sr_by_ratio, ...)`` picks the DEEPEST ratio whose measured
   closed-loop SR still clears the recovery floor. That measured ratio -- not 0.28 --
   is the model's sweet spot. Closed-loop SR is the only arbiter (warning W5).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..core.registry import register


@register("mask_selection", "ratio")
class PruneRatioSweep:
    """Map prune ratios to layer counts and DISCOVER the per-model sweet spot."""

    name = "ratio"
    #: cross-model SEED only (StarVLA/OFT) -- a hypothesis to confirm per-model, NOT a truth.
    #: FastWAM's real sweet spot was 0.20, so never hard-code this as the budget.
    SWEET_SPOT_SEED = 0.28
    #: kept as an alias for backward-compat; do NOT read this as "the sweet spot".
    SWEET_SPOT = SWEET_SPOT_SEED
    CLIFF_ABOVE = 0.44

    def __init__(self, n_layers: int):
        self.n_layers = n_layers

    def k_for_ratio(self, ratio: float) -> int:
        return max(1, int(round(ratio * self.n_layers)))

    def ratio_for_k(self, k: int) -> float:
        return k / self.n_layers

    def seed_k(self) -> int:
        """The first budget to PROBE (the cross-model seed), not the answer."""
        return self.k_for_ratio(self.SWEET_SPOT_SEED)

    # backward-compat: old callers expect a single starting k from the seed.
    def sweet_spot_k(self) -> int:
        return self.seed_k()

    def sweep(self, ratios: Optional[List[float]] = None) -> List[dict]:
        """BIDIRECTIONAL ratio sweep (default explores BELOW the seed too, down to 0.10).

        Returns one row per ratio; ``below_seed`` flags the down-search points the old
        upward-only sweep would have skipped (and where FastWAM's real sweet spot lived).
        """
        if ratios is None:
            ratios = [0.10, 0.14, 0.20, 0.28, 0.40, 0.53, 0.75]
        out = []
        for r in ratios:
            out.append({
                "ratio": r,
                "k": self.k_for_ratio(r),
                "below_seed": r < self.SWEET_SPOT_SEED,
                "past_cliff": r > self.CLIFF_ABOVE,
                "is_seed": abs(r - self.SWEET_SPOT_SEED) < 1e-6,
            })
        return out

    def discover_sweet_spot(self, sr_by_ratio: Dict[float, float],
                            recover_floor: float = 0.85, dense_sr: Optional[float] = None
                            ) -> dict:
        """Pick the model's REAL sweet spot from MEASURED closed-loop SR per ratio.

        Parameters
        ----------
        sr_by_ratio : {prune_ratio -> measured recovered SR}
            From real prune+recover+eval at each swept ratio (not a probe surrogate).
        recover_floor : absolute SR (or, if ``dense_sr`` given, fraction-of-dense)
            below which a ratio is "not near-full recovery".
        dense_sr : if given, ``recover_floor`` is interpreted as a fraction of it
            (e.g. 0.90 -> require >= 0.90*dense_sr).

        Returns the DEEPEST ratio that still clears the floor -- the largest
        compression with near-full recovery -- plus the full sorted evidence.
        """
        floor = recover_floor * dense_sr if dense_sr else recover_floor
        rows = sorted(sr_by_ratio.items())  # by ratio ascending
        passing = [(r, sr) for r, sr in rows if sr is not None and sr >= floor]
        best_ratio = max((r for r, _ in passing), default=None)
        return {
            "sweet_spot_ratio": best_ratio,
            "sweet_spot_k": self.k_for_ratio(best_ratio) if best_ratio is not None else None,
            "floor": floor,
            "passing": passing,
            "evidence": rows,
            "note": ("DISCOVERED from measured SR (not the 0.28 seed)."
                     if best_ratio is not None else
                     "no swept ratio cleared the floor; widen the sweep DOWNWARD."),
        }
