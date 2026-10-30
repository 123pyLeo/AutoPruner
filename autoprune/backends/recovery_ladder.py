"""
The unified RECOVERY LADDER -- one cost-ordered vocabulary for every model's recovery.
======================================================================================

Every VLA in AutoPrune recovers differently at the *training-script* level (that is
architecture-determined and intentional -- see each plugin's docstring), but they all
recover at one of the SAME four rungs of a cost-ordered ladder. This module is the
single source of truth for that vocabulary; each ``ModelPlugin`` declares which rungs
its architecture supports via ``recovery_ladder`` (tier id -> its native recipe id).

The four tiers, cheapest first:

    tier 0  direct            No training at all: prune, then eval. The SR floor.
                              Sufficient only at shallow depths.
    tier 1  module_finetune   Retrain ONE module of the pruned model on its native
                              task loss -- either a zero-init boundary adapter at the
                              cut, or the model's own action module (FastWAM action
                              expert, CogACT DiT head) when its architecture routes
                              actions around the cut so no adapter is needed.
    tier 2  joint_distill     Co-adapt several modules together (adapter + action
                              head + proprio / kept layers) while distilling from the
                              frozen dense teacher on offline demos. The verified
                              workhorse for deep pruning (OFT k=28 -> 96.5%).
    tier 3  onpolicy_distill  Continue tier 2 on the student's OWN rollout states
                              (DAgger) with a trust-region anchor. The only recipe
                              that breaks the ~0.80 frozen-backbone ceiling
                              (starVLA 0.788 -> 0.828).

A rung declared as unsupported (None) means one of THREE distinct things -- the
plugin comment must say which, because they carry different weight in the paper:
  (a) ARCHITECTURE-excluded: the rung makes no sense for this model (CogACT's DiT
      head reads the residual stream, so a boundary adapter has nothing to bridge);
  (b) EVIDENCE-excluded: the rung was tried and measurably fails (starVLA
      adapter-with-frozen-downstream -> SR 0.005, so there is no tier-1 rung);
  (c) NOT-YET-BUILT: nothing forbids it, the recipe just isn't implemented
      (FastWAM tiers 2-3 would need a dense-teacher distill path; unnecessary so
      far because tier 1 already reaches its sweet spot).

The menu is EVIDENCE-CURATED: recipes verified harmful or useless never enter it
(LoRA-only -> 0% SR, linear/SVD adapter -> 0%, frozen-downstream -> 0.005; see
``recovery_lib/negative_baselines.py``). The agent picks a rung from the declared
menu -- cheapest first, escalating on SR failure -- and never invents a recipe.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

#: canonical tier ids, cheapest first. Order IS the escalation order.
TIER_ORDER: Tuple[str, ...] = (
    "direct",
    "module_finetune",
    "joint_distill",
    "onpolicy_distill",
)

#: one-line description per tier (used in reports / agent context).
TIER_DESCRIPTIONS: Dict[str, str] = {
    "direct":           "no training: prune + closed-loop eval (the SR floor)",
    "module_finetune":  "retrain one module (boundary adapter OR the model's own "
                        "action module) on its native loss",
    "joint_distill":    "co-adapt adapter + action modules jointly, distilling from "
                        "the frozen dense teacher on offline demos",
    "onpolicy_distill": "continue joint distill on the student's own rollout states "
                        "with a trust-region anchor (breaks the ~0.80 ceiling)",
}


def tier_rank(tier: str) -> int:
    """Position of ``tier`` in the escalation order (raises if unknown)."""
    return TIER_ORDER.index(tier)


def validate_ladder(ladder: Dict[str, Optional[str]],
                    recovery_strategies: Tuple[str, ...]) -> None:
    """Sanity-check one plugin's declared ladder against its native recipe ids.

    * every key must be a canonical tier id;
    * every non-None value must be one of the plugin's ``recovery_strategies``;
    * every native recipe the plugin implements must appear on exactly one rung
      (so the two vocabularies stay in sync as recipes are added).
    Raises ``ValueError`` with a precise message on any violation.
    """
    for tier in ladder:
        if tier not in TIER_ORDER:
            raise ValueError(f"unknown recovery tier '{tier}' (valid: {TIER_ORDER})")
    natives = [v for v in ladder.values() if v is not None]
    if len(natives) != len(set(natives)):
        raise ValueError(f"a native recipe appears on more than one rung: {ladder}")
    for native in natives:
        if native not in recovery_strategies:
            raise ValueError(
                f"ladder maps to '{native}' which is not in recovery_strategies "
                f"{recovery_strategies}")
    unmapped = set(recovery_strategies) - set(natives)
    if unmapped:
        raise ValueError(
            f"native recipes {sorted(unmapped)} are not placed on any ladder rung")
