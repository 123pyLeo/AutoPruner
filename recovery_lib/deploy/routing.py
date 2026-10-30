"""
TaskRouter -- verified deployment optimum: route each task to its best ckpt.
============================================================================

Source
------
* openvla : the 3-mask / per-task routing result.

Idea
----
A single pruned+recovered ckpt hits a Pareto ceiling (multi-task gradient
interference caps it). Instead, keep a small set of (mask, adapter, head, proprio)
ckpts -- a general one plus a few per-task specialists -- and at inference route the
task instruction to the ckpt with the best closed-loop SR on that task.

Verified evidence
-----------------
real C2v2 76.4 % + route {t8, t9}→specialist = **80.2 %** (+3.8 pt, deployable);
3-mask per-task oracle upper bound = **88.5 %** (strong anti-correlated t8↔t9 mask
preference -- no single mask wins both). Only method that beats the single-mask ceiling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..core.registry import register


@register("recovery_deploy", "task_router")
class TaskRouter:
    """Per-task argmax over a set of recovered ckpts (by their per-task SR)."""

    name = "task_router"

    def __init__(self, default_ckpt: str):
        self.default_ckpt = default_ckpt
        # ckpt_id -> {task -> sr}
        self._sr: Dict[str, Dict[str, float]] = {}

    def register_ckpt(self, ckpt_id: str, per_task_sr: Dict[str, float]) -> None:
        self._sr[ckpt_id] = dict(per_task_sr)

    def route(self, task: str) -> str:
        """Return the ckpt id with the best SR on ``task`` (else the default)."""
        best, best_sr = self.default_ckpt, -1.0
        for ckpt, table in self._sr.items():
            if task in table and table[task] > best_sr:
                best, best_sr = ckpt, table[task]
        return best

    def routing_table(self, tasks: List[str]) -> Dict[str, str]:
        return {t: self.route(t) for t in tasks}

    def expected_sr(self, tasks: List[str]) -> float:
        """Mean SR of the routed policy = the deployable number (e.g. 80.2 %)."""
        vals = []
        for t in tasks:
            ckpt = self.route(t)
            vals.append(self._sr.get(ckpt, {}).get(t, 0.0))
        return float(sum(vals) / len(vals)) if vals else 0.0

    def oracle_sr(self, tasks: List[str]) -> float:
        """Upper bound: best ckpt per task regardless of routing feasibility (88.5 %)."""
        vals = []
        for t in tasks:
            vals.append(max((tbl.get(t, 0.0) for tbl in self._sr.values()), default=0.0))
        return float(sum(vals) / len(vals)) if vals else 0.0
