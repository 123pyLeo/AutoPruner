"""DiagnosisModule — pattern matching on results → triggered rules.

After each iteration's eval lookup, driver calls diagnose() with:
  - the result (sr, mask, recovery)
  - the history of prior results

Returns a list of triggered patterns (each with an id + suggested rule text).
Agent in next iteration sees these and decides which to accept as persistent
rules.
"""
import json
from collections import Counter
from pathlib import Path

from .grid import cell_of, cell_label, concentration, K_BINS, CENTROID_BINS


class DiagnosisModule:
    def __init__(self, data_path: Path, dense_sr: float,
                  layer_static_facts: list[dict] | None = None,
                  n_layers: int = 32):
        self.data_path = Path(data_path)
        self.dense_sr = dense_sr
        self.n_layers = n_layers
        # Build {layer_idx: training_free_sr} lookup if static facts provided.
        # Used by inverted_anchor_priority pattern to spot mask choices that
        # contradict the per-layer training-free SR evidence.
        self.training_free_sr_by_layer: dict[int, float] = {}
        if layer_static_facts:
            for L in layer_static_facts:
                idx = L.get("idx")
                sr = L.get("single_delete_training_free_sr")
                if idx is not None and sr is not None:
                    self.training_free_sr_by_layer[int(idx)] = float(sr)
        with open(self.data_path) as f:
            self._data = json.load(f)

    def list_patterns(self) -> list[dict]:
        return list(self._data["patterns"])

    def diagnose(self, result: dict, recipe_id: str, mask: list[int],
                  history: list[dict]) -> list[dict]:
        """Run pattern matching. Returns list of triggered patterns."""
        sr = result["sr"]
        triggered: list[dict] = []

        # Pattern 1: direct_collapse
        if recipe_id == "direct" and sr < self.dense_sr * 0.5:
            triggered.append({
                "pattern_id": "direct_collapse",
                "rule_template": f"Direct pruning of {len(mask)}+ layers requires a recovery module; do not propose 'direct' recovery for k>={len(mask)} again.",
                "evidence": f"k={len(mask)} direct → SR={sr:.3f} (< dense_sr * 0.5 = {self.dense_sr * 0.5:.3f})",
            })

        # Pattern 2: recovery_insufficient
        if recipe_id == "adapter_only" and 0.6 <= sr < 0.9 and len(mask) >= 16:
            triggered.append({
                "pattern_id": "recovery_insufficient",
                "rule_template": f"For mask_size>={len(mask)}, propose joint recovery (adapter + action_head + proprio) instead of adapter_only.",
                "evidence": f"k={len(mask)} adapter_only → SR={sr:.3f} in [0.6, 0.9); joint recovery likely better.",
            })

        # Pattern 3: recovery_match_dense (validated config)
        if sr >= self.dense_sr - 0.02:
            triggered.append({
                "pattern_id": "recovery_match_dense",
                "rule_template": f"Mask k={len(mask)} with {recipe_id} reaches dense-level SR ({sr:.3f}); record as validated configuration.",
                "evidence": f"k={len(mask)} {recipe_id} → SR={sr:.3f} >= dense_sr - 0.02.",
            })

        # Pattern 4: anchor_violation — if previous twin mask (same recovery, ±1 layer) had >10pt higher SR
        twin_mask_set = set(mask)
        for h in history:
            if h.get("recovery") != recipe_id or h.get("sr") is None:
                continue
            h_set = set(h["mask"])
            diff_added = twin_mask_set - h_set
            diff_removed = h_set - twin_mask_set
            # current mask has exactly 1 extra layer vs h
            if len(diff_added) == 1 and len(diff_removed) == 0:
                if h["sr"] - sr >= 0.10:
                    added_layer = list(diff_added)[0]
                    triggered.append({
                        "pattern_id": "anchor_violation",
                        "rule_template": f"L{added_layer} is a functional anchor; removing it drops SR by {(h['sr']-sr)*100:.1f}pt. Do NOT include L{added_layer} in mask without explicit ablation justification.",
                        "evidence": f"k={len(h['mask'])} {recipe_id} SR={h['sr']:.3f} → k={len(mask)} adding L{added_layer} → SR={sr:.3f}",
                        "_anchor_layer": added_layer,
                    })

        # Pattern 6: k_progression_violation — soft warning when agent jumps k aggressively
        # without consolidating wins at smaller k. Methodology: sweep at k, find best start,
        # then push k by +4..+8. A jump of >8 without prior SR >= 0.85 is premature.
        K_JUMP_LIMIT = 8
        SR_CONSOLIDATION_THRESHOLD = 0.85
        sr_seen = [(len(h["mask"]), h["sr"]) for h in history
                    if h.get("sr") is not None]
        if sr_seen:
            max_k_seen = max(k for k, _ in sr_seen)
            max_sr_so_far = max(s for _, s in sr_seen)
            current_k = len(mask)
            if current_k > max_k_seen + K_JUMP_LIMIT and max_sr_so_far < SR_CONSOLIDATION_THRESHOLD:
                triggered.append({
                    "pattern_id": "k_progression_violation",
                    "rule_template": (
                        f"Soft warning: current k={current_k} jumps >{K_JUMP_LIMIT} layers beyond "
                        f"prior max k={max_k_seen}, but best SR so far is only {max_sr_so_far:.3f} "
                        f"(< {SR_CONSOLIDATION_THRESHOLD}). Consider consolidating at smaller k first "
                        f"(sweep different starts to find a strong base) before pushing k further. "
                        f"Override if you have a specific hypothesis."
                    ),
                    "evidence": (
                        f"current k={current_k} (mask start={min(mask) if mask else '-'}); "
                        f"history max_k={max_k_seen} max_sr={max_sr_so_far:.3f}"
                    ),
                })

        # Pattern 5: sr_cliff_detected — generalized anchor discovery via near-twin mask comparison
        # Two iterations with same recovery, symmetric_difference of masks in {1, 2}, |sr diff| > 0.10
        # → the differing layers are anchor candidates (covers L30-style cliff at k=28 vs k=29)
        SR_CLIFF_THRESHOLD = 0.10
        for h in history:
            if h.get("recovery") != recipe_id or h.get("sr") is None:
                continue
            h_set = set(h["mask"])
            sym_diff = twin_mask_set.symmetric_difference(h_set)
            if not (1 <= len(sym_diff) <= 2):
                continue
            delta = abs(h["sr"] - sr)
            if delta <= SR_CLIFF_THRESHOLD:
                continue
            sr_lower, sr_higher = sorted([sr, h["sr"]])
            diff_layers_sorted = sorted(sym_diff)
            diff_str = ",".join(f"L{i}" for i in diff_layers_sorted)
            # describe the asymmetry: which iter has which layers
            in_curr_only = sorted(twin_mask_set - h_set)
            in_hist_only = sorted(h_set - twin_mask_set)
            asym = []
            if in_curr_only:
                asym.append(f"current adds {','.join(f'L{i}' for i in in_curr_only)}")
            if in_hist_only:
                asym.append(f"prior had {','.join(f'L{i}' for i in in_hist_only)}")
            asym_str = "; ".join(asym)
            triggered.append({
                "pattern_id": "sr_cliff_detected",
                "rule_template": (
                    f"Layers {{{diff_str}}} produce an SR cliff "
                    f"{sr_lower:.3f}->{sr_higher:.3f} (Δ={delta*100:.1f}pt) "
                    f"between near-twin masks differing by {len(sym_diff)} layer(s); "
                    f"treat {diff_str} as anchor candidates and consider preserving in future masks."
                ),
                "evidence": (
                    f"current k={len(mask)} SR={sr:.3f} vs prior k={len(h['mask'])} {recipe_id} SR={h['sr']:.3f}; "
                    f"{asym_str}"
                ),
                "_anchor_layer": diff_layers_sorted[0] if diff_layers_sorted else None,
            })

        # Pattern 7: inverted_anchor_priority — agent's mask choice contradicts
        # the per-layer training-free SR evidence: a layer with HIGH training-free
        # SR (≥ 0.85, easily removable) is being KEPT, while a layer with LOW
        # training-free SR (< 0.50, hard anchor) is being REMOVED. Suggests a
        # potential swap. Fires regardless of current iter's SR — pure structural
        # cross-check against per-layer evidence.
        if self.training_free_sr_by_layer:
            HIGH_KEEP_THRESHOLD = 0.85   # kept layer this easy = candidate to remove
            LOW_REMOVE_THRESHOLD = 0.50  # removed layer this hard = candidate to keep
            mask_set = set(mask)
            kept_set = set(range(self.n_layers)) - mask_set
            # find pairs
            kept_easy = sorted(
                [(L, self.training_free_sr_by_layer[L]) for L in kept_set
                  if L in self.training_free_sr_by_layer
                  and self.training_free_sr_by_layer[L] >= HIGH_KEEP_THRESHOLD],
                key=lambda x: -x[1],  # highest SR first (most "removable" first)
            )
            removed_hard = sorted(
                [(L, self.training_free_sr_by_layer[L]) for L in mask_set
                  if L in self.training_free_sr_by_layer
                  and self.training_free_sr_by_layer[L] < LOW_REMOVE_THRESHOLD],
                key=lambda x: x[1],  # lowest SR first (most "must keep" first)
            )
            if kept_easy and removed_hard:
                kept_easy_str = ", ".join(f"L{i}(tf-SR={s:.2f})" for i, s in kept_easy[:5])
                removed_hard_str = ", ".join(f"L{i}(tf-SR={s:.2f})" for i, s in removed_hard[:3])
                triggered.append({
                    "pattern_id": "inverted_anchor_priority",
                    "rule_template": (
                        f"INVERTED anchor priority detected: your mask KEEPS layer(s) "
                        f"[{kept_easy_str}] whose training-free single-delete SR ≥ {HIGH_KEEP_THRESHOLD:.2f} "
                        f"(individually easy to remove), while REMOVING layer(s) "
                        f"[{removed_hard_str}] whose training-free SR < {LOW_REMOVE_THRESHOLD:.2f} "
                        f"(individually hard anchors — pipeline collapses on their single deletion). "
                        f"Consider a swap: bring a hard-anchor layer back into kept, drop an easy one. "
                        f"⚠️ CAVEAT 1: recovery training MAY rescue hard anchors (closed-loop SR is the only true arbiter). "
                        f"⚠️ CAVEAT 2: training-free SR MISSES subtle anchors like format-aligners (e.g., a late-stack "
                        f"layer can have tf-SR≈0.99 yet cliff at high k — cross-check kept_easy candidates against "
                        f"prior sr_cliff_detected diagnoses before swapping them out)."
                    ),
                    "evidence": (
                        f"mask k={len(mask)} SR={sr:.3f}; "
                        f"kept easy-to-remove (tf-SR≥{HIGH_KEEP_THRESHOLD:.2f}): {kept_easy_str}; "
                        f"removed hard-anchor (tf-SR<{LOW_REMOVE_THRESHOLD:.2f}): {removed_hard_str}"
                    ),
                    "_anchor_layer": removed_hard[0][0] if removed_hard else None,
                })

        # Pattern 8: exploration_bias — SOFT warning when the agent over-clusters
        # recent proposals in ONE grid cell AND that cell has NOT produced a
        # near-best SR (clustering + no payoff = orbiting a weak region). It is
        # SUPPRESSED when the over-sampled cell IS the current global-best region
        # (then clustering = legitimate exploitation — leave it alone). Soft, like
        # k_progression: it only nudges toward an under-explored cell; agent may ignore.
        # Thresholds: window short (W=6) because closed-loop iters are few; conc
        # 0.60 = "clearly biased to one cell but not fully locked". Exemption uses a
        # RELATIVE line (within EB_BEST_MARGIN of global best) — avoids hard-coding an
        # absolute target SR (ledger §9: align to target SR later if desired).
        EB_WINDOW = 6
        EB_CONC_THRESHOLD = 0.60
        EB_BEST_MARGIN = 0.05
        recent_masks = [h["mask"] for h in history if h.get("mask")]
        if mask:
            recent_masks = recent_masks + [list(mask)]
        frac, top_cell, n_recent = concentration(recent_masks, EB_WINDOW)
        if n_recent >= EB_WINDOW and top_cell is not None and frac >= EB_CONC_THRESHOLD:
            cell_srs = [h["sr"] for h in history
                        if h.get("mask") and h.get("sr") is not None
                        and cell_of(h["mask"]) == top_cell]
            if sr is not None and mask and cell_of(mask) == top_cell:
                cell_srs.append(sr)
            best_in_cell = max(cell_srs) if cell_srs else 0.0
            all_srs = [h["sr"] for h in history if h.get("sr") is not None]
            if sr is not None:
                all_srs.append(sr)
            global_best = max(all_srs) if all_srs else 0.0
            # Fire ONLY if the over-sampled cell is NOT the global-best region.
            if best_in_cell < global_best - EB_BEST_MARGIN:
                seen_cells = Counter(cell_of(h["mask"]) for h in history if h.get("mask"))
                all_cells = [(kb, cb) for kb in range(K_BINS) for cb in range(CENTROID_BINS)]
                cand = sorted(all_cells, key=lambda c: (seen_cells.get(c, 0), c))
                suggest = next((c for c in cand if c != top_cell), None)
                suggest_str = cell_label(suggest) if suggest else "any unexplored region"
                triggered.append({
                    "pattern_id": "exploration_bias",
                    "rule_template": (
                        f"Soft warning: {frac*100:.0f}% of your last {n_recent} proposals fell in "
                        f"region {cell_label(top_cell)}, whose best SR so far ({best_in_cell:.3f}) is "
                        f"below your global best ({global_best:.3f}) — you may be orbiting a weak region. "
                        f"Consider an under-explored region such as {suggest_str} (different k and/or "
                        f"prune position). Only a nudge — if you have a specific reason to keep refining "
                        f"here (e.g. you ARE near the best region), ignore it."
                    ),
                    "evidence": (
                        f"window={n_recent} concentration={frac:.2f} in cell {top_cell}"
                        f"({cell_label(top_cell)}); cell_best={best_in_cell:.3f} < "
                        f"global_best-{EB_BEST_MARGIN}={global_best - EB_BEST_MARGIN:.3f}"
                    ),
                })

        # Post-process: enrich every triggered dict with structured fields the
        # RuleStore consumes (rho/label/strategy/problem/scope). Anchor-class
        # patterns carry a concrete anchor_layer = model-specific anatomy (not
        # transferable); methodology patterns transfer across models.
        # (related_work_mining_2026-05-30 §4)
        _LABEL = {
            "direct_collapse": "failure", "recovery_insufficient": "failure",
            "recovery_match_dense": "success", "anchor_violation": "failure",
            "proxy_disagrees_closed_loop": "failure", "sr_cliff_detected": "failure",
            "k_progression_violation": "failure", "inverted_anchor_priority": "failure",
            "exploration_bias": "failure",
        }
        _LEVEL = {
            "direct_collapse": "methodology", "recovery_insufficient": "methodology",
            "k_progression_violation": "methodology", "exploration_bias": "methodology",
            "recovery_match_dense": "model_specific", "anchor_violation": "model_specific",
            "sr_cliff_detected": "model_specific", "inverted_anchor_priority": "model_specific",
            "proxy_disagrees_closed_loop": "model_specific",
        }
        for t in triggered:
            pid = t["pattern_id"]
            t.setdefault("rho", pid)
            t.setdefault("label", _LABEL.get(pid, "failure"))
            t.setdefault("strategy", t.get("rule_template", ""))
            t.setdefault("problem", t.get("evidence", ""))
            t.setdefault("scope", {
                "level": _LEVEL.get(pid, "model_specific"),
                "model_id": None,
                "anchor_layer": t.pop("_anchor_layer", None),
            })

        return triggered
