"""Driver — orchestrates Agent + ModuleSet through N iterations.

Each iteration:
  1. Build AgentContext from current modules state + last result + last diagnoses
  2. agent.decide(ctx) → AgentDecision
  3. Validate decision (recipe known, mask within range, not already tried optional)
  4. Lookup SR from eval module (or train+eval if real GPU mode)
  5. Run diagnosis on result + history → triggered patterns
  6. Promote accepted_diagnoses to rules
  7. Record mask in mask_search; write iter_NNN/{proposal,result,diagnoses,rules_after}.json

Run end-to-end + dumps a final trace.json summary.
"""
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

# Allow imports from anywhere
ROOT = Path(__file__).resolve().parents[1]  # <research-workspace>
sys.path.insert(0, str(ROOT))

from agentic_pruning.modules import ModuleSet
from agentic_pruning.modules.rule_store import RuleStore
from agentic_pruning.modules.gate import GateConfig, gate_with_escalation
from agentic_pruning.agents.base import Agent, AgentContext, AgentDecision


SEED_RULES = [
    "closed_loop_dominates_proxy: closed-loop robot SR is the final judge; static metrics (FIM/CKA) can only rank candidates, not decide.",
]

HARD_ANCHOR_SR_THRESHOLD = 0.50      # training-free SR below this = confirmed hard anchor
SOFT_ANCHOR_SR_THRESHOLD = 0.85      # training-free SR in [0.50, 0.85) = soft anchor (handle with care)


def derive_training_free_anchor_rules(per_layer_table: list[dict],
                                        dense_sr: float) -> list[str]:
    """Auto-emit seed rules from training-free single-delete SR per layer.

    HARD anchor: single-delete SR < HARD_ANCHOR_SR_THRESHOLD (severe immediate
        dependency — must keep). SOFT anchor: between HARD and SOFT thresholds
        (notable drop — handle with care). Returns a list of rule strings.
    No rule is emitted for layers without single_delete_training_free_sr field.
    """
    hard: list[tuple[int, float]] = []
    soft: list[tuple[int, float]] = []
    for L in per_layer_table:
        sr = L.get("single_delete_training_free_sr")
        if sr is None:
            continue
        if sr < HARD_ANCHOR_SR_THRESHOLD:
            hard.append((L["idx"], sr))
        elif sr < SOFT_ANCHOR_SR_THRESHOLD:
            soft.append((L["idx"], sr))
    rules: list[str] = []
    if hard:
        listing = ", ".join(f"L{i}(SR={s:.2f})" for i, s in hard)
        rules.append(
            f"[training_free_hard_anchor] Layers {listing} have training-free "
            f"single-delete SR < {HARD_ANCHOR_SR_THRESHOLD:.2f} (vs dense {dense_sr:.2f}). "
            f"These are CONFIRMED HARD ANCHORS — any mask including them will collapse SR. "
            f"NEVER include these layers in a remove-mask without an explicit ablation hypothesis."
        )
    if soft:
        listing = ", ".join(f"L{i}(SR={s:.2f})" for i, s in soft)
        rules.append(
            f"[training_free_soft_anchor] Layers {listing} have training-free "
            f"single-delete SR in [{HARD_ANCHOR_SR_THRESHOLD:.2f}, {SOFT_ANCHOR_SR_THRESHOLD:.2f}) "
            f"(notable drop). Treat as SOFT ANCHORS — keep by default; only remove if a "
            f"strong recovery recipe is justified."
        )
    return rules


def _is_contiguous_block(mask: list[int]) -> bool:
    """True iff mask is exactly one contiguous run of layer indices."""
    if not mask:
        return False
    s = sorted(mask)
    return s == list(range(s[0], s[0] + len(s)))


def propose_with_validation_retry(agent: Agent, ctx: AgentContext,
                                    tried_records,
                                    max_retries: int = 3) -> tuple[AgentDecision, dict]:
    """Call agent.decide; reject + retry if proposal violates either:
      - contiguous block constraint (mask must be one consecutive run)
      - dedup constraint (mask not previously tried)

    `tried_records` may be either:
      - a list of dicts with keys mask / sr / iter_idx / recovery (preferred, richer reject notes), OR
      - a set of mask-tuples (legacy callers; degrades to minimal reject note)

    After max_retries, accept the last proposal anyway with `exhausted` flag.
    Mutates ctx.rules transiently with rejection notes (capped at last 5).
    """
    # Normalize tried_records → (lookup_set, record_by_key) for rich reject notes
    if isinstance(tried_records, set):
        tried_key_set = tried_records
        record_by_key: dict = {}
    else:
        record_by_key = {}
        for r in tried_records:
            key = tuple(sorted(r["mask"]))
            record_by_key[key] = r
        tried_key_set = set(record_by_key.keys())

    REJECT_HISTORY_CAP = 5  # only keep last N reject notes in ctx.rules to avoid prompt bloat

    for attempt in range(max_retries + 1):
        decision = agent.decide(ctx)
        mask_sorted = sorted(decision.mask)
        mask_key = tuple(mask_sorted)
        is_contig = _is_contiguous_block(mask_sorted)
        is_dup = mask_key in tried_key_set

        if is_contig and not is_dup:
            return decision, {"attempts": attempt + 1, "exhausted": False,
                                "reason": "ok"}
        if attempt >= max_retries:
            reason = "non_contiguous" if not is_contig else "duplicate"
            print(f"[validation] iter {ctx.iter_idx} exhausted {max_retries} retries "
                  f"(reason={reason}); accepting mask {mask_sorted} anyway")
            return decision, {"attempts": attempt + 1, "exhausted": True,
                                "reason": reason}
        # Build reject note (with rich context when available)
        if not is_contig:
            mask_repr = mask_sorted if len(mask_sorted) <= 12 else f"{mask_sorted[:6]}...{mask_sorted[-3:]}"
            reject_note = (
                f"[REJECT attempt {attempt+1}/{max_retries}] mask={mask_repr} is NOT one "
                f"contiguous block. mask MUST be [start, start+1, ..., start+k-1] for some "
                f"start and k. Re-propose with a SINGLE consecutive run."
            )
        else:  # is_dup
            prior = record_by_key.get(mask_key)
            if prior is not None:
                prior_iter = prior.get("iter_idx", prior.get("iter", "?"))
                prior_sr = prior.get("sr")
                prior_recovery = prior.get("recovery", "?")
                if prior_sr is not None:
                    sr_str = f"SR={prior_sr:.3f}"
                    learn_hint = ("If that SR was high, you do not need to re-test the same mask. "
                                    "If it was low, propose a meaningfully different mask "
                                    "(different start, different k, or different recovery).")
                else:
                    sr_str = "SR=n/a (cache miss / no eval)"
                    learn_hint = ("Propose a meaningfully different mask "
                                    "(different start, different k, or different recovery).")
                reject_note = (
                    f"[REJECT attempt {attempt+1}/{max_retries}] mask={mask_sorted} (start={mask_sorted[0]}, "
                    f"k={len(mask_sorted)}) was ALREADY tried at iter {prior_iter} "
                    f"with recovery={prior_recovery} → {sr_str}. {learn_hint}"
                )
            else:
                reject_note = (
                    f"[REJECT attempt {attempt+1}/{max_retries}] mask={mask_sorted} was ALREADY "
                    f"tried in a prior iteration. Propose a DIFFERENT contiguous mask "
                    f"(different start or different k)."
                )
        # Append reject note; cap history of REJECTs in ctx.rules to last N
        prior_rejects = [r for r in ctx.rules if isinstance(r, str) and r.startswith("[REJECT")]
        non_rejects = [r for r in ctx.rules if not (isinstance(r, str) and r.startswith("[REJECT"))]
        kept_rejects = (prior_rejects + [reject_note])[-REJECT_HISTORY_CAP:]
        ctx.rules = non_rejects + kept_rejects
        print(f"[validation] iter {ctx.iter_idx} attempt {attempt+1}: "
              f"{'non_contiguous' if not is_contig else 'duplicate'} — retrying")
    return decision, {"attempts": max_retries + 1, "exhausted": True, "reason": "fallback"}


# Backward-compat alias (older callers used dedup-only name)
propose_with_dedup_retry = propose_with_validation_retry


def compute_best_so_far(tried: list[dict]) -> Optional[dict]:
    """Pick the (mask, recovery, sr) record with highest SR among those with non-null SR.

    Returns a normalized dict: {mask, k, start, sr, recovery, iter}.
    `start` = min(mask) when mask is contiguous (None if not contiguous or empty mask).
    Returns None if no record has a non-null SR.
    """
    scored = [t for t in tried if t.get("sr") is not None]
    if not scored:
        return None
    best = max(scored, key=lambda t: t["sr"])
    mask = sorted(best["mask"])
    is_contig = bool(mask) and mask == list(range(mask[0], mask[0] + len(mask)))
    return {
        "mask": mask,
        "k": len(mask),
        "start": mask[0] if (mask and is_contig) else None,
        "contiguous": is_contig,
        "sr": best["sr"],
        "recovery": best.get("recovery", "?"),
        "iter": best.get("iter_idx", best.get("iter", "?")),
    }


def build_context(modules: ModuleSet, rules: list[str], iter_idx: int,
                   last_proposal: Optional[dict], last_result: Optional[dict],
                   last_diagnoses: list[dict], pruning_mode: str = "contiguous",
                   no_probe: bool = False) -> AgentContext:
    """``no_probe`` (added 2026-07-17, quad_cell_4 experiment): strip EVERY
    probe-evidence channel from the context — per-layer static facts, safety
    ranking, probe methodology/reliability/anchor-recipe/warnings/landmarks/
    metric semantics, and the probe-query catalog. The agent keeps only the
    closed-loop channels (tried history + SR, best_so_far, archive, diagnoses,
    rules) plus suite/dense_sr/n_layers/recovery menu. Default False keeps the
    historical behavior byte-identical."""
    probe = modules.probe
    tried = modules.mask_search.list_tried()
    if no_probe:
        return AgentContext(
            iter_idx=iter_idx,
            rules=list(rules),
            last_proposal=last_proposal,
            last_result=last_result,
            last_diagnoses=list(last_diagnoses),
            probe_summary={},
            recovery_options=modules.recovery.list_recipes(),
            tried_masks=tried,
            search_space=modules.mask_search.get_search_space(),
            suite=probe.suite,
            dense_sr=probe.dense_sr,
            n_layers=probe.n_layers,
            pruning_mode=pruning_mode,
            probe_reliability={},
            anchor_recipe={},
            warnings=[],
            qualitative_landmarks={},
            metric_semantics={},
            layer_static_facts=[],
            available_probes=[],
            probe_results={},
            probe_round_idx=0,
            probe_rounds_max=0,
            best_so_far=compute_best_so_far(tried),
            layer_safety_table=[],
            archive_map=(modules.mask_search.get_archive_map()
                         if hasattr(modules.mask_search, "get_archive_map") else []),
            per_layer_table=[],
        )
    available_probes = (modules.probe_query.list_available()
                         if modules.probe_query is not None else [])
    # ⑤ z-score safety ranking by training-free single-delete SR (safest first).
    facts = probe.get_layer_static_facts()
    srs = [L["single_delete_training_free_sr"] for L in facts
           if L.get("single_delete_training_free_sr") is not None]
    layer_safety_table: list[dict] = []
    if len(srs) >= 2:
        mu = statistics.mean(srs)
        sd = statistics.pstdev(srs) or 1e-9
        for L in facts:
            s = L.get("single_delete_training_free_sr")
            if s is None:
                continue
            tag = ("HARD-ANCHOR" if s < 0.50 else "soft-anchor" if s < 0.85 else "safe")
            layer_safety_table.append({"idx": L["idx"], "sr": s,
                                        "z": (s - mu) / sd, "tag": tag})
        layer_safety_table.sort(key=lambda r: r["z"], reverse=True)  # safest first
    archive_map = (modules.mask_search.get_archive_map()
                   if hasattr(modules.mask_search, "get_archive_map") else [])
    return AgentContext(
        iter_idx=iter_idx,
        rules=list(rules),
        last_proposal=last_proposal,
        last_result=last_result,
        last_diagnoses=list(last_diagnoses),
        probe_summary=probe.get_summary(),
        recovery_options=modules.recovery.list_recipes(),
        tried_masks=tried,
        search_space=modules.mask_search.get_search_space(),
        suite=probe.suite,
        dense_sr=probe.dense_sr,
        n_layers=probe.n_layers,
        pruning_mode=pruning_mode,

        probe_reliability=probe.get_probe_reliability(),
        anchor_recipe=probe.get_anchor_recipe(),
        warnings=probe.get_warnings(),
        qualitative_landmarks=probe.get_qualitative_landmarks(),
        metric_semantics=probe.get_metric_semantics(),
        layer_static_facts=probe.get_layer_static_facts(),
        available_probes=available_probes,
        probe_results={},
        probe_round_idx=0,
        probe_rounds_max=3,
        best_so_far=compute_best_so_far(tried),
        layer_safety_table=layer_safety_table,
        archive_map=archive_map,
        # back-compat (unused after refactor):
        per_layer_table=[],
    )


def propose_with_probe_tool_loop(agent: Agent, ctx: AgentContext,
                                    probe_query_service,
                                    tried_records,
                                    max_retries: int = 3,
                                    max_probe_rounds: int = 3,
                                    dedup_key: str = "mask+recovery",
                                    ) -> tuple[AgentDecision, dict]:
    """Unified per-iter loop. Each agent.decide() call may either:
      - emit probe_queries only (mask=[]) → execute, accumulate probe_results, re-call
      - emit a mask → validate (contiguous + dedup), accept if good, else reject+retry

    Bounded by: max_probe_rounds (query phase) + max_retries (validation phase).
    Returns the FIRST valid (contiguous + non-duplicate) mask, or the last
    attempted one when limits are hit (with exhausted flag).

    ``dedup_key`` (added 2026-07-16, open_menu_replay experiment):
      - "mask" (default, historical behavior): a proposal is a duplicate iff its
        MASK was tried before, regardless of recovery. This makes recovery
        escalation on the same mask impossible.
      - "mask+recovery": duplicate iff the (mask, recovery) PAIR was tried —
        required for cheap-first / escalate-on-failure recovery search.
    """
    probe_call_log: list[dict] = []
    ctx.probe_results = {}
    ctx.probe_round_idx = 0
    ctx.probe_rounds_max = max_probe_rounds
    last_intent_rationale = ""

    def _key(mask_sorted: tuple, recovery: str):
        if dedup_key == "mask+recovery":
            return (mask_sorted, recovery)
        return mask_sorted

    # Normalize tried_records (list of dicts OR set) → record_by_key + tried_key_set
    if isinstance(tried_records, set):
        tried_key_set = tried_records
        record_by_key: dict = {}
    else:
        record_by_key = {_key(tuple(sorted(r["mask"])), r.get("recovery", "")): r
                         for r in tried_records}
        tried_key_set = set(record_by_key.keys())

    REJECT_HISTORY_CAP = 5
    probe_rounds_used = 0
    validation_attempts = 0
    total_calls_cap = max_probe_rounds + max_retries + 2  # safety
    # Safe default so an all-failing iter still returns a usable (empty-mask) decision
    # instead of None (which would AttributeError in run_loop).
    last_decision = AgentDecision(mask=[], recovery="", rationale="agent produced no parseable decision",
                                  success_gate=0.0, stop_gate=0.0, evidence_used=[])
    last_validation_reason = "ok"
    exhausted = False

    while (probe_rounds_used + validation_attempts) < total_calls_cap:
        # round_idx grows past max → prompt switches into "phase over" mode
        ctx.probe_round_idx = probe_rounds_used
        try:
            decision = agent.decide(ctx)
        except Exception as exc:  # noqa: BLE001 - one bad LLM turn must not kill the run
            validation_attempts += 1
            print(f"[validation] iter {ctx.iter_idx} agent.decide raised "
                  f"{type(exc).__name__}: {exc}")
            if validation_attempts >= max_retries:
                last_validation_reason = "agent_error"
                exhausted = True
                break
            stern = (f"[AGENT_ERROR v{validation_attempts}/{max_retries}] Your previous response "
                     f"failed validation: {str(exc)[:300]}. Output ONLY one valid JSON "
                     f"object with an integer-array mask following the declared pruning geometry and a valid recovery. "
                     f"No prose, no trailing commas.")
            ctx.rules = list(ctx.rules) + [stern]
            continue
        last_decision = decision

        # --- Branch 1: agent committed a mask --------------------------------
        if decision.mask:
            mask_sorted = sorted(decision.mask)
            mask_key = _key(tuple(mask_sorted), decision.recovery)
            is_contig = _is_contiguous_block(mask_sorted)
            contiguity_required = (ctx.pruning_mode == "contiguous")
            # Geometry must match the model's mode in BOTH directions: contiguous mode needs
            # one block; SKIP mode needs a SCATTERED set — a contiguous run of >1 layers would
            # span an anchor / transition band and collapse SR, so reject it and make the agent
            # retry. (k<=1 is trivially valid in either mode.)
            skip_required = (ctx.pruning_mode == "skip") and len(mask_sorted) > 1
            contiguity_ok = ((is_contig or not contiguity_required)
                             and not (skip_required and is_contig))
            is_dup = mask_key in tried_key_set
            range_ok = (len(set(mask_sorted)) == len(mask_sorted)
                        and all(isinstance(x, int) and 0 <= x < ctx.n_layers for x in mask_sorted)
                        and len(mask_sorted) < ctx.n_layers)
            if contiguity_ok and range_ok and not is_dup:
                last_validation_reason = "ok"
                meta = {
                    "probe_rounds_used": probe_rounds_used,
                    "probe_calls": probe_call_log,
                    "validation": {"attempts": validation_attempts + 1,
                                     "exhausted": False, "reason": "ok"},
                    "intent_rationale": last_intent_rationale,
                }
                return decision, meta
            # Invalid mask → reject and try again (count as a validation attempt)
            validation_attempts += 1
            if validation_attempts >= max_retries:
                last_validation_reason = (
                    "out_of_range_or_duplicate_indices" if not range_ok else
                    "non_contiguous" if (contiguity_required and not is_contig)
                    else "contiguous_in_skip_mode" if (skip_required and is_contig)
                    else "duplicate")
                exhausted = True
                break
            if skip_required and is_contig:
                reject_note = (f"[REJECT v{validation_attempts}/{max_retries}] mask={mask_sorted} "
                                 f"is a CONTIGUOUS run. This model uses SKIP pruning — the removed "
                                 f"layers MUST be SCATTERED (non-adjacent), e.g. {{3,5,8,10,13,17,21,25,27,30}}. "
                                 f"Query 'knockout_dall', KEEP its top anchors, and pick scattered "
                                 f"low-knockout layers.")
            elif contiguity_required and not is_contig:
                reject_note = (f"[REJECT v{validation_attempts}/{max_retries}] mask={mask_sorted} "
                                 f"is NOT contiguous. This model uses CONTIGUOUS pruning — mask MUST "
                                 f"be [start, start+1, ..., start+k-1].")
            else:
                prior = record_by_key.get(mask_key, {})
                prior_iter = prior.get("iter_idx", prior.get("iter", "?"))
                prior_sr = prior.get("sr")
                sr_s = f"SR={prior_sr:.3f}" if isinstance(prior_sr, (int, float)) else "SR=n/a"
                dup_what = ("(mask, recovery) pair" if dedup_key == "mask+recovery" else "mask")
                dup_fix = ("Propose a different (start, k) or a DIFFERENT recovery for this mask."
                           if dedup_key == "mask+recovery" else "Propose a different (start, k).")
                reject_note = (f"[REJECT v{validation_attempts}/{max_retries}] {dup_what} "
                                 f"mask={mask_sorted} already tried at iter {prior_iter} ({sr_s}). "
                                 f"{dup_fix}")
            prior_rejects = [r for r in ctx.rules if isinstance(r, str) and r.startswith("[REJECT")]
            non_rejects = [r for r in ctx.rules if not (isinstance(r, str) and r.startswith("[REJECT"))]
            ctx.rules = non_rejects + (prior_rejects + [reject_note])[-REJECT_HISTORY_CAP:]
            print(f"[validation] iter {ctx.iter_idx} reject ({validation_attempts}/{max_retries}): "
                  f"{'non_contig' if not is_contig else 'dup'}")
            continue

        # --- Branch 2: agent emitted probe_queries (mask=[]) ----------------
        if decision.probe_queries and probe_rounds_used < max_probe_rounds:
            last_intent_rationale = decision.rationale
            for q in decision.probe_queries:
                probe_name = q.get("probe")
                layers = q.get("layers", [])
                if not probe_name or not layers:
                    continue
                try:
                    res = probe_query_service.query(probe_name, [int(L) for L in layers])
                except Exception as exc:
                    probe_call_log.append({"probe": probe_name, "layers": list(layers),
                                              "error": str(exc), "round": probe_rounds_used})
                    continue
                probe_call_log.append({"probe": probe_name, "layers": list(layers),
                                          "result": res, "round": probe_rounds_used})
                ctx.probe_results.setdefault(probe_name, {}).update(
                    {int(k): v for k, v in res.items()}
                )
            probe_rounds_used += 1
            if probe_rounds_used >= max_probe_rounds:
                stern = (f"[FORCE_COMMIT] Used all {max_probe_rounds} probe rounds. "
                          f"Next call MUST emit a non-empty mask (no more probe_queries).")
                ctx.rules = list(ctx.rules) + [stern]
            continue

        # --- Branch 3: nothing useful (neither mask nor probes left) --------
        validation_attempts += 1
        if validation_attempts >= max_retries:
            last_validation_reason = "no_output"
            exhausted = True
            break
        stern = (f"[FORCE_COMMIT v{validation_attempts}/{max_retries}] You emitted neither "
                  f"a mask nor a valid probe_query. Output a contiguous non-empty mask NOW.")
        ctx.rules = list(ctx.rules) + [stern]

    # Loop exited without success — accept last decision (mask may be empty)
    meta = {
        "probe_rounds_used": probe_rounds_used,
        "probe_calls": probe_call_log,
        "validation": {"attempts": validation_attempts, "exhausted": exhausted,
                          "reason": last_validation_reason},
        "intent_rationale": last_intent_rationale,
    }
    print(f"[validation] iter {ctx.iter_idx} EXHAUSTED — accepting "
            f"mask={sorted(last_decision.mask) if last_decision else None}")
    return last_decision, meta


def write_iter_trace(out_dir: Path, iter_idx: int, ctx: AgentContext,
                      decision: AgentDecision, result: Optional[dict],
                      diagnoses: list[dict], rules_after: list[str]) -> None:
    d = out_dir / f"iter_{iter_idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "context.json").write_text(json.dumps({
        "iter_idx": ctx.iter_idx,
        "rules_before": ctx.rules,
        "last_proposal": ctx.last_proposal,
        "last_result": ctx.last_result,
        "last_diagnoses": ctx.last_diagnoses,
        # skip probe_summary/recovery_options to keep small; in full prompt
    }, indent=2, ensure_ascii=False))
    (d / "proposal.json").write_text(json.dumps(asdict(decision), indent=2, ensure_ascii=False))
    (d / "result.json").write_text(json.dumps(result or {}, indent=2, ensure_ascii=False))
    (d / "diagnoses.json").write_text(json.dumps(diagnoses, indent=2, ensure_ascii=False))
    (d / "rules_after.json").write_text(json.dumps(rules_after, indent=2, ensure_ascii=False))


def run_loop(agent: Agent, modules: ModuleSet, n_iters: int, out_dir: Path,
             *, extra_seed_rules: Optional[list] = None,
             pruning_mode: str = "contiguous",
             dedup_key: str = "mask+recovery",
             no_probe: bool = False,
             experience_dir: Optional[Path] = None,
             gate_cfg: Optional[GateConfig] = None) -> dict:
    """Run the agentic loop. ``extra_seed_rules`` are appended to the seed rule set
    (used to inject the user's goal — target SR / compression — so LLM agents search
    toward it instead of blind, then post-hoc selection enforces the constraint).

    ``no_probe`` (2026-07-17, quad_cell_4): evidence-gating ablation cell ④ —
    strips probe evidence from BOTH the seed rules (no training-free anchor
    rules) and the per-iter context (see build_context), and disables the
    probe-query tool (max_probe_rounds=0). Closed-loop SR feedback is kept.

    ``experience_dir`` (2026-07-17, quad_cell_4): RuleStore directory override.
    The default (historical) store persists across runs and already contains
    anchor rules learned by earlier experiments — a leak for evidence-gating
    ablations. Point this at a fresh dir for a clean-slate run.

    ``gate_cfg`` (2026-07-21, final_protocol_replay): Wilson 95% three-zone
    acceptance gate (paper §3.5). When set, cached lookups are judged
    pass/fail/uncertain against tau via the confidence interval; 'uncertain'
    escalates the budget one step (200→500) using a higher-budget cached
    measurement if present, else the cell is flagged into
    out_dir/escalation_queue.json for a REAL re-evaluation. Uncertain at max
    budget = pre-registered fail. Default None keeps the historical
    point-estimate behavior byte-identical."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # RuleStore: structured, self-sharpening, cross-model-transferable rule
    # library (related_work_mining_2026-05-30 §4). Replaces the old free-text
    # `rules` list. seed() registers SEED_RULES + derived anchor rules as
    # methodology-level priors; render() produces the prompt-facing strings.
    store = RuleStore(store_dir=(Path(experience_dir) if experience_dir is not None
                                 else out_dir / "experience"),
                      model_id=modules.probe.suite)
    store.set_dense_sr(modules.probe.dense_sr)
    seed_rules = list(SEED_RULES)
    if not no_probe:
        # Training-free single-delete anchor rules are PROBE evidence — cell ④
        # must not see them.
        seed_rules += derive_training_free_anchor_rules(
            modules.probe.get_per_layer_table(), modules.probe.dense_sr)
    store.seed(seed_rules + list(extra_seed_rules or []))
    n_rules_now = len([r for r in store.all_rules() if r["status"] == "active"])
    last_proposal: Optional[dict] = None
    last_result: Optional[dict] = None
    last_diagnoses: list[dict] = []
    history: list[dict] = []
    escalation_queue: list[dict] = []

    trace_rows: list[dict] = []
    for i in range(n_iters):
        rules = store.render(max_m=15)
        ctx = build_context(modules, rules, i, last_proposal, last_result, last_diagnoses,
                            pruning_mode=pruning_mode, no_probe=no_probe)

        tried_records = modules.mask_search.list_tried()
        decision, tool_meta = propose_with_probe_tool_loop(
            agent, ctx, None if no_probe else modules.probe_query, tried_records,
            max_retries=3, max_probe_rounds=0 if no_probe else 3, dedup_key=dedup_key,
        )
        dedup_meta = tool_meta.get("validation", {})
        # Save tool_meta (probe call log + intent_rationale) alongside proposal
        iter_dir = out_dir / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "tool_meta.json").write_text(json.dumps({
            "probe_rounds_used": tool_meta["probe_rounds_used"],
            "probe_calls": tool_meta["probe_calls"],
            "intent_rationale": tool_meta["intent_rationale"],
            "validation": tool_meta["validation"],
        }, indent=2, ensure_ascii=False))

        requested_recovery = decision.recovery
        # Execute only the requested, model-supported recipe; invalid proposals
        # remain explicit failures rather than silently changing the experiment.

        # Validate
        if dedup_meta.get("exhausted"):
            # The propose loop never produced a valid (contiguous + novel) mask. Do
            # NOT eval or record it: an empty/duplicate/non-contiguous mask pollutes
            # the tried log and — if a non-contiguous mask happens to sit in the eval
            # cache — could even surface an invalid winner.
            print(f"[iter {i}] validation exhausted (reason={dedup_meta.get('reason')}); "
                  f"skipping eval + record.")
            result = {"sr": None, "error": f"validation_exhausted:{dedup_meta.get('reason')}",
                      "source": "validate"}
        elif not modules.recovery.is_valid(decision.recovery):
            print(f"[iter {i}] WARN: unknown recovery '{decision.recovery}'; skipping eval.")
            result = {"sr": None, "error": f"unknown_recovery:{decision.recovery}", "source": "validate"}
        else:
            # Lookup eval (gated three-zone lookup when the Wilson gate is on)
            if gate_cfg is not None:
                looked = gate_with_escalation(modules.eval, decision.mask,
                                              decision.recovery, gate_cfg)
            else:
                looked = modules.eval.lookup_cached(decision.mask, decision.recovery)
            if looked is None:
                result = {"sr": None, "error": "mask_not_in_cache", "source": "lookup",
                          "known_recoveries_for_mask": modules.eval.list_known_recoveries_for_mask(decision.mask)}
            else:
                result = looked
                g = result.get("gate")
                if g and g.get("needs_real_eval"):
                    escalation_queue.append({
                        "iter": i, "mask": sorted(decision.mask),
                        "recovery": decision.recovery,
                        "current": g["chain"][-1],
                        "needs_budget": g.get("needs_budget"),
                    })
                    (out_dir / "escalation_queue.json").write_text(
                        json.dumps(escalation_queue, indent=2, ensure_ascii=False))

        result["requested_recovery"] = requested_recovery
        result["executed_recovery"] = decision.recovery

        # Diagnose (SR-dependent patterns). Anchor identification is the
        # AGENT'S job — derived from raw probe table + methodology rules.
        # Framework only diagnoses post-hoc SR-driven patterns (collapse,
        # recovery_insufficient, anchor_violation when SR drops on a layer
        # the agent had reason to believe was safe, etc.).
        diagnoses: list[dict] = []
        if result.get("sr") is not None:
            diagnoses = modules.diagnosis.diagnose(
                result, decision.recovery, decision.mask, history,
            )

        # Self-evolution via RuleStore: (1) upsert the diagnoses the agent
        # ACCEPTED from last iter (model-specific ones get tagged with model_id),
        # (2) reconcile counts on existing anchor rules with this iter's fresh SR,
        # (3) prune stale/contradictory rules, (4) persist for cross-run reuse.
        accept_ids = set(decision.accept_diagnoses) if decision.accept_diagnoses else set()
        for d in ctx.last_diagnoses:
            if d.get("pattern_id") in accept_ids:
                sc = d.get("scope")
                if sc and sc.get("level") == "model_specific" and sc.get("model_id") is None:
                    d["scope"]["model_id"] = store.model_id
                store.upsert(d)
        store.reconcile(result, decision.mask, history)
        store.prune()
        store.persist()
        n_rules_now = len([r for r in store.all_rules() if r["status"] == "active"])

        # Record in mask_search (skip exhausted/invalid proposals — else we re-log
        # duplicates or pollute the tried set with empty/non-contiguous masks).
        if not dedup_meta.get("exhausted"):
            modules.mask_search.record_proposed(
                mask=decision.mask, recovery=decision.recovery,
                sr=result.get("sr"), iter_idx=i, rationale=decision.rationale,
            )

        # Write iter trace
        write_iter_trace(out_dir, i, ctx, decision, result, diagnoses, store.render(max_m=50))

        # Print row
        sr_str = f"{result['sr']:.3f}" if result.get("sr") is not None else "n/a"
        g = result.get("gate") or {}
        gate_str = ""
        if g:
            fin = g["chain"][-1]
            gate_str = (f" | gate[{g['interval']}@{fin['n']}ep "
                        f"CI=({fin['lcb']:.3f},{fin['ucb']:.3f})]={g['verdict']}"
                        + (" (needs_real_eval)" if g.get("needs_real_eval") else "")
                        + (" (pre-registered fail)" if g.get("pre_registered_fail") else ""))
        print(f"[iter {i:02d}] k={len(decision.mask):2d} recipe={decision.recovery:24s} → SR={sr_str}  "
              f"| diagnoses={[d['pattern_id'] for d in diagnoses]}{gate_str}")

        trace_rows.append({
            "iter": i,
            "mask": decision.mask,
            "k": len(decision.mask),
            "recovery": decision.recovery,
            "sr": result.get("sr"),
            "n_trials": result.get("n_trials"),
            "gate_verdict": g.get("verdict"),
            "gate_chain": g.get("chain"),
            "rationale_short": decision.rationale[:80],
            "diagnoses": [d["pattern_id"] for d in diagnoses],
            "n_rules_active": n_rules_now,
        })

        # Update history + rolling state
        history.append({
            "iter": i,
            "mask": decision.mask,
            "recovery": decision.recovery,
            "sr": result.get("sr"),
        })
        last_proposal = {"mask": decision.mask, "recovery": decision.recovery,
                         "rationale": decision.rationale}
        last_result = result
        last_diagnoses = diagnoses

    # The final diagnoses remain in iter_NNN/diagnoses.json for review.
    # Without a subsequent agent acceptance, they must not enter the persistent
    # rule store. Accepted rules were already reconciled and persisted above.

    summary = {
        "agent": agent.name,
        "n_iters": n_iters,
        "final_rules": store.render(max_m=50),
        "rule_store": store.all_rules(),
        "trace": trace_rows,
        "suite": modules.probe.suite,
        "total_layers": modules.mask_search.total_layers,
        "dense_sr": modules.probe.dense_sr,
        "gate": gate_cfg.as_dict() if gate_cfg is not None else None,
        "escalation_queue": escalation_queue,
    }
    (out_dir / "trace.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


