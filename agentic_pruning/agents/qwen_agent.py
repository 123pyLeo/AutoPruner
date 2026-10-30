"""QwenAgent — local Qwen3-8B vLLM backend.

Reads context → builds JSON-output-only prompt → calls vLLM OpenAI-compatible
endpoint → parses structured decision.

Requires: vLLM server running at base_url (default http://127.0.0.1:8000/v1)
Start with: tmux new -d -s qwen3 'bash agentic_pruning/scripts/serve_qwen3_vllm.sh'
"""
import json
import os
from typing import Optional

from .base import Agent, AgentContext, AgentDecision, coerce_mask, coerce_probe_queries, decision_from_json


def _geometry_and_strategy(pruning_mode: str, n_layers: int) -> str:
    geometry = "one contiguous deletion block" if pruning_mode == "contiguous" else "a noncontiguous deletion set"
    return (f"Use {geometry}. Indices must be unique integers in [0, {n_layers-1}]. "
            "Leave at least one layer. Duplicate means the same deletion set AND recovery recipe.")


def _recipe_block(recovery_options) -> str:
    """Recovery-recipe instruction built from THIS model's actual menu (not hardcoded)."""
    ids = [r.get("id") for r in (recovery_options or []) if r.get("id")]
    if not ids:
        ids = ["adapter+head+proprio"]
    default = ids[0]
    return (f"""## Recipe selection
Choose "recovery" from THIS model's menu: {ids}. If unsure, emit the first one ("{default}").
Do NOT emit a recipe outside this list — it will be rejected.""")


def build_system_prompt(pruning_mode: str = "contiguous", n_layers: int = 32, recovery_options=None) -> str:
    return (f"You control a {n_layers}-layer VLA pruning experiment. Follow the user's explicit SR "
            "threshold and budget. Maximize deleted layers subject to measured closed-loop success. "
            "Functional probes guide proposals; they do not establish deployment success. "
            "Never treat a known result as a newly executed experiment. "
            + _geometry_and_strategy(pruning_mode, n_layers) + " " + _recipe_block(recovery_options) +
            " Query knockout_dall or other available probes using probe_queries, each with probe and layers. "
            "A query-only turn uses an empty mask. After bounded queries, propose a mask and recovery. "
            "You may try another recovery on the same structure. Do not assume specific layers are anchors. "
            "Return STRICT JSON with probe_queries, mask, recovery, rationale, success_gate, stop_gate, "
            "evidence_used, and accept_diagnoses (pattern IDs from the latest diagnoses). "
            "Types are mandatory: mask is an array of unique integer indices; recovery and rationale are strings; "
            "success_gate and stop_gate are JSON NUMBERS between 0.0 and 1.0, never sentences or objects. "
            "evidence_used and accept_diagnoses are arrays of strings. probe_queries is an array of "
            "objects with a probe string and layers integer array. "
            'Typed example: {"probe_queries":[],"mask":[],"recovery":"","rationale":"request probes",'
            '"success_gate":0.9,"stop_gate":0.5,"evidence_used":[],"accept_diagnoses":[]}. '

            "Use the observed history and measured evidence to justify the next experiment.")


#: Backward-compatible default (OFT / 32-layer / contiguous). Agents that know the model's
#: pruning_mode / n_layers / recovery menu should call build_system_prompt(...) per-iter instead.
SYSTEM_PROMPT = build_system_prompt()


def build_user_prompt(ctx: AgentContext) -> str:
    parts = []
    parts.append(f"## Iteration {ctx.iter_idx}  suite={ctx.suite}  dense_sr={ctx.dense_sr}")
    parts.append("")
    parts.append("### Current rules (evolved across iterations)")
    for i, r in enumerate(ctx.rules):
        parts.append(f"  R{i+1}. {r}")
    parts.append("")
    parts.append("### METHODOLOGY (transferable RULES from prior VLA research — no current-model data)")
    # Probe reliability (which probes work / mislead / partial — RULE-level)
    pr = ctx.probe_reliability
    if pr:
        parts.append("probe_reliability:")
        work = [p["probe"] for p in pr.get("work", [])]
        misleading = [p["probe"] for p in pr.get("misleading", [])]
        partial = [p["probe"] for p in pr.get("partial", [])]
        parts.append(f"  WORK (cite as evidence):     {work}")
        parts.append(f"  MISLEADING (do NOT cite):    {misleading}")
        parts.append(f"  PARTIAL (combine with SR):   {partial}")
        for p in pr.get("misleading", []):
            parts.append(f"    why {p['probe']} misleads: {p['why']}")
    # Metric semantics — explicit direction-of-interpretation for each metric
    # (prevents misreading like "high linear_probe_l1 = redundant")
    ms = ctx.metric_semantics
    if ms:
        parts.append("metric_semantics (READ FIRST — directional meaning of each metric):")
        for k, v in ms.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            parts.append(f"  [{k}] direction={v.get('direction','?')}")
            if v.get("interpretation_low"):
                parts.append(f"    LOW : {v['interpretation_low']}")
            if v.get("interpretation_high"):
                parts.append(f"    HIGH: {v['interpretation_high']}")
            if v.get("interpretation"):
                parts.append(f"    : {v['interpretation']}")
            if v.get("do_NOT_infer"):
                parts.append(f"    !! {v['do_NOT_infer']}")
    # Anchor recipe (keep/remove rules — RULE-level, no specific layer indices)
    ar = ctx.anchor_recipe
    if ar:
        parts.append("anchor_recipe:")
        for r in ar.get("keep_rules", []):
            parts.append(f"  KEEP: {r}")
        for r in ar.get("remove_rules", []):
            parts.append(f"  REMOVE: {r}")
        for cm in ar.get("common_misreadings", []):
            parts.append(f"  MISREAD TRAP: {cm['trap']}")
            parts.append(f"    why_wrong: {cm['why_wrong']}")
            parts.append(f"    correct_check: {cm['correct_check']}")
        for s in ar.get("discovery_protocol", []):
            parts.append(f"  PROTOCOL: {s}")
    # Warnings (hard rules)
    if ctx.warnings:
        parts.append("warnings (violation auto-fails proposal):")
        for w in ctx.warnings:
            parts.append(f"  {w['id']}: {w['rule']} — {w['reason']}")
    # Qualitative landmarks (rules-of-thumb, NO numbers)
    ql = ctx.qualitative_landmarks
    if ql:
        parts.append("qualitative_landmarks (rule-of-thumb expectations):")
        for k, v in ql.items():
            if k.startswith("_"):
                continue
            parts.append(f"  {k}: {v}")
    parts.append("")
    # Per-layer STATIC facts: ONLY idx + training-free single_delete_SR
    # (allowed pre-loaded exception; all other metrics require probe query)
    parts.append("### Per-layer STATIC facts (no query needed — these are the only pre-loaded per-layer numbers)")
    if ctx.layer_static_facts:
        keys = [k for k in ctx.layer_static_facts[0].keys() if k != "idx"]
        parts.append("  L#  | " + " | ".join(f"{k:>32}" for k in keys))
        parts.append("  ----+-" + "-+-".join("-" * 32 for _ in keys))
        for L in ctx.layer_static_facts:
            vals = []
            for k in keys:
                v = L.get(k)
                if isinstance(v, float):
                    vals.append(f"{v:>32.4f}")
                else:
                    vals.append(f"{str(v):>32}")
            parts.append(f"  L{L['idx']:>2} | " + " | ".join(vals))
    parts.append("")

    # ⑤ Layer prune-safety ranking (z-score of training-free single-delete SR):
    # decision-time evidence with safest-to-prune at the top, HARD-ANCHORs at the
    # bottom. Makes anchor priority an INPUT, not just a post-hoc diagnosis.
    if ctx.layer_safety_table:
        parts.append("### Layer prune-safety ranking (training-free single-delete SR, z-scored; SAFEST first)")
        parts.append("   L#  | tf-SR |   z    | tag")
        for r in ctx.layer_safety_table:
            parts.append(f"   L{r['idx']:>2}  | {r['sr']:.2f}  | {r['z']:+.2f} | {r['tag']}")
        parts.append("  ^ HARD-ANCHOR = single-delete collapses SR → keep unless deliberately ablating.")
        parts.append("    CAVEAT: tf-SR misses format-aligners (a high-tf-SR layer can still cliff at high k —")
        parts.append("    cross-check with probe queries / prior sr_cliff diagnoses before trusting).")
        parts.append("")

    # Probe tool API — hide entire section once rounds are exhausted to remove
    # the temptation for the agent to keep querying.
    probe_phase_over = ctx.probe_round_idx >= ctx.probe_rounds_max
    if probe_phase_over:
        parts.append("### Probe phase OVER — commit a mask")
        parts.append(f"  You have used all {ctx.probe_rounds_max} probe-query rounds. "
                      f"Any probe_queries you emit now will be IGNORED. Output a "
                      f"non-empty contiguous mask in this turn.")
        parts.append("")
    else:
        parts.append("### Available probes (call via probe_queries in your JSON)")
        parts.append(f"  rounds_used: {ctx.probe_round_idx}/{ctx.probe_rounds_max}")
        if not ctx.available_probes:
            parts.append("  (no probe service configured)")
        for p in ctx.available_probes:
            if not p.get("available"):
                continue
            parts.append(f"  - {p['name']:24s} ({p.get('value_range','')})")
            parts.append(f"      desc: {p.get('description','')}")
        parts.append("")

    # Probe results accumulated so far this iter (across previous rounds)
    parts.append("### Probe results so far this iter (from your previous probe_queries)")
    if not ctx.probe_results:
        parts.append("  (none yet — emit probe_queries in your JSON to obtain values)")
    else:
        for probe_name, layer_vals in ctx.probe_results.items():
            sorted_pairs = sorted(layer_vals.items())
            preview = ", ".join(f"L{k}={v:.4f}" if isinstance(v, float) else f"L{k}={v}"
                                  for k, v in sorted_pairs)
            parts.append(f"  {probe_name}: {{{preview}}}")
    parts.append("")
    parts.append("### Available recovery recipes")
    for r in ctx.recovery_options:
        parts.append(f"  - {r['id']}: trainable={r['trainable']}  ({r.get('description','')})")
    parts.append("")
    if ctx.best_so_far is not None:
        b = ctx.best_so_far
        if b.get("contiguous"):
            shape = f"start={b['start']}, k={b['k']}, span=L{b['mask'][0]}..L{b['mask'][-1]}"
        else:
            shape = f"mask={b['mask']} (NON-CONTIGUOUS)"
        gap = ctx.dense_sr - b['sr']
        parts.append("### Best result so far (the score to beat)")
        parts.append(f"  iter {b['iter']}: {shape}  recovery={b['recovery']} → SR={b['sr']:.3f}  "
                      f"(dense={ctx.dense_sr:.3f}, gap={gap:+.3f})")
        parts.append("  Next proposal should aim to EITHER: (a) match/exceed this SR while pushing k larger, "
                      "OR (b) match/exceed this SR with a meaningfully different (start, k, recovery). "
                      "Do not propose strictly worse-on-all-axes masks unless probing a specific hypothesis.")
        parts.append("")
    # ③ QD archive map — per-region champion so far (the tradeoff map, not just
    # one global best). Regions absent below are UNEXPLORED opportunities.
    if ctx.archive_map:
        parts.append("### QD archive — best SR seen per region (k × prune-position cell)")
        for r in ctx.archive_map:
            m = r["mask"]
            span = f"L{m[0]}..L{m[-1]}" if m else "-"
            parts.append(f"  [{r['label']}] best SR={r['sr']:.3f}  (k={r['k']}, {span}, iter {r['iter']})")
        parts.append("  Regions NOT listed are unexplored — consider one if you're orbiting a weak region.")
        parts.append("")
    parts.append("### Already-tried masks  (each as (start, k) for contiguous blocks)")
    if not ctx.tried_masks:
        parts.append("  (none yet)")
    for t in ctx.tried_masks[-10:]:  # last 10
        sr_s = f"{t['sr']:.3f}" if t.get("sr") is not None else "n/a"
        m = sorted(t['mask']) if t.get('mask') else []
        is_contig = bool(m) and m == list(range(m[0], m[0] + len(m)))
        if is_contig:
            shape = f"contiguous (start={m[0]}, k={len(m)}, span=L{m[0]}..L{m[-1]})"
        else:
            shape = f"NON-CONTIGUOUS {m}  ← would be rejected under current rules"
        parts.append(f"  iter {t['iter']:2d}: {shape}  recovery={t['recovery']} → SR={sr_s}")
    parts.append("")
    parts.append("### Last proposal & result")
    if ctx.last_proposal:
        parts.append(f"  proposal: mask={ctx.last_proposal['mask']} recovery={ctx.last_proposal['recovery']}")
        parts.append(f"  rationale: {ctx.last_proposal.get('rationale','')[:200]}")
    if ctx.last_result:
        if ctx.last_result.get("sr") is not None:
            parts.append(f"  result: SR={ctx.last_result['sr']:.3f}  n_trials={ctx.last_result.get('n_trials','?')}")
        else:
            parts.append(f"  result: SR unavailable ({ctx.last_result.get('error','?')})")
    parts.append("")
    parts.append("### Diagnoses triggered by last proposal (you can promote to rules)")
    if not ctx.last_diagnoses:
        parts.append("  (none)")
    for d in ctx.last_diagnoses:
        parts.append(f"  - {d['pattern_id']}: {d['rule_template']}")
        parts.append(f"      evidence: {d['evidence']}")
    parts.append("")
    parts.append("### Search space")
    parts.append(f"  total_layers={ctx.search_space.get('total_layers')}  "
                 f"compression_budget_options={ctx.search_space.get('compression_budget_options')}")
    # Per-model sweet spot DISCOVERED from measured SR (not the 28% cross-model seed).
    # The budget options include below-seed ratios (~10-20%): search DOWN too, do not
    # only ratchet up from 28%. FastWAM's real sweet spot was 20% (k=6), not 28% (k=8).
    _dsw = ctx.search_space.get("discovered_sweet_spot")
    if _dsw and _dsw.get("sweet_spot_ratio") is not None:
        parts.append(f"  DISCOVERED sweet spot so far: ratio={_dsw['sweet_spot_ratio']} "
                     f"k={_dsw['sweet_spot_k']} (deepest budget still clearing the recovery floor; "
                     f"from MEASURED SR, NOT the 28% seed). Center the next budget here, and probe "
                     f"DOWNWARD if even this is under the floor.")
    elif "discovered_sweet_spot" in ctx.search_space:
        parts.append("  DISCOVERED sweet spot so far: none yet — sweep budgets BOTH ways "
                     "(include ~10-20%), do not assume 28%.")
    parts.append(f"  constraint: {ctx.search_space.get('constraint','')}")
    parts.append("")
    parts.append("### Your task")
    if probe_phase_over:
        parts.append("**FINAL DECISION TURN**: emit a non-empty contiguous `mask` NOW. "
                      "Do NOT emit probe_queries — they will be ignored. "
                      "Use the probe results you already gathered + static facts + methodology.")
        parts.append("Output STRICT JSON: {mask: [int,...], recovery: \"adapter+head+proprio\", rationale, success_gate, stop_gate, evidence_used, accept_diagnoses}")
    elif ctx.probe_round_idx == ctx.probe_rounds_max - 1:
        parts.append(f"This is your LAST probe-query round ({ctx.probe_round_idx+1}/{ctx.probe_rounds_max}). "
                      f"You may emit FINAL probe_queries here, OR commit a mask now. On the NEXT call after this, "
                      f"probe_queries are disabled.")
        parts.append("Output STRICT JSON with keys: probe_queries, mask, recovery, rationale, success_gate, stop_gate, evidence_used, accept_diagnoses")
    else:
        parts.append(f"You are on probe-query round {ctx.probe_round_idx+1} of {ctx.probe_rounds_max}. "
                      f"You MAY emit probe_queries to gather more evidence (mask: []), OR commit a mask now.")
        parts.append("Output STRICT JSON with keys: probe_queries, mask, recovery, rationale, success_gate, stop_gate, evidence_used, accept_diagnoses")
    return "\n".join(parts)


def parse_json_response(text: str) -> dict:
    """Robust parse: handle Qwen3 <think>...</think> chain-of-thought, markdown
    fences, and raw {...} blocks. Tries multiple strategies."""
    import re
    if not text:
        raise ValueError("parse_json_response: empty/None response from LLM")
    # 1. Strip <think>...</think> blocks (Qwen3 reasoning mode)
    s = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 2. Try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 3. Strip markdown fences
    if s.startswith("```"):
        s = s.lstrip("`").lstrip()
        if s.startswith("json"):
            s = s[4:].lstrip()
        if s.endswith("```"):
            s = s[:-3].rstrip()
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
    # 4. Find outermost { ... } in post-think text
    a, b = s.find("{"), s.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(s[a:b+1])
        except json.JSONDecodeError:
            pass
    # 5. Last resort: search whole original text
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        try:
            return json.loads(text[a:b+1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from response:\n{text[:1000]}")


class QwenAgent(Agent):
    """Zero-dep HTTP client (urllib) to local vLLM OpenAI-compatible endpoint.
    Avoids requiring the `openai` Python pkg in whatever env the driver runs in.
    Bypasses HTTP proxies by default (vLLM is local)."""
    def __init__(self,
                  base_url: Optional[str] = None,
                  api_key: Optional[str] = None,
                  model: Optional[str] = None,
                  temperature: float = 0.1,
                  max_tokens: int = 6000):  # generous; Qwen3 uses tokens for <think> too
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "local")
        self.model = model or os.environ.get("AGENT_MODEL", "qwen3-8b")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _post_chat(self, messages: list[dict]) -> str:
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        # Build a proxy-bypassing opener (vLLM is on localhost)
        proxy_handler = urllib.request.ProxyHandler({})  # empty dict = no proxy
        opener = urllib.request.build_opener(proxy_handler)
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with opener.open(req, timeout=300) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        return obj["choices"][0]["message"]["content"]

    def decide(self, ctx: AgentContext) -> AgentDecision:
        user_prompt = build_user_prompt(ctx)
        text = self._post_chat([
            {"role": "system", "content": build_system_prompt(
                ctx.pruning_mode, ctx.n_layers, ctx.recovery_options)},
            {"role": "user", "content": user_prompt},
        ])
        obj = parse_json_response(text)
        return decision_from_json(obj, ctx, note=f"qwen raw response: {text[:120]}")
