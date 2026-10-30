"""Agent ABC + context/decision dataclasses.

Any backend (Qwen, GPT, Claude, rule-based) implements Agent.decide(ctx)
to produce a structured AgentDecision.
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


def coerce_mask(raw, n_layers: Optional[int] = None) -> list[int]:
    """Tolerantly coerce an LLM ``mask`` field into a sorted, de-duplicated int list.

    A single malformed mask must NOT crash the whole run (``[int(x) for x in "22-29"]``
    iterates characters and dies on ``'-'``). Accepts a list of ints / numeric strings /
    ``"a-b"`` inclusive-range strings, or a single ``"22-29"`` / ``"2,3,4"`` string.
    Non-numeric tokens are skipped; unparseable input yields ``[]``.

    Indices are bounded to ``[0, n_layers)`` when ``n_layers`` is given (negatives are
    always dropped): a coerced range like ``"30-39"`` must NOT reach prune/eval as an
    out-of-bounds index — that crashes ``prune_model`` (subprocess) or ``self.imp[mask]``
    (mock IndexError / negative-index silent wrong SR).
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = re.split(r"[,\s]+", raw.strip())
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    out: set[int] = set()
    for tok in raw:
        if isinstance(tok, bool):
            continue
        if isinstance(tok, (int, float)):
            out.add(int(tok))
            continue
        s = str(tok).strip()
        if not s:
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)   # "22-29" inclusive range
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out.update(range(min(a, b), max(a, b) + 1))
            continue
        try:
            out.add(int(float(s)))
        except (TypeError, ValueError):
            continue
    hi = n_layers if isinstance(n_layers, int) and n_layers > 0 else None
    return sorted(i for i in out if i >= 0 and (hi is None or i < hi))


def coerce_probe_queries(raw, n_layers: Optional[int] = None) -> list[dict]:
    """Normalize an LLM ``probe_queries`` field into ``[{"probe": str, "layers": [int]}]``.

    Mirrors :func:`coerce_mask`'s tolerance: drops non-dict entries, keeps only in-range
    layer indices, and tolerates each layer token individually (one bad element like
    ``"9a"`` no longer discards the whole query). ``n_layers`` bounds the layer indices
    (defaults to no upper bound) so this isn't hardcoded to 32-layer models.
    """
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    hi = n_layers if isinstance(n_layers, int) and n_layers > 0 else None
    for q in raw:
        if not isinstance(q, dict):
            continue
        probe = str(q.get("probe", "")).strip()
        layers: list[int] = []
        for x in (q.get("layers") or []):
            try:
                v = int(x)
            except (TypeError, ValueError):
                continue
            if v >= 0 and (hi is None or v < hi):
                layers.append(v)
        if probe and layers:
            out.append({"probe": probe, "layers": layers})
    return out


@dataclass
class AgentContext:
    """What the agent sees on each iteration."""
    iter_idx: int
    rules: list[str]                          # accumulated rules so far
    last_proposal: Optional[dict] = None      # what agent proposed last iter
    last_result: Optional[dict] = None        # SR + metadata
    last_diagnoses: list[dict] = field(default_factory=list)  # patterns triggered by last result
    probe_summary: dict = field(default_factory=dict)
    recovery_options: list[dict] = field(default_factory=list)
    tried_masks: list[dict] = field(default_factory=list)
    search_space: dict = field(default_factory=dict)
    suite: str = ""
    dense_sr: float = 0.0
    n_layers: int = 32                        # layer count; bounds mask / probe-query indices
    pruning_mode: str = "contiguous"          # "contiguous" -> mask must be one block;
                                              # "skip" -> non-adjacent layers allowed
    # Transferable methodology priors (rules only, no current-model data):
    # which probes are reliable / which mislead, anchor-identification recipe,
    # hard warnings, qualitative rules-of-thumb. Agent applies these to the
    # raw per-layer probe table to derive its OWN role labels + mask choices.
    probe_reliability: dict = field(default_factory=dict)
    anchor_recipe: dict = field(default_factory=dict)
    warnings: list[dict] = field(default_factory=list)
    qualitative_landmarks: dict = field(default_factory=dict)
    metric_semantics: dict = field(default_factory=dict)
    # Per-suite STATIC facts the agent gets WITHOUT querying a probe — currently
    # idx + single_delete_training_free_sr per layer (allowed pre-loaded
    # exception per project policy). For NEW models, single_delete must also be
    # measured first (32-layer ablation sweep).
    layer_static_facts: list[dict] = field(default_factory=list)
    # Catalog of probes the agent can call this turn (from ProbeQueryService).
    available_probes: list[dict] = field(default_factory=list)
    # Probe results accumulated across this iter's probe-query rounds.
    # Shape: {probe_name: {layer_id: value}}
    probe_results: dict = field(default_factory=dict)
    # Probe-query round counter this iter (driver increments).
    probe_round_idx: int = 0
    probe_rounds_max: int = 3
    # Best (mask, recovery) → SR observed so far across all tried masks with
    # non-null SR. None if no iter has produced a real SR yet.
    best_so_far: Optional[dict] = None
    # Backward-compat shim: old callers may still reference per_layer_table.
    # Kept as empty list — no longer populated.
    per_layer_table: list[dict] = field(default_factory=list)
    # ⑤ z-score safety ranking of layers by training-free single-delete SR
    # (safest-to-prune first; HARD-ANCHOR flagged). Decision-time evidence so
    # anchor priority is an INPUT, not just a post-hoc diagnosis.
    layer_safety_table: list[dict] = field(default_factory=list)
    # ③ QD archive map: per behavioral-cell champion (mask, sr) — the tradeoff
    # map, so the agent sees more than one global best.
    archive_map: list[dict] = field(default_factory=list)


@dataclass
class AgentDecision:
    """What the agent outputs per turn (either a probe-query round OR a final mask commit)."""
    mask: list[int]                           # empty list = "still investigating, see probe_queries"
    recovery: str                             # recipe_id
    rationale: str
    success_gate: float                       # expected SR target for this proposal (when mask committed)
    stop_gate: float                          # below this = abort related directions
    evidence_used: list[str]                  # which rules / probes were cited
    accept_diagnoses: list[str] = field(default_factory=list)
    note: str = ""
    # Tool-use channel: agent can request probe values BEFORE committing a mask.
    # Each entry: {"probe": "<name>", "layers": [int, ...]}
    # If non-empty AND mask is empty AND probe_round_idx < probe_rounds_max,
    # driver runs the queries and re-calls the agent.
    probe_queries: list[dict] = field(default_factory=list)


class Agent(ABC):
    """Abstract agent. Subclass to back with a specific LLM (or rule engine)."""

    @abstractmethod
    def decide(self, ctx: AgentContext) -> AgentDecision:
        ...

    @property
    def name(self) -> str:
        return type(self).__name__


def decision_from_json(obj, ctx: AgentContext, *, note="") -> AgentDecision:
    """Shared provider-independent response contract; invalid proposals are retried."""
    import math
    if not isinstance(obj,dict):
        raise ValueError('Agent response must be a JSON object')
    def number(name):
        value=obj.get(name,0.)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError(f'{name} must be a JSON number in [0,1]; put explanations in rationale')
        return float(value)
    mask=obj.get('mask',[])
    if not isinstance(mask,list) or any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<ctx.n_layers for x in mask):
        raise ValueError(f'mask must be an array of integer layer indices in [0,{ctx.n_layers-1}]')
    if len(set(mask))!=len(mask):
        raise ValueError('mask must contain unique indices')
    queries=obj.get('probe_queries',[])
    if not isinstance(queries,list):raise ValueError('probe_queries must be an array')
    for query in queries:
        if not isinstance(query,dict) or not isinstance(query.get('probe'),str) or not isinstance(query.get('layers'),list):
            raise ValueError('Each probe query needs a probe string and layers integer array')
        if any(isinstance(x,bool) or not isinstance(x,int) or not 0<=x<ctx.n_layers for x in query['layers']):
            raise ValueError('Probe layer indices are out of range or not integers')
    def strings(name):
        value=obj.get(name,[])
        if not isinstance(value,list) or any(not isinstance(x,str) for x in value):
            raise ValueError(f'{name} must be an array of strings')
        return value
    recovery=obj.get('recovery','')
    if not isinstance(recovery,str):raise ValueError('recovery must be a string')
    allowed={r['id'] for r in ctx.recovery_options}
    if mask and allowed and recovery not in allowed:
        raise ValueError(f'recovery must be in {sorted(allowed)}')
    return AgentDecision(mask=sorted(mask),recovery=recovery,rationale=str(obj.get('rationale','')),
        success_gate=number('success_gate'),stop_gate=number('stop_gate'),
        evidence_used=strings('evidence_used'),accept_diagnoses=strings('accept_diagnoses'),
        probe_queries=queries,note=note)
