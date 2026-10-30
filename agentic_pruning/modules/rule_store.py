"""RuleStore — structured, self-sharpening, cross-model-transferable rule library.

Design: ledger related_work_mining_2026-05-30 §4 (④ 规则库升级).
Four meshed pieces:
  (a) structured schema (Robo-Cortex 5-tuple ρ,d,a,c,y instead of free text);
  (b) count-based Laplace confidence (BrainMem (s+1)/(t+2), new rule = 0.5);
  (c) self-pruning + contradiction detection (NAS-4a support∧¬conflict, BrainMem
      low-utility eviction) — treats the "t9 +16pt was a hybrid-head artifact"
      class of stale rules;
  (d) scope tiering for cross-run transfer: methodology-level = transferable
      (search-methodology) → global rule_store.json; model_specific-level =
      non-transferable (L30 / tf-SR anatomy) → rule_store_<model_id>.json.

Stdlib only (json / pathlib / collections / typing).

Each rule is a dict::

    {
      "rho": str,                      # pattern id, e.g. "sr_cliff_detected"
      "problem": str,                  # problem description d
      "strategy": str,                 # suggested action a
      "label": "success" | "failure", # y
      "confidence": float,             # c, Laplace-smoothed
      "n_success": int,
      "n_total": int,
      "scope": {"level": "methodology"|"model_specific",
                "model_id": str|None, "anchor_layer": int|None},
      "evidence": str,
      "status": "active" | "retired",
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# --- tuning constants (ledger §4c/§4 risk note) ----------------------------
RETIRE_NEG_MIN = 2            # (n_total - n_success) >= this …
RETIRE_CONF_MAX = 0.4        # … AND confidence < this → retire
CAP_M = 15                   # capacity: at most this many active rules
PROTECT_CONF = 0.8           # protected-from-eviction confidence floor …
PROTECT_NSUCCESS = 3         # … AND n_success floor (ledger: N_success>=5 too high → 3)
DEFAULT_DENSE_SR = 0.965
SR_CLIFF_MARGIN = 0.10       # anchor confirmation/refutation margin vs dense_sr

# rho values that reconcile() acts on (anchor-class rules keyed on a layer).
ANCHOR_RHOS = {"sr_cliff_detected", "anchor_violation", "inverted_anchor_priority"}

# substrings that flag a seed string as encoding a FAILURE-derived lesson.
_FAILURE_TOKENS = ("anchor", "never", "collapse")


def laplace_confidence(n_success: int, n_total: int) -> float:
    """BrainMem Laplace-smoothed confidence; new rule (0,0) -> 0.5."""
    return (n_success + 1) / (n_total + 2)


def _dedup_key(rule: dict) -> tuple:
    """Dedup key = (rho, scope.anchor_layer, problem[:60])."""
    scope = rule.get("scope") or {}
    return (rule.get("rho"),
            scope.get("anchor_layer"),
            (rule.get("problem") or "")[:60])


class RuleStore:
    def __init__(self, store_dir, model_id: str):
        self.store_dir = Path(store_dir)
        self.model_id = model_id
        self.dense_sr = DEFAULT_DENSE_SR
        self._rules: list[dict] = []
        self.load()

    # ----- dense-SR injection (reconcile can't take it in its signature) ----
    def set_dense_sr(self, v: float) -> None:
        self.dense_sr = float(v)

    # ----- internal helpers -------------------------------------------------
    def _index(self) -> dict[tuple, dict]:
        """Map dedup-key -> rule (last wins) for the in-memory list."""
        return {_dedup_key(r): r for r in self._rules}

    def _merge_in(self, rule: dict) -> dict:
        """Insert rule if its dedup key is unseen; otherwise keep existing.

        Returns the rule object that now lives in the store for that key.
        """
        key = _dedup_key(rule)
        existing = self._index().get(key)
        if existing is not None:
            return existing
        self._rules.append(rule)
        return rule

    @staticmethod
    def _norm_scope(scope: Optional[dict]) -> dict:
        scope = dict(scope or {})
        return {
            "level": scope.get("level", "model_specific"),
            "model_id": scope.get("model_id"),
            "anchor_layer": scope.get("anchor_layer"),
        }

    def _refresh_confidence(self, rule: dict) -> None:
        rule["confidence"] = laplace_confidence(rule["n_success"], rule["n_total"])

    # ----- seeding existing free-text rules ---------------------------------
    def seed(self, rule_strings: list[str]) -> None:
        """Register existing seed strings (SEED_RULES + derived anchor rules) as
        methodology-level active rules. Label guessed from content:
        containing "anchor"/"NEVER"/"collapse" -> failure, else success. Deduped.
        """
        for s in rule_strings:
            if not s or not s.strip():
                continue
            text = s.strip()
            low = text.lower()
            label = "failure" if any(t in low for t in _FAILURE_TOKENS) else "success"
            # derive a stable rho from a leading [tag] or "name:" prefix.
            rho = self._rho_from_seed(text)
            rule = {
                "rho": rho,
                "problem": text,
                "strategy": text,
                "label": label,
                "n_success": 0,
                "n_total": 0,
                "confidence": laplace_confidence(0, 0),
                "scope": {"level": "methodology", "model_id": None,
                          "anchor_layer": None},
                "evidence": "seed",
                "status": "active",
            }
            self._merge_in(rule)

    @staticmethod
    def _rho_from_seed(text: str) -> str:
        t = text.strip()
        if t.startswith("[") and "]" in t:
            return t[1:t.index("]")].strip()
        if ":" in t:
            head = t.split(":", 1)[0].strip()
            # only treat as an id if it looks like one (no spaces-heavy prose)
            if head and len(head.split()) <= 4:
                return head
        return "seed_rule"

    # ----- agent-emitted diagnoses -----------------------------------------
    def upsert(self, diag: dict) -> None:
        """Insert a diagnosis as a new rule (n_success=n_total=0 -> conf=0.5).

        diag carries rho/problem/strategy/label/scope/evidence. Deduped on
        (rho, scope.anchor_layer, problem[:60]); existing rules are left intact.
        """
        rule = {
            "rho": diag.get("rho", "unknown"),
            "problem": diag.get("problem", ""),
            "strategy": diag.get("strategy", ""),
            "label": diag.get("label", "failure"),
            "n_success": 0,
            "n_total": 0,
            "confidence": laplace_confidence(0, 0),
            "scope": self._norm_scope(diag.get("scope")),
            "evidence": diag.get("evidence", ""),
            "status": "active",
        }
        self._merge_in(rule)

    # ----- reconciliation against fresh closed-loop results -----------------
    def reconcile(self, result: dict, mask: list[int], history: list[dict]) -> None:
        """Confirm / refute anchor-class rules with a fresh result.

        Only acts on rules whose rho is in ANCHOR_RHOS and whose
        scope.anchor_layer is not None. If the new `mask` INCLUDES that
        anchor_layer (i.e. the anchor was deleted):
          - result["sr"] not None and < dense_sr - 0.10 → CONFIRMED
            (deleting the anchor really crashed SR): n_success+=1, n_total+=1.
          - result["sr"] not None and >= dense_sr - 0.10 → REFUTED
            (deleted the anchor but SR held — recovery rescued it):
            n_total+=1 only (a counter-example in the denominator).
        Confidence recomputed. Does NOT auto-prune (caller calls prune()).
        """
        sr = result.get("sr")
        if sr is None:
            return
        mask_set = set(mask or [])
        floor = self.dense_sr - SR_CLIFF_MARGIN
        for rule in self._rules:
            if rule.get("status") != "active":
                continue
            if rule.get("rho") not in ANCHOR_RHOS:
                continue
            anchor = (rule.get("scope") or {}).get("anchor_layer")
            if anchor is None:
                continue
            if anchor not in mask_set:
                continue  # anchor wasn't deleted in this run → no evidence
            if sr < floor:
                rule["n_success"] += 1
                rule["n_total"] += 1
            else:
                rule["n_total"] += 1
            self._refresh_confidence(rule)

    # ----- self-pruning + contradiction detection ---------------------------
    def prune(self) -> None:
        # (1) low-utility retirement.
        for rule in self._rules:
            if rule.get("status") != "active":
                continue
            neg = rule["n_total"] - rule["n_success"]
            if neg >= RETIRE_NEG_MIN and rule["confidence"] < RETIRE_CONF_MAX:
                rule["status"] = "retired"

        # (2) contradiction detection per anchor_layer: a "keep" (success) and a
        # "delete" (failure) active rule on the SAME layer conflict → keep the
        # higher-confidence one, retire the other.
        by_layer: dict[Any, list[dict]] = {}
        for rule in self._rules:
            if rule.get("status") != "active":
                continue
            anchor = (rule.get("scope") or {}).get("anchor_layer")
            if anchor is None:
                continue
            by_layer.setdefault(anchor, []).append(rule)
        for anchor, rules in by_layer.items():
            successes = [r for r in rules if r["label"] == "success"]
            failures = [r for r in rules if r["label"] == "failure"]
            if successes and failures:
                winner = max(rules, key=lambda r: r["confidence"])
                for r in rules:
                    if r is winner:
                        continue
                    # only retire the opposing-label rules vs the winner
                    if r["label"] != winner["label"]:
                        r["status"] = "retired"

        # (3) capacity: retire lowest-confidence active rules beyond CAP_M,
        # protecting confidence>=0.8 AND n_success>=3.
        active = [r for r in self._rules if r.get("status") == "active"]
        if len(active) <= CAP_M:
            return
        protected = [r for r in active
                     if r["confidence"] >= PROTECT_CONF
                     and r["n_success"] >= PROTECT_NSUCCESS]
        evictable = [r for r in active if r not in protected]
        # keep slots for protected ones; how many evictable may stay
        slots = CAP_M - len(protected)
        if slots < 0:
            slots = 0
        # ascending confidence → the lowest are retired first
        evictable_sorted = sorted(evictable, key=lambda r: r["confidence"])
        n_to_retire = len(evictable) - slots
        for r in evictable_sorted[:max(0, n_to_retire)]:
            r["status"] = "retired"

    # ----- rendering for the prompt -----------------------------------------
    def render(self, max_m: int = 15) -> list[str]:
        visible = [
            r for r in self._rules
            if r.get("status") == "active"
            and ((r["scope"].get("level") == "methodology")
                 or (r["scope"].get("model_id") == self.model_id))
        ]
        visible.sort(key=lambda r: r["confidence"], reverse=True)
        out: list[str] = []
        for r in visible[:max_m]:
            tag = "KEEP" if r["label"] == "success" else "AVOID"
            line = (f"[{r['rho']} | conf={r['confidence']:.2f} "
                    f"n={r['n_total']} | {tag}] {r['strategy']} "
                    f"(ctx: {r['problem']})")
            if r["scope"].get("level") == "methodology":
                line += " (transferable)"
            out.append(line)
        return out

    # ----- persistence ------------------------------------------------------
    def _methodology_path(self) -> Path:
        return self.store_dir / "rule_store.json"

    def _model_path(self) -> Path:
        return self.store_dir / f"rule_store_{self.model_id}.json"

    def persist(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        methodology = [r for r in self._rules
                       if r["scope"].get("level") == "methodology"]
        model_specific = [r for r in self._rules
                          if r["scope"].get("level") != "methodology"]
        with open(self._methodology_path(), "w") as f:
            json.dump(methodology, f, indent=2)
        with open(self._model_path(), "w") as f:
            json.dump(model_specific, f, indent=2)

    def load(self) -> None:
        """Read both files (if present) and merge into memory, deduped."""
        loaded: list[dict] = []
        for path in (self._methodology_path(), self._model_path()):
            if path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    data = []
                if isinstance(data, list):
                    loaded.extend(data)
        # merge loaded on top of whatever is already in memory, deduped.
        merged: dict[tuple, dict] = {}
        for r in self._rules:
            merged[_dedup_key(r)] = r
        for r in loaded:
            r = self._normalize_loaded(r)
            merged.setdefault(_dedup_key(r), r)
        self._rules = list(merged.values())

    @staticmethod
    def _normalize_loaded(r: dict) -> dict:
        r = dict(r)
        r["scope"] = RuleStore._norm_scope(r.get("scope"))
        r.setdefault("n_success", 0)
        r.setdefault("n_total", 0)
        r.setdefault("label", "failure")
        r.setdefault("status", "active")
        r.setdefault("evidence", "")
        r.setdefault("problem", "")
        r.setdefault("strategy", "")
        r.setdefault("rho", "unknown")
        if "confidence" not in r:
            r["confidence"] = laplace_confidence(r["n_success"], r["n_total"])
        return r

    # ----- introspection ----------------------------------------------------
    def all_rules(self) -> list[dict]:
        """Deep-ish copy of the internal rule list (for tests / debugging)."""
        out = []
        for r in self._rules:
            c = dict(r)
            c["scope"] = dict(r["scope"])
            out.append(c)
        return out


# ===========================================================================
# self-test
# ===========================================================================
if __name__ == "__main__":
    import shutil

    smoke_dir = Path("/tmp/rs_smoke")
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)

    MID = "openvla-7b-oft-finetuned-libero-object"
    rs = RuleStore(smoke_dir, model_id=MID)

    # 1) seed a few strings → render non-empty
    seeds = [
        "closed_loop_dominates_proxy: closed-loop robot SR is the final judge; "
        "static metrics can only rank candidates, not decide.",
        "[training_free_hard_anchor] Layers L30(SR=0.07) collapse SR — NEVER "
        "include in a remove-mask without an explicit ablation hypothesis.",
        "[recovery_insufficient] For mask_size>=16, propose joint recovery.",
    ]
    rs.seed(seeds)
    rendered = rs.render()
    print("=== after seed ===")
    for line in rendered:
        print("  ", line)
    assert rendered, "render empty after seed"
    # labels: anchor/NEVER/collapse → failure; others → success
    labels = {r["rho"]: r["label"] for r in rs.all_rules()}
    print("  seed labels:", labels)
    assert labels["training_free_hard_anchor"] == "failure"
    assert labels["closed_loop_dominates_proxy"] == "success"
    assert labels["recovery_insufficient"] == "success"

    # 2) upsert one anchor failure rule (rho=sr_cliff_detected, anchor_layer=30)
    rs.upsert({
        "rho": "sr_cliff_detected",
        "problem": "L30 cliff between near-twin masks at high k",
        "strategy": "Preserve L30 in future masks (anchor candidate).",
        "label": "failure",
        "scope": {"level": "model_specific", "model_id": MID, "anchor_layer": 30},
        "evidence": "k=28 SR=0.76 vs k=29 (adds L30) SR=0.07",
    })
    anchor_rule = next(r for r in rs.all_rules()
                       if r["rho"] == "sr_cliff_detected"
                       and r["scope"]["anchor_layer"] == 30)
    print("=== after upsert anchor failure (layer 30) ===")
    print("  conf:", anchor_rule["confidence"],
          "n_success:", anchor_rule["n_success"],
          "n_total:", anchor_rule["n_total"])
    assert abs(anchor_rule["confidence"] - 0.5) < 1e-9, "new rule conf must be 0.5"

    # set dense_sr explicitly (default already 0.965)
    rs.set_dense_sr(0.965)

    # 3) reconcile confirm twice (mask includes 30, sr low) → conf up, n_success=2
    for _ in range(2):
        rs.reconcile({"sr": 0.07}, mask=list(range(14, 31)), history=[])
    anchor_rule = next(r for r in rs.all_rules()
                       if r["rho"] == "sr_cliff_detected"
                       and r["scope"]["anchor_layer"] == 30)
    print("=== after 2 confirmations (deleted L30, SR=0.07 < 0.865) ===")
    print("  conf:", round(anchor_rule["confidence"], 4),
          "n_success:", anchor_rule["n_success"],
          "n_total:", anchor_rule["n_total"])
    assert anchor_rule["n_success"] == 2
    assert anchor_rule["n_total"] == 2
    assert anchor_rule["confidence"] > 0.5  # (2+1)/(2+2)=0.75

    conf_before_refute = anchor_rule["confidence"]
    # 4) reconcile refute once (mask includes 30, sr high) → n_total up, conf down
    rs.reconcile({"sr": 0.90}, mask=list(range(14, 31)), history=[])
    anchor_rule = next(r for r in rs.all_rules()
                       if r["rho"] == "sr_cliff_detected"
                       and r["scope"]["anchor_layer"] == 30)
    print("=== after 1 refutation (deleted L30, SR=0.90 >= 0.865) ===")
    print("  conf:", round(anchor_rule["confidence"], 4),
          "n_success:", anchor_rule["n_success"],
          "n_total:", anchor_rule["n_total"])
    assert anchor_rule["n_success"] == 2
    assert anchor_rule["n_total"] == 3       # (2+1)/(3+2)=0.6
    assert anchor_rule["confidence"] < conf_before_refute

    # 5) upsert a competing SUCCESS rule on the SAME anchor_layer=30, then prune.
    #    Give it more confirmations so the failure rule (conf 0.6) wins on
    #    contradiction detection — verify the lower one is retired.
    rs.upsert({
        "rho": "sr_cliff_detected",
        "problem": "L30 is removable — recovery rescues it, keep deletable",
        "strategy": "L30 may be removed (recovery covers it).",
        "label": "success",
        "scope": {"level": "model_specific", "model_id": MID, "anchor_layer": 30},
        "evidence": "observed SR held after deleting L30 once",
    })
    success_rule = next(r for r in rs.all_rules()
                        if r["rho"] == "sr_cliff_detected"
                        and r["scope"]["anchor_layer"] == 30
                        and r["label"] == "success")
    print("=== after upsert competing success rule (conf 0.5 < failure 0.6) ===")
    rs.prune()
    statuses = {(r["label"]): r["status"] for r in rs.all_rules()
                if r["rho"] == "sr_cliff_detected" and r["scope"]["anchor_layer"] == 30}
    print("  per-label status:", statuses)
    # failure conf=0.6 > success conf=0.5 → success retired, failure stays active
    assert statuses["failure"] == "active"
    assert statuses["success"] == "retired"

    # 6) persist → fresh RuleStore.load → all_rules count matches (round-trip)
    rs.persist()
    n_before = len(rs.all_rules())
    rs2 = RuleStore(smoke_dir, model_id=MID)
    n_after = len(rs2.all_rules())
    print("=== round-trip persist/load ===")
    print("  n_before:", n_before, "n_after:", n_after)
    assert n_before == n_after, f"round-trip mismatch {n_before} != {n_after}"

    print("RULESTORE-SMOKE OK")
