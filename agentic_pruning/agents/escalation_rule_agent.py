"""EscalationRuleAgent — hand-coded cheap-first / escalate-on-failure agent (no LLM).

Built for the open_menu_replay experiment (2026-07-16). Unlike RuleAgent (which
hardcodes ONE recovery per iteration index), this agent's recovery choice is a
FEEDBACK-DRIVEN policy over the open recovery menu:

  Ladder    = the recovery menu in file order (recovery_recipes_open.json orders
              recipes cheap -> expensive: direct < adapter_mlp_0.5M < adapter_only
              < adapter+head+proprio).
  (P1) On a NEW mask start at the cheapest plausible rung ("cheap first"):
       - very first mask: rung 0 (direct, zero cost);
       - later masks: WARM START at the rung that was SUFFICIENT (SR >= target)
         on the previous, smaller mask. Documented heuristic: "recovery need is
         monotone non-decreasing in k" — do not re-pay failures already observed
         at smaller k.
  (P2) If SR < target (0.90): ESCALATE one rung on the SAME mask (do NOT change
       the mask — recovery insufficiency is not evidence against the mask).
  (P3) If SR is None (offline cache miss): the driver's result carries
       `known_recoveries_for_mask` (which recoveries have a cached 200-trial SR
       for this mask). Escalate to the CHEAPEST cached rung above the current
       one; if none is cached, escalate one rung blindly; if already at the top,
       give the mask up and advance.
  (P4) If SR >= target: mask validated — GROW to the next mask in the curriculum
       (P1 warm start applies).
  (P5) If the TOP rung still misses target: the mask is infeasible even with the
       strongest recovery — record and fall back to the best validated config.

HONESTY NOTE (must be reported as such in the paper): every part of this policy
— the cost ladder, the escalate-on-failure rule, the warm-start heuristic, the
cache-aware step (P3) and the mask curriculum below — is EXPLICITLY HAND-CODED
framework policy. Nothing here is emergent LLM behavior. The mask curriculum
(k8 tail -> k16 mid -> k28 keep-anchors -> k29 L30-ablation) is copied verbatim
from RuleAgent so that iteration counts are comparable with the pinned-menu
paper_comparison.md runs; only the RECOVERY-selection logic differs.
"""
from .base import Agent, AgentContext, AgentDecision

TARGET_SR = 0.90  # experiment goal: maximize deleted layers subject to SR >= 0.90

# Mask curriculum — identical masks to agents/rule_agent.py for comparability.
CURRICULUM: list[list[int]] = [
    list(range(22, 30)),  # stage 0: k=8  late block [22-29]
    list(range(8, 24)),   # stage 1: k=16 mid block  [8-23]
    list(range(2, 30)),   # stage 2: k=28 keep anchors L0/L1/L30/L31
    list(range(2, 31)),   # stage 3: k=29 ablation — also drop L30 (anchor test)
]
STAGE_WHY = [
    "late layers most often redundant; k=8 [22-29] is the cheapest structural probe",
    "k=8 validated; push depth to k=16 mid-block [8-23]",
    "k=16 validated; extreme k=28 [2-29] keeping probe anchors L0/L1/L30/L31",
    "k=28 validated; ablation: also drop L30 to falsify/confirm the anchor claim",
]


class EscalationRuleAgent(Agent):
    def decide(self, ctx: AgentContext) -> AgentDecision:
        accept_all = [d["pattern_id"] for d in ctx.last_diagnoses]
        ladder = [r["id"] for r in ctx.recovery_options]  # menu order = cost order
        if not ladder:
            raise RuntimeError("empty recovery menu")

        def propose(mask, rung, why, evidence):
            rung = max(0, min(rung, len(ladder) - 1))
            return AgentDecision(
                mask=list(mask), recovery=ladder[rung], rationale=why,
                success_gate=TARGET_SR, stop_gate=0.50,
                evidence_used=evidence, accept_diagnoses=accept_all,
            )

        tried = [t for t in ctx.tried_masks if t.get("mask")]
        if not tried:
            # (P1) very first proposal: first curriculum mask, CHEAPEST rung.
            return propose(
                CURRICULUM[0], 0,
                f"Cheap-first policy (P1): new mask [{CURRICULUM[0][0]}-{CURRICULUM[0][-1]}] "
                f"(k={len(CURRICULUM[0])}, {STAGE_WHY[0]}); start with the cheapest recovery "
                f"'{ladder[0]}' ({self._cost(ctx, 0)}) — only pay for training if SR proves it necessary.",
                [f"dense_sr={ctx.dense_sr}", "recovery menu ordered cheap->expensive"],
            )

        last = tried[-1]
        last_mask = sorted(last["mask"])
        last_sr = last.get("sr")
        rung = ladder.index(last["recovery"]) if last["recovery"] in ladder else 0
        stage = self._stage_of(last_mask)
        top = len(ladder) - 1

        # (P3) cache miss — steer toward recoveries that HAVE a cached datapoint.
        if last_sr is None:
            known = (ctx.last_result or {}).get("known_recoveries_for_mask", []) or []
            known_higher = sorted(ladder.index(r) for r in known
                                  if r in ladder and ladder.index(r) > rung)
            if known_higher:
                nxt = known_higher[0]
                return propose(
                    last_mask, nxt,
                    f"Escalation on cache miss (P3): ({self._m(last_mask)}, {ladder[rung]}) has no "
                    f"cached closed-loop SR. Cheapest recovery ABOVE the current rung with a cached "
                    f"200-trial result is '{ladder[nxt]}' ({self._cost(ctx, nxt)}) — escalate to it "
                    f"rather than changing the mask or paying for an un-cached run.",
                    [f"eval cache miss for ({self._m(last_mask)}, {ladder[rung]})",
                     f"known_recoveries_for_mask={known}"],
                )
            if rung < top:
                return propose(
                    last_mask, rung + 1,
                    f"Escalation on cache miss (P3, blind): no cached recovery above rung "
                    f"'{ladder[rung]}' for {self._m(last_mask)}; escalate one rung to "
                    f"'{ladder[rung + 1]}'.",
                    [f"eval cache miss for ({self._m(last_mask)}, {ladder[rung]})"],
                )
            # top rung and still nothing cached — give up on this mask
            return self._advance(ctx, stage, rung, propose,
                                 reason=f"{self._m(last_mask)} has no cached SR even at the top rung")

        # Gate-aware failure test (final protocol, 2026-07-21): when the driver
        # ran the Wilson three-zone gate, its verdict REPLACES the point
        # comparison — 'fail' (UCB < tau) and 'uncertain' (pre-registered fail
        # at max budget, or real re-eval pending; the search may still climb a
        # rung from it, paper §3.5) both count as failure; only 'pass'
        # (LCB >= tau) validates the cell. Without a gate, historical
        # point-estimate behavior is byte-identical.
        gate = ((ctx.last_result or {}).get("gate") or {})
        gate_verdict = gate.get("verdict")
        if gate_verdict:
            fin = gate["chain"][-1]
            gate_note = (f"gate[{gate.get('interval')}@{fin['n']}ep] "
                         f"CI=({fin['lcb']:.3f},{fin['ucb']:.3f}) vs tau={gate.get('tau')} "
                         f"-> {gate_verdict}")
            failed = (gate_verdict != "pass")
        else:
            gate_note = ""
            failed = (last_sr < TARGET_SR)

        # (P2) failed the acceptance rule -> escalate on the SAME mask.
        if failed:
            if rung < top:
                why_failed = (gate_note if gate_verdict
                              else f"SR={last_sr:.3f} < target {TARGET_SR}")
                return propose(
                    last_mask, rung + 1,
                    f"Escalation (P2): {self._m(last_mask)} with '{ladder[rung]}' failed acceptance "
                    f"({why_failed}). Recovery under-recovered — keep the mask, "
                    f"upgrade recovery one rung to '{ladder[rung + 1]}' ({self._cost(ctx, rung + 1)}).",
                    [f"iter {last.get('iter', '?')}: ({self._m(last_mask)}, {ladder[rung]}) -> SR={last_sr:.3f}"]
                    + ([gate_note] if gate_note else []),
                )
            # (P5) top rung failed -> mask infeasible.
            best = ctx.best_so_far or {}
            fallback = best.get("mask") or CURRICULUM[max(0, stage - 1)]
            return AgentDecision(
                mask=list(fallback), recovery=best.get("recovery", ladder[top]),
                rationale=(
                    f"Top-rung failure (P5): {self._m(last_mask)} misses target even with the strongest "
                    f"recovery '{ladder[top]}' (SR={last_sr:.3f}) — the mask itself is infeasible "
                    f"(anchor violated), not the recovery. Fall back to best validated config "
                    f"{self._m(sorted(fallback))} @ SR={best.get('sr')}."),
                success_gate=TARGET_SR, stop_gate=0.50,
                evidence_used=[f"({self._m(last_mask)}, {ladder[top]}) -> SR={last_sr:.3f} < {TARGET_SR}"],
                accept_diagnoses=accept_all,
                note="converged: escalation exhausted on the ablation mask",
            )

        # (P4) accepted -> grow to next curriculum stage, warm-started (P1).
        reason = f"{self._m(last_mask)} validated at SR={last_sr:.3f} with '{ladder[rung]}'"
        if gate_note:
            reason += f" ({gate_note})"
        return self._advance(ctx, stage, rung, propose, reason=reason)

    # ------------------------------------------------------------------ utils
    def _advance(self, ctx, stage, warm_rung, propose, reason: str):
        ladder = [r["id"] for r in ctx.recovery_options]
        if stage + 1 < len(CURRICULUM):
            nxt_mask = CURRICULUM[stage + 1]
            return propose(
                nxt_mask, warm_rung,
                f"Grow (P4): {reason}. Next stage: {STAGE_WHY[stage + 1]}. Warm start (P1) at "
                f"'{ladder[min(warm_rung, len(ladder) - 1)]}' — the rung that was sufficient at the "
                f"previous (smaller) mask; recovery need is assumed monotone non-decreasing in k.",
                [reason, "warm-start heuristic: monotone recovery need in k"],
            )
        # curriculum exhausted — re-state the best config (converged).
        best = ctx.best_so_far or {}
        fallback = best.get("mask") or CURRICULUM[-2]
        return propose(
            fallback, len(ladder) - 1,
            f"Converged: curriculum exhausted ({reason}). Best validated config "
            f"{self._m(sorted(fallback))} @ SR={best.get('sr')}.",
            [reason],
        )

    @staticmethod
    def _stage_of(mask_sorted: list[int]) -> int:
        for s, m in enumerate(CURRICULUM):
            if sorted(m) == mask_sorted:
                return s
        return 0

    @staticmethod
    def _m(mask_sorted) -> str:
        if not mask_sorted:
            return "[]"
        return f"[{mask_sorted[0]}-{mask_sorted[-1]}](k={len(mask_sorted)})"

    @staticmethod
    def _cost(ctx: AgentContext, rung: int) -> str:
        opts = ctx.recovery_options
        if 0 <= rung < len(opts):
            for key in ("cost_estimate",):
                if key in opts[rung]:
                    return f"cost {opts[rung][key]}"
            desc = opts[rung].get("description", "")
            return desc[:40]
        return "?"
