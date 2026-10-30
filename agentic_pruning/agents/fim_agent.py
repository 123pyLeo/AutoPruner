"""FimRankingAgent — static-proxy baseline (contiguous-segment version).

Sliding-window FIM: for each k ∈ {8, 12, 16, 20, 24, 28}, picks the
contiguous block of length k whose mean FIM is lowest (= "block of most
removable layers" by FIM). Ignores closed-loop feedback and diagnoses.
Always uses adapter+head+proprio recovery (strongest option) so the
COMPARISON with closed-loop LLM agent is fair: same recovery + same
contiguous-block constraint, different selection method (static FIM vs.
LLM reasoning over multi-probe evidence).

Paper purpose: shows that FIM ranking — popular in NLP layer pruning —
fails on VLA because it ranks the format-aligner / final-calibration
anchors as lowest importance, while closed-loop evidence shows they
are critical.
"""
from .base import Agent, AgentContext, AgentDecision


def lowest_fim_window(fim: list[float], k: int) -> int:
    """Return start index of the contiguous window of length k with the
    smallest mean FIM. Assumes len(fim) >= k."""
    best_start, best_mean = 0, float("inf")
    for start in range(0, len(fim) - k + 1):
        m = sum(fim[start:start + k]) / k
        if m < best_mean:
            best_mean = m
            best_start = start
    return best_start


class FimRankingAgent(Agent):
    def __init__(self, fim_per_layer: list[float]):
        self.fim = list(fim_per_layer)

    def decide(self, ctx: AgentContext) -> AgentDecision:
        ks = [8, 12, 16, 20, 24, 28]
        k = ks[ctx.iter_idx] if ctx.iter_idx < len(ks) else ks[-1]
        start = lowest_fim_window(self.fim, k)
        mask = list(range(start, start + k))
        window_mean = sum(self.fim[start:start + k]) / k
        recovery = "adapter+head+proprio"
        rationale = (
            f"Static FIM sliding-window baseline. Contiguous k={k} block with "
            f"lowest mean FIM: start={start}, mask={mask}, mean_fim={window_mean:.4f}."
        )
        evidence = [f"FIM sliding window k={k}: start={start} (mean FIM {window_mean:.4f})"]
        return AgentDecision(
            mask=mask, recovery=recovery, rationale=rationale,
            success_gate=0.90, stop_gate=0.50,
            evidence_used=evidence, accept_diagnoses=[],
            note="static FIM sliding-window agent; does not adapt to closed-loop feedback",
        )
