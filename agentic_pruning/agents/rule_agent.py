"""Deterministic synthetic demonstration; never a real-model search agent.

Enumerate untried contiguous masks and the declared recovery menu using the
current model depth. No planted OFT layer numbers or assumed measured outcomes
are needed to exercise the shared driver.
"""
from .base import Agent, AgentContext, AgentDecision


class RuleAgent(Agent):
    def decide(self, ctx: AgentContext) -> AgentDecision:
        n = ctx.n_layers
        if n < 2 or not ctx.recovery_options:
            raise ValueError('Synthetic demo requires at least two layers and a recovery menu')
        recipes = [r['id'] for r in ctx.recovery_options]
        tried = {(tuple(row['mask']), row.get('recovery')) for row in ctx.tried_masks}
        first_width = max(1, n // 4)
        widths = list(range(first_width, n)) + list(range(1, first_width))
        for width in widths:
            starts = sorted(range(n-width+1), key=lambda start: (abs(start-(n-width)/2), start))
            for start in starts:
                mask = list(range(start,start+width))
                for recipe in recipes:
                    if (tuple(mask),recipe) not in tried:
                        return AgentDecision(mask=mask,recovery=recipe,
                            rationale='Synthetic demonstration: try the next unmeasured mask and declared recipe.',
                            success_gate=0.9,stop_gate=0.5,evidence_used=[],
                            accept_diagnoses=[d['pattern_id'] for d in ctx.last_diagnoses])
        raise ValueError('Synthetic demo has exhausted the declared search space')
