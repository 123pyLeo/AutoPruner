"""Wilson three-zone acceptance gate — final protocol (2026-07-21).

Paper §3.5 ("Acceptance under noise"): SR is a binomial estimate; every
evaluation runs at a pre-registered episode budget (200 or 500) and acceptance
uses the Wilson 95% interval:

  - PASS      if LCB >= tau
  - FAIL      if UCB <  tau
  - UNCERTAIN otherwise -> escalate budget one step (200 -> 500).
    A cell still uncertain at the maximum budget is, by pre-registration,
    treated as FAILING for acceptance purposes (the search may still climb a
    rung from it).

`interval="wald"` is provided as a sensitivity variant only: the 2026-07-21
planning numbers (k16 adapter_only UCB~0.904, k28 LCB~0.939) were computed
with the normal-approximation (Wald) interval; the paper text specifies
Wilson. Verdicts can differ on borderline cells — both are reported in the
final-protocol replay.
"""
import math
from statistics import NormalDist
from typing import Optional

DEFAULT_TAU = 0.90
DEFAULT_ALPHA = 0.05
DEFAULT_BUDGETS = (200, 500)


def z_from_alpha(alpha: float = DEFAULT_ALPHA) -> float:
    """Two-sided z critical value (alpha=0.05 -> 1.959964)."""
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wilson_interval(successes: int, n: int, alpha: float = DEFAULT_ALPHA) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 1.0)
    z = z_from_alpha(alpha)
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def wald_interval(successes: int, n: int, alpha: float = DEFAULT_ALPHA) -> tuple:
    """Normal-approximation interval p +/- z*sqrt(p(1-p)/n) (sensitivity only)."""
    if n <= 0:
        return (0.0, 1.0)
    z = z_from_alpha(alpha)
    p = successes / n
    half = z * math.sqrt(p * (1 - p) / n)
    return (max(0.0, p - half), min(1.0, p + half))


_INTERVALS = {"wilson": wilson_interval, "wald": wald_interval}


def three_zone(sr: float, n_trials: int, tau: float = DEFAULT_TAU,
               alpha: float = DEFAULT_ALPHA, interval: str = "wilson",
               successes: Optional[int] = None) -> dict:
    """Single-measurement three-zone verdict.

    successes defaults to round(sr * n_trials) (cached ledger points store SR,
    not raw counts, for some entries).
    Returns {successes, n, sr, lcb, ucb, verdict} with verdict in
    {"pass", "fail", "uncertain"}.
    """
    s = int(round(sr * n_trials)) if successes is None else int(successes)
    lcb, ucb = _INTERVALS[interval](s, n_trials, alpha)
    if lcb >= tau:
        verdict = "pass"
    elif ucb < tau:
        verdict = "fail"
    else:
        verdict = "uncertain"
    return {"successes": s, "n": n_trials, "sr": s / n_trials if n_trials else None,
            "lcb": lcb, "ucb": ucb, "verdict": verdict,
            "tau": tau, "alpha": alpha, "interval": interval}


class GateConfig:
    def __init__(self, tau: float = DEFAULT_TAU, alpha: float = DEFAULT_ALPHA,
                 interval: str = "wilson", budgets: tuple = DEFAULT_BUDGETS):
        assert interval in _INTERVALS, interval
        self.tau = tau
        self.alpha = alpha
        self.interval = interval
        self.budgets = tuple(sorted(budgets))

    def as_dict(self) -> dict:
        return {"tau": self.tau, "alpha": self.alpha, "interval": self.interval,
                "budgets": list(self.budgets)}


def gate_with_escalation(eval_module, mask: list, recovery: str,
                         cfg: GateConfig) -> Optional[dict]:
    """Full budget-escalation gate for one (mask, recovery) cell against the
    cached ledger.

    Walks the budget ladder: judge at the lowest cached budget >= budgets[0];
    on 'uncertain', escalate to the next budget IF a cached measurement exists
    there, else stop and flag `needs_real_eval` (a real re-evaluation at the
    next budget is required — the replay harness collects these into an
    escalation queue). Uncertain at max budget => pre-registered fail.

    Returns None on total cache miss (no measurement at any budget), else a
    result dict compatible with the driver's `result` (sr/n_trials of the
    DECIDING measurement) plus a "gate" sub-dict with the full chain.
    """
    chain = []
    entry = eval_module.lookup_at_budget(mask, recovery, min_trials=0)
    if entry is None:
        return None
    needs_real_eval = False
    next_budget = None
    v = three_zone(entry["sr"], entry["n_trials"], tau=cfg.tau, alpha=cfg.alpha,
                   interval=cfg.interval)
    chain.append(v)
    while v["verdict"] == "uncertain":
        # escalate one budget step above the current measurement
        higher = [b for b in cfg.budgets if b > (entry.get("n_trials") or 0)]
        if not higher:
            break  # already at/above max budget -> pre-registered fail below
        e2 = eval_module.lookup_at_budget(mask, recovery, min_trials=higher[0])
        if e2 is None:
            needs_real_eval = True   # real re-eval at higher[0] required
            next_budget = higher[0]
            break
        entry = e2
        v = three_zone(entry["sr"], entry["n_trials"], tau=cfg.tau,
                       alpha=cfg.alpha, interval=cfg.interval)
        chain.append(v)

    final = chain[-1]
    verdict = final["verdict"]
    pre_registered_fail = False
    if verdict == "uncertain" and not needs_real_eval and \
            final["n"] >= cfg.budgets[-1]:
        # uncertain at the maximum budget: pre-registered fail
        pre_registered_fail = True
    accepted = (verdict == "pass")
    return {
        "sr": final["sr"],
        "n_trials": final["n"],
        "raw": f"{final['successes']}/{final['n']}",
        "name": entry.get("name"),
        "source": "cached+gate",
        "gate": {
            **cfg.as_dict(),
            "chain": chain,
            "verdict": verdict,
            "accepted": accepted,
            "pre_registered_fail": pre_registered_fail,
            "needs_real_eval": needs_real_eval,
            "needs_budget": next_budget,
        },
    }
