"""
recovery_lib.core.losses
========================

Pure, verified recovery loss kernels. Each is written to work on **both** numpy
arrays (for the mock backend and unit tests) and torch tensors (real backends --
the graph is preserved so ``backend.apply_gradients`` can backprop). No model, no
I/O. Every term here appears in a VERIFIED recovery recipe; see ``METHODS.md`` for
the source and the SR evidence.

Two terms carry a caveat: ``awr_weight`` (GRACE/AWR) and ``feature_anchor`` used
*offline* were verified to HURT in the fragile pruned basin -- they live here so
the negative baselines can compute them, but the strategies do not use them.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

_EPS = 1e-12


def _is_torch(x) -> bool:
    return hasattr(x, "detach") and hasattr(x, "dim")


# --------------------------------------------------------------------------- #
#  Behaviour cloning + distillation (the Phase-2 objective)                    #
# --------------------------------------------------------------------------- #
def bc_action_l1(pred, gt):
    """Action behaviour-cloning L1 -- the core supervised term in every recipe."""
    return (pred - gt).abs().mean() if _is_torch(pred) else float(np.abs(np.asarray(pred) - np.asarray(gt)).mean())


def action_distill_l1(student_action, teacher_action):
    """Distill the *action chunk* from the frozen dense teacher (starVLA distill term)."""
    return bc_action_l1(student_action, teacher_action)


def feature_mse(student_hidden, teacher_hidden, relative: bool = False):
    """MSE between student and teacher action-token hidden states (HKD / hidden_mse).

    ``relative=True`` normalises by the teacher-hidden energy → scale-free, which is
    the form the verified on-policy recipe uses for its multi-layer anchor.
    """
    if _is_torch(student_hidden):
        diff = (student_hidden - teacher_hidden) ** 2
        num = diff.mean()
        if relative:
            num = num / ((teacher_hidden ** 2).mean() + _EPS)
        return num
    a, b = np.asarray(student_hidden), np.asarray(teacher_hidden)
    num = float(((a - b) ** 2).mean())
    return num / (float((b ** 2).mean()) + _EPS) if relative else num


def multi_hidden_mse(student_hiddens: Dict[int, object], teacher_hiddens: Dict[int, object],
                     skip_post: int = 2, relative: bool = True, removed_after: Sequence[int] = ()):
    """Anchor EVERY kept layer's hidden to the teacher (scale-free), skipping the
    ``skip_post`` "absorber" layers right after each prune boundary.

    Verified necessary in the on-policy recipe (weight ~10). NOTE: applied to
    *offline* states it over-constrains and HURTS (multitraj 0.708 < 0.788) -- the
    benefit comes from pairing it with on-policy states.
    """
    skip = set()
    for b in removed_after:
        for d in range(1, skip_post + 1):
            skip.add(b + d)
    layers = [l for l in student_hiddens if l in teacher_hiddens and l not in skip]
    if not layers:
        return 0.0 if not _is_torch(next(iter(student_hiddens.values()), np.zeros(1))) else None
    terms = [feature_mse(student_hiddens[l], teacher_hiddens[l], relative=relative) for l in layers]
    if _is_torch(terms[0]):
        out = terms[0]
        for t in terms[1:]:
            out = out + t
        return out / len(terms)
    return float(np.mean(terms))


def distill_kl(student_logits, teacher_logits, temperature: float = 1.0):
    """KL(teacher || student) over the action-range logits (openvla Phase-2 KL).

    Returns ``T^2 * KL`` (batch-mean), matching the source distillation scaling.
    """
    T = temperature
    if _is_torch(student_logits):
        import torch
        import torch.nn.functional as F
        log_p_student = F.log_softmax(student_logits / T, dim=-1)
        p_teacher = F.softmax(teacher_logits / T, dim=-1)
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)
    s = np.asarray(student_logits, dtype=np.float64) / T
    t = np.asarray(teacher_logits, dtype=np.float64) / T
    s = s - s.max(-1, keepdims=True)
    t = t - t.max(-1, keepdims=True)
    log_p_s = s - np.log(np.exp(s).sum(-1, keepdims=True) + _EPS)
    p_t = np.exp(t) / (np.exp(t).sum(-1, keepdims=True) + _EPS)
    kl = (p_t * (np.log(p_t + _EPS) - log_p_s)).sum(-1).mean()
    return float(kl * T * T)


# --------------------------------------------------------------------------- #
#  Trust-region anchor (the verified anti-basin-collapse guardrail)            #
# --------------------------------------------------------------------------- #
def smooth_l1(x, y, beta: float = 1.0):
    if _is_torch(x):
        import torch.nn.functional as F
        return F.smooth_l1_loss(x, y, beta=beta)
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    d = np.abs(a - b)
    return float(np.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta).mean())


def trust_region_anchor(student_action, anchor_action, weight: float = 0.5):
    """Penalise drift from a FROZEN snapshot of the pre-continuation student.

    The load-bearing term that let on-policy distillation break the frozen-backbone
    0.80 ceiling: it stops self-imitation from collapsing the basin. weight ~0.5.
    """
    tr = smooth_l1(student_action, anchor_action, beta=1.0)
    return weight * tr if _is_torch(student_action) else weight * tr


# --------------------------------------------------------------------------- #
#  AWR weighting (GRACE) -- documented; VERIFIED to HURT, used only by baseline #
# --------------------------------------------------------------------------- #
def awr_weight(advantage, tau: float = 1.0, gamma: float = 0.0, gate=None, w_max: float = 20.0):
    """Advantage-weighted-regression per-frame weight ``clip(exp(A/tau)*(1+gamma*gate), 0, w_max)``.

    VERIFIED to HURT (GRACE 0.732 < 0.828; self-imitation regression). Kept for the
    negative baseline only.
    """
    A = np.asarray(advantage, dtype=np.float64)
    g = np.zeros_like(A) if gate is None else np.asarray(gate, dtype=np.float64)
    return np.clip(np.exp(A / tau) * (1.0 + gamma * g), 0.0, w_max)
