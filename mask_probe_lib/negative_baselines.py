"""
mask_probe_lib.negative_baselines
==================================

Methods that were tried for layer/mask selection and **verified to FAIL** as
selectors.  They live here -- outside the main library -- so you can (a) reproduce
the negative result, (b) use them as honest baselines in a paper, and (c) avoid
re-discovering the same traps.  Knowing *what does not work* is itself a verified
finding from both repos.

The cross-cutting reason they fail: they score layers in **parameter space** or by
**representation similarity**, both of which are anti-correlated or uncorrelated
with closed-loop SR for a *recoverable* prune.  The verified methods instead score
**action-output sensitivity** (knockout) or **recoverability** (can a small adapter
restore the block) -- see ``METHODS.md``.

Each entry records: the metric, the selection rule people use, and the concrete
failure evidence.  Computable ones (CKA, BI, isometry) are implemented by calling
the same kernels; gradient ones (FIM, Taylor) need backend gradients and are
described with the exact formula + a runnable stub signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .core.backend import ModelBackend
from .core.kernels import block_influence, effective_rank, isometry, linear_cka
from .core.types import ActivationBundle, LayerScore


@dataclass
class FailedMethod:
    name: str
    measures: str
    selection_rule: str
    failure_evidence: str
    source: str


#: registry of verified-failed selectors (DO NOT use to choose a mask)
FAILED_METHODS: Dict[str, FailedMethod] = {
    "fim": FailedMethod(
        name="FIM / Fisher information (per layer)",
        measures="||dL/dW_l||^2 summed per layer (parameter-space sensitivity)",
        selection_rule="remove lowest-FIM layers (standard NLP pruning)",
        failure_evidence=(
            "FLAGSHIP COUNTEREXAMPLE: L30 has the LOWEST FIM (0.006) yet deleting it "
            "drops SR ~12 pts. The FIM agent deletes L0/L30/L31 every iter (18 anchor "
            "violations) and never finds k28. Per-task top-5 identical across all tasks "
            "([1,11,10,14,12]) -> no task discrimination. (grad-norm/weight-magnitude "
            "fail identically since FIM = grad-norm^2.)"
        ),
        source="openvla scripts/fim_per_layer.py; agentic_pruning/agents/fim_agent.py",
    ),
    "taylor": FailedMethod(
        name="Taylor importance |grad . h| (per layer)",
        measures="first-order action-loss Taylor importance at layer output",
        selection_rule="remove lowest-Taylor layers",
        failure_evidence=(
            "Same parameter-/activation-space family as FIM; does not predict "
            "recoverable-prune SR. Useful only as a within-family comparison, not a "
            "selector."
        ),
        source="openvla scripts/compute_action_aware_metric.py",
    ),
    "cka": FailedMethod(
        name="Adjacent-layer CKA (representation similarity)",
        measures="centred linear CKA between consecutive layer outputs",
        selection_rule="remove layers in high-CKA (redundant) runs",
        failure_evidence=(
            "MISLEADING: L29-L30-L31 all CKA > 0.97-0.99, which marks L30 deletable, "
            "but deleting L30 costs 11.5 pts. Rotation invariance hides the head-format "
            "projection direction."
        ),
        source="openvla scripts/cka_between_layers.py",
    ),
    "isometry_alone": FailedMethod(
        name="Isometry score (alone)",
        measures="Spearman of pairwise distances in vs out of a layer",
        selection_rule="remove high-isometry (low-distortion) layers",
        failure_evidence=(
            "L30 isometry = 0.985 (almost no distortion) yet is non-removable. "
            "Distance preservation != information preservation. Only valid INSIDE the "
            "combined ManifoldGeometryProbe."
        ),
        source="openvla scripts/per_layer_manifold.py",
    ),
    "block_influence": FailedMethod(
        name="ShortGPT Block-Influence / per-layer residual cos (alone)",
        measures="1 - mean cos(h_in, h_out) at action tokens, per layer",
        selection_rule="remove lowest-BI (most redundant) single layers",
        failure_evidence=(
            "WEAK: Spearman(cos, single-drop SR) = +0.38, wrong sign for some layers "
            "(L1 cos 0.77 but SR 0; L31 had the LARGEST residual L2 yet near-anchor). "
            "Single-layer drift is too small / absorbed by redundancy; BI only matters "
            "at multi-layer scale, where it is non-additive."
        ),
        source="openvla scripts/compute_layer_metrics.py, perlayer_residual_vs_sr.py",
    ),
    "act3k": FailedMethod(
        name="act3k short-train LoRA proxy",
        measures="action-loss window value at ~3k LoRA steps",
        selection_rule="rank masks by lower act3k (predicted higher SR)",
        failure_evidence=(
            "FAILED as a ranker: single-feature Spearman with SR only +0.29; cannot "
            "distinguish ceiling cases (act3k normal but SR=0.04). Superseded by the "
            "DriftPredictor."
        ),
        source="openvla phase1/analyze_mask_regression.py",
    ),
    "structural_ols": FailedMethod(
        name="Structural-prior OLS / surrogate-LoRA ranker",
        measures="OLS of SR on structural features (k, gaps, max_run, ...) + act3k",
        selection_rule="rank by predicted_sr from the linear structural model",
        failure_evidence=(
            "OVER-FITS n=13: LOO R² = -0.84 (worse than predicting the mean). The "
            "two-feature DriftPredictor beat it at LOO R²=0.71. Always leave-one-out "
            "check an SR surrogate."
        ),
        source="openvla phase1/analyze_mask_regression.py, surrogate.py",
    ),
    "single_layer_sr_sum": FailedMethod(
        name="Summing single-layer ablation SR",
        measures="sum of per-layer single-drop SR losses for a multi-layer mask",
        selection_rule="estimate a mask's damage by adding its layers' single-drop losses",
        failure_evidence=(
            "NON-ADDITIVE: k=8 cumulative loss != sum of 8 single-layer losses (~3x "
            "non-linear amplification). Single-layer ablation is a valid CORNERSTONE "
            "labeller (L0/L1 SR=0) but its values must not be summed."
        ),
        source="openvla phase1 ledger (b1_tf_layer_X)",
    ),
    "full_seq_cosine": FailedMethod(
        name="Full-sequence (non-action-token) hidden cosine",
        measures="boundary cosine averaged over ALL token positions",
        selection_rule="rank masks by full-sequence drift",
        failure_evidence=(
            "Averages away the action signal: full-seq cos 0.59 vs action-token cos "
            "0.48 for the same mask. ALWAYS restrict probes to action-token positions "
            "(warning W3)."
        ),
        source="openvla scripts/validate_boundary_drift_crossbatch.py",
    ),
    "effrank_geometry_veto": FailedMethod(
        name="Effective rank used to CHOOSE mask geometry",
        measures="effective rank of the cumulative residual Δ",
        selection_rule="veto non-contiguous masks whose Δ is higher-rank",
        failure_evidence=(
            "Effective rank explains the recoverability MECHANISM (Δ is low-rank) but "
            "has NO mapping to SR for choosing between geometries. The multi-anchor "
            "Probe C (+0.06 R²) explicitly overturned this veto. Use effective rank as "
            "a recoverability signal, never as a geometry selector."
        ),
        source="openvla scripts/probe_mask_sweep.py vs probe_deeper_designs.py (Probe C)",
    ),
    "open_loop_l1_as_sr": FailedMethod(
        name="Open-loop action L1 as a closed-loop SR predictor",
        measures="student's raw action L1 vs teacher",
        selection_rule="pick the mask/student with the lowest open-loop L1",
        failure_evidence=(
            "L1 != SR: a student with L1 0.0334 (LOWER than the successful teacher's "
            "0.0378) still failed closed-loop. Open-loop metrics RANK, closed-loop SR "
            "DECIDES (warning W5)."
        ),
        source="starVLA pruning_recovery_full_writeup_2026-06-09.md",
    ),
}


def describe_all() -> str:
    lines = ["FAILED layer/mask-selection methods (DO NOT use as selectors):", ""]
    for key, m in FAILED_METHODS.items():
        lines += [
            f"[{key}] {m.name}",
            f"    measures : {m.measures}",
            f"    rule     : {m.selection_rule}",
            f"    WHY FAILS: {m.failure_evidence}",
            f"    source   : {m.source}",
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Runnable implementations (so you can reproduce the negative result).        #
#  These compute the metric; they are deliberately NOT registered as probes.   #
# --------------------------------------------------------------------------- #
def adjacent_cka_curve(bundle: ActivationBundle) -> List[LayerScore]:
    """Adjacent-layer CKA. High == redundant. WARNING: hides format-aligner anchors."""
    idx = bundle.layer_indices
    out = []
    for L in idx:
        nxt = L + 1
        val = linear_cka(bundle.hidden[L], bundle.hidden[nxt]) if nxt in bundle.hidden else 1.0
        out.append(LayerScore(layer=L, score=float(val), raw={"adjacent_cka": float(val)}))
    return out


def block_influence_curve(bundle: ActivationBundle) -> List[LayerScore]:
    """ShortGPT BI = 1 - cos(in,out). WARNING: Spearman with SR only +0.38."""
    idx = bundle.layer_indices
    out = []
    for L in idx:
        nxt = L + 1
        val = block_influence(bundle.hidden[L], bundle.hidden[nxt]) if nxt in bundle.hidden else 0.0
        out.append(LayerScore(layer=L, score=float(val), raw={"block_influence": float(val)}))
    return out


def fim_per_layer(backend: ModelBackend, *args, **kwargs):
    """Per-layer Fisher trace F_l = E[ sum_p ||dL_action/dW_p||^2 ].

    Needs gradient access the abstract ModelBackend does not expose; implement
    ``backend.layer_grad_sq(layer)`` to run it.  Kept as a documented stub because
    FIM is the project's flagship FAILURE (see FAILED_METHODS['fim']).
    """
    raise NotImplementedError(
        "FIM requires per-layer weight gradients. It is a VERIFIED-FAILED selector "
        "(L30 lowest FIM but essential) -- see FAILED_METHODS['fim']. Implement only "
        "to reproduce the negative baseline."
    )
