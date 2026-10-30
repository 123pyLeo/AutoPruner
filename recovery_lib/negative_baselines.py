"""
recovery_lib.negative_baselines
===============================

Recovery strategies that were tried and **verified to FAIL or under-perform** in
the fragile pruned basin. They live here -- outside the main library -- so you can
reproduce the negative result and not re-walk the same dead ends. Across both repos
the failures cluster into a few lessons, recorded with each entry.

The cross-cutting verified principle: the ONLY thing that beats the frozen-backbone
~0.80 ceiling is the matched triad **on-policy states + co-adapting weights +
anti-collapse trust-region anchor**. Every offline / frozen / init-only / flat-minima
/ self-imitation trick below is neutral-or-harmful. And in every case, offline loss
falling did NOT imply closed-loop SR rising (W5: SR is the only arbiter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class FailedRecovery:
    name: str
    recipe: str
    lesson: str
    evidence: str
    source: str


FAILED_RECOVERIES: Dict[str, FailedRecovery] = {
    # ---- architecture falsifiers (prove the verified adapter is necessary) ---- #
    "lora_only": FailedRecovery(
        name="LoRA-only on kept layers (no boundary adapter)",
        recipe="LoRA r=128 on all kept LLaMA layers + sidecars, KL+L1, 10k",
        lesson="a boundary adapter that rebuilds the removed block is mandatory",
        evidence="openvla E13 -> 0% SR.",
        source="openvla vla-scripts/finetune_phase13_lora_only.py",
    ),
    "svd_linear_adapter": FailedRecovery(
        name="SVD / closed-form linear boundary adapter",
        recipe="closed-form ridge/OLS linear adapter (R²=0.967 offline)",
        lesson="the boundary adapter must be NONLINEAR + SGD-trained",
        evidence="openvla E14 -> 0% SR despite near-perfect offline fit.",
        source="openvla scripts/solve_svd_adapter.py / E14",
    ),
    "frozen_downstream": FailedRecovery(
        name="Frozen downstream (adapter-only, sidecars+kept layers frozen)",
        recipe="train ONLY the boundary adapter; head/proprio/kept layers = teacher",
        lesson="recovery is JOINT co-adaptation, not adapter reconstruction",
        evidence="starVLA frozen-downstream -> 0.005; openvla adapter-only k16 85.5% vs joint 98.0%.",
        source="starVLA k10cont_scp_frozen config; openvla freeze_sidecars",
    ),
    # ---- init / offline / flat-minima tricks (neutral-harmful) ---- #
    "scp_warmstart": FailedRecovery(
        name="SCP low-rank adapter warm-start",
        recipe="pre-regress the adapter to the removed residual offline, then distill",
        lesson="init is not a lever; offline-perfect init lands outside the co-adapt point",
        evidence="starVLA k10cont_scp 0.640 < zero-init 0.680.",
        source="starVLA scripts/recovery/scp_init_precompute.py",
    ),
    "offline_multi_anchor": FailedRecovery(
        name="Offline multi-anchor (multitraj warm)",
        recipe="multi_hidden_mse=20 to teacher on OFFLINE states",
        lesson="anchoring downstream to teacher on offline states = soft-freeze",
        evidence="starVLA multitraj_warm 0.708 < 0.788. (The SAME anchor on-policy helps.)",
        source="starVLA k10anchor_multitraj_warm config",
    ),
    "swa": FailedRecovery(
        name="SWA (stochastic weight averaging)",
        recipe="average late checkpoints (flat-minima hypothesis)",
        lesson="averaging harms the fragile pruned basin",
        evidence="starVLA swa_k10dist_late3 0.660 < 0.758.",
        source="starVLA scripts/recovery/swa_average.py",
    ),
    # ---- self-imitation / RL (hurt or capped) ---- #
    "grace_awr": FailedRecovery(
        name="GRACE / AWR (advantage-weighted self-imitation)",
        recipe="AWR-weighted, target switches to executed action when advantage>0",
        lesson="self-imitation regression hurts; pure teacher-distill+anchor stays best",
        evidence="starVLA grace_recovnf 0.732 < 0.828; hard task8 crashed to 3/50.",
        source="starVLA finetune_grace.py",
    ),
    "rl_head": FailedRecovery(
        name="RL head (REINFORCE on action-head + behavior-KL)",
        recipe="REINFORCE on ~66M head weights, frozen backbone, behavior-KL trust region",
        lesson="a frozen backbone caps RL at the platform; can't break 0.828",
        evidence="starVLA locked ~0.81-0.825; learner running_sr drifted 0.787->0.767.",
        source="starVLA deployment/model_server/rl_learner_head.py",
    ),
    "naive_ppo": FailedRecovery(
        name="Naive online PPO on the recovery basin",
        recipe="PPO on adapter+head LoRA + value head, Gaussian macro-action, sparse reward",
        lesson="deterministic L1 head + Gaussian noise = exploration is degradation",
        evidence="openvla S0 -> SR 18-27% (vs 50% base) in ~10 updates; positive actor loss.",
        source="openvla s0_online_rl/",
    ),
    # ---- offline-supervised relabel / KD (decoupled from SR) ---- #
    "dagger_v1_supervised": FailedRecovery(
        name="DAgger v1 supervised hard-state relabel (OFFLINE step)",
        recipe="relabel student rollout failures with teacher action L1, 50/50 RLDS anchor",
        lesson="offline relabel: safe-LR = no signal, higher-LR = catastrophic collapse",
        evidence="openvla v2 t0 64 vs 68 (no change); v1/v3 collapse to 3-18%.",
        source="openvla dagger/code/*",
    ),
    "anchor_kd_intermediate": FailedRecovery(
        name="Anchor-KD / intermediate hidden-state distillation",
        recipe="2 boundary adapters + MSE(adapter_out, dense intermediate hidden), KD w=1",
        lesson="offline hidden-shape match does not translate to closed-loop SR",
        evidence="openvla E2 -> 60% (probe predicted 90%+), -22.5pt vs C2v2 82.5%.",
        source="openvla vla-scripts/finetune_phase3_anchor_kd.py",
    ),
    "e34_relation_distill": FailedRecovery(
        name="E34 state-aligned relation distillation + replay anchor",
        recipe="action-token cosine-relation MSE + replay anchor, head LoRA",
        lesson="relation-loss dropped 55% while action-L1/SR never moved (decoupled)",
        evidence="openvla t9 +6pt (insignificant), flat slope at 500 & 2000 steps.",
        source="openvla e34_state_align/",
    ),
    "theseus_softinterp": FailedRecovery(
        name="R-series Theseus (hidden-MSE warmup + soft interp + L30/L31 LoRA)",
        recipe="phase0 hidden-MSE warmup, phase1 soft interp teacher/adapter + cos anchor",
        lesson="C2v2 adapter geometry is 'alien' to teacher hidden; structurally infeasible",
        evidence="openvla best pilot final eval -24 to -56pt (t8=0, t9=6).",
        source="openvla r1_theseus/",
    ),
    "full_unfreeze_multitask": FailedRecovery(
        name="Full unfreeze of kept layers (multi-task)",
        recipe="16 kept LLaMA layers + adapter + sidecars = 3.43B, multi-task RLDS",
        lesson="multi-task gradient interference, not capacity, caps a single ckpt",
        evidence="openvla E24 ~72% < 82.5% ceiling; 150x capacity scaling no gain -> use routing.",
        source="openvla logs/E24_master.log",
    ),
    "task_module_lora_orar": FailedRecovery(
        name="E29 task module (LoRA on head+proprio + ORAR, frozen base)",
        recipe="LoRA r16-64 + ORAR output-residual on frozen C2v2 adapter, per-task",
        lesson="the per-task adapter delta is HIGH-RANK (~565); LoRA/ORAR rank << needed",
        evidence="openvla all variants t8 52-60% = real baseline 56% (0pt). Use full unfreeze.",
        source="openvla vla-scripts/finetune_phase18_task_module.py",
    ),
    # ---- on-policy data variants that didn't add over the win ---- #
    "best_of_n": FailedRecovery(
        name="Best-of-N branch-and-splice recovery data",
        recipe="branch at transition frames, keep successful continuations, splice into npz",
        lesson="more recovery DATA didn't beat the 0.828 platform (the loss/anchor was the lever)",
        evidence="starVLA best-of-N best 0.822 <= 0.828 (within noise).",
        source="starVLA collect_bestofn_recovery.py",
    ),
    "corrective_buffer": FailedRecovery(
        name="Corrective buffer (gripper-emphasis on-policy fine-tune)",
        recipe="mix demo stream with a small student-rollout corrective buffer, weighted",
        lesson="a tiny corrective buffer fossilizes the wrong task-phase",
        evidence="starVLA corrective_l1 fell to 0.009 but key episode still failed.",
        source="starVLA CorrectiveBufferSampler",
    ),
}


def describe_all() -> str:
    lines = ["FAILED / under-performing recovery strategies (do not use):", ""]
    for key, m in FAILED_RECOVERIES.items():
        lines += [
            f"[{key}] {m.name}",
            f"    recipe : {m.recipe}",
            f"    LESSON : {m.lesson}",
            f"    evidence: {m.evidence}",
            f"    source : {m.source}",
            "",
        ]
    return "\n".join(lines)
