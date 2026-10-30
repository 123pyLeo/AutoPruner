"""
write_report -- consolidate one run into a single report.
=========================================================

The automated feedback the framework promises (ARCHITECTURE.md §8): after the agentic
loop + winner selection, emit ONE report (json + human-readable md) carrying
    (a) success rate,
    (b) parameter / compute reduction,
    (c) a pointer to the final recovered weights folder.
Written next to the run's selection.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def _winning_weights(run_dir: Path, winner: Optional[dict]) -> Optional[dict]:
    """Best-effort: read the winning iteration's result.json for the recovery run_dir /
    adapter weights produced during real recovery (None on the mock path)."""
    if not winner or winner.get("iter") is None:
        return None
    p = run_dir / f"iter_{int(winner['iter']):03d}" / "result.json"
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not (r.get("run_dir") or r.get("weights")):
        return None
    return {"run_dir": r.get("run_dir") or r.get("weights"),
            "adapter_step": r.get("adapter_step"), "source": r.get("source")}


def write_report(result: dict, *, model: str) -> dict:
    """Write report.json + report.md into the run dir; return their paths."""
    run_dir = Path(result["run_dir"])
    spec, sel = result["spec"], result["selection"]
    winner = sel.get("winner")
    weights = _winning_weights(run_dir, winner)

    report = {
        "synthetic": bool(result["summary"].get("synthetic", False)),
        "probe_metric": "paper_l2_v1",
        "model": model,
        "suite": getattr(spec, "suite", None),
        "dense_sr": sel.get("dense_sr"),
        "sr_threshold": getattr(spec, "sr_threshold", None),
        "meets_spec": sel.get("winner_meets_spec"),
        "plan": None,
        "weights": weights,
        "frontier": sel.get("frontier"),
    }
    if winner:
        p = winner.get("params", {}) or {}
        report["plan"] = {
            "mask_removed": sorted(winner["mask"]),
            "recovery": winner.get("recovery"),
            "sr": winner.get("sr"),
            "layers_removed": p.get("layers_removed"),
            "layers_total": p.get("layers_total"),
            "layer_compression": p.get("layer_compression"),
            "param_compression_decoder": p.get("param_compression_decoder"),
            "est_decoder_compute_reduction": p.get("est_decoder_compute_reduction"),
        }

    rj = run_dir / "report.json"
    rj.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    # human-readable
    L = [f"# AutoPrune 报告 — {model} / {report['suite']}", "",
         f"- dense SR: {report['dense_sr']}",
         f"- SR 下限: {report['sr_threshold']}  ·  达标: {report['meets_spec']}"]
    pl = report["plan"]
    if pl:
        lc = (pl["layer_compression"] or 0) * 100
        L += ["", "## 最佳方案",
              f"- 剪除层 (mask): {pl['mask_removed']}",
              f"- 恢复策略: {pl['recovery']}",
              f"- **成功率 SR: {pl['sr']}**",
              f"- 层压缩: {pl['layers_removed']}/{pl['layers_total']} ({lc:.1f}%)"]
        if pl.get("param_compression_decoder") is not None:
            L.append(f"- 参数量下降(decoder): {pl['param_compression_decoder']*100:.1f}%")
        if pl.get("est_decoder_compute_reduction") is not None:
            L.append(f"- 估算算力下降: {pl['est_decoder_compute_reduction']*100:.1f}%")
    else:
        L.append("\n（没有候选产出 SR，无可选方案。）")
    L += ["", "## 最终权重"]
    if weights:
        L += [f"- 目录: {weights['run_dir']}",
              f"- 来源: {weights.get('source')}  step={weights.get('adapter_step')}"]
    else:
        L.append("- (mock 路径无真实权重;真模型恢复后此处指向 adapter 权重目录)")

    rm = run_dir / "report.md"
    rm.write_text("\n".join(L) + "\n")
    return {"report_json": str(rj), "report_md": str(rm)}
