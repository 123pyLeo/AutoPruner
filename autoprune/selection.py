"""
Result selection + size accounting for the AutoPrune pruning loop.
==================================================================

The agentic loop produces a trace of (mask, recovery, SR) candidates. The product
goal is NOT "highest SR" (that is always the *least*-pruned candidate) but:

    among candidates with SR >= threshold, return the one that removes the MOST
    layers (ties broken by higher SR).

That objective already lives in ``agentic_pruning/e2e/run_e2e.select_winning_iter``;
this module wraps it, adds a Pareto frontier (so callers can see the SR/compression
trade-off when nothing clears the bar), and a parameter/compute report so eval
surfaces "各项参数量" next to SR.

All functions are model-agnostic: param numbers come from ``backend.model_config()``
when the backend exposes one (GenericVLABackend reads the ckpt config; the mock
returns a nominal OFT-7B estimate). When it doesn't, the report degrades to the
layer-count compression, which for identical decoder layers IS the exact fraction of
decoder-stack parameters removed.
"""

from __future__ import annotations

from typing import Optional
from autoprune.validation import valid_sr

from agentic_pruning.e2e.prompt_parser import ProblemSpec
from agentic_pruning.e2e.run_e2e import select_winning_iter


# --------------------------------------------------------------------------- #
# parameter / compute accounting
# --------------------------------------------------------------------------- #
def _decoder_layer_params(hidden: int, intermediate: int) -> int:
    """Params in ONE LLaMA-style decoder layer (attn q/k/v/o + MLP gate/up/down).

    Norm/bias terms are negligible and omitted; this is the dominant term and is
    what structural layer pruning removes.
    """
    attn = 4 * hidden * hidden
    mlp = 3 * hidden * intermediate
    return attn + mlp


def param_report(mask, backend) -> dict:
    """Size/compute reduction for removing ``mask`` from ``backend``.

    Always returns layer-level numbers; adds absolute parameter + estimated
    compute reduction when ``backend.model_config()`` is available.
    """
    k = len(mask)
    n = int(getattr(backend, "n_layers", 0)) or (k or 1)
    rep = {
        "layers_removed": k,
        "layers_total": n,
        "layers_kept": n - k,
        "layer_compression": round(k / n, 4) if n else 0.0,
    }
    cfg = None
    getter = getattr(backend, "model_config", None)
    if callable(getter):
        try:
            cfg = getter()
        except Exception:  # noqa: BLE001 - reporting must never break the loop
            cfg = None
    if cfg and cfg.get("hidden_size") and cfg.get("intermediate_size"):
        h, i = int(cfg["hidden_size"]), int(cfg["intermediate_size"])
        nl = int(cfg.get("num_hidden_layers", n)) or n
        per = _decoder_layer_params(h, i)
        decoder_total = per * nl
        rep.update({
            "params_per_decoder_layer": per,
            "decoder_params_removed": per * k,
            "decoder_params_total": decoder_total,
            "param_compression_decoder": round(per * k / decoder_total, 4) if decoder_total else 0.0,
            # per-layer FLOPs/latency are ~equal, so kept-fraction is the compute reduction
            "est_decoder_compute_reduction": round(k / nl, 4) if nl else 0.0,
            "params_estimate_is_nominal": bool(cfg.get("nominal", False)),
        })
    return rep


def fmt_params(p: Optional[int]) -> str:
    """Human-readable parameter count (e.g. 1_616_904_192 -> '1.62B')."""
    if p is None:
        return "n/a"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(p) >= div:
            return f"{p / div:.2f}{suf}"
    return str(p)


# --------------------------------------------------------------------------- #
# winner + trade-off frontier
# --------------------------------------------------------------------------- #
def pareto_frontier(trace_rows: list[dict]) -> list[dict]:
    """Non-dominated (k, SR) candidates: no other row has >= layers AND >= SR.

    Returned sorted by k descending — the compression/SR trade-off curve.
    """
    scored = [r for r in trace_rows if valid_sr(r.get("sr"))]
    front = []
    for r in scored:
        dominated = any(
            o is not r
            and o["k"] >= r["k"] and o["sr"] >= r["sr"]
            and (o["k"] > r["k"] or o["sr"] > r["sr"])
            for o in scored
        )
        if not dominated:
            front.append(r)
    # collapse exact (k, sr) duplicates, keep the earliest iter
    seen, uniq = set(), []
    for r in sorted(front, key=lambda r: (-r["k"], -r["sr"])):
        key = (r["k"], round(r["sr"], 6))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq


def select_winner(summary: dict, spec: ProblemSpec, backend=None) -> dict:
    """Pick the most-pruned candidate meeting ``spec``, with a frontier fallback.

    Returns a dict:
      {winner, winner_meets_spec, frontier, dense_sr, spec_summary}
    ``winner`` is the spec-satisfying most-pruned row when one exists; otherwise
    the most-pruned row on the Pareto frontier (clearly flagged as not meeting the
    SR bar). Each row is enriched with a ``params`` report when ``backend`` is given.
    """
    frontier = pareto_frontier(summary["trace"])
    winner = select_winning_iter(summary, spec)
    meets = winner is not None
    if winner is None and frontier:
        # Nothing cleared SR>=threshold. Do NOT fall back to the most-pruned point
        # (that is the catastrophic-collapse end of the frontier). The useful single
        # suggestion is the one CLOSEST to the SR goal = highest SR (ties -> more
        # pruned); the full frontier is returned so the caller can relax the bar.
        winner = max(frontier, key=lambda r: (r["sr"], r["k"]))

    def enrich(row):
        if row is None or backend is None:
            return row
        row = dict(row)
        row["params"] = param_report(row["mask"], backend)
        return row

    return {
        "winner": enrich(winner),
        "winner_meets_spec": meets,
        "frontier": [enrich(r) for r in frontier],
        "dense_sr": summary.get("dense_sr"),
        "spec_summary": {
            "suite": spec.suite, "sr_threshold": spec.sr_threshold,
            "min_compression": spec.min_compression,
            "min_k": spec.min_k, "max_k": spec.max_k,
            "sr_margin_from_dense": spec.sr_margin_from_dense,
        },
    }
