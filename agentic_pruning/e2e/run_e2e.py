"""Historical result selector retained for API compatibility. Use autoprune.run_e2e to run."""
from __future__ import annotations

from autoprune.validation import valid_sr

def select_winning_iter(trace: dict, spec) -> dict | None:
    """Find the best iter that meets the spec constraints."""
    candidates = []
    dense_sr = trace["dense_sr"]
    # New traces record the model's actual depth. Keep 32 only as a compatibility
    # fallback for historical OFT/CogACT traces that predate this metadata.
    total_layers = int(trace.get("total_layers", 32))
    if total_layers <= 0:
        raise ValueError(f"total_layers must be positive, got {total_layers}")
    for row in trace["trace"]:
        sr = row.get("sr")
        if not valid_sr(sr):
            continue
        k = row["k"]
        compression = k / total_layers
        # check constraints
        if sr < spec.sr_threshold:
            continue
        if spec.sr_margin_from_dense is not None and (dense_sr - sr) > spec.sr_margin_from_dense:
            continue
        if compression < spec.min_compression:
            continue
        if spec.min_k is not None and k < spec.min_k:
            continue
        if spec.max_k is not None and k > spec.max_k:
            continue
        candidates.append(row)
    if not candidates:
        return None
    # Sort by: (compression desc, sr desc)  → prefer most compressed at acceptable SR
    candidates.sort(key=lambda r: (r["k"], r["sr"]), reverse=True)
    return candidates[0]
