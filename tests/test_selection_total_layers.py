from types import SimpleNamespace

from agentic_pruning.e2e.run_e2e import select_winning_iter


def _spec(min_compression):
    return SimpleNamespace(
        sr_threshold=0.90,
        sr_margin_from_dense=None,
        min_compression=min_compression,
        min_k=None,
        max_k=None,
    )


def test_non_32_layer_compression_uses_trace_depth():
    trace = {
        "dense_sr": 0.95,
        "total_layers": 30,
        "trace": [{"k": 6, "sr": 0.91, "mask": list(range(6))}],
    }
    assert select_winning_iter(trace, _spec(0.20)) is trace["trace"][0]


def test_non_32_layer_candidate_below_ratio_is_rejected():
    trace = {
        "dense_sr": 0.95,
        "total_layers": 30,
        "trace": [{"k": 5, "sr": 0.94, "mask": list(range(5))}],
    }
    assert select_winning_iter(trace, _spec(0.20)) is None
