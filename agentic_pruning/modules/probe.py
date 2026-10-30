"""ProbeModule — read-only access to per-layer raw probe measurements + transferable methodology rules.

Two data sources:
  - probe_object.json:  raw per-layer numbers (manifold/linear-probe/CKA/FIM/residual stats).
                        Represents what the agent WOULD get back by calling probe tools
                        on this model+suite. NO derived conclusions (no role labels,
                        no anchor evidence, no scaling-curve answers).
  - probe_methodology.json:  TRANSFERABLE RULES distilled from prior research on other
                        VLA models — which probes are reliable / which mislead, anchor
                        recipe, hard warnings, qualitative landmarks. NO concrete
                        layer indices or SR numbers from the current model.

The split exists so that the agent reads PRIOR KNOWLEDGE (methodology) + RAW DATA (probe
results), and must derive the conclusions itself (which layer is the format aligner,
where the sweet spot is). The framework does NOT spoon-feed answers.
"""
import json
from pathlib import Path
from typing import Optional


class ProbeModule:
    def __init__(self, data_path: Path, methodology_path: Optional[Path] = None):
        self.data_path = Path(data_path)
        with open(self.data_path) as f:
            self._data = json.load(f)
        self.methodology_path = (
            Path(methodology_path) if methodology_path is not None
            else self.data_path.parent / "probe_methodology.json"
        )
        if self.methodology_path.exists():
            with open(self.methodology_path) as f:
                self._methodology = json.load(f)
        else:
            self._methodology = {}

    @property
    def suite(self) -> str:
        return self._data["suite"]

    @property
    def dense_sr(self) -> float:
        return self._data["dense_sr"]

    @property
    def n_layers(self) -> int:
        return self._data["n_layers_total"]

    def get_layer_evidence(self, idx: int) -> dict:
        """Raw probe measurements for one layer. No role label."""
        for L in self._data["layers"]:
            if L["idx"] == idx:
                return dict(L)
        raise KeyError(f"layer {idx} not in probe data")

    def get_summary(self) -> dict:
        """Compact view: suite + dense SR + n_layers + per-metric source pointers.
        Does NOT include role labels, anchor evidence, or scaling-curve answers —
        agent must derive those from raw layer table + methodology rules."""
        return {
            "suite": self.suite,
            "dense_sr": self.dense_sr,
            "n_layers": self.n_layers,
            "metric_sources": self._data.get("metric_sources", {}),
        }

    def get_per_layer_table(self, metrics: Optional[list[str]] = None) -> list[dict]:
        """Static per-layer table — currently only idx + single_delete_training_free_sr
        (the allowed pre-loaded exception). All OTHER probe metrics have been moved
        to experiments/cache/probes_<model>_<suite>.json and must be obtained via
        the probe_query.query() tool API."""
        if metrics is None:
            return [dict(L) for L in self._data["layers"]]
        return [
            {k: L.get(k) for k in (["idx"] + metrics)}
            for L in self._data["layers"]
        ]

    def get_layer_static_facts(self) -> list[dict]:
        """Alias of get_per_layer_table — clearer name post-refactor. Returns
        only the fields directly stored in probe_object.json (no on-demand probes)."""
        return [dict(L) for L in self._data["layers"]]

    # ---------------- Methodology priors (transferable rules) ----------------

    def get_methodology(self) -> dict:
        return dict(self._methodology)

    def get_probe_reliability(self) -> dict:
        return self._methodology.get("probe_reliability", {})

    def get_anchor_recipe(self) -> dict:
        return self._methodology.get("anchor_recipe", {})

    def get_warnings(self) -> list[dict]:
        return self._methodology.get("warnings", [])

    def get_qualitative_landmarks(self) -> dict:
        return self._methodology.get("qualitative_landmarks", {})

    def get_metric_semantics(self) -> dict:
        return self._methodology.get("metric_semantics", {})
