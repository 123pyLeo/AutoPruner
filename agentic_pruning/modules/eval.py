"""EvalModule — closed-loop SR oracle.

Two modes:
  - lookup_cached(mask, recovery, suite): query the cached results table
    (eval_lookup_object.json). Returns SR or None if unknown.
  - run_closed_loop(...): real eval (not implemented in offline demo).

The agent never sees the full results table — driver only returns the SR
for the agent's own proposal.
"""
import json
from pathlib import Path
from typing import Optional


class EvalModule:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        with open(self.data_path) as f:
            self._data = json.load(f)
        self._index = {}
        self._index_multi: dict = {}  # key -> [entries sorted by n_trials asc]
        for r in self._data["results"]:
            key = (tuple(sorted(r["mask"])), r["recovery"])
            # Historical single-entry index: FIRST entry in file order wins, so
            # appending higher-budget (500ep) entries later never changes the
            # default (non-gated) replay behavior.
            self._index.setdefault(key, r)
            self._index_multi.setdefault(key, []).append(r)
        for lst in self._index_multi.values():
            lst.sort(key=lambda e: e.get("n_trials") or 0)

    @property
    def suite(self) -> str:
        return self._data["suite"]

    def lookup_cached(self, mask: list[int], recovery: str) -> Optional[dict]:
        """Return full result dict (sr + metadata) or None if not in cache."""
        key = (tuple(sorted(mask)), recovery)
        r = self._index.get(key)
        if r is None:
            return None
        return {
            "sr": r["sr"],
            "n_trials": r.get("n_trials"),
            "raw": r.get("raw"),
            "name": r.get("name"),
            "source": "cached",
        }

    def lookup_at_budget(self, mask: list[int], recovery: str,
                          min_trials: int = 0) -> Optional[dict]:
        """Budget-aware lookup (final-protocol Wilson gate, 2026-07-21).

        Returns the cached entry with the SMALLEST n_trials >= min_trials for
        this (mask, recovery) cell, or None. min_trials=0 returns the lowest
        budget measurement (the original search-time point).
        """
        key = (tuple(sorted(mask)), recovery)
        for r in self._index_multi.get(key, []):
            n = r.get("n_trials") or 0
            if n >= min_trials:
                return {
                    "sr": r["sr"],
                    "n_trials": n,
                    "raw": r.get("raw"),
                    "name": r.get("name"),
                    "source": "cached",
                }
        return None

    def list_known_recoveries_for_mask(self, mask: list[int]) -> list[str]:
        """Helper: which recoveries have been tried for this mask in cache."""
        key = tuple(sorted(mask))
        return [r["recovery"] for r in self._data["results"]
                if tuple(sorted(r["mask"])) == key]

    def run_closed_loop(self, ckpt_path: str, n_trials: int = 200) -> dict:
        """Real closed-loop eval. Not implemented in offline demo."""
        raise NotImplementedError("Offline demo uses lookup_cached only.")
