"""MaskSearchModule — search-space config + accumulating tried_masks log + QD archive.

Agent reads list_tried() to avoid repeats and get_search_space() for budget
constraints. Driver calls record_proposed() after each iteration.

QD archive (related_work_mining_2026-05-30 §10): instead of tracking only ONE
global champion, we keep the best (mask, SR) seen PER behavioral cell (see
modules/grid.py). This is a derived index over tried_masks — no extra GPU — and
is rendered into the agent prompt so it sees the whole tradeoff map, not just the
single best. Deployment still ships ONE mask; the archive is only a search aid.
Passive use only for now (agent reads the map); "actively spend SR to fill empty
cells" is deferred until the §3 proxy funnel (条①) lands to gate it.
"""
import json
from pathlib import Path
from typing import List, Optional

from .grid import cell_of, cell_label

try:
    from mask_probe_lib.mask_selection.ratio import PruneRatioSweep
except Exception:  # mask_probe_lib optional at import time
    PruneRatioSweep = None


class MaskSearchModule:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        with open(self.data_path) as f:
            self._data = json.load(f)
        if "tried_masks" not in self._data:
            self._data["tried_masks"] = []
        # QD archive: {cell_key "kb,cb": {cell, label, mask, k, sr, iter}}
        if "archive" not in self._data:
            self._data["archive"] = {}

    @property
    def suite(self) -> str:
        return self._data["suite"]

    @property
    def total_layers(self) -> int:
        return self._data["total_layers"]

    def get_search_space(self) -> dict:
        return {
            "suite": self.suite,
            "total_layers": self.total_layers,
            "mask_form": self._data.get("mask_form"),
            "mask_form_doc": self._data.get("mask_form_doc", ""),
            "k_options": self._data.get("k_options", []),
            # BIDIRECTIONAL budget options: preset if the data file provides them, else
            # generated from PruneRatioSweep (which now searches BELOW the 28% seed too,
            # down to ~10% — so the agent explores small budgets where FastWAM's real
            # sweet spot lived, instead of only ratcheting UP from the cross-model prior).
            "compression_budget_options": self._budget_options(),
            # Running per-model sweet spot DISCOVERED from measured SR so far (not the
            # 0.28 seed). None until some ratio clears the floor.
            "discovered_sweet_spot": self.discover_budget(),
            "constraint": self._data.get("constraint", ""),
        }

    # ---------------- budget discovery (wires ratio.discover_sweet_spot into the loop) ----
    def _budget_options(self) -> list:
        preset = self._data.get("compression_budget_options")
        if preset:
            return preset
        if PruneRatioSweep is None:
            return []
        return [{"ratio": r["ratio"], "k": r["k"], "below_seed": r["below_seed"]}
                for r in PruneRatioSweep(self.total_layers).sweep()]

    def _sr_by_ratio(self) -> dict:
        """Best measured recovered SR per prune ratio (k/total_layers), over tried_masks."""
        best: dict = {}
        for t in self._data["tried_masks"]:
            sr = t.get("sr")
            if sr is None or not t.get("mask"):
                continue
            ratio = round(len(t["mask"]) / self.total_layers, 3)
            if ratio not in best or sr > best[ratio]:
                best[ratio] = sr
        return best

    def discover_budget(self, dense_sr: Optional[float] = None,
                        recover_floor: float = 0.90) -> Optional[dict]:
        """Per-model sweet spot from MEASURED SR (PruneRatioSweep.discover_sweet_spot).

        ``recover_floor`` is a fraction of ``dense_sr`` when ``dense_sr`` is given (defaults
        to the data file's ``dense_sr``), else an absolute SR. Returns None until enough
        SR is measured. This is what makes the loop CONVERGE to the model's own budget
        instead of trusting the 28% seed."""
        if PruneRatioSweep is None:
            return None
        sr_by_ratio = self._sr_by_ratio()
        if not sr_by_ratio:
            return None
        dense = dense_sr if dense_sr is not None else self._data.get("dense_sr")
        return PruneRatioSweep(self.total_layers).discover_sweet_spot(
            sr_by_ratio, recover_floor=recover_floor, dense_sr=dense)

    def list_tried(self) -> list[dict]:
        return list(self._data["tried_masks"])

    def is_tried(self, mask: list[int], recovery: str) -> bool:
        key = (tuple(sorted(mask)), recovery)
        for t in self._data["tried_masks"]:
            if (tuple(sorted(t["mask"])), t["recovery"]) == key:
                return True
        return False

    # ---------------- QD archive ----------------
    @staticmethod
    def _cell_key(cell) -> str:
        return f"{cell[0]},{cell[1]}"

    def _update_archive(self, mask: list[int], sr: Optional[float], iter_idx: int) -> None:
        """Elitism: a (mask, sr) enters its cell only if the cell is empty or sr
        beats the incumbent. Masks with sr=None (e.g. loss-floor abort) are skipped."""
        if sr is None or not mask:
            return
        cell = cell_of(mask)
        ck = self._cell_key(cell)
        cur = self._data["archive"].get(ck)
        if cur is None or sr > cur["sr"]:
            self._data["archive"][ck] = {
                "cell": list(cell), "label": cell_label(cell),
                "mask": sorted(mask), "k": len(mask), "sr": sr, "iter": iter_idx,
            }

    def get_archive_map(self) -> list[dict]:
        """Per-cell champions sorted by SR desc — the tradeoff map for the prompt."""
        rows = list(self._data["archive"].values())
        rows.sort(key=lambda r: r["sr"], reverse=True)
        return rows

    def record_proposed(self, mask: list[int], recovery: str, sr: Optional[float],
                         iter_idx: int, rationale: str = "") -> None:
        self._data["tried_masks"].append({
            "iter": iter_idx,
            "mask": sorted(mask),
            "k": len(mask),
            "recovery": recovery,
            "sr": sr,
            "rationale": rationale,
        })
        self._update_archive(mask, sr, iter_idx)
        self._flush()

    def _flush(self) -> None:
        self.data_path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False) + "\n")
