"""RecoveryModule — recipe enum + train() hook (stub for offline demo).

For the offline demo, train() is a no-op; the driver looks up the cached SR
via EvalModule.lookup_cached(mask, recovery). When wiring up real GPU runs
later, train() can be replaced with a subprocess call to the actual training
script.
"""
import json
from pathlib import Path
from typing import Optional


class RecoveryModule:
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        with open(self.data_path) as f:
            self._data = json.load(f)

    def list_recipes(self) -> list[dict]:
        return list(self._data["recipes"])

    def get_recipe(self, recipe_id: str) -> Optional[dict]:
        for r in self._data["recipes"]:
            if r["id"] == recipe_id:
                return dict(r)
        return None

    def is_valid(self, recipe_id: str) -> bool:
        return self.get_recipe(recipe_id) is not None

    def get_summary(self) -> dict:
        return {
            "recipes": [
                {"id": r["id"], "trainable": r["trainable"], "description": r["description"]}
                for r in self._data["recipes"]
            ],
        }

    def train(self, mask: list[int], recipe_id: str, **kwargs) -> Optional[str]:
        """Stub for offline demo. Returns None (caller uses EvalModule.lookup_cached).
        Replace with subprocess to real training script when ready."""
        return None
