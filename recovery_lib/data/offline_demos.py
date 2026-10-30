"""OfflineDemoSource -- streamed offline demonstration batches (Phase-2 default)."""

from __future__ import annotations

from typing import Iterable, Optional

from ..core.backend import RecoveryBackend
from ..core.registry import register
from ..core.types import Batch


@register("recovery_data", "offline_demos")
class OfflineDemoSource:
    """Thin wrapper over ``backend.iter_offline_batches`` (optionally task-filtered)."""

    name = "offline_demos"

    def __init__(self, batch_size: int = 4, task_filter: Optional[str] = None):
        self.batch_size = batch_size
        self.task_filter = task_filter

    def batches(self, backend: RecoveryBackend, steps: int) -> Iterable[Batch]:
        yield from backend.iter_offline_batches(steps, self.batch_size, self.task_filter)
