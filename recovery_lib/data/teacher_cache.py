"""
TeacherCache -- interface for cached frozen-teacher targets.
============================================================

Source
------
* openvla : ``prismatic/training/teacher_cache.py`` (TeacherOutputCache) +
            ``cache_teacher_*.py`` builders.

Caching the dense teacher's action-range logits (keyed by input hash) makes
distillation cheap and reproducible for OFFLINE recovery. For ON-POLICY recovery
the teacher must be queried LIVE on the student's visited states (the cache can't
contain them) -- so this is an offline-only accelerator.
"""

from __future__ import annotations

from typing import Optional


class TeacherCache:
    """Minimal cache interface. A real backend implements ``lookup``/``has`` over a
    hash of ``(input_ids, attention_mask)`` returning action-range logits + gt loss.
    """

    def has(self, key) -> bool:                      # pragma: no cover - interface
        raise NotImplementedError

    def lookup(self, key) -> Optional[object]:       # pragma: no cover - interface
        raise NotImplementedError

    @staticmethod
    def is_usable_for_onpolicy() -> bool:
        """Always False: on-policy states are not in any offline cache (query live)."""
        return False
