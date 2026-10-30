"""
mask_probe_lib.core.registry
=============================

A tiny name -> class registry so every verified method can be discovered and
constructed uniformly::

    from mask_probe_lib.core.registry import build, available

    probe = build("knockout")          # -> AblationKnockoutProbe()
    print(available("layer_importance"))

Each probe module registers itself with :func:`register`.  This keeps the calling
convention identical across the five pipeline layers, which is the whole point of
the "standardised interface" the library exists to provide.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

# category -> {name -> factory}
_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "layer_importance": {},
    "mask_recoverability": {},
    "sr_surrogate": {},
    "mask_selection": {},
    "validation": {},
}


def register(category: str, name: str):
    """Class decorator: ``@register("layer_importance", "knockout")``."""
    if category not in _REGISTRY:
        _REGISTRY[category] = {}

    def deco(cls):
        _REGISTRY[category][name] = cls
        cls.registry_name = name
        cls.registry_category = category
        return cls

    return deco


def build(name: str, category: Optional[str] = None, **kwargs):
    """Instantiate a registered method by name (optionally scoped to a category)."""
    cats = [category] if category else list(_REGISTRY)
    for cat in cats:
        if name in _REGISTRY[cat]:
            return _REGISTRY[cat][name](**kwargs)
    raise KeyError(f"no registered method {name!r} in {cats}")


def available(category: Optional[str] = None) -> Dict[str, List[str]]:
    """List registered method names, by category."""
    if category:
        return {category: sorted(_REGISTRY.get(category, {}))}
    return {c: sorted(m) for c, m in _REGISTRY.items()}
