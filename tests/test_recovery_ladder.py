"""Smoke test: every plugin's recovery_ladder is valid and in sync with its
recovery_strategies, and the tier helpers resolve both vocabularies.

Run:  python -m pytest tests/test_recovery_ladder.py -q   (or just execute it)
"""
import importlib

import pytest

from autoprune.backends.recovery_ladder import TIER_ORDER, validate_ladder

PLUGIN_MODULES = {
    "oft":     ("autoprune.backends.oft_plugin", "OftPlugin"),
}


def _load(name):
    mod_name, cls_name = PLUGIN_MODULES[name]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


@pytest.mark.parametrize("name", sorted(PLUGIN_MODULES))
def test_ladder_valid_and_in_sync(name):
    cls = _load(name)
    validate_ladder(cls.recovery_ladder, cls.recovery_strategies)
    # direct is universal: every model can at least prune + eval
    assert cls.recovery_ladder.get("direct") == "direct"


@pytest.mark.parametrize("name", sorted(PLUGIN_MODULES))
def test_resolve_both_vocabularies(name):
    cls = _load(name)
    plugin = cls.__new__(cls)   # attrs are class-level; skip heavy __init__
    # tier ids resolve to the native recipe recover_and_eval dispatches on
    for tier, native in cls.recovery_ladder.items():
        assert plugin.resolve_recovery(tier) == (native if native is not None
                                                 else cls.recovery_ladder.get(tier))
        if native is not None:
            assert plugin.supports_recovery(tier)
    # native ids still resolve to themselves (backward compatible)
    for native in cls.recovery_strategies:
        assert plugin.resolve_recovery(native) == native
        assert plugin.supports_recovery(native)
    # escalation order is respected and unknown names fail closed
    tiers = plugin.supported_tiers()
    assert list(tiers) == [t for t in TIER_ORDER if t in tiers]
    assert plugin.resolve_recovery("made_up_recipe") is None
    assert not plugin.supports_recovery("made_up_recipe")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
