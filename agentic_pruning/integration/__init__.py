"""
agentic_pruning.integration
============================

Upgrades the agentic_pruning loop from a CACHED-lookup demo into a *relatively
complete* loop whose three live seams are backed by the standalone libraries:

    probe_query  ->  mask_probe_lib   (real per-layer importance / recoverability probes)
    recovery     ->  recovery_lib     (real recovery strategies: phase2 / on-policy)
    eval         ->  backend SR        (the recovered student's success rate)

A single ``recovery_lib.RecoveryBackend`` (which is also a ``mask_probe_lib.ModelBackend``)
feeds all three. With the synthetic ``MockRecoveryBackend`` the whole agent loop runs
end-to-end with no GPU; swap in the real OFT backend (once its skeleton is filled in)
and the same loop runs on the model.

This package is ADDITIVE: it does not modify driver.py / modules/ / data/. The
original cached demo still works unchanged.
"""

import sys
from pathlib import Path

# Vendored in AutoPrune: mask_probe_lib / recovery_lib / agentic_pruning live at the
# repo root. Walk up until we find it, and put it on sys.path so the top-level
# package imports resolve regardless of where the loop is launched from.
_here = Path(__file__).resolve()
for _cand in _here.parents:
    if (_cand / "mask_probe_lib").is_dir() and (_cand / "recovery_lib").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from .lib_modules import (  # noqa: E402
    LibEvalModule,
    LibProbeQueryService,
    LibRecoveryModule,
    build_lib_module_set,
)
from .lib_engine import LibEngine  # noqa: E402

__all__ = [
    "LibEngine",
    "LibProbeQueryService",
    "LibRecoveryModule",
    "LibEvalModule",
    "build_lib_module_set",
]
