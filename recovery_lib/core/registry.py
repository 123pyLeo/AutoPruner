"""Re-export the shared registry from ``mask_probe_lib`` so recovery methods register
into the same namespace (categories: recovery_adapter / recovery_objective /
recovery_strategy / recovery_data / recovery_deploy / validation)."""

from mask_probe_lib.core.registry import available, build, register

__all__ = ["register", "build", "available"]
