"""
[5] Deployment composition.

    task_router -- TaskRouter (route each task to its best ckpt; 80.2% deployable / 88.5% oracle)
"""

from .routing import TaskRouter

__all__ = ["TaskRouter"]
