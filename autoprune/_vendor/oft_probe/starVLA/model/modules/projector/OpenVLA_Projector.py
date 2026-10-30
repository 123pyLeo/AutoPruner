from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from autoprune._vendor.oft_probe.starVLA.model.modules.openvla_utils import find_component_checkpoint, load_component_state_dict


class OpenVLAProprioProjector(nn.Module):
    """
    Simple 2-layer GELU MLP copied into StarVLA so OpenVLA-OFT no longer depends on
    `prismatic.models.projectors.ProprioProjector`.
    """

    def __init__(self, llm_dim: int, proprio_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(proprio_dim, llm_dim, bias=True)
        self.fc2 = nn.Linear(llm_dim, llm_dim, bias=True)
        self.act_fn1 = nn.GELU()

    def forward(self, proprio: torch.Tensor | None = None) -> torch.Tensor:
        projected = self.fc1(proprio)
        projected = self.act_fn1(projected)
        projected = self.fc2(projected)
        return projected


def get_proprio_projector(
    config: Optional[dict] = None,
    llm_dim: Optional[int] = None,
    proprio_dim: int = 8,
    **kwargs,
) -> OpenVLAProprioProjector:
    if config is None:
        raise ValueError("OpenVLA proprio projector requires the global config")
    if llm_dim is None:
        raise ValueError("OpenVLA proprio projector requires `llm_dim` from the loaded VLM")

    checkpoint_dir = Path(config.framework.openvla.pretrained_checkpoint)
    projector = OpenVLAProprioProjector(llm_dim=int(llm_dim), proprio_dim=int(proprio_dim))
    checkpoint_path = find_component_checkpoint(checkpoint_dir, "proprio_projector")
    projector.load_state_dict(load_component_state_dict(checkpoint_path))
    projector = projector.to(dtype=torch.bfloat16)
    projector.eval()
    return projector
