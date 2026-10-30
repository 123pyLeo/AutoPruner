from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from autoprune._vendor.oft_probe.starVLA.model.modules.openvla_utils import find_component_checkpoint, load_component_state_dict


class MLPResNetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) + x


class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList([MLPResNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = self.relu(x)
        for block in self.mlp_resnet_blocks:
            x = block(x)
        x = self.layer_norm2(x)
        return self.fc2(x)


class OpenVLAL1RegressionActionHead(nn.Module):
    """
    OpenVLA-OFT action head.

    Important difference from StarVLA's generic `MLP_ActionHeader`:
    OpenVLA tokenizes one action step into `action_dim` action tokens, so the head consumes
    `[B, chunk_len * action_dim, hidden_dim]`, reshapes it to `[B, chunk_len, hidden_dim * action_dim]`,
    and regresses one continuous action vector per step.
    """

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int, num_actions_chunk: int) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.num_actions_chunk = int(num_actions_chunk)
        self.model = MLPResNet(
            num_blocks=2,
            input_dim=input_dim * self.action_dim,
            hidden_dim=hidden_dim,
            output_dim=self.action_dim,
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = actions_hidden_states.shape[0]
        x = actions_hidden_states.reshape(batch_size, self.num_actions_chunk, -1)
        return self.model(x)

    def forward(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        return self.predict_action(actions_hidden_states)


def get_action_model(
    config: Optional[dict] = None,
    llm_dim: Optional[int] = None,
    **kwargs,
) -> OpenVLAL1RegressionActionHead:
    if config is None:
        raise ValueError("OpenVLA action head requires the global config")
    if llm_dim is None:
        raise ValueError("OpenVLA action head requires `llm_dim` from the loaded VLM")

    action_cfg = config.framework.action_model
    action_dim = int(action_cfg.action_dim)
    num_actions_chunk = int(action_cfg.past_action_window_size) + int(action_cfg.future_action_window_size) + 1
    checkpoint_dir = Path(config.framework.openvla.pretrained_checkpoint)

    action_head = OpenVLAL1RegressionActionHead(
        input_dim=int(llm_dim),
        hidden_dim=int(llm_dim),
        action_dim=action_dim,
        num_actions_chunk=num_actions_chunk,
    )
    checkpoint_path = find_component_checkpoint(checkpoint_dir, "action_head")
    action_head.load_state_dict(load_component_state_dict(checkpoint_path))
    action_head = action_head.to(dtype=torch.bfloat16)
    action_head.eval()
    return action_head
