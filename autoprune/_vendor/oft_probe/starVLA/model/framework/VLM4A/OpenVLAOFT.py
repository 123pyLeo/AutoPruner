from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from autoprune._vendor.oft_probe.starVLA.model.framework.base_framework import baseframework
from autoprune._vendor.oft_probe.starVLA.model.framework.share_tools import merge_framework_config
from autoprune._vendor.oft_probe.starVLA.model.modules.action_model.OpenVLA_ActionHead import get_action_model
from autoprune._vendor.oft_probe.starVLA.model.modules.openvla_utils import normalize_bounds
from autoprune._vendor.oft_probe.starVLA.model.modules.projector.OpenVLA_Projector import get_proprio_projector
from autoprune._vendor.oft_probe.starVLA.model.modules.vlm import get_vlm_model
from autoprune._vendor.oft_probe.starVLA.model.tools import FRAMEWORK_REGISTRY, FrameworkTools

IGNORE_INDEX = -100


@dataclass
class OpenVLAOFTDefaultConfig:
    name: str = "OpenVLAOFT"
    openvla: dict = field(default_factory=lambda: {
        "pretrained_checkpoint": "./playground/Pretrained_models/OpenVLA/openvla-7b-oft-finetuned-libero-goal",
        "trust_remote_code": True,
    })
    action_model: dict = field(default_factory=lambda: {
        "action_model_type": "OpenVLAL1Regression",
        "action_dim": 7,
        "future_action_window_size": 7,
        "past_action_window_size": 0,
    })
    inference: dict = field(default_factory=lambda: {
        "num_images_in_input": 2,
        "use_proprio": True,
        "center_crop": True,
        "image_size": 224,
        "unnorm_key": None,
    })


@FRAMEWORK_REGISTRY.register("OpenVLAOFT")
class OpenVLAOFT(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(OpenVLAOFTDefaultConfig, config)

        self.vlm_interface = get_vlm_model(config=self.config)
        self.openvla_interface = self.vlm_interface
        self.openvla_model = self.vlm_interface.model
        self.openvla_processor = self.vlm_interface.processor
        self.norm_stats = self.vlm_interface.norm_stats

        self.unnorm_key = FrameworkTools.check_unnorm_key(
            self.norm_stats,
            self.config.framework.inference.unnorm_key,
        )

        self.action_dim = int(self.config.framework.action_model.action_dim)
        self.future_action_window_size = int(self.config.framework.action_model.future_action_window_size)
        self.past_action_window_size = int(self.config.framework.action_model.past_action_window_size)
        self.chunk_len = self.future_action_window_size + self.past_action_window_size + 1
        self.action_token_count = self.chunk_len * self.action_dim

        self.action_model = get_action_model(config=self.config, llm_dim=self.openvla_model.llm_dim)
        self.action_head = self.action_model

        self.use_proprio = bool(self.config.framework.inference.use_proprio)
        self.default_num_images_in_input = int(self.config.framework.inference.num_images_in_input)
        self.proprio_dim = 8
        self.proprio_projector = (
            get_proprio_projector(
                config=self.config,
                llm_dim=self.openvla_model.llm_dim,
                proprio_dim=self.proprio_dim,
            )
            if self.use_proprio
            else None
        )

        self.l1_loss = nn.L1Loss()

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        return next(module.parameters()).device

    @staticmethod
    def _module_dtype(module: nn.Module) -> torch.dtype:
        return next(module.parameters()).dtype

    def _normalize_actions_for_starvla(self, actions: np.ndarray) -> np.ndarray:
        stats = self.norm_stats[self.unnorm_key]["action"]
        normalized = normalize_bounds(actions, stats)
        if normalized.shape[-1] > 6:
            normalized[..., 6] = np.where(actions[..., 6] < 0.5, -1.0, 1.0)
        return normalized.astype(np.float32)

    def _normalize_proprio(self, state: np.ndarray) -> np.ndarray:
        proprio_stats = self.norm_stats[self.unnorm_key]["proprio"]
        return normalize_bounds(state, proprio_stats).astype(np.float32)

    def _prepare_batch_inputs(self, examples: List[dict]) -> tuple[dict[str, torch.Tensor], Optional[torch.Tensor], int]:
        model_device = self._module_device(self.openvla_model)
        model_dtype = self._module_dtype(self.openvla_model)

        instructions = [example["lang"] for example in examples]
        image_batches = [self.vlm_interface.prepare_image_list(example["image"]) for example in examples]
        inputs, num_images_in_input = self.vlm_interface.build_openvla_inputs(
            image_batches=image_batches,
            instructions=instructions,
            num_images_in_input=self.default_num_images_in_input,
            device=model_device,
            dtype=model_dtype,
        )
        self.vlm_interface.set_num_images_in_input(num_images_in_input)

        proprio = None
        if self.use_proprio:
            raw_states = []
            for example in examples:
                raw_state = np.asarray(
                    example.get("state", np.zeros(self.proprio_dim, dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1)
                if raw_state.shape[0] != self.proprio_dim:
                    raise ValueError(
                        f"OpenVLA-OFT expects proprio dim {self.proprio_dim}, got shape {raw_state.shape}"
                    )
                raw_states.append(raw_state)
            proprio = torch.as_tensor(
                self._normalize_proprio(np.stack(raw_states, axis=0)),
                device=model_device,
                dtype=model_dtype,
            )
        return inputs, proprio, num_images_in_input

    def _prepare_action_prediction_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        input_ids, attention_mask = self.openvla_model._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = self.openvla_model._prepare_labels_for_action_prediction(labels, input_ids)
        return input_ids, attention_mask, labels

    def _extract_action_hidden_states(
        self,
        model_output,
        labels: torch.Tensor,
        include_proprio_token: bool,
    ) -> torch.Tensor:
        if hasattr(self.openvla_model, "get_num_patches"):
            num_patch_tokens = int(self.openvla_model.get_num_patches())
        else:
            num_patch_tokens = int(
                self.openvla_model.vision_backbone.get_num_patches()
                * self.openvla_model.vision_backbone.get_num_images_in_input()
            )
        num_patch_tokens += int(include_proprio_token)
        last_hidden_states = model_output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patch_tokens:-1]
        token_labels = labels[:, 1:]
        all_actions_mask = self.openvla_model._process_action_masks(token_labels)
        batch_size = text_hidden_states.shape[0]
        return text_hidden_states[all_actions_mask].reshape(batch_size, self.action_token_count, -1)

    def forward(self, examples: List[dict] = None, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        inputs, proprio, _ = self._prepare_batch_inputs(examples)
        input_ids, attention_mask, labels = self._prepare_action_prediction_inputs(
            inputs["input_ids"],
            inputs["attention_mask"],
        )

        autocast_enabled = self._module_device(self.openvla_model).type == "cuda"
        with torch.autocast("cuda", dtype=self._module_dtype(self.openvla_model), enabled=autocast_enabled):
            output = self.openvla_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=inputs["pixel_values"],
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                proprio=proprio,
                proprio_projector=self.proprio_projector if proprio is not None else None,
            )

        action_dtype = self._module_dtype(self.action_model)
        action_device = self._module_device(self.action_model)
        with torch.autocast("cuda", dtype=torch.float32, enabled=action_device.type == "cuda"):
            action_hidden_states = self._extract_action_hidden_states(
                output,
                labels=labels,
                include_proprio_token=proprio is not None,
            ).to(device=action_device, dtype=action_dtype)
            predicted_actions = self.action_model.predict_action(action_hidden_states)

            actions_target = torch.as_tensor(
                np.asarray([example["action"] for example in examples]),
                device=predicted_actions.device,
                dtype=predicted_actions.dtype,
            )
            actions_target = actions_target[:, -self.chunk_len :, :]
            action_loss = self.l1_loss(predicted_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        normalized_actions_batch = []
        for example in examples:
            inputs, proprio, _ = self._prepare_batch_inputs([example])
            prediction = self.openvla_model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False,
                proprio=None if proprio is None else proprio[0].detach().float().cpu().numpy(),
                proprio_projector=self.proprio_projector if proprio is not None else None,
                action_head=self.action_model,
            )
            actions = prediction[0] if isinstance(prediction, tuple) else prediction
            normalized_actions_batch.append(self._normalize_actions_for_starvla(np.asarray(actions)))

        return {"normalized_actions": np.stack(normalized_actions_batch, axis=0)}
