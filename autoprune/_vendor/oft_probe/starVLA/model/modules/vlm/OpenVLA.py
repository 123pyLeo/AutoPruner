from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from autoprune._vendor.oft_probe.deployment.model_server.tools.image_tools import to_pil_preserve
from autoprune._vendor.oft_probe.starVLA.model.modules.openvla_utils import move_batch_to_device, resize_and_center_crop
from autoprune._vendor.oft_probe.starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _OpenVLA_Interface(nn.Module):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        openvla_cfg = config.framework.get("openvla", {})
        model_path = openvla_cfg.get(
            "pretrained_checkpoint",
            "./playground/Pretrained_models/OpenVLA/openvla-7b-oft-finetuned-libero-goal",
        )
        trust_remote_code = bool(openvla_cfg.get("trust_remote_code", True))

        inference_cfg = config.framework.get("inference", {})
        self.center_crop = bool(inference_cfg.get("center_crop", True))
        self.image_size = int(inference_cfg.get("image_size", 224))

        checkpoint_dir = Path(model_path)
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"OpenVLA checkpoint directory not found: {checkpoint_dir}")

        self.processor = AutoProcessor.from_pretrained(str(checkpoint_dir), trust_remote_code=trust_remote_code)
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = "left"

        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                str(checkpoint_dir),
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
        except Exception as exc:
            raise RuntimeError(
                "OpenVLA checkpoints require a compatible environment. "
                "Use the `openvla-oft` conda env (transformers==4.40.1, timm 0.9.x)."
            ) from exc

        with open(checkpoint_dir / "dataset_statistics.json", "r") as file:
            self.norm_stats = json.load(file)
        self.model.norm_stats = self.norm_stats

    @staticmethod
    def build_prompt(instruction: str) -> str:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def prepare_image_list(self, images) -> list[Image.Image]:
        pil_images = to_pil_preserve(images)
        pil_images = pil_images if isinstance(pil_images, list) else [pil_images]
        prepared: list[Image.Image] = []
        for image in pil_images:
            image = image.convert("RGB")
            if self.center_crop:
                image = resize_and_center_crop(image, image_size=self.image_size)
            elif image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), resample=Image.BICUBIC)
            prepared.append(image)
        return prepared

    def set_num_images_in_input(self, num_images_in_input: int) -> None:
        if hasattr(self.model, "set_num_images_in_input"):
            self.model.set_num_images_in_input(int(num_images_in_input))
        elif hasattr(self.model, "vision_backbone") and hasattr(self.model.vision_backbone, "set_num_images_in_input"):
            self.model.vision_backbone.set_num_images_in_input(int(num_images_in_input))

    def build_openvla_inputs(
        self,
        image_batches: List[list[Image.Image]],
        instructions: List[str],
        num_images_in_input: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[dict[str, torch.Tensor], int]:
        if len(image_batches) != len(instructions):
            raise ValueError("image_batches and instructions must have the same batch size")
        if not image_batches:
            raise ValueError("OpenVLA input batch is empty")

        available_num_images = min(len(images) for images in image_batches)
        if available_num_images <= 0:
            raise ValueError("Each OpenVLA example must provide at least one image")

        resolved_num_images = available_num_images if num_images_in_input is None else min(
            int(num_images_in_input), available_num_images
        )
        prompts = [self.build_prompt(instruction) for instruction in instructions]
        primary_images = [images[0] for images in image_batches]
        inputs = self.processor(text=prompts, images=primary_images, return_tensors="pt", padding=True)

        if resolved_num_images > 1:
            pixel_values = [inputs["pixel_values"]]
            for image_index in range(1, resolved_num_images):
                extra_inputs = self.processor(
                    text=prompts,
                    images=[images[image_index] for images in image_batches],
                    return_tensors="pt",
                    padding=True,
                )
                pixel_values.append(extra_inputs["pixel_values"])
            inputs["pixel_values"] = torch.cat(pixel_values, dim=1)

        if device is not None:
            inputs = move_batch_to_device(inputs, device=device, dtype=dtype)
        return inputs, resolved_num_images


def get_vlm_model(config: Optional[dict] = None, **kwargs) -> _OpenVLA_Interface:
    return _OpenVLA_Interface(config=config, **kwargs)
