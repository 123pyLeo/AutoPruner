from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import numpy as np
import torch
from PIL import Image


def load_component_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        cleaned_state_dict[key[7:] if key.startswith("module.") else key] = value
    return cleaned_state_dict


def find_component_checkpoint(ckpt_dir: str | Path, prefix: str) -> Path:
    checkpoint_dir = Path(ckpt_dir)
    matches = sorted(checkpoint_dir.glob(f"{prefix}*checkpoint.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one `{prefix}*checkpoint.pt` under {checkpoint_dir}, found {len(matches)}"
        )
    return matches[0]


def resize_and_center_crop(image: Image.Image, image_size: int = 224, crop_scale: float = 0.9) -> Image.Image:
    """Resize then center-crop using OpenVLA-OFT's area-based crop scale."""
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), resample=Image.BICUBIC)

    crop_size = max(1, int(round(image_size * math.sqrt(crop_scale))))
    left = (image_size - crop_size) // 2
    top = (image_size - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((image_size, image_size), resample=Image.BICUBIC)


def normalize_bounds(value: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    if "q01" in stats and "q99" in stats:
        high, low = np.asarray(stats["q99"]), np.asarray(stats["q01"])
    else:
        high, low = np.asarray(stats["max"]), np.asarray(stats["min"])
    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2 * (value - low) / (high - low + 1e-8) - 1, value)
    return np.clip(normalized, -1.0, 1.0)


def move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype | None = None) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype or value.dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved
