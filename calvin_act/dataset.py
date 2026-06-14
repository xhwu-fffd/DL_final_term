# -*- coding: utf-8 -*-
"""A PyTorch ``Dataset`` of action-chunk windows over preloaded CALVIN data.

Each item is one observation (multi-camera images + proprioceptive state) paired
with the next ``chunk_size`` actions and a padding mask - exactly what the ACT
policy consumes. Images are read lazily per item; the small state/action arrays
are already in RAM (see :class:`calvin_act.calvin_data.EnvData`).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .calvin_data import EnvData
from .normalize import Normalizer

# ImageNet stats (used when the vision backbone is pretrained).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prep_image(img_hwc: np.ndarray, size: int,
               mean: np.ndarray = IMAGENET_MEAN,
               std: np.ndarray = IMAGENET_STD) -> torch.Tensor:
    """uint8 HxWxC -> normalised float CHW tensor at ``size`` x ``size``."""
    from PIL import Image
    arr = np.asarray(img_hwc)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] > 3:
        arr = arr[..., :3]
    if arr.shape[0] != size or arr.shape[1] != size:
        arr = np.asarray(Image.fromarray(arr.astype(np.uint8)).resize(
            (size, size), Image.BILINEAR))
    x = arr.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class ChunkDataset(Dataset):
    """Windows of ``(images, state, action_chunk, pad_mask)``."""

    def __init__(self, env_data: EnvData, chunk_size: int, image_size: int,
                 state_norm: Normalizer, action_norm: Normalizer,
                 pretrained_backbone: bool = True):
        self.ed = env_data
        self.chunk_size = int(chunk_size)
        self.image_size = int(image_size)
        self.state_norm = state_norm
        self.action_norm = action_norm
        self.image_keys = env_data.image_keys
        if pretrained_backbone:
            self.mean, self.std = IMAGENET_MEAN, IMAGENET_STD
        else:                                          # plain [-1, 1]
            self.mean = np.array([0.5, 0.5, 0.5], np.float32)
            self.std = np.array([0.5, 0.5, 0.5], np.float32)

    def __len__(self) -> int:
        return len(self.ed)

    def __getitem__(self, row: int) -> dict[str, torch.Tensor]:
        imgs = self.ed.load_images(row)
        cam = [prep_image(imgs[k], self.image_size, self.mean, self.std)
               for k in self.image_keys if k in imgs]
        images = torch.stack(cam, dim=0)               # [num_cam, 3, H, W]

        state = torch.from_numpy(self.state_norm.transform(self.ed.states[row]))
        chunk, pad = self.ed.chunk(row, self.chunk_size)
        action = torch.from_numpy(self.action_norm.transform(chunk))   # [H, A]
        return {
            "images": images.float(),
            "state": state.float(),
            "action": action.float(),
            "pad_mask": torch.from_numpy(pad),         # True = padded
        }

    @property
    def num_cameras(self) -> int:
        return len(self.image_keys)
