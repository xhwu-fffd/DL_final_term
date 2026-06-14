# -*- coding: utf-8 -*-
"""The ACT configuration - architecture + hyperparameters in one place.

Crucially, the **same** config object is used for both the env-A model and the
env-A/B/C joint model, so the only difference between the two runs is the
training data (the assignment requires identical architecture & hyperparameters).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class ACTConfig:
    # -- data / observation ------------------------------------------------
    # LeRobot column names: "image" (static 200x200), "wrist_image" (84x84).
    image_keys: tuple[str, ...] = ("image",)
    image_size: int = 96
    state_key: str = "state"
    action_key: str = "actions"          # CALVIN rel_actions, stored as "actions"
    chunk_size: int = 10                 # action-chunk horizon H (the analysed knob)
    state_dim: int = 15                  # filled from data at build time
    action_dim: int = 7                  # filled from data at build time

    # -- vision backbone ---------------------------------------------------
    vision_backbone: str = "resnet18"    # "resnet18" | "small"
    pretrained_backbone: bool = True

    # -- transformer -------------------------------------------------------
    hidden_dim: int = 256
    n_heads: int = 8
    n_enc_layers: int = 4
    n_dec_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1

    # -- CVAE --------------------------------------------------------------
    use_vae: bool = True
    latent_dim: int = 32
    kl_weight: float = 10.0

    # -- optimisation ------------------------------------------------------
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    batch_size: int = 32
    steps: int = 20000
    log_every: int = 50
    save_every: int = 2000
    seed: int = 42

    # -- validation curve (assignment §4.1 asks for a validation metric curve) ---
    val_fraction: float = 0.05           # episodes held out of train env(s) for val
    eval_every: int = 500                # log val Action-L1 every N steps (0 = off)
    val_batches: int = 16                # #batches used per validation estimate

    # -- inference ---------------------------------------------------------
    temporal_ensemble_coeff: float = 0.01   # exp-weight m in exp(-m*i); 0 disables

    # -- normalisation -----------------------------------------------------
    state_norm: str = "mean_std"
    action_norm: str = "min_max"

    @property
    def num_cameras(self) -> int:
        return len(self.image_keys)

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_keys"] = list(self.image_keys)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "ACTConfig":
        d = dict(d)
        if "image_keys" in d:
            d["image_keys"] = tuple(d["image_keys"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "ACTConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
