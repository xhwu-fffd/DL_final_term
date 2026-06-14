# -*- coding: utf-8 -*-
"""calvin_act - ACT cross-environment generalization on CALVIN (HW3 Task 2).

Train Action-Chunking-with-Transformers (ACT) on CALVIN environment A, then on
A+B+C jointly with identical architecture/hyperparameters, and zero-shot test both
on the unseen environment D to study how action chunking holds up under visual
distribution shift.

The data / normalisation / metrics layers are NumPy-only; the model, trainer and
evaluator use PyTorch and are imported lazily, so ``import calvin_act`` stays cheap
when you only need the data utilities.
"""

from __future__ import annotations

import importlib
import typing

# -- eager, lightweight (NumPy-only) ----------------------------------------
from .calvin_data import (
    EnvData, find_data_root, default_data_root, env_dir, available_envs,
    load_info, load_episodes_meta, ENVS,
)
from .normalize import Normalizer, fit_normalizers
from .metrics import (
    action_l1, action_mse, per_dim_l1, per_chunk_step_l1, success_rate,
    calvin_sequence_metrics, write_comparison_table,
)
from .act_config import ACTConfig

# -- lazy, torch-backed -----------------------------------------------------
_LAZY = {
    "ACTPolicy": "act_model", "count_parameters": "act_model",
    "ChunkDataset": "dataset",
    "TemporalEnsembler": "temporal_ensemble",
    "train": "train", "save_checkpoint": "train",
    "evaluate_offline": "evaluate", "load_policy": "evaluate",
    "closed_loop_rollout": "evaluate",
    "describe_lerobot": "lerobot_export",
}


def __getattr__(name: str):                            # PEP 562 lazy attributes
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'calvin_act' has no attribute {name!r}")
    return getattr(importlib.import_module(f".{mod}", __name__), name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    "EnvData", "find_data_root", "default_data_root", "env_dir", "available_envs",
    "load_info", "load_episodes_meta", "ENVS",
    "Normalizer", "fit_normalizers",
    "action_l1", "action_mse", "per_dim_l1", "per_chunk_step_l1", "success_rate",
    "calvin_sequence_metrics", "write_comparison_table",
    "ACTConfig",
    "ACTPolicy", "count_parameters", "ChunkDataset", "TemporalEnsembler",
    "train", "save_checkpoint", "evaluate_offline", "load_policy",
    "closed_loop_rollout", "describe_lerobot",
]

__version__ = "0.1.0"
