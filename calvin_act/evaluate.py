# -*- coding: utf-8 -*-
"""Zero-shot cross-environment evaluation of a trained ACT policy (Task 3).

Primary (runs anywhere, no simulator): **open-loop action error** on the unseen
environment's play data. The policy predicts an action chunk from each observation
and we compare it (in raw action units) to the ground-truth teleop actions. Each
model applies *its own training-set normaliser*, and we report errors in raw space
so the env-A and env-A/B/C models are compared fairly. Lower error under env D =
better generalisation under visual distribution shift.

Also provided: :func:`closed_loop_rollout`, a generic loop (with ACT temporal
ensembling) that drives any gym-like environment — wire CALVIN's PyBullet env into
it for the success-rate metric (see README; the simulator is an extra install).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .act_config import ACTConfig
from .act_model import ACTPolicy
from .calvin_data import EnvData, find_data_root
from .dataset import ChunkDataset
from .metrics import action_l1, action_mse, per_dim_l1, per_chunk_step_l1
from .normalize import normalizers_from_dict
from .temporal_ensemble import TemporalEnsembler
from .train import pick_device


def load_policy(ckpt_path: str | Path, device: str | None = None):
    """Load a checkpoint into ``(model, cfg, normalizers)``."""
    device = pick_device(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ACTConfig.from_dict(ckpt["config"])
    model = ACTPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    norms = normalizers_from_dict(ckpt["normalizers"])
    return model, cfg, norms


def evaluate_offline(ckpt_path: str | Path, data_root: str | Path | None, eval_envs,
                     device: str | None = None, max_steps: int | None = 5000,
                     batch_size: int = 64, num_workers: int = 0,
                     max_episodes: int | None = None,
                     out: str | Path | None = None) -> dict:
    device = pick_device(device)
    model, cfg, norms = load_policy(ckpt_path, device)

    root = find_data_root(data_root)
    ed = EnvData.build(root, eval_envs, image_keys=cfg.image_keys,
                       state_key=cfg.state_key, action_key=cfg.action_key,
                       max_episodes=max_episodes, subset="all", verbose=False)
    ds = ChunkDataset(ed, cfg.chunk_size, cfg.image_size, norms["state"],
                      norms["action"], pretrained_backbone=cfg.pretrained_backbone)
    if max_steps is not None and len(ds) > max_steps:
        idx = np.linspace(0, len(ds) - 1, max_steps).astype(int)
        ds = Subset(ds, idx.tolist())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    preds, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            state = batch["state"].to(device)
            pred = model.predict_chunk(images, state).cpu().numpy()    # normalised
            preds.append(pred)
            targets.append(batch["action"].numpy())
            masks.append(batch["pad_mask"].numpy())
    pred = np.concatenate(preds)          # [N, H, A] normalised
    target = np.concatenate(targets)
    mask = np.concatenate(masks)

    # raw action space (fair across models with different normalisers)
    inv = norms["action"].inverse
    pred_raw = inv(pred.reshape(-1, cfg.action_dim)).reshape(pred.shape)
    target_raw = inv(target.reshape(-1, cfg.action_dim)).reshape(target.shape)

    # first-step error = what closed-loop executes before ensembling
    first_mask = mask[:, :1]
    result = {
        "checkpoint": str(ckpt_path),
        "eval_envs": list(eval_envs),
        "num_windows": int(pred.shape[0]),
        "chunk_size": cfg.chunk_size,
        "action_l1_raw": round(action_l1(pred_raw, target_raw, mask), 6),
        "action_mse_raw": round(action_mse(pred_raw, target_raw, mask), 6),
        "action_l1_norm": round(action_l1(pred, target, mask), 6),
        "first_step_l1_raw": round(action_l1(pred_raw[:, :1], target_raw[:, :1], first_mask), 6),
        "per_dim_l1_raw": per_dim_l1(pred_raw, target_raw, mask),
        "per_chunk_step_l1_raw": per_chunk_step_l1(pred_raw, target_raw, mask),
    }
    print(f"[eval] {Path(ckpt_path).parent.name} on envs={list(eval_envs)}: "
          f"raw L1={result['action_l1_raw']:.4f} (first-step "
          f"{result['first_step_l1_raw']:.4f}), MSE={result['action_mse_raw']:.4f}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[eval] wrote {out}")
    return result


# ---------------------------------------------------------------------------
# Closed-loop rollout (generic; plug in CALVIN's simulator for success rate)
# ---------------------------------------------------------------------------
@torch.no_grad()
def closed_loop_rollout(model: ACTPolicy, cfg: ACTConfig, norms: dict, env,
                        max_steps: int = 360, device: str | None = None,
                        use_temporal_ensemble: bool = True,
                        is_success=None) -> dict:
    """Drive ``env`` with the policy until ``max_steps`` or success.

    ``env`` must expose ``reset() -> obs`` and ``step(action) -> (obs, reward,
    done, info)`` where ``obs`` is a dict with the configured image keys and a
    ``cfg.state_key`` state. Uses ACT temporal ensembling to turn overlapping
    chunks into a stable command. Returns steps / success.
    """
    from .dataset import prep_image, IMAGENET_MEAN, IMAGENET_STD
    device = pick_device(device)
    model.eval()
    ens = TemporalEnsembler(cfg.chunk_size, cfg.temporal_ensemble_coeff)
    ens.reset()
    if cfg.pretrained_backbone:
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean = np.array([0.5, 0.5, 0.5], np.float32)
        std = np.array([0.5, 0.5, 0.5], np.float32)

    obs = env.reset()
    success = False
    steps = 0
    for steps in range(1, max_steps + 1):
        cams = [prep_image(np.asarray(obs[k]), cfg.image_size, mean, std)
                for k in cfg.image_keys if k in obs]
        images = torch.stack(cams).unsqueeze(0).to(device)             # [1,K,3,H,W]
        state_np = norms["state"].transform(np.asarray(obs[cfg.state_key]).reshape(-1))
        state = torch.from_numpy(state_np).float().unsqueeze(0).to(device)

        chunk = model.predict_chunk(images, state)                     # [1,H,A] norm
        if use_temporal_ensemble:
            act_norm = ens.update(chunk)                               # [1,A]
        else:
            act_norm = chunk[:, 0]
        action = norms["action"].inverse(act_norm.cpu().numpy().reshape(-1))

        obs, _, done, info = env.step(action)
        if is_success is not None and is_success(info):
            success = True
            break
        if done:
            break
    return {"steps": steps, "success": bool(success)}


def rollout_calvin(*args, **kwargs):
    """Closed-loop CALVIN success-rate eval. Requires the CALVIN simulator
    (``calvin_env`` + PyBullet) which is a separate install — see README."""
    try:
        import calvin_env  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "closed-loop CALVIN rollout needs the simulator: install `calvin_env` "
            "(PyBullet) from https://github.com/mees/calvin and build the CALVIN "
            "task environment, then construct the env and call "
            "`closed_loop_rollout(model, cfg, norms, env, is_success=...)`. "
            "Until then use the offline `evaluate_offline` action-error metric.") from e
