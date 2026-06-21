# -*- coding: utf-8 -*-
"""Train an ACT policy on a chosen set of CALVIN environments.

The same :class:`ACTConfig` drives both required runs — env **A** only (Task 1)
and env **A+B+C** jointly (Task 2) — so the only difference is ``train_envs``. We
log the **Action L1 loss** (the assignment's key convergence metric) plus the KL
term and a held-out **validation Action-L1** curve (§4.1), checkpoint periodically,
and save the config + normalisers + loss history for plotting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .act_config import ACTConfig
from .act_model import ACTPolicy, count_parameters
from .calvin_data import EnvData, find_data_root
from .dataset import ChunkDataset
from .logger import RunLogger
from .normalize import fit_normalizers, normalizers_to_dict


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _to_device(batch: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def save_checkpoint(path: str | Path, model: ACTPolicy, cfg: ACTConfig,
                    normalizers: dict, history: list, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "normalizers": normalizers_to_dict(normalizers),
        "history": history,
        "extra": extra or {},
    }, path)


@torch.no_grad()
def _validation_l1(model: ACTPolicy, loader, device: str, max_batches: int) -> float:
    """Mean masked Action-L1 in *normalised* space over a few val batches
    (directly comparable to the training L1 plotted on the same axis)."""
    model.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = _to_device(batch, device)
        pred = model.predict_chunk(b["images"], b["state"])
        valid = (~b["pad_mask"]).unsqueeze(-1).float()
        l1 = ((pred - b["action"]).abs() * valid).sum() / (
            valid.sum() * b["action"].shape[-1]).clamp(min=1)
        tot += float(l1)
        n += 1
    model.train()
    return tot / max(n, 1)


def train(cfg: ACTConfig, data_root: str | Path | None, train_envs,
          out_dir: str | Path, device: str | None = None,
          logger_backend: str = "none", run_name: str | None = None,
          num_workers: int = 0, max_steps: int | None = None,
          max_train_steps_data: int | None = None, max_episodes: int | None = None,
          val_fraction: float | None = None, use_disk_cache: bool = True) -> dict:
    device = pick_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    root = find_data_root(data_root)
    vf = cfg.val_fraction if val_fraction is None else float(val_fraction)
    use_val = vf and vf > 0.0

    # -- training data -----------------------------------------------------
    ed = EnvData.build(root, train_envs, image_keys=cfg.image_keys,
                       state_key=cfg.state_key, action_key=cfg.action_key,
                       max_episodes=max_episodes, max_steps=max_train_steps_data,
                       subset=("train" if use_val else "all"), val_fraction=vf,
                       split_seed=cfg.seed, use_disk_cache=use_disk_cache)
    cfg.state_dim, cfg.action_dim = ed.state_dim, ed.action_dim       # lock dims to data
    normalizers = fit_normalizers(ed, cfg.state_norm, cfg.action_norm)
    dataset = ChunkDataset(ed, cfg.chunk_size, cfg.image_size,
                           normalizers["state"], normalizers["action"],
                           pretrained_backbone=cfg.pretrained_backbone)
    loader = DataLoader(dataset, batch_size=min(cfg.batch_size, len(dataset)),
                        shuffle=True, num_workers=num_workers, drop_last=True,
                        pin_memory=(device == "cuda"),
                        persistent_workers=(num_workers > 0))

    # -- held-out validation data (shares the training normalisers) --------
    val_loader = None
    if use_val:
        try:
            ved = EnvData.build(root, train_envs, image_keys=cfg.image_keys,
                                state_key=cfg.state_key, action_key=cfg.action_key,
                                max_episodes=max_episodes, subset="val",
                                val_fraction=vf, split_seed=cfg.seed,
                                use_disk_cache=use_disk_cache, verbose=False)
            val_ds = ChunkDataset(ved, cfg.chunk_size, cfg.image_size,
                                  normalizers["state"], normalizers["action"],
                                  pretrained_backbone=cfg.pretrained_backbone)
            val_loader = DataLoader(val_ds, batch_size=min(cfg.batch_size, len(val_ds)),
                                    shuffle=False, num_workers=num_workers)
        except Exception as e:
            print(f"[train] (warning) no validation split ({e}); skipping val curve.")

    # -- model / optim -----------------------------------------------------
    model = ACTPolicy(cfg).to(device)
    backbone_ids = set(id(p) for p in model.backbone.parameters())
    groups = [
        {"params": [p for p in model.parameters()
                    if p.requires_grad and id(p) not in backbone_ids], "lr": cfg.lr},
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": cfg.lr_backbone},
    ]
    optim = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = int(max_steps if max_steps is not None else cfg.steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=steps)

    logger = RunLogger(logger_backend, name=run_name or f"act_{'_'.join(train_envs)}",
                       config={**cfg.to_dict(), "train_envs": list(train_envs)})
    print(f"[train] envs={list(train_envs)} device={device} "
          f"params={count_parameters(model):,} steps={steps} "
          f"batches/epoch={len(loader)}"
          + (f"  val_steps={len(val_loader.dataset)}" if val_loader else ""))

    # -- loop --------------------------------------------------------------
    history: list[dict] = []
    data_iter = _cycle(loader)
    model.train()
    t0 = time.perf_counter()
    run_l1 = run_kl = run_loss = 0.0
    eval_every = int(cfg.eval_every)
    for step in range(1, steps + 1):
        batch = _to_device(next(data_iter), device)
        out = model(batch)
        optim.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        sched.step()

        run_loss += float(out["loss"].detach()); run_l1 += float(out["l1"]); run_kl += float(out["kl"])
        if step % cfg.log_every == 0:
            n = cfg.log_every
            rec = {"step": step, "loss": round(run_loss / n, 6),
                   "action_l1": round(run_l1 / n, 6), "kl": round(run_kl / n, 6),
                   "lr": optim.param_groups[0]["lr"]}
            if val_loader is not None and eval_every > 0 and step % eval_every == 0:
                rec["val_action_l1"] = round(
                    _validation_l1(model, val_loader, device, cfg.val_batches), 6)
            history.append(rec)
            logger.log({k: v for k, v in rec.items() if k != "step"}, step)
            msg = (f"  step {step:>6}/{steps}  loss {rec['loss']:.4f}  "
                   f"L1 {rec['action_l1']:.4f}  kl {rec['kl']:.4f}")
            if "val_action_l1" in rec:
                msg += f"  val_L1 {rec['val_action_l1']:.4f}"
            print(msg)
            run_loss = run_l1 = run_kl = 0.0
        if step % cfg.save_every == 0:
            save_checkpoint(out_dir / "checkpoint.pt", model, cfg, normalizers, history)

    elapsed = time.perf_counter() - t0
    summary = {"train_envs": list(train_envs), "steps": steps,
               "train_steps_data": len(ed), "num_segments": len(ed.segments),
               "num_params": count_parameters(model), "device": device,
               "train_time_s": round(elapsed, 2),
               "final_action_l1": history[-1]["action_l1"] if history else None,
               "final_val_action_l1": next(
                   (h["val_action_l1"] for h in reversed(history)
                    if "val_action_l1" in h), None)}

    # -- save artifacts ----------------------------------------------------
    save_checkpoint(out_dir / "model_final.pt", model, cfg, normalizers, history, extra=summary)
    cfg.save(out_dir / "config.json")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.finish()
    print(f"[train] done in {elapsed:.1f}s -> {out_dir}  (final L1 {summary['final_action_l1']})")
    return {"history": history, "summary": summary, "ckpt": str(out_dir / "model_final.pt")}
