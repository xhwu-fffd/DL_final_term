# -*- coding: utf-8 -*-
"""Shared utilities for the Task-2 deep-analysis suite.

Everything here is read-only w.r.t. the trained runs: it loads the checkpoints in
``outputs/`` and the LeRobot data in ``data/calvin-lerobot`` and produces metrics
+ figures under ``analyze/results`` and ``analyze/figures``.

Key design choices (see analyze/README.md for the rationale):

* **Full-split re-evaluation.** The ``val_action_l1`` logged *during* training is a
  small, fixed first-512-window estimate (high variance, pessimistic). Here we
  re-evaluate every model on a *representative* held-out sample of each
  environment (evenly spaced over the whole val split) so the numbers are
  low-variance and comparable across models/envs.
* **Held-out everywhere.** We always evaluate on ``subset="val"`` with the same
  ``val_fraction`` / ``split_seed`` used at train time, so no evaluated frame was
  a training frame for *either* model.
* **Raw action units.** Errors are reported after inverting each model's own
  normaliser, exactly like ``calvin_act.evaluate``, so the env-A and env-A/B/C
  models are compared fairly.
* **B / C as the unseen-env proxy.** Env D is not present in this dataset, so the
  A-only model's error on B and C (environments it never saw) is our stand-in for
  the zero-shot env-D test of Task 3 — a genuine visual-distribution-shift signal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# make `calvin_act` importable when running these scripts directly
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from calvin_act.calvin_data import EnvData, find_data_root          # noqa: E402
from calvin_act.dataset import ChunkDataset                          # noqa: E402
from calvin_act.evaluate import load_policy                          # noqa: E402

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
OUTPUTS = REPO / "outputs"
ANALYZE = REPO / "analyze"
RESULTS = ANALYZE / "results"
FIGURES = ANALYZE / "figures"
ARRAYS = RESULTS / "arrays"
for _d in (RESULTS, FIGURES, ARRAYS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# What we analyse
# ---------------------------------------------------------------------------
# The two "headline" models (identical config, only the training data differs).
MAIN_MODELS = {
    "ACT-A (A only)":     OUTPUTS / "task1_envA_40k" / "model_final.pt",
    "ACT-ABC (A+B+C)":    OUTPUTS / "task2_envABC_40k" / "model_final.pt",
}
# Chunk-size ablation (all A+B+C, 40k, identical except H).
CHUNK_MODELS = {
    5:  OUTPUTS / "task2_envABC_H5_40k" / "model_final.pt",
    10: OUTPUTS / "task2_envABC_40k" / "model_final.pt",
    20: OUTPUTS / "task2_envABC_H20_40k" / "model_final.pt",
}
EVAL_ENVS = ["A", "B", "C"]            # D is absent; B/C stand in as unseen for ACT-A

# CALVIN rel_actions layout (7-D)
ACTION_GROUPS = [("translation", [0, 1, 2]), ("rotation", [3, 4, 5]), ("gripper", [6])]
ACTION_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
GRIPPER_DIM = 6

# consistent colours
ENV_COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728"}
MODEL_COLORS = {"ACT-A (A only)": "#d62728", "ACT-ABC (A+B+C)": "#1f77b4"}

VAL_FRACTION = 0.05
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Plot style (lazy so the data-only scripts don't need matplotlib)
# ---------------------------------------------------------------------------
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
        "figure.facecolor": "white", "savefig.bbox": "tight",
    })
    return plt


# ---------------------------------------------------------------------------
# Held-out evaluation of one checkpoint on one environment
# ---------------------------------------------------------------------------
def evaluate_model_on_env(ckpt: str | Path, env: str, n_windows: int = 3000,
                          device: str | None = None, num_workers: int = 4,
                          batch_size: int = 64):
    """Open-loop predict the action chunk for a representative held-out sample.

    Returns a dict of numpy arrays:
      pred_raw  [N, H, A]   predicted chunk in raw action units
      tgt_raw   [N, H, A]   ground-truth teleop chunk in raw units
      mask      [N, H]      True = padded (invalid) step
      pos       [N]         position of the observation within its episode (0..1)
      chunk_size, env, ckpt
    """
    model, cfg, norms = load_policy(ckpt, device)
    dev = next(model.parameters()).device
    root = find_data_root(None)
    ed = EnvData.build(root, [env], image_keys=cfg.image_keys, state_key=cfg.state_key,
                       action_key=cfg.action_key, subset="val",
                       val_fraction=VAL_FRACTION, split_seed=SPLIT_SEED, verbose=False)

    n = len(ed)
    rows = np.linspace(0, n - 1, min(n_windows, n)).astype(int)
    # position within episode for each sampled row
    pos = np.empty(len(rows), dtype=np.float32)
    for i, r in enumerate(rows):
        s, e = ed.segment_of(int(r))
        pos[i] = (int(r) - s) / max(e - s - 1, 1)

    ds = ChunkDataset(ed, cfg.chunk_size, cfg.image_size, norms["state"],
                      norms["action"], pretrained_backbone=cfg.pretrained_backbone)
    from torch.utils.data import Subset
    loader = DataLoader(Subset(ds, rows.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    preds, tgts, masks = [], [], []
    with torch.no_grad():
        for b in loader:
            pr = model.predict_chunk(b["images"].to(dev), b["state"].to(dev)).cpu().numpy()
            preds.append(pr)
            tgts.append(b["action"].numpy())
            masks.append(b["pad_mask"].numpy())
    pred = np.concatenate(preds)
    tgt = np.concatenate(tgts)
    mask = np.concatenate(masks)

    inv = norms["action"].inverse
    A = cfg.action_dim
    pred_raw = inv(pred.reshape(-1, A)).reshape(pred.shape)
    tgt_raw = inv(tgt.reshape(-1, A)).reshape(tgt.shape)
    return {"pred_raw": pred_raw, "tgt_raw": tgt_raw, "mask": mask, "pos": pos,
            "chunk_size": int(cfg.chunk_size), "env": env, "ckpt": str(ckpt)}


# ---------------------------------------------------------------------------
# Metric helpers operating on the arrays above
# ---------------------------------------------------------------------------
def _valid(mask):
    return ~mask.astype(bool)


def overall_l1(pred, tgt, mask):
    v = _valid(mask)[..., None]
    return float((np.abs(pred - tgt) * v).sum() / v.sum() / pred.shape[-1])


def overall_mse(pred, tgt, mask):
    v = _valid(mask)[..., None]
    return float((((pred - tgt) ** 2) * v).sum() / v.sum() / pred.shape[-1])


def first_step_l1(pred, tgt, mask):
    v = _valid(mask)[:, :1, None]
    return float((np.abs(pred[:, :1] - tgt[:, :1]) * v).sum() / v.sum() / pred.shape[-1])


def per_dim_l1(pred, tgt, mask):
    v = _valid(mask)[..., None]
    num = (np.abs(pred - tgt) * v).sum(axis=(0, 1))
    den = v.sum(axis=(0, 1))
    return (num / den).astype(float)            # [A]


def per_chunk_step_l1(pred, tgt, mask):
    """Mean L1 (averaged over action dims) at each position in the chunk -> [H]."""
    err = np.abs(pred - tgt).mean(axis=-1)      # [N, H]
    v = _valid(mask)                            # [N, H]
    out = np.full(err.shape[1], np.nan)
    for h in range(err.shape[1]):
        col = err[:, h][v[:, h]]
        if len(col):
            out[h] = col.mean()
    return out


def gripper_accuracy(pred, tgt, mask, dim=GRIPPER_DIM):
    """Sign-agreement of the (binary +-1) gripper command over valid steps."""
    v = _valid(mask)
    p = np.sign(pred[..., dim])[v]
    t = np.sign(tgt[..., dim])[v]
    return float((p == t).mean())


def save_arrays(name: str, rec: dict):
    np.savez_compressed(ARRAYS / f"{name}.npz",
                        pred_raw=rec["pred_raw"], tgt_raw=rec["tgt_raw"],
                        mask=rec["mask"], pos=rec["pos"],
                        chunk_size=rec["chunk_size"])


def load_arrays(name: str) -> dict:
    z = np.load(ARRAYS / f"{name}.npz")
    return {k: z[k] for k in z.files}


def safe_name(s: str) -> str:
    return (s.replace(" ", "_").replace("(", "").replace(")", "")
            .replace("+", "").replace("/", "_"))


def load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def dump_json(p: Path, obj):
    Path(p).write_text(json.dumps(obj, indent=2), encoding="utf-8")
