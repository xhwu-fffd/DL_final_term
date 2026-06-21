# -*- coding: utf-8 -*-
"""Policy case-study figure.

For one representative frame from each of env A, B, D (the moment the gripper
is about to close on an object), show:
  - static camera  |  wrist camera
  - ACT-A  predicted 10-step XY trajectory
  - ACT-ABC predicted 10-step XY trajectory
Ground-truth trajectory is overlaid as a dashed line in both panels.

Saves: analyze/figures/fig_policy_cases.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from calvin_act.evaluate import load_policy
from calvin_act.calvin_data import EnvData, find_data_root
from calvin_act.dataset import prep_image, IMAGENET_MEAN, IMAGENET_STD

CKPT_A   = ROOT / "outputs/task1_envA_40k/model_final.pt"
CKPT_ABC = ROOT / "outputs/task2_envABC_40k/model_final.pt"
FIGURES  = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

GRIPPER_DIM = 6
COLOR_A   = "#d62728"   # red for ACT-A
COLOR_ABC = "#1f77b4"   # blue for ACT-ABC
COLOR_GT  = "#2ca02c"   # green for GT


# ── helpers ──────────────────────────────────────────────────────────────────

def infer(model, cfg, norms, imgs_dict, state_np, device):
    """Return raw action chunk [H, 7] for one observation."""
    cams = [prep_image(np.asarray(imgs_dict[k]), cfg.image_size,
                       IMAGENET_MEAN, IMAGENET_STD)
            for k in cfg.image_keys if k in imgs_dict]
    images = torch.stack(cams).unsqueeze(0).to(device)
    st = torch.from_numpy(
        norms["state"].transform(state_np.astype(np.float32))
    ).float().unsqueeze(0).to(device)
    with torch.no_grad():
        chunk_n = model.predict_chunk(images, st)          # [1,H,7] norm
    raw = norms["action"].inverse(
        chunk_n.cpu().numpy().reshape(-1, cfg.action_dim)
    ).reshape(cfg.chunk_size, cfg.action_dim)
    return raw


def find_grasp_frame(ed: EnvData, start: int = 50, window: int = 2000):
    """Return a row index just before a gripper-close event."""
    end = min(start + window, len(ed) - 2)
    for i in range(start, end):
        if ed.actions[i, GRIPPER_DIM] > 0.2 and ed.actions[i + 1, GRIPPER_DIM] < -0.2:
            return i
    # fallback: frame with highest translation magnitude
    mag = np.abs(ed.actions[start:end, :3]).sum(axis=1)
    return start + int(np.argmax(mag))


def traj_xy(chunk):
    """Cumulative XY displacement path including origin: shape [H+1, 2]."""
    xy = np.cumsum(chunk[:, :2], axis=0)
    return np.vstack([[0.0, 0.0], xy])


def plot_traj(ax, chunk, gt_chunk, color, label):
    """Plot predicted + GT trajectory on ax."""
    pred = traj_xy(chunk)
    gt   = traj_xy(gt_chunk)
    grip = chunk[:, GRIPPER_DIM]

    # GT dashed
    ax.plot(gt[:, 0], gt[:, 1], '--', color=COLOR_GT, lw=1.2, alpha=0.7,
            label="GT", zorder=2)

    # Prediction path
    ax.plot(pred[:, 0], pred[:, 1], '-', color=color, lw=1.8, alpha=0.85,
            label=label, zorder=3)

    # Per-step dots: green = gripper open, red = closed
    for i in range(len(chunk)):
        dot_c = "#2ecc71" if grip[i] > 0 else "#e74c3c"
        ax.scatter(pred[i + 1, 0], pred[i + 1, 1],
                   color=dot_c, s=22, zorder=5,
                   edgecolors=color, linewidths=0.6)

    # Origin star
    ax.scatter(0, 0, marker="*", s=90, color="black", zorder=6)

    # Limit / style
    all_xy = np.vstack([pred, gt])
    span = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]), 0.06) * 0.65
    cx, cy = all_xy[:, 0].mean(), all_xy[:, 1].mean()
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.4, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.4)
    ax.set_xlabel("Δx (m)", fontsize=7)
    ax.set_ylabel("Δy (m)", fontsize=7)
    ax.tick_params(labelsize=6)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading checkpoints ...")
    model_A,   cfg_A,   norms_A   = load_policy(CKPT_A,   device)
    model_ABC, cfg_ABC, norms_ABC = load_policy(CKPT_ABC, device)

    data_root = find_data_root(None)
    envs_to_show = [("A", "in-domain (env A)"),
                    ("B", "OOD for ACT-A (env B)"),
                    ("D", "zero-shot (env D)")]

    # ── collect one representative frame per env ──────────────────────────
    rows_data = []
    for env_letter, env_label in envs_to_show:
        print(f"  Loading env {env_letter} ...")
        ed = EnvData.build(data_root, [env_letter],
                           image_keys=("image", "wrist_image"),
                           subset="val", val_fraction=0.05,
                           split_seed=42, verbose=False)
        row = find_grasp_frame(ed)
        imgs  = ed.load_images(row)
        state = ed.states[row]
        gt_chunk, _ = ed.chunk(row, cfg_A.chunk_size)

        pred_A   = infer(model_A,   cfg_A,   norms_A,   imgs, state, device)
        pred_ABC = infer(model_ABC, cfg_ABC, norms_ABC, imgs, state, device)

        rows_data.append(dict(env=env_letter, label=env_label,
                              imgs=imgs, gt=gt_chunk,
                              pred_A=pred_A, pred_ABC=pred_ABC))
        print(f"    frame {row}: "
              f"ACT-A L1={np.abs(pred_A - gt_chunk).mean():.3f}  "
              f"ACT-ABC L1={np.abs(pred_ABC - gt_chunk).mean():.3f}")

    # ── build figure: 3 rows × 5 cols ─────────────────────────────────────
    # cols: static | wrist | ACT-A traj | ACT-ABC traj | legend/info
    n_rows = len(rows_data)
    fig = plt.figure(figsize=(14, 3.6 * n_rows))
    outer = gridspec.GridSpec(n_rows, 1, figure=fig, hspace=0.45)

    for ri, rd in enumerate(rows_data):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[ri],
            width_ratios=[1, 0.85, 1.15, 1.15, 0.55],
            wspace=0.25)

        # ---- static camera ----
        ax_s = fig.add_subplot(inner[0])
        ax_s.imshow(np.asarray(rd["imgs"]["image"]))
        ax_s.set_xticks([]); ax_s.set_yticks([])
        ax_s.set_title("Static camera", fontsize=8)

        # ---- wrist camera ----
        ax_w = fig.add_subplot(inner[1])
        ax_w.imshow(np.asarray(rd["imgs"]["wrist_image"]))
        ax_w.set_xticks([]); ax_w.set_yticks([])
        ax_w.set_title("Wrist camera", fontsize=8)

        # ---- ACT-A trajectory ----
        ax_ta = fig.add_subplot(inner[2])
        l1_a = np.abs(rd["pred_A"] - rd["gt"]).mean()
        plot_traj(ax_ta, rd["pred_A"], rd["gt"], COLOR_A, "ACT-A")
        ax_ta.set_title(f"ACT-A prediction  (L1={l1_a:.3f})", fontsize=8)

        # ---- ACT-ABC trajectory ----
        ax_tb = fig.add_subplot(inner[3])
        l1_abc = np.abs(rd["pred_ABC"] - rd["gt"]).mean()
        plot_traj(ax_tb, rd["pred_ABC"], rd["gt"], COLOR_ABC, "ACT-ABC")
        ax_tb.set_title(f"ACT-ABC prediction  (L1={l1_abc:.3f})", fontsize=8)

        # ---- row label + legend ----
        ax_l = fig.add_subplot(inner[4])
        ax_l.axis("off")
        ax_l.text(0.5, 0.72, f"env {rd['env']}",
                  ha="center", va="center", fontsize=13, fontweight="bold",
                  transform=ax_l.transAxes)
        ax_l.text(0.5, 0.52, rd["label"],
                  ha="center", va="center", fontsize=7.5, color="#555",
                  transform=ax_l.transAxes)
        # dot legend
        patches = [
            mpatches.Patch(color=COLOR_GT,  label="GT trajectory"),
            mpatches.Patch(color=COLOR_A,   label="ACT-A"),
            mpatches.Patch(color=COLOR_ABC, label="ACT-ABC"),
            mpatches.Patch(color="#2ecc71", label="gripper open"),
            mpatches.Patch(color="#e74c3c", label="gripper close"),
        ]
        ax_l.legend(handles=patches, loc="lower center",
                    fontsize=6.5, framealpha=0.8,
                    bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Policy case study: predicted 10-step EE trajectory vs. ground truth\n"
        "(★ = current position; dots = step 1–10; colour = gripper state)",
        fontsize=10, y=1.01)

    out = FIGURES / "fig_policy_cases.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
