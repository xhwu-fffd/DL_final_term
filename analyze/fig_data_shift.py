# -*- coding: utf-8 -*-
"""Characterise the distribution shift between environments A / B / C.

The whole premise of the task is *visual* distribution shift across CALVIN
environments. This script quantifies it directly from the data (no model needed):

  fig_visual_shift.png   per-channel RGB means + brightness distribution per env,
                         and a quantitative "visual distance" matrix between envs
  fig_image_montage.png  sample static-camera frames from A / B / C side by side
  fig_action_shift.png   per-dimension action distributions + gripper balance,
                         showing the shift is (mostly) visual, not in action space

Writes the numbers to analyze/results/data_shift.json.
"""

from __future__ import annotations

import numpy as np

import common as C
from calvin_act.calvin_data import EnvData, find_data_root


def _sample_images(env: str, n_frames: int, key: str = "image"):
    """Decode `n_frames` evenly-spaced static-camera frames from env's val split."""
    root = find_data_root(None)
    ed = EnvData.build(root, [env], image_keys=(key,), subset="val",
                       val_fraction=C.VAL_FRACTION, split_seed=C.SPLIT_SEED,
                       verbose=False)
    rows = np.linspace(0, len(ed) - 1, n_frames).astype(int)
    imgs = []
    for r in rows:
        d = ed.load_images(int(r))
        if key in d:
            imgs.append(np.asarray(d[key]))
    return imgs


def visual_shift(plt, n_frames=200):
    stats = {}
    chan_means, bright_samples, montage, arrs = {}, {}, {}, {}
    for env in C.EVAL_ENVS:
        imgs = _sample_images(env, n_frames)
        arr = np.stack([im.astype(np.float32) for im in imgs])          # [N,H,W,3]
        arrs[env] = arr                                                  # reuse below
        chan_means[env] = arr.reshape(-1, 3).mean(0)
        bright_samples[env] = arr.mean(-1).reshape(len(arr), -1).mean(1)  # per-frame brightness
        montage[env] = imgs[:4]
        stats[env] = {
            "rgb_mean": [round(float(v), 2) for v in chan_means[env]],
            "rgb_std": [round(float(v), 2) for v in arr.reshape(-1, 3).std(0)],
            "brightness_mean": round(float(arr.mean()), 2),
        }

    # pairwise "visual distance" = L2 between per-env mean RGB histograms (64 bins/ch)
    def hist(arr):
        h = np.concatenate([np.histogram(arr[..., c], bins=32, range=(0, 255),
                                          density=True)[0] for c in range(3)])
        return h
    hists = {e: hist(arrs[e]) for e in C.EVAL_ENVS}
    envs = C.EVAL_ENVS
    dist = np.zeros((len(envs), len(envs)))
    for i, a in enumerate(envs):
        for j, b in enumerate(envs):
            dist[i, j] = np.linalg.norm(hists[a] - hists[b])

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    x = np.arange(len(envs))
    w = 0.25
    for ci, (cname, col) in enumerate(zip("RGB", ["#d62728", "#2ca02c", "#1f77b4"])):
        ax.bar(x + (ci - 1) * w, [chan_means[e][ci] for e in envs], w, label=cname, color=col)
    ax.set_xticks(x); ax.set_xticklabels([f"env {e}" for e in envs])
    ax.set_ylabel("mean channel value (0-255)")
    ax.set_title("(a) Per-channel image brightness by env")
    ax.legend()

    ax = axes[1]
    for e in envs:
        ax.hist(bright_samples[e], bins=40, alpha=0.55, label=f"env {e}",
                color=C.ENV_COLORS[e], density=True)
    ax.set_xlabel("per-frame mean brightness")
    ax.set_ylabel("density")
    ax.set_title("(b) Brightness distribution by env")
    ax.legend()

    ax = axes[2]
    im = ax.imshow(dist, cmap="magma")
    ax.set_xticks(x); ax.set_xticklabels(envs)
    ax.set_yticks(x); ax.set_yticklabels(envs)
    for i in range(len(envs)):
        for j in range(len(envs)):
            ax.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center",
                    color="white" if dist[i, j] < dist.max() * 0.6 else "black", fontsize=9)
    ax.set_title("(c) Visual distance (RGB-hist L2)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Visual distribution shift between CALVIN environments", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_visual_shift.png")
    plt.close(fig)

    # montage
    fig, axes = plt.subplots(len(envs), 4, figsize=(10, 2.6 * len(envs)))
    for i, e in enumerate(envs):
        for j in range(4):
            ax = axes[i, j]
            ax.imshow(montage[e][j])
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if j == 0:
                ax.set_ylabel(f"env {e}", fontsize=12, rotation=0, labelpad=28, va="center")
    fig.suptitle("Sample static-camera frames (held-out) per environment", y=1.0)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_image_montage.png")
    plt.close(fig)

    stats["pairwise_visual_distance"] = {a: {b: round(float(dist[i, j]), 4)
                                             for j, b in enumerate(envs)}
                                         for i, a in enumerate(envs)}
    return stats


def action_shift(plt, max_episodes=600):
    root = find_data_root(None)
    acts, states = {}, {}
    for e in C.EVAL_ENVS:
        ed = EnvData.build(root, [e], subset="all", max_episodes=max_episodes, verbose=False)
        acts[e] = ed.actions
        states[e] = ed.states

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.ravel()
    for d in range(7):
        ax = axes[d]
        for e in C.EVAL_ENVS:
            col = acts[e][:, d]
            if d == C.GRIPPER_DIM:
                ax.hist(col, bins=np.linspace(-1.2, 1.2, 25), alpha=0.5,
                        label=f"env {e}", color=C.ENV_COLORS[e], density=True)
            else:
                ax.hist(col, bins=60, range=(-1, 1), alpha=0.45, label=f"env {e}",
                        color=C.ENV_COLORS[e], density=True, histtype="stepfilled")
        ax.set_title(f"action[{d}] = {C.ACTION_LABELS[d]}")
        ax.set_yticks([])
        if d == 0:
            ax.legend(fontsize=9)
    # gripper open fraction
    ax = axes[7]
    x = np.arange(len(C.EVAL_ENVS))
    fr = [float(np.mean(acts[e][:, C.GRIPPER_DIM] > 0)) for e in C.EVAL_ENVS]
    ax.bar(x, fr, color=[C.ENV_COLORS[e] for e in C.EVAL_ENVS])
    ax.set_xticks(x); ax.set_xticklabels([f"env {e}" for e in C.EVAL_ENVS])
    ax.set_ylim(0, 1); ax.axhline(0.5, ls="--", c="k", alpha=0.4)
    ax.set_title("gripper: fraction 'open' (+1)")
    for xi, v in zip(x, fr):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center")
    fig.suptitle("Action distribution across environments "
                 "(translation/rotation overlap closely -> shift is mostly visual)",
                 y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_action_shift.png")
    plt.close(fig)

    return {e: {"action_mean": [round(float(v), 4) for v in acts[e].mean(0)],
                "action_std": [round(float(v), 4) for v in acts[e].std(0)],
                "gripper_open_frac": round(float(np.mean(acts[e][:, C.GRIPPER_DIM] > 0)), 4)}
            for e in C.EVAL_ENVS}


def main():
    plt = C.setup_mpl()
    print("[data-shift] visual statistics ...")
    vstats = visual_shift(plt)
    print("[data-shift] action statistics ...")
    astats = action_shift(plt)
    C.dump_json(C.RESULTS / "data_shift.json", {"visual": vstats, "action": astats})
    print(f"wrote {C.FIGURES/'fig_visual_shift.png'}, fig_image_montage.png, "
          f"fig_action_shift.png and results/data_shift.json")


if __name__ == "__main__":
    main()
