# -*- coding: utf-8 -*-
"""Where does the error live? Per-action-dimension and per-action-group breakdown.

Reads the cached arrays (analyze/results/arrays/*.npz). Splits the 7-D CALVIN
rel_action into translation (x,y,z) / rotation (roll,pitch,yaw) / gripper, and
shows which components carry the cross-environment penalty.

  fig_per_dim_error.png   L1 per action dim, grouped by env, for ACT-A vs ACT-ABC
  fig_error_groups.png    L1 by action group (transl / rot / grip-as-error-rate)
"""

from __future__ import annotations

import numpy as np

import common as C


def main():
    plt = C.setup_mpl()
    models = list(C.MAIN_MODELS.keys())
    envs = C.EVAL_ENVS

    # per-dim L1 [model][env] -> [7]
    pd = {}
    for m in models:
        pd[m] = {}
        for e in envs:
            a = C.load_arrays(f"{C.safe_name(m)}__env{e}")
            pd[m][e] = C.per_dim_l1(a["pred_raw"], a["tgt_raw"], a["mask"])

    # ---- figure 1: per-dim grouped bars (one subplot per env) ----
    fig, axes = plt.subplots(1, len(envs), figsize=(16, 4.6), sharey=True)
    x = np.arange(7); w = 0.38
    for j, e in enumerate(envs):
        ax = axes[j]
        for i, m in enumerate(models):
            ax.bar(x + (i - 0.5) * w, pd[m][e], w, label=m, color=C.MODEL_COLORS.get(m))
        ax.set_xticks(x); ax.set_xticklabels(C.ACTION_LABELS, rotation=0)
        ax.set_title(f"env {e}" + ("" if e == "A" else " (unseen by ACT-A)"))
        ax.axvspan(-0.5, 2.5, color="gray", alpha=0.06)
        ax.axvspan(2.5, 5.5, color="gray", alpha=0.12)
        if j == 0:
            ax.set_ylabel("per-dim L1 (raw)"); ax.legend(fontsize=8)
    fig.suptitle("Per-action-dimension error  [grey: translation | darker: rotation | grip]",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_per_dim_error.png")
    plt.close(fig)

    # ---- figure 2: by action group ----
    # translation/rotation: mean L1; gripper: error rate (1 - sign acc)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
    groups = ["translation", "rotation"]
    gidx = {"translation": [0, 1, 2], "rotation": [3, 4, 5]}
    xg = np.arange(len(groups)); w = 0.38
    # plot grouped by env for clarity: x = envs, separate panels per group not needed;
    # show translation & rotation as two clusters, bars = model x env
    width = 0.12
    base = np.arange(len(groups))
    k = 0
    for m in models:
        for e in envs:
            vals = [np.mean([pd[m][e][d] for d in gidx[g]]) for g in groups]
            ax1.bar(base + k * width, vals, width,
                    label=f"{m.split(' ')[0]}/{e}",
                    color=C.MODEL_COLORS.get(m),
                    alpha=0.45 + 0.18 * envs.index(e))
            k += 1
    ax1.set_xticks(base + width * (k - 1) / 2)
    ax1.set_xticklabels(groups)
    ax1.set_ylabel("mean L1 (raw)")
    ax1.set_title("(a) Translation vs rotation error")
    ax1.legend(fontsize=7, ncol=2)

    # gripper error rate
    xb = np.arange(len(envs)); w = 0.38
    for i, m in enumerate(models):
        er = []
        for e in envs:
            a = C.load_arrays(f"{C.safe_name(m)}__env{e}")
            er.append(1 - C.gripper_accuracy(a["pred_raw"], a["tgt_raw"], a["mask"]))
        ax2.bar(xb + (i - 0.5) * w, er, w, label=m, color=C.MODEL_COLORS.get(m))
        for xi, v in zip(xb + (i - 0.5) * w, er):
            ax2.text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
    ax2.set_xticks(xb); ax2.set_xticklabels([f"env {e}" for e in envs])
    ax2.set_ylabel("gripper error rate (1 - sign acc)")
    ax2.set_title("(b) Gripper mistakes rise most under shift for ACT-A")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_error_groups.png")
    plt.close(fig)

    # numbers
    out = {m: {e: {"per_dim": [round(float(v), 4) for v in pd[m][e]],
                   "translation": round(float(np.mean(pd[m][e][:3])), 4),
                   "rotation": round(float(np.mean(pd[m][e][3:6])), 4),
                   "gripper_l1": round(float(pd[m][e][6]), 4)}
               for e in envs} for m in models}
    C.dump_json(C.RESULTS / "error_breakdown.json", out)
    print("wrote fig_per_dim_error.png, fig_error_groups.png")


if __name__ == "__main__":
    main()
