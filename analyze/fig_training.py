# -*- coding: utf-8 -*-
"""Training-dynamics deep dive from the logged histories (outputs/*/history.json).

  fig_training_curves.png   smoothed train Action-L1 + KL for ACT-A vs ACT-ABC
                            (40k), with the unreliable in-training val markers and
                            the *full held-out* re-eval (from run_evals) overlaid.
  fig_convergence_speed.png steps-to-threshold for each run + a note that the
                            logged val curve is a 512-window estimate, far above
                            the true full-split L1.

Why overlay the full-split number: `val_action_l1` logged during training is the
mean over only the first `val_batches*batch_size` (=512) windows of the val set —
a high-variance, pessimistic slice. The single markers show the representative
held-out L1 computed by run_evals over a 3000-window even sample.
"""

from __future__ import annotations

import numpy as np

import common as C

RUNS = {
    "ACT-A 40k":   (C.OUTPUTS / "task1_envA_40k" / "history.json", "#d62728"),
    "ACT-ABC 40k": (C.OUTPUTS / "task2_envABC_40k" / "history.json", "#1f77b4"),
    "ACT-A 20k":   (C.OUTPUTS / "task1_envA" / "history.json", "#f4a6a6"),
    "ACT-ABC 20k": (C.OUTPUTS / "task2_envABC" / "history.json", "#a6c8e8"),
}
ACTION_DIM = 7   # CALVIN 7-D action; history action_l1 was summed over dims, divide to get per-dim


def smooth(x, k=9):
    """Edge-preserving moving average (pads with edge values, no boundary dip)."""
    x = np.asarray(x, float)
    if len(x) < k or k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(k) / k, mode="valid")[:len(x)]


def load_hist(p):
    h = C.load_json(p)
    step = np.array([r["step"] for r in h])
    # history action_l1 used old formula: sum_abs_err / n_valid_steps (not divided by
    # action_dim). Divide by ACTION_DIM here to get the true per-dim per-step mean so
    # the training curve is on the same formula as eval metrics.
    l1 = np.array([r["action_l1"] for r in h]) / ACTION_DIM
    kl = np.array([r.get("kl", np.nan) for r in h])
    vstep = np.array([r["step"] for r in h if "val_action_l1" in r])
    vval = np.array([r["val_action_l1"] for r in h if "val_action_l1" in r]) / ACTION_DIM
    return step, l1, kl, vstep, vval


def main():
    plt = C.setup_mpl()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    for name, (p, col) in RUNS.items():
        if not p.exists():
            continue
        step, l1, kl, vstep, vval = load_hist(p)
        lw = 2.2 if "40k" in name else 1.2
        alpha = 1.0 if "40k" in name else 0.6
        ax1.plot(step, smooth(l1), color=col, lw=lw, alpha=alpha, label=name)
        ax2.plot(step, np.clip(smooth(kl), 1e-5, None), color=col, lw=lw,
                 alpha=alpha, label=name)

    # overlay full-held-out re-eval on env A on a secondary (right) y-axis.
    # Training curves are per-dim L1 in *normalised* action space; eval stars are
    # per-dim L1 in *raw* action units.  Two different spaces → two axes.
    ax1_raw = ax1.twinx()
    try:
        rec = C.load_json(C.RESULTS / "eval_summary.json")
        a_full = rec["ACT-A_A_only__envA"]["l1_raw"]
        abc_full = rec["ACT-ABC_ABC__envA"]["l1_raw"]
        ax1_raw.scatter([40000], [a_full], marker="*", s=240, color="#d62728",
                        edgecolor="k", zorder=5,
                        label=f"ACT-A held-out raw L1={a_full:.3f}")
        ax1_raw.scatter([40000], [abc_full], marker="*", s=240, color="#1f77b4",
                        edgecolor="k", zorder=5,
                        label=f"ACT-ABC held-out raw L1={abc_full:.3f}")
        ax1_raw.set_ylabel("Action L1 (raw units, right axis)", color="gray", fontsize=9)
        ax1_raw.tick_params(axis="y", labelcolor="gray")
        ax1_raw.legend(fontsize=7, loc="upper right")
    except Exception as ex:
        print("(skip full-eval overlay):", ex)

    ax1.set_xlabel("training step")
    ax1.set_ylabel("Action L1 per-dim (normalised space)")
    ax1.set_title("(a) Train Action-L1 convergence  (★ = held-out raw L1, right axis)")
    ax1.legend(fontsize=8, loc="upper left")
    ax2.set_xlabel("training step"); ax2.set_ylabel("KL (latent), log scale")
    ax2.set_yscale("log")
    ax2.set_title("(b) CVAE KL term collapses early (latent ~ prior)")
    ax2.legend(fontsize=8)
    fig.suptitle("Training dynamics: ACT-A vs ACT-ABC (joint sees 3x the visual "
                 "variety, so train-L1 sits higher but generalises better)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_training_curves.png")
    plt.close(fig)

    # convergence speed: steps to reach per-dim normalised train-L1 thresholds
    thresholds = [0.20, 0.15, 0.10]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    names = [n for n in RUNS if RUNS[n][0].exists()]
    x = np.arange(len(names)); w = 0.25
    for ti, th in enumerate(thresholds):
        steps_to = []
        for n in names:
            step, l1, *_ = load_hist(RUNS[n][0])
            sm = smooth(l1)
            below = np.where(sm <= th)[0]
            steps_to.append(step[below[0]] if len(below) else np.nan)
        ax.bar(x + (ti - 1) * w, steps_to, w, label=f"per-dim norm L1 ≤ {th}")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("training step reached")
    ax.set_title("Convergence speed: steps to reach per-dim normalised train-L1 thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_convergence_speed.png")
    plt.close(fig)
    print("wrote fig_training_curves.png, fig_convergence_speed.png")


if __name__ == "__main__":
    main()
