# -*- coding: utf-8 -*-
"""Headline result: does training on A+B+C buy robustness to a *shifted* env?

This is the Task-3 question, evaluated with B and C standing in for the absent
env D (the A-only model has never seen them). Reads analyze/results/eval_summary.json.

  fig_generalization_matrix.png  L1 heatmap (model x eval-env) + grouped bars
  fig_generalization_gap.png     the in-domain -> shifted degradation, per model
"""

from __future__ import annotations

import numpy as np

import common as C


def main():
    plt = C.setup_mpl()
    rec = C.load_json(C.RESULTS / "eval_summary.json")
    models = list(C.MAIN_MODELS.keys())
    envs = C.EVAL_ENVS

    def get(tag, env, key="l1_raw"):
        return rec[f"{C.safe_name(tag)}__env{env}"][key]

    L1 = np.array([[get(m, e) for e in envs] for m in models])          # [2,3]

    # ---- figure 1: heatmap + grouped bars ----
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(13.5, 4.6))

    im = axh.imshow(L1, cmap="viridis", aspect="auto")
    axh.set_xticks(range(len(envs))); axh.set_xticklabels([f"env {e}" for e in envs])
    axh.set_yticks(range(len(models))); axh.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(envs)):
            axh.text(j, i, f"{L1[i, j]:.3f}", ha="center", va="center",
                     color="white" if L1[i, j] < L1.max() * 0.7 else "black", fontsize=11)
    axh.set_title("(a) Held-out action L1 (raw) — model x eval env")
    fig.colorbar(im, ax=axh, fraction=0.046, label="L1")

    x = np.arange(len(envs)); w = 0.38
    for i, m in enumerate(models):
        bars = axb.bar(x + (i - 0.5) * w, L1[i], w, label=m,
                       color=C.MODEL_COLORS.get(m))
        for b, v in zip(bars, L1[i]):
            axb.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
                     ha="center", fontsize=8)
    axb.set_xticks(x); axb.set_xticklabels([f"env {e}" for e in envs])
    axb.set_ylabel("action L1 (raw)")
    axb.set_title("(b) ACT-A degrades off-domain; ACT-ABC stays flat")
    # annotate which envs are unseen for ACT-A
    axb.text(1, L1.max() * 1.05, "B,C unseen by ACT-A", fontsize=9, color="#d62728",
             ha="center")
    axb.legend()
    axb.set_ylim(0, L1.max() * 1.18)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_generalization_matrix.png")
    plt.close(fig)

    # ---- figure 2: degradation gap (relative to each model's env-A score) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
    # absolute gap vs in-domain A
    for i, m in enumerate(models):
        base = L1[i, 0]
        gaps = (L1[i] - base)
        ax1.plot(envs, gaps, "o-", lw=2, ms=8, color=C.MODEL_COLORS.get(m), label=m)
        for e, g in zip(envs, gaps):
            ax1.annotate(f"{g:+.3f}", (e, g), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=8)
    ax1.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax1.set_ylabel("L1 increase vs that model's env-A score")
    ax1.set_title("(a) Cross-env degradation (lower = more robust)")
    ax1.legend()

    # gripper accuracy across envs
    G = np.array([[get(m, e, "gripper_acc") for e in envs] for m in models])
    xb = np.arange(len(envs)); w = 0.38
    for i, m in enumerate(models):
        ax2.bar(xb + (i - 0.5) * w, G[i], w, label=m, color=C.MODEL_COLORS.get(m))
    ax2.set_xticks(xb); ax2.set_xticklabels([f"env {e}" for e in envs])
    ax2.set_ylim(0.7, 1.0)
    ax2.set_ylabel("gripper sign accuracy")
    ax2.set_title("(b) Gripper command accuracy across envs")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_generalization_gap.png")
    plt.close(fig)

    # numeric takeaways
    out = {
        "l1_matrix": {m: {e: round(float(L1[i, j]), 4) for j, e in enumerate(envs)}
                      for i, m in enumerate(models)},
        "act_A_shift_penalty_B_pct": round(100 * (L1[0, 1] / L1[0, 0] - 1), 1),
        "act_A_shift_penalty_C_pct": round(100 * (L1[0, 2] / L1[0, 0] - 1), 1),
        "act_ABC_shift_penalty_B_pct": round(100 * (L1[1, 1] / L1[1, 0] - 1), 1),
        "act_ABC_shift_penalty_C_pct": round(100 * (L1[1, 2] / L1[1, 0] - 1), 1),
        "abc_error_reduction_on_B_pct": round(100 * (1 - L1[1, 1] / L1[0, 1]), 1),
        "abc_error_reduction_on_C_pct": round(100 * (1 - L1[1, 2] / L1[0, 2]), 1),
        "abc_in_domain_cost_on_A_pct": round(100 * (L1[1, 0] / L1[0, 0] - 1), 1),
    }
    C.dump_json(C.RESULTS / "generalization_findings.json", out)
    print("generalization findings:", out)
    print("wrote fig_generalization_matrix.png, fig_generalization_gap.png")


if __name__ == "__main__":
    main()
