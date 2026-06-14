# -*- coding: utf-8 -*-
"""Action-chunking robustness analysis (the assignment's required deep dive).

Two complementary views, both from analyze/results/eval_summary.json:

  fig_chunk_step_growth.png   per-position-in-chunk L1 for ACT-A vs ACT-ABC on
                              each env. Does error grow along the predicted chunk,
                              and is the growth steeper when the visual input is
                              out-of-distribution (ACT-A on B/C)?

  fig_chunk_size_ablation.png how the chunk horizon H itself (5/10/20, A+B+C)
                              trades off: mean L1, first-step L1, gripper accuracy,
                              and the per-step error profile for each H.
"""

from __future__ import annotations

import numpy as np

import common as C


def chunk_step_growth(plt, rec):
    models = list(C.MAIN_MODELS.keys())
    envs = C.EVAL_ENVS
    fig, axes = plt.subplots(1, len(envs), figsize=(15, 4.4), sharey=True)
    for j, e in enumerate(envs):
        ax = axes[j]
        for m in models:
            pcs = rec[f"{C.safe_name(m)}__env{e}"]["per_chunk_step_l1"]
            pcs = [v for v in pcs if v is not None]
            x = np.arange(1, len(pcs) + 1)
            ax.plot(x, pcs, "o-", lw=2, ms=5, color=C.MODEL_COLORS.get(m), label=m)
        unseen = " (unseen by ACT-A)" if e != "A" else " (in-domain)"
        ax.set_title(f"env {e}{unseen}")
        ax.set_xlabel("position in action chunk")
        if j == 0:
            ax.set_ylabel("L1 (raw)")
            ax.legend(fontsize=9)
    fig.suptitle("Per-chunk-step error growth — chunking lets error grow with "
                 "horizon, and faster under visual shift", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_chunk_step_growth.png")
    plt.close(fig)

    # slope (last - first) per model/env as a compact robustness number
    out = {}
    for m in models:
        out[m] = {}
        for e in envs:
            pcs = [v for v in rec[f"{C.safe_name(m)}__env{e}"]["per_chunk_step_l1"]
                   if v is not None]
            out[m][e] = {"first": round(pcs[0], 4), "last": round(pcs[-1], 4),
                         "growth": round(pcs[-1] - pcs[0], 4)}
    return out


def chunk_size_ablation(plt, rec):
    Hs = sorted(C.CHUNK_MODELS.keys())
    envs = C.EVAL_ENVS
    # mean over envs of each metric
    def metric(H, key):
        return float(np.mean([rec[f"ABC-H{H}__env{e}"][key] for e in envs]))

    mean_l1 = [metric(H, "l1_raw") for H in Hs]
    first_l1 = [metric(H, "first_step_l1") for H in Hs]
    grip = [metric(H, "gripper_acc") for H in Hs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    ax.plot(Hs, mean_l1, "o-", lw=2, ms=8, label="chunk-mean L1")
    ax.plot(Hs, first_l1, "s--", lw=2, ms=7, label="first-step L1")
    ax.set_xlabel("chunk size H"); ax.set_ylabel("L1 (raw, mean over A/B/C)")
    ax.set_title("(a) Accuracy vs chunk horizon")
    ax.set_xticks(Hs); ax.legend()
    for h, v in zip(Hs, mean_l1):
        ax.annotate(f"{v:.3f}", (h, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)

    ax = axes[1]
    ax.plot(Hs, grip, "o-", lw=2, ms=8, color="#2ca02c")
    ax.set_xlabel("chunk size H"); ax.set_ylabel("gripper sign accuracy")
    ax.set_title("(b) Gripper accuracy vs chunk horizon")
    ax.set_xticks(Hs)
    for h, v in zip(Hs, grip):
        ax.annotate(f"{v:.3f}", (h, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)

    ax = axes[2]
    cmap = plt.get_cmap("plasma")
    for k, H in enumerate(Hs):
        # average the per-step profile over envs (pad to same length by position)
        profs = [np.array([v for v in rec[f"ABC-H{H}__env{e}"]["per_chunk_step_l1"]
                           if v is not None]) for e in envs]
        prof = np.mean(profs, axis=0)
        ax.plot(np.arange(1, len(prof) + 1), prof, "o-", ms=4, lw=2,
                color=cmap(k / max(len(Hs) - 1, 1)), label=f"H={H}")
    ax.set_xlabel("position in action chunk")
    ax.set_ylabel("L1 (raw, mean over A/B/C)")
    ax.set_title("(c) Per-step error profile by H")
    ax.legend()
    fig.suptitle("Chunk-size ablation (A+B+C, identical except H): "
                 "shorter chunks are more accurate per-step but more reactive",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_chunk_size_ablation.png")
    plt.close(fig)

    return {"H": Hs, "mean_l1": [round(v, 4) for v in mean_l1],
            "first_step_l1": [round(v, 4) for v in first_l1],
            "gripper_acc": [round(v, 4) for v in grip]}


def main():
    plt = C.setup_mpl()
    rec = C.load_json(C.RESULTS / "eval_summary.json")
    growth = chunk_step_growth(plt, rec)
    abl = chunk_size_ablation(plt, rec)
    C.dump_json(C.RESULTS / "chunking_findings.json",
                {"per_chunk_growth": growth, "chunk_size_ablation": abl})
    print("chunk growth (ACT-A):", growth.get("ACT-A (A only)"))
    print("chunk-size ablation:", abl)
    print("wrote fig_chunk_step_growth.png, fig_chunk_size_ablation.png")


if __name__ == "__main__":
    main()
