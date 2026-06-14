# -*- coding: utf-8 -*-
"""Where in a trajectory does the policy struggle? Error vs position-in-episode.

CALVIN play episodes go approach -> reach -> manipulate. We bin each evaluated
observation by its normalised position within its episode (0 = start, 1 = end)
and plot the first-step L1 per bin. This shows whether the cross-environment
penalty is uniform or concentrated in particular phases (e.g. contact/grasp).

Reads the cached arrays (need the per-window `pos`). Writes fig_temporal_error.png.
"""

from __future__ import annotations

import numpy as np

import common as C


def binned_first_step(a, n_bins=10):
    pos = a["pos"]
    v = (~a["mask"][:, 0].astype(bool))
    err = np.abs(a["pred_raw"][:, 0] - a["tgt_raw"][:, 0]).mean(-1)      # first-step L1 per window
    edges = np.linspace(0, 1, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = (pos >= edges[b]) & (pos < edges[b + 1] if b < n_bins - 1 else pos <= 1.0) & v
        if sel.sum():
            out[b] = err[sel].mean()
    return mids, out


def main():
    plt = C.setup_mpl()
    models = list(C.MAIN_MODELS.keys())
    envs = C.EVAL_ENVS

    fig, axes = plt.subplots(1, len(envs), figsize=(15, 4.4), sharey=True)
    findings = {}
    for j, e in enumerate(envs):
        ax = axes[j]
        for m in models:
            a = C.load_arrays(f"{C.safe_name(m)}__env{e}")
            mids, vals = binned_first_step(a)
            ax.plot(mids, vals, "o-", lw=2, ms=5, color=C.MODEL_COLORS.get(m), label=m)
            findings.setdefault(m, {})[e] = [round(float(x), 4) for x in vals]
        ax.set_title(f"env {e}" + ("" if e == "A" else " (unseen by ACT-A)"))
        ax.set_xlabel("position within episode (0=start, 1=end)")
        if j == 0:
            ax.set_ylabel("first-step L1 (raw)"); ax.legend(fontsize=9)
    fig.suptitle("Action error along the trajectory — is the shift penalty "
                 "uniform or phase-specific?", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_temporal_error.png")
    plt.close(fig)
    C.dump_json(C.RESULTS / "temporal_error.json", findings)
    print("wrote fig_temporal_error.png")


if __name__ == "__main__":
    main()
