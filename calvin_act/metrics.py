# -*- coding: utf-8 -*-
"""Metrics for the ACT study: action error (the convergence + zero-shot metric),
success-rate aggregation (closed-loop), and a markdown/CSV comparison table.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Action error
# ---------------------------------------------------------------------------
def _masked(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if mask is None:
        return pred.reshape(-1, pred.shape[-1]), target.reshape(-1, target.shape[-1])
    keep = ~np.asarray(mask, dtype=bool).reshape(-1)
    return (pred.reshape(-1, pred.shape[-1])[keep],
            target.reshape(-1, target.shape[-1])[keep])


def action_l1(pred, target, mask=None) -> float:
    p, t = _masked(pred, target, mask)
    return float(np.abs(p - t).mean()) if len(p) else float("nan")


def action_mse(pred, target, mask=None) -> float:
    p, t = _masked(pred, target, mask)
    return float(((p - t) ** 2).mean()) if len(p) else float("nan")


def per_dim_l1(pred, target, mask=None) -> list[float]:
    p, t = _masked(pred, target, mask)
    return [round(float(v), 6) for v in np.abs(p - t).mean(axis=0)]


def per_chunk_step_l1(pred, target, mask=None) -> list[float]:
    """Mean L1 at each position within the chunk (0 = first / most immediate).

    Reveals how prediction error grows further into the action chunk - central to
    the action-chunking robustness analysis.
    """
    pred = np.asarray(pred, dtype=np.float64)          # [N, H, A]
    target = np.asarray(target, dtype=np.float64)
    err = np.abs(pred - target).mean(axis=-1)          # [N, H]
    if mask is not None:
        m = ~np.asarray(mask, dtype=bool)              # [N, H] valid
        out = []
        for h in range(err.shape[1]):
            v = err[:, h][m[:, h]]
            out.append(round(float(v.mean()), 6) if len(v) else float("nan"))
        return out
    return [round(float(v), 6) for v in err.mean(axis=0)]


# ---------------------------------------------------------------------------
# Success rate (closed-loop)
# ---------------------------------------------------------------------------
def success_rate(successes) -> float:
    s = np.asarray(successes, dtype=bool)
    return float(s.mean()) if len(s) else float("nan")


def calvin_sequence_metrics(chain_lengths, num_tasks: int = 5) -> dict:
    """CALVIN long-horizon metrics: per-length success + average sequence length
    over rollouts that chain up to ``num_tasks`` instructions."""
    cl = np.asarray(chain_lengths, dtype=np.int64)
    out = {"avg_seq_len": round(float(cl.mean()), 4) if len(cl) else float("nan")}
    for k in range(1, num_tasks + 1):
        out[f"success_{k}"] = round(float((cl >= k).mean()), 4) if len(cl) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Comparison table (markdown + optional CSV)
# ---------------------------------------------------------------------------
def write_comparison_table(rows: list[dict], md_path: str | Path,
                           csv_path: str | Path | None = None) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    md = ["| " + " | ".join(cols) + " |",
          "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).write_text("\n".join(md) + "\n", encoding="utf-8")
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})


def moving_average(x, window: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2 or window <= 1:
        return x
    window = min(window, len(x))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")
