# -*- coding: utf-8 -*-
"""Step 1 — run every (model x env) held-out evaluation once and cache the results.

Produces:
  analyze/results/arrays/<model>__env<E>.npz   raw pred/target/mask/pos per eval
  analyze/results/eval_summary.json            scalar + vector metrics per eval
  analyze/results/eval_summary.csv             flat table (for the report)
  analyze/results/eval_summary.md              same as a markdown table

All downstream figure scripts read these caches, so the (slow) GPU inference runs
only once. Re-run with `--n-windows` to change the held-out sample size.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

import common as C


def evaluate_all(n_windows: int, device: str | None, num_workers: int):
    records = {}        # name -> metrics dict

    def run(tag: str, ckpt, env: str):
        print(f"  [eval] {tag:28s} on env {env} ...", flush=True)
        rec = C.evaluate_model_on_env(ckpt, env, n_windows=n_windows,
                                      device=device, num_workers=num_workers)
        name = f"{C.safe_name(tag)}__env{env}"
        C.save_arrays(name, rec)
        pcs = C.per_chunk_step_l1(rec["pred_raw"], rec["tgt_raw"], rec["mask"])
        pdl = C.per_dim_l1(rec["pred_raw"], rec["tgt_raw"], rec["mask"])
        m = {
            "tag": tag, "env": env, "chunk_size": rec["chunk_size"],
            "n_windows": int(rec["pred_raw"].shape[0]),
            "l1_raw": C.overall_l1(rec["pred_raw"], rec["tgt_raw"], rec["mask"]),
            "mse_raw": C.overall_mse(rec["pred_raw"], rec["tgt_raw"], rec["mask"]),
            "first_step_l1": C.first_step_l1(rec["pred_raw"], rec["tgt_raw"], rec["mask"]),
            "gripper_acc": C.gripper_accuracy(rec["pred_raw"], rec["tgt_raw"], rec["mask"]),
            "per_dim_l1": [round(float(v), 6) for v in pdl],
            "per_chunk_step_l1": [None if np.isnan(v) else round(float(v), 6) for v in pcs],
        }
        records[name] = m
        print(f"        L1(raw)={m['l1_raw']:.4f}  first-step={m['first_step_l1']:.4f}  "
              f"grip-acc={m['gripper_acc']:.3f}")
        return m

    print("== MAIN models (ACT-A vs ACT-ABC) on held-out A / B / C ==")
    for tag, ckpt in C.MAIN_MODELS.items():
        if not ckpt.exists():
            print(f"  (skip, missing: {ckpt})")
            continue
        for env in C.EVAL_ENVS:
            run(tag, ckpt, env)

    print("== CHUNK-SIZE ablation (A+B+C, H in {5,10,20}) on held-out A / B / C ==")
    for H, ckpt in C.CHUNK_MODELS.items():
        if not ckpt.exists():
            print(f"  (skip H={H}, missing: {ckpt})")
            continue
        for env in C.EVAL_ENVS:
            run(f"ABC-H{H}", ckpt, env)

    return records


def write_tables(records: dict):
    C.dump_json(C.RESULTS / "eval_summary.json", records)
    cols = ["tag", "env", "chunk_size", "n_windows", "l1_raw", "mse_raw",
            "first_step_l1", "gripper_acc"]
    rows = [{k: m[k] for k in cols} for m in records.values()]
    with open(C.RESULTS / "eval_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for r in rows:
        md.append("| " + " | ".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    (C.RESULTS / "eval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {C.RESULTS/'eval_summary.json'} (+ .csv/.md), "
          f"{len(records)} evaluations.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=3000,
                    help="held-out windows sampled per (model, env)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    records = evaluate_all(args.n_windows, args.device, args.num_workers)
    write_tables(records)


if __name__ == "__main__":
    main()
