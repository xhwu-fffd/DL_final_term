# `analyze/` — deep analysis & error analysis for Task 2

Extra analyses on top of the trained Task-1 / Task-2 runs in `../outputs/`:
cross-environment generalisation, action-chunking robustness under visual shift,
per-dimension / gripper error breakdown, trajectory-phase error, visual & action
distribution shift, and training dynamics. All figures are saved to `figures/` and
all numbers to `results/`.

**The report-ready write-up with every number and interpretation is
[`FINDINGS.md`](FINDINGS.md).** This README is just how to run the code.

## Run it

From this folder (GPU recommended for the eval step; everything else is CPU):

```bash
python run_all.py                 # re-evaluate models on held-out A/B/C, then all figures
python run_all.py --skip-evals    # reuse cached evals (results/arrays/*.npz), rebuild figures only
python run_all.py --n-windows 5000  # larger held-out sample (default 3000)
```

Or run any stage on its own:

```bash
python run_evals.py --n-windows 3000   # step 1: GPU inference -> results/arrays + eval_summary.*
python fig_generalization.py           # headline matrix + degradation gap
python fig_chunking.py                 # per-chunk-step growth + chunk-size ablation
python fig_error_breakdown.py          # per-action-dim + gripper
python fig_temporal.py                 # error vs position-in-episode
python fig_training.py                 # training dynamics (reads outputs/*/history.json)
python fig_data_shift.py               # visual + action distribution shift (no model needed)
```

`fig_*` scripts (except `fig_training` / `fig_data_shift`) read the cache written by
`run_evals.py`, so run that first (or use `run_all.py`).

## What each script produces

| script | figures | results JSON |
| --- | --- | --- |
| `run_evals.py` | — | `eval_summary.{json,csv,md}`, `arrays/*.npz` |
| `fig_generalization.py` | `fig_generalization_matrix`, `fig_generalization_gap` | `generalization_findings.json` |
| `fig_chunking.py` | `fig_chunk_step_growth`, `fig_chunk_size_ablation` | `chunking_findings.json` |
| `fig_error_breakdown.py` | `fig_per_dim_error`, `fig_error_groups` | `error_breakdown.json` |
| `fig_temporal.py` | `fig_temporal_error` | `temporal_error.json` |
| `fig_training.py` | `fig_training_curves`, `fig_convergence_speed` | — |
| `fig_data_shift.py` | `fig_visual_shift`, `fig_image_montage`, `fig_action_shift` | `data_shift.json` |

## Method notes (so the numbers are defensible)

* **Held-out everywhere.** Every model is evaluated on `subset="val"` with the same
  `val_fraction=0.05` / `split_seed=42` used at train time, so no evaluated frame was
  a training frame for either model.
* **Representative sampling.** We evaluate an evenly-spaced 3000-window sample over
  the *whole* val split (not the first-512-window slice the training loop logs), for
  low-variance, comparable numbers — see `FINDINGS.md §0`.
* **Raw action units.** Errors invert each model's own normaliser before comparison,
  exactly like `calvin_act/evaluate.py`, so ACT-A and ACT-ABC are compared fairly.
* **B / C = unseen-env proxy for the absent env D.** ACT-A never trained on B or C,
  so its error there is a true zero-shot visual-shift signal. Add `splitD` and put
  `"D"` in `EVAL_ENVS` (`common.py`) to extend every figure to the real Task-3 env.

Config of what is analysed (models, eval envs, action-dim labels, colours) lives in
[`common.py`](common.py).
