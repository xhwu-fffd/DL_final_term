# Task 2 — Deep Analysis & Error Analysis (report-ready)

This document collects the extra analyses for the HW3 题目二 report. Every number
below is reproduced by the scripts in this folder (`python run_all.py`) and every
claim points at a figure in `analyze/figures/` and a JSON in `analyze/results/`.

> **Scope note.** The dataset ships environments **A, B, C** only — there is no
> `splitD`, so the literal Task-3 zero-shot test on env D could not be run. We
> therefore use **B and C as the unseen-environment proxy**: the A-only model
> (`ACT-A`) never saw B or C during training, so its error there is a genuine
> zero-shot, visual-distribution-shift signal — exactly the phenomenon Task 3
> asks us to analyse. When `splitD` is added, `run_evals.py` picks it up by adding
> `"D"` to `EVAL_ENVS` in `common.py` and every figure regenerates unchanged.

---

## 0. Are the Task 1 / Task 2 results as expected?  → Yes.

| run | envs | H | steps | final train L1 | logged val L1 | **full held-out L1 (env A)** |
| --- | --- | --- | --- | --- | --- | --- |
| task1_envA_40k | A | 10 | 40k | 0.563 | 0.542 | **0.090** |
| task2_envABC_40k | A+B+C | 10 | 40k | 0.740 | 0.585 | **0.102** |

Both runs converge cleanly (`fig_training_curves.png`). The joint model sits at a
*higher train L1* — expected and healthy: it fits 3× the visual variety with the
same capacity, so it cannot specialise to A as tightly. This is the standard
specialisation-vs-generalisation trade and is the whole point of Task 2.

**A measurement caveat we fixed.** The `val_action_l1` logged during training
(~0.54) is computed over only the **first 512 windows** of the val split
(`cfg.val_batches=16 × batch=32`, no shuffle) — a high-variance, pessimistic slice
of one or two episodes. Re-evaluating on a representative 3000-window even sample
of the *full* held-out split gives **0.090 / 0.102** — ~6× lower and far more
stable. Use the full-split numbers for any cross-run comparison in the report; the
logged curve is still fine as a *relative* training-progress signal. (★ markers in
`fig_training_curves.png` show the gap.)

---

## 1. Headline: does A+B+C training buy robustness to a shifted env?

`fig_generalization_matrix.png`, `fig_generalization_gap.png` ·
`results/generalization_findings.json`

**Held-out action L1 (raw units), model × eval-env:**

| model | env A | env B | env C |
| --- | --- | --- | --- |
| **ACT-A** (A only) | **0.090** | 0.198 | 0.163 |
| **ACT-ABC** (A+B+C) | 0.102 | **0.119** | **0.105** |

* ACT-A collapses off-domain: **+119 %** error on B, **+80 %** on C vs its own env A.
* ACT-ABC is almost flat: **+16 %** on B, **+2 %** on C — it has effectively
  *absorbed* the shift by seeing those domains in training.
* Multi-env training cuts the cross-env error by **~40 % on B** and **~36 % on C**.
* The price is a modest **+13 %** in-domain cost on A (0.090 → 0.102) — the classic
  generalist-vs-specialist trade.

**Takeaway for the report:** training on more environments is a strong, cheap
robustness lever — it converts a 2× off-domain blow-up into a near-flat response,
at a small in-domain cost.

---

## 2. The shift is *visual*, not in the action space

`fig_visual_shift.png`, `fig_image_montage.png`, `fig_action_shift.png` ·
`results/data_shift.json`

* **Visual:** env B is far brighter (mean brightness **176**) than A (**119**) and
  C (**129**); the montage shows A = dark red desk, B = light tan desk, C = mid
  brown with a different object layout. The RGB-histogram **visual distance**
  A→B = **0.080** is the largest pairwise gap, A→C = **0.047**.
* **Action:** the per-dimension action histograms of A/B/C **overlap almost
  perfectly**, and the gripper open-fraction is 0.47–0.48 in every env. So the
  teleop action statistics are essentially identical across environments.

**Why this matters:** it lets us attribute *all* of ACT-A's degradation to the
**visual** distribution shift — there is no confounding action-distribution shift.
And the ranking lines up: the visually-most-distant env (B) is also where ACT-A
hurts most (+119 %), while the closer env (C) hurts less (+80 %). Visual distance
predicts transfer error.

---

## 3. Action-chunking robustness under visual shift  (the required deep analysis)

`fig_chunk_step_growth.png` · `results/chunking_findings.json`

We plot the L1 at each position *inside* the predicted chunk (0 = most immediate).

**Error growth across the 10-step chunk (first-step → last-step L1):**

| model | env A (in-domain) | env B (unseen) | env C (unseen) |
| --- | --- | --- | --- |
| ACT-A | 0.087 → 0.099 (**+0.012**) | 0.163 → 0.237 (**+0.074**) | 0.140 → 0.182 (**+0.043**) |
| ACT-ABC | 0.095 → 0.112 (+0.017) | 0.109 → 0.131 (+0.022) | 0.097 → 0.114 (+0.017) |

**Interpretation (mechanism).** ACT predicts a whole chunk from *one* observation.
In-domain the perception is reliable, so error is low and barely grows along the
chunk (ACT-A on A: +0.012). Under visual shift the single observation is
*misread*, and because the chunk is generated open-loop from that one latent, the
error not only **starts higher** but **compounds with horizon** — ACT-A's growth is
**6× steeper on B** than on A. The multi-env model, having learned B/C visually,
keeps both the level *and* the slope low. This is the concrete failure mode of
action chunking under visual distribution shift: **commitment to a chunk built on
an out-of-distribution percept propagates that misread across the whole horizon.**

The same mechanism has an upside in-domain (next section): committing to a chunk
smooths per-frame perception noise.

---

## 4. Chunk-size ablation: H ∈ {5, 10, 20}

`fig_chunk_size_ablation.png` · `results/chunking_findings.json`
(all A+B+C, 40k, identical config except H; metrics averaged over A/B/C)

| H | chunk-mean L1 | first-step L1 | gripper acc |
| --- | --- | --- | --- |
| 5  | **0.103** | **0.097** | **0.946** |
| 10 | 0.109 | 0.100 | 0.928 |
| 20 | 0.118 | 0.103 | 0.902 |

* Open-loop accuracy *worsens* monotonically with H. Panel (c) shows why: all H
  share the **same per-step error profile**; a longer chunk simply extends further
  into the high-error far-future, dragging the average up.
* Gripper accuracy drops from 0.946 (H=5) to 0.902 (H=20): committing 20 steps
  ahead makes the discrete grasp decision stale.

**Important caveat to state in the report.** This open-loop metric structurally
favours *small* H. It does **not** capture the closed-loop benefit of large H +
temporal ensembling (smoothness, robustness to per-frame jitter, fewer
compounding-error restarts). So "small H wins" holds for open-loop action error;
the closed-loop sweet spot is typically larger. With the CALVIN PyBullet simulator
one would re-run this as success-rate vs H. The cross-env *gap* (B − A) stays ~0.015
at every H, i.e. multi-env training neutralises the shift independently of H.

---

## 5. Where the error lives: per-dimension & gripper

`fig_per_dim_error.png`, `fig_error_groups.png` · `results/error_breakdown.json`

* Under shift, ACT-A's error concentrates in **yaw** (action[5]) and the
  **gripper** (action[6]) — the most *vision-dependent* commands (orientation and
  grasp timing need to localise the object). Translation degrades less.
* **Gripper error rate (1 − sign acc):** ACT-A 0.053 (A) → **0.162 (B)**, a 3×
  jump; ACT-ABC holds at 0.066 → 0.082. Note ACT-ABC is *slightly worse* than ACT-A
  on env-A gripper (0.066 vs 0.053) — the visible in-domain specialisation cost.

---

## 6. Error along the trajectory (phase analysis)

`fig_temporal_error.png` · `results/temporal_error.json`

Binning each observation by its position within its episode reveals a **U-shape**:
error is highest at the **start** (initial approach, high motion/uncertainty) and
**end** (fine manipulation / contact), lowest mid-trajectory. The cross-env penalty
(ACT-A red above ACT-ABC blue) is present across all phases but largest at the
episode start on env B — the policy is least anchored when it must localise the
scene from scratch under an unfamiliar appearance.

---

## 7. One-paragraph summary for the report

> Training the ACT policy jointly on environments A+B+C converts a brittle,
> domain-specialised policy into a robust one: the A-only model's open-loop action
> error inflates by 80–120 % on environments it never saw, whereas the joint model
> stays within 2–16 % of its in-domain error, a ~40 % cross-environment error
> reduction, bought for a ~13 % in-domain cost. Because the A/B/C action statistics
> are identical and only the appearance differs, this degradation is purely a
> visual-distribution-shift effect, and its magnitude tracks the measured visual
> distance between environments. The action-chunking mechanism is the locus of the
> failure: an out-of-distribution observation is misread and, because the chunk is
> rolled out open-loop from that single percept, the error compounds along the
> horizon — 6× faster off-domain than in-domain — and concentrates in the
> vision-dependent yaw and gripper commands. Shorter chunks reduce this open-loop
> compounding (and improve gripper timing), while multi-environment training
> removes the shift sensitivity at every chunk length.

---

### Files

Figures (`analyze/figures/`): `fig_generalization_matrix`, `fig_generalization_gap`,
`fig_chunk_step_growth`, `fig_chunk_size_ablation`, `fig_per_dim_error`,
`fig_error_groups`, `fig_temporal_error`, `fig_visual_shift`, `fig_image_montage`,
`fig_action_shift`, `fig_training_curves`, `fig_convergence_speed`.

Data (`analyze/results/`): `eval_summary.{json,csv,md}`,
`generalization_findings.json`, `chunking_findings.json`, `error_breakdown.json`,
`temporal_error.json`, `data_shift.json`, raw per-eval arrays in `arrays/*.npz`.
