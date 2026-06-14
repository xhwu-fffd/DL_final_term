# Report Outline — Task 2 (ACT Cross-Environment Generalization)

Scaffold for the PDF report (assignment §4.1). Use a NeurIPS/CVPR LaTeX template.
Fill the bracketed `[…]` with your runs. The graded core is the **analysis** of
cross-environment generalisation and the action-chunking mechanism — not just
curves.

---

## Title page
- Title; **all** team members' name + student ID; **division of labour** (分工).
  1–2 people: note it (extra credit).
- GitHub repo link + cloud-drive link to model weights (`model_final.pt` for both
  runs; include extraction code if any).

## 1. Introduction / Background
- Embodied AI imitation learning; the environment-generalisation problem.
- **ACT**: action chunking + a CVAE transformer; why it's a strong, *lightweight*
  baseline. The question: does training on more environments (A+B+C) buy
  robustness to an unseen environment (D)?

## 2. Dataset
- **CALVIN**: long-horizon table-top manipulation; 4 environments A/B/C/D differing
  in desk/textures/object layout (the **visual distribution shift**).
- Format: **LeRobot v2.x**, one dataset folder per environment
  (`data/calvin-lerobot/split{A,B,C,D}/`), one parquet per episode. #episodes and
  #frames per env (paste `calvin_act info` output), observation keys used
  (`image` 200×200[`, wrist_image` 84×84], `state` 15-D), action space (`actions`
  = CALVIN `rel_actions`, 7-D), fps=10.

## 3. Method
- **ACT architecture**: ResNet-18 visual tokens (+ 2-D pos), proprio token, CVAE
  style latent `z`; transformer encoder → memory; decoder over `H` learned action
  queries → action chunk `[H, action_dim]`. Loss = masked **L1** + `β·KL`.
- **Action chunking**: predict `H` future actions per inference; **temporal
  ensembling** (`exp(-m·age)`) at closed-loop time.
- **Normalisation**: state mean/std, action min/max, fit on the *training* env(s),
  saved with each checkpoint (no test-time leakage).
- **Two trainings, one config**: env-A vs env-A/B/C use identical architecture &
  hyperparameters — the only change is the training data.

## 4. Experimental setup (tables)

**Hyperparameters** (identical for both runs — paste from `configs/act_calvin.json`):

| Item | Value |
| --- | --- |
| vision backbone | ResNet-18 (pretrained) |
| chunk size `H` | [10] |
| hidden dim / heads | [256 / 8] |
| enc / dec layers | [4 / 6] |
| latent dim / KL weight `β` | [32 / 10] |
| optimizer | AdamW |
| lr / lr_backbone / weight decay | [1e-4 / 1e-5 / 1e-4] |
| batch size / steps | [32 / 20000] |
| image size / cameras | [96 / static(+gripper)] |
| action space | rel_actions (7-D) |

**Logging**: embed the **WandB/SwanLab** curves — Action L1 (and KL) vs step for
both runs on one axis.

## 5. Results & Analysis

### 5.1 Training convergence (A vs A+B+C)
- The `convergence.png` Action-L1 curves. Does the joint model converge to a
  similar/lower L1? Is it slower per-step but richer? (Note: joint sees more
  visual variety, so a *slightly* higher train L1 is normal and not bad.)

### 5.2 Zero-shot transfer to env D (required comparison)
Paste from the eval JSONs / `comparison.md`:

| Model | Train envs | Train L1 | **D: Action L1 (raw)** | D: first-step L1 | D: MSE | (closed-loop succ.) |
| --- | --- | --- | --- | --- | --- | --- |
| ACT-A | A | [ ] | [ ] | [ ] | [ ] | [ ] |
| ACT-ABC | A+B+C | [ ] | [ ] | [ ] | [ ] | [ ] |

- Expectation: **ACT-ABC generalises better to D** (lower action error / higher
  success) because multi-environment training reduces overfitting to one visual
  domain. Quantify the gap; if it's small, discuss why (synthetic vs real shift,
  capacity, #steps).

### 5.3 Action-chunking robustness under visual shift (required, deep)
- **Per-chunk-step error** (`per_chunk_step_l1_raw`): plot error vs position in the
  chunk for ACT-A and ACT-ABC *on env D*. Does error grow with horizon? Is the
  growth steeper for the A-only model (it must extrapolate visually)?
- **Chunk-size ablation** (`--chunk-size` sweep `H ∈ {1,5,10,20}`): plot zero-shot
  D error vs `H`. Interpret: small `H` → reactive but jittery / compounding error;
  large `H` → smoother & temporally consistent, but can **overcommit** to a wrong
  plan when the *visual input is out of distribution*. Where is the sweet spot,
  and does it shift between in-domain and OOD?
- Tie back to mechanism: chunking + temporal ensembling trade reactivity for
  stability; under visual shift the policy's per-frame perception is noisier, so
  committing to a chunk can **mask** transient perception errors (robustness) but
  also **propagate** a misread (brittleness). Argue which dominates in your data.

### 5.4 (Optional) closed-loop success rate
- If you ran the CALVIN simulator: `success_1..5` and avg-sequence-length for both
  models on D; relate to the offline action error.

## 6. Conclusion & limitations
- Does environmental diversity improve zero-shot robustness here? What would help
  more (domain randomisation, larger/pretrained vision, language conditioning,
  more data)? Open-loop-error vs closed-loop-success caveats.

## Appendix
- Exact commands (copy from README); hardware; per-run wall-clock
  (`train_summary.json`); repo + weights links (repeat from title page).
