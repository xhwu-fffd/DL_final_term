# HW3 题目二：基于 LeRobot 的 ACT 策略跨环境泛化

在 **CALVIN** 数据集（以 **LeRobot v2.1** 格式提供）上训练轻量级 **ACT**（Action
Chunking with Transformers）视觉-动作策略，研究它在**不同环境间的泛化能力**。三个子任务：

1. **任务 1 — 基础策略**：仅用环境 **A** 训练 ACT。
2. **任务 2 — 联合训练**：用 **A+B+C** 混合数据、**完全相同的网络结构与超参数**重训一个模型，对比训练收敛（Action L1）。
3. **任务 3 — 零样本跨环境测试**：把两个模型部署到**未见过的环境 D** 上零样本评测，重点分析**动作分块机制在视觉分布偏移下的鲁棒性**。

> **当前进度**：任务 1、任务 2 **已完成**（含 20k/40k 步、以及 chunk size H∈{5,10,20} 的消融实验），并补充了一整套**深入分析**（见 [§5 分析部分](#5-分析部分deep-analysis)）。
> **任务 3 的代码已就绪但尚未运行** —— 因为本数据集只提供了 A/B/C，**没有 `splitD`**。待拿到环境 D 数据后即可直接运行（见 [§6 给同伴：任务 3 待办](#6-给同伴任务-3-待办)）。

```
        ┌──────────── 完全相同的 ACT 配置 ────────────┐
 env A    ──► 任务1 ──► model_A   ─┐                    │
 env A+B+C ──► 任务2 ──► model_ABC ─┤─► 任务3：在 env D 上零样本评测 ─► 对比 A vs ABC
        └──────────────────────────┘    (Action L1 / 逐分块步误差) + 分块鲁棒性分析
```

---

## 1. 我做了什么（工作总览）

| 项目 | 状态 | 产物 |
| --- | --- | --- |
| 任务 1：env A 训练（20k & 40k 步） | ✅ | `outputs/task1_envA`, `outputs/task1_envA_40k` |
| 任务 2：env A+B+C 联合训练（20k & 40k 步） | ✅ | `outputs/task2_envABC`, `outputs/task2_envABC_40k` |
| chunk size 消融（H=5 / 10 / 20，A+B+C，40k） | ✅ | `outputs/task2_envABC_H5_40k`, `..._H20_40k`（H10 即 `task2_envABC_40k`） |
| 训练收敛对比图（A vs ABC、20k vs 40k 等） | ✅ | `outputs/convergence_*.png` |
| 全部实验汇总表 | ✅ | `outputs/training_summary_all_runs.{md,csv}` |
| 深入分析 + 误差分析（12 张图） | ✅ | `analyze/figures/`, `analyze/results/`，报告稿见 `analyze/FINDINGS.md` |
| 任务 3：零样本 env D 评测 | ⏳ 待运行 | 代码：`calvin_act task3` / `calvin_act eval` |

**一句话结论**：联合训练（A+B+C）把"只在 A 上训练"的脆弱策略变成了鲁棒策略 —— ACT-A 在没见过的环境上动作误差暴涨 80–120%，而 ACT-ABC 只比自身 in-domain 高 2–16%（跨环境误差降低约 40%），代价仅是 in-domain 上约 13% 的小幅损失。

---

## 2. 仓库结构

```
task2_share/
├── README.md                    # 本文件
├── REPORT_OUTLINE.md            # 实验报告大纲（撰写报告时参考）
├── requirements.txt / environment.yml / pyproject.toml
├── configs/
│   ├── act_calvin.json          # 主配置（ResNet-18, H=10, ...）—— 任务1/2共用
│   └── act_calvin_small.json    # 笔记本/小显存的快速配置
├── calvin_act/                  # 核心代码包（python -m calvin_act <command>）
│   ├── cli.py                   # 命令行入口（task1/task2/task3/train/eval/info/...）
│   ├── calvin_data.py           # LeRobot 数据读取（按文件夹选择环境）
│   ├── dataset.py               # 动作分块窗口的 PyTorch Dataset
│   ├── act_model.py             # ACT 策略（ResNet-18 + CVAE + Transformer 解码器）
│   ├── normalize.py             # 状态/动作归一化（仅用训练集统计量）
│   ├── train.py                 # 训练循环（含验证曲线、SwanLab/WandB 日志）
│   ├── evaluate.py              # 零样本开环动作误差评测（任务3）+ 闭环 rollout 接口
│   ├── metrics.py               # Action L1/MSE、逐分块步误差、对比表
│   ├── temporal_ensemble.py     # ACT 时间集成（闭环推理用）
│   └── lerobot_export.py        # 用官方 LeRobot ACT 训练的说明
├── outputs/                     # 训练结果（已去除 *.pt 权重，权重见网盘）
│   └── <run>/{config,history,train_summary}.json + convergence_*.png + 汇总表
└── analyze/                     # 深入分析与画图（详见 §5）
    ├── README.md                # 分析代码使用说明
    ├── FINDINGS.md              # ★ 报告可直接引用的分析结论（每个数字都有图/数据支撑）
    ├── run_all.py               # 一键跑完整套分析
    ├── common.py / run_evals.py / fig_*.py
    ├── figures/                 # 12 张分析图（PNG）
    └── results/                 # 分析得到的 JSON/CSV 指标
```

> 注意：模型权重 `*.pt`（每个约 89 MB）和原始数据 `data/`、SwanLab 日志 `swanlog/`
> 均**不入库**。权重请从网盘下载（见 [§7](#7-模型权重)）后放回对应 `outputs/<run>/` 目录。

---

## 3. 环境配置

需要 Python ≥ 3.9：

```bash
pip install torch torchvision        # 选择与你 GPU 匹配的 CUDA 版本
pip install -e .                     # 或：pip install -r requirements.txt
pip install -e ".[plot,logging]"     # 可选：matplotlib 画图 + SwanLab/WandB 日志
```

核心依赖：`numpy, torch, torchvision, Pillow, pandas, pyarrow`（pandas/pyarrow 读取
LeRobot 的 parquet）。Conda 用户：`conda env create -f environment.yml`。

---

## 4. 数据准备

数据是**一组 LeRobot v2.x 数据集，每个 CALVIN 环境一个文件夹**，放在同一个根目录下。
默认根目录为 `data/calvin-lerobot/`（即本目录下）：

```
data/calvin-lerobot/
  splitA/                                   # 环境 A
    meta/info.json                          # 特征定义、fps、路径模板
    meta/episodes.jsonl                     # 每行一个 episode（长度、scene 等）
    meta/tasks.jsonl
    data/chunk-000/episode_000000.parquet   # 每个 parquet = 一个 episode
    ...
  splitB/  splitC/  splitD/                  # 同样的结构（D 待补充）
```

每行 parquet 是一个时间步，列包含：`image`（200×200×3，PNG 编码）、`wrist_image`
（84×84×3）、`state`（float32[15]）、`actions`（float32[7]，即 CALVIN `rel_actions`）。
**环境 = 文件夹**：env A → `splitA`，…，env D → `splitD`。

**让代码找到数据**（优先级从高到低）：

1. 默认：放在 `task2_share/data/calvin-lerobot/`，无需任何参数；
2. 环境变量：`$env:CALVIN_DATA_ROOT = "D:\path\to\calvin-lerobot"`（PowerShell）/ `export CALVIN_DATA_ROOT=...`（bash）；
3. 命令行参数：`--data-root <path>`。

确认加载器看到的数据：

```bash
python -m calvin_act info
#   env A: 6089 episodes, 366693 frames, fps=10 (splitA) ...
#   env B: 6115 episodes ... ; env C: 5666 episodes ...
```

本次使用的数据规模：env A = 6089 episodes / 366693 frames，env B = 6115，env C = 5666，fps=10。

---

## 5. 运行训练与评测

每个任务**独立可运行**。默认输出到 `outputs/task1_envA`、`outputs/task2_envABC`、`outputs/task3_zeroshot_D`。

### 任务 1 —— 仅在环境 A 上训练

```bash
python -m calvin_act task1 --config configs/act_calvin.json --logger swanlab --num-workers 4
```

### 任务 2 —— 在 A+B+C 上联合训练（配置完全相同）

```bash
python -m calvin_act task2 --config configs/act_calvin.json --logger swanlab --num-workers 4
```

（需要 `splitB/`、`splitC/` 存在。）若任务 1 也已完成，任务 2 会自动输出
`outputs/convergence_A_vs_ABC.png` 对比两者的 Action-L1 曲线。

### 任务 3 —— 在未见过的环境 D 上零样本评测两个模型

```bash
python -m calvin_act task3        # 默认使用 outputs/task1_envA 和 outputs/task2_envABC
```

会输出每个模型的评测 JSON、`comparison.md/.csv` 以及 `per_chunk_step_D.png`
（动作分块鲁棒性曲线）。可用 `--ckpt-a` / `--ckpt-abc` 指定权重，用 `--test-env` 改测试环境。

### chunk size 消融（已做）

```bash
# H=5/10/20，A+B+C，相同数据，仅改 chunk-size
python -m calvin_act train --envs ABC --config configs/act_calvin.json --chunk-size 5  --out outputs/task2_envABC_H5_40k  --steps 40000
python -m calvin_act train --envs ABC --config configs/act_calvin.json --chunk-size 20 --out outputs/task2_envABC_H20_40k --steps 40000
```

### 常用参数（所有命令通用）

| 参数 | 含义 |
| --- | --- |
| `--data-root PATH` | `split{A,B,C,D}/` 所在目录（默认 `data/calvin-lerobot`） |
| `--config FILE` | ACT 配置 JSON（默认 `configs/act_calvin.json`） |
| `--device cuda\|cpu` | 计算设备（默认自动） |
| `--num-workers N` | DataLoader 进程数（GPU 训练建议 4–8） |
| `--logger swanlab\|wandb\|none` | 实验记录后端（导出 Loss/验证曲线） |
| `--steps N` / `--batch-size N` / `--chunk-size H` | 覆盖配置项 |
| `--max-episodes N` | 每个环境最多用多少 episode（快速调试/省显存） |

---

## 6. 分析部分（Deep Analysis）

> 这是本次工作的重点。完整的、可直接写进报告的结论（含全部数字）在
> **[`analyze/FINDINGS.md`](analyze/FINDINGS.md)**；分析代码用法见
> [`analyze/README.md`](analyze/README.md)。一键复现：`cd analyze && python run_all.py`
> （需要先把权重放回 `outputs/`）。

**方法学要点**（保证数字站得住脚）：

- **全集重测**：训练时记录的 `val_action_l1`（≈0.54）只在验证集**前 512 个窗口**上算，方差大、偏悲观；分析里在整个 held-out 集上均匀采样 3000 个窗口重测，得到稳定值（env A ≈ 0.09）。
- **统一在 held-out 上评测**：用与训练相同的 `val_fraction=0.05`、`split_seed=42`，保证评测帧对两个模型都没在训练里见过。
- **原始动作单位**：误差先逆归一化再比较，A 模型与 ABC 模型公平可比。
- **用 B/C 代理未见环境 D**：因为缺 `splitD`，而 ACT-A 从未见过 B/C，所以它在 B/C 上的误差就是真实的零样本视觉偏移信号 —— 正是任务 3 要分析的现象。补上 D 后，把 `analyze/common.py` 里 `EVAL_ENVS` 加上 `"D"` 即可让所有图自动扩展。

### 各张图的含义

| 图（`analyze/figures/`） | 分析内容 | 关键结论 |
| --- | --- | --- |
| **fig_generalization_matrix.png** | 模型 × 评测环境的 held-out 动作 L1 热图 + 柱状图 | ACT-A：A=0.090 → B=0.198 / C=0.163（暴涨）；ACT-ABC：0.102 / 0.119 / 0.105（几乎持平） |
| **fig_generalization_gap.png** | 相对各自 in-domain 的跨环境退化 + 夹爪准确率 | 联合训练把跨环境退化从 +120% 压到 +16% |
| **fig_chunk_step_growth.png** | **逐分块步**误差（chunk 内第 1…H 步）in-domain vs 未见环境 | ★ 核心：ACT-A 误差在 A 上沿 chunk 仅 +0.012，在 B 上 +0.074（**陡 6 倍**）—— 误读一帧 OOD 观测后，开环 rollout 把错误沿整个 horizon 放大 |
| **fig_chunk_size_ablation.png** | H=5/10/20 的精度、夹爪准确率、逐步误差曲线 | 开环下 H 越小逐步误差越低、夹爪越准；所有 H 共享同一逐步误差曲线，长 chunk 只是延伸到更难的远期 |
| **fig_per_dim_error.png** | 7 维动作（xyz/roll-pitch-yaw/夹爪）逐维误差 | 视觉偏移下 ACT-A 在 **yaw 和夹爪**（最依赖视觉）上误差最大 |
| **fig_error_groups.png** | 平移/旋转误差 + 夹爪错误率 | ACT-A 夹爪错误率在 env B 上从 0.053 升到 0.162（3 倍） |
| **fig_temporal_error.png** | 误差随 episode 内位置（0=起点,1=终点）的变化 | 呈 U 形：起点（接近）和终点（精细操作/抓取）误差最高 |
| **fig_visual_shift.png** | 各环境图像的 RGB 均值、亮度分布、视觉距离矩阵 | env B 明显更亮（亮度 176 vs A 的 119）；视觉距离 A→B(0.08) 最大，与 ACT-A 在 B 上误差最大一致 |
| **fig_image_montage.png** | A/B/C 静态相机样例帧拼图 | 直观看到桌面颜色/纹理/物体布局差异（A 深红桌、B 浅黄桌、C 中棕桌） |
| **fig_action_shift.png** | 各环境动作分布直方图 + 夹爪开合比例 | 三个环境动作分布几乎重合 → **偏移纯粹是视觉的**，不是动作分布偏移 |
| **fig_training_curves.png** | A vs ABC 训练 L1 + KL 曲线（叠加全集重测点） | 联合模型 train-L1 更高（看更多视觉变化），但泛化更好；CVAE 的 KL 早期就坍缩 |
| **fig_convergence_speed.png** | 各 run 达到 train-L1 阈值所需步数 | 收敛速度对比 |

### 分析结果对应的数据文件（`analyze/results/`）

`eval_summary.{json,csv,md}`（所有 模型×环境 的指标）、`generalization_findings.json`、
`chunking_findings.json`、`error_breakdown.json`、`temporal_error.json`、`data_shift.json`。

---

## 7. 任务 3 

任务 3 代码已经写好：

1. 把 `splitD/`（与 A/B/C 同样的 LeRobot 结构）放进 `data/calvin-lerobot/`；
2. 从网盘下载权重，放回 `outputs/task1_envA_40k/model_final.pt` 和 `outputs/task2_envABC_40k/model_final.pt`；
3. 运行零样本评测：
   ```bash
   python -m calvin_act task3 \
     --ckpt-a   outputs/task1_envA_40k/model_final.pt \
     --ckpt-abc outputs/task2_envABC_40k/model_final.pt \
     --test-env D
   ```
   得到 `comparison.md`、`eval_*_on_D.json`、`per_chunk_step_D.png`。
4. 让整套分析图自动覆盖到 env D：编辑 `analyze/common.py`，把 `EVAL_ENVS = ["A","B","C"]`
   改成 `["A","B","C","D"]`，然后 `cd analyze && python run_all.py`。所有图会自动加入 env D 的结果。
5. （可选，更标准的指标）闭环成功率需要 CALVIN 的 PyBullet 仿真器 `calvin_env`
   （单独安装）；`evaluate.closed_loop_rollout(...)` 已提供带时间集成的通用闭环接口。

报告中重点分析（任务要求）：**动作分块机制在跨环境视觉偏移下的鲁棒性** ——
可直接复用 `analyze/FINDINGS.md` 第 3、4 节的论证，并把其中的"未见环境 B/C"替换/补充为真正的 env D。

---

## 8. 模型权重

训练好的 6 个模型权重（每个约 89 MB，已从仓库中排除）请上传网盘并在此处填写链接：

```
网盘链接：<待填写>
提取码：  <待填写>
```

下载后按原目录结构放回 `outputs/<run>/model_final.pt` 即可被 `task3` / `eval` / `analyze` 使用。

---

## 9. 超参数（任务 1/2 完全一致，摘自 `configs/act_calvin.json`）

| 项 | 取值 |
| --- | --- |
| 视觉骨干 | ResNet-18（ImageNet 预训练） |
| chunk size H | 10 |
| hidden dim / heads | 256 / 8 |
| encoder / decoder 层数 | 4 / 6 |
| latent dim / KL 权重 β | 32 / 10 |
| 优化器 | AdamW（cosine 退火） |
| lr / lr_backbone / weight_decay | 1e-4 / 1e-5 / 1e-4 |
| batch size / steps | 32 / 40000 |
| 图像尺寸 / 相机 | 96 / static + wrist |
| 动作空间 | rel_actions（7 维） |
| 归一化 | state: mean/std；action: min/max |

完整的逐 run 汇总见 `outputs/training_summary_all_runs.md`。
