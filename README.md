# HW3 题目二：基于 ACT 的跨环境操作策略学习与分布迁移分析

在 **CALVIN** 数据集（LeRobot v2.1 格式）上训练 **ACT**（Action Chunking with Transformers）视觉-动作策略，系统研究视觉分布偏移对开环动作分块策略的影响。三个子任务均已完成：

1. **任务一**：仅用环境 A 训练 ACT-A（基础策略）
2. **任务二**：用 A+B+C 混合数据训练 ACT-ABC（相同网络结构与超参数），分析联合训练的泛化收益
3. **任务三**：将两个模型零样本部署到未见过的环境 D，分析动作分块机制在视觉分布偏移下的鲁棒性

```
 env A        ──► 任务一 ──► ACT-A   ─┐
 env A+B+C    ──► 任务二 ──► ACT-ABC ─┤──► 任务三：env D 零样本评测
                                       └──► 对比 Action L1 / 逐分块步误差 / 逐维误差
```

**核心结论**：ACT-A 在未见环境上误差暴涨 80–119%，开环动作分块将误读的单帧感知沿预测序列放大（域外误差增长斜率为域内 6 倍）；多环境联合训练（ACT-ABC）将跨环境误差降低 36–40%，零样本迁移 env D 改善 16.0%，仅付出域内 13% 的小幅代价。

---

## 1. 实验完成情况

| 任务 | 状态 | 输出目录 |
| --- | --- | --- |
| 任务一：env A 训练（40k 步） | ✅ | `outputs/task1_envA_40k/` |
| 任务二：env A+B+C 联合训练（40k 步） | ✅ | `outputs/task2_envABC_40k/` |
| chunk size 消融（H=5/10/20，A+B+C，40k） | ✅ | `outputs/task2_envABC_H5_40k/`、`_H20_40k/` |
| 任务三：env D 零样本评测 | ✅ | `outputs/task3_zeroshot_D/` |
| 深入分析（12 张图 + 5 个 JSON 结果） | ✅ | `analyze/figures/`、`analyze/results/` |
| 策略预测可视化（case study） | ✅ | `analyze/figures/fig_policy_cases.png` |

> 模型权重 `*.pt`（每个约 89 MB）不入库，请从网盘下载（见 [§7 模型权重](#7-模型权重)）。

---

## 2. 仓库结构

```
DL_final_term/
├── README.md
├── requirements.txt / environment.yml / pyproject.toml
├── configs/
│   ├── act_calvin.json          # 主配置（ResNet-18, H=10）—— 任务一/二共用
│   └── act_calvin_small.json    # 低显存快速调试配置
├── calvin_act/                  # 核心代码包（python -m calvin_act <command>）
│   ├── cli.py                   # 命令行入口
│   ├── calvin_data.py           # LeRobot 数据读取
│   ├── dataset.py               # 动作分块窗口 PyTorch Dataset
│   ├── act_model.py             # ACT 策略（ResNet-18 + CVAE + Transformer）
│   ├── normalize.py             # 状态/动作归一化
│   ├── train.py                 # 训练循环（SwanLab/WandB 日志）
│   ├── evaluate.py              # 开环误差评测 + 闭环 rollout 接口
│   ├── metrics.py               # Action L1/MSE、逐分块步误差
│   ├── temporal_ensemble.py     # ACT 时间集成（闭环推理）
│   └── lerobot_export.py        # 官方 LeRobot ACT 导出说明
├── outputs/                     # 训练与评测结果（*.pt 权重已排除）
│   ├── task1_envA_40k/          # config.json / history.json / train_summary.json
│   ├── task2_envABC_40k/
│   ├── task2_envABC_H5_40k/
│   ├── task2_envABC_H20_40k/
│   └── task3_zeroshot_D/        # eval_A_on_D.json / eval_ABC_on_D.json / per_chunk_step_D.png
└── analyze/                     # 深入分析脚本与结果
    ├── README.md                # 分析代码使用说明
    ├── FINDINGS.md              # 完整分析结论（含全部数字与图表索引）
    ├── common.py                # 共享配置与工具函数
    ├── run_evals.py             # 批量重评测（生成 analyze/results/arrays/）
    ├── run_all.py               # 一键复现全套分析图
    ├── fig_training.py          # 训练曲线 + 收敛速度
    ├── fig_generalization.py    # 泛化矩阵与跨环境退化
    ├── fig_chunking.py          # 逐分块步误差 + chunk size 消融
    ├── fig_error_breakdown.py   # 逐动作维度误差分析
    ├── fig_temporal.py          # episode 内时序误差
    ├── fig_data_shift.py        # 视觉分布偏移量化
    ├── fig_policy_cases.py      # 策略预测轨迹可视化（case study）
    ├── figures/                 # 生成的分析图（PNG）
    └── results/                 # 分析指标 JSON
        ├── eval_summary.json
        ├── error_breakdown.json
        ├── chunking_findings.json
        ├── generalization_findings.json
        ├── temporal_error.json
        └── data_shift.json
```

---

## 3. 环境配置

Python ≥ 3.9：

```bash
pip install torch torchvision        # 选择与 GPU 匹配的 CUDA 版本
pip install -e .                     # 或：pip install -r requirements.txt
pip install -e ".[plot,logging]"     # 可选：matplotlib + SwanLab/WandB
```

Conda 用户：`conda env create -f environment.yml`

核心依赖：`torch, torchvision, numpy, Pillow, pandas, pyarrow`

---

## 4. 数据准备

数据为 LeRobot v2.x 格式，每个 CALVIN 环境一个文件夹，默认根目录为 `data/calvin-lerobot/`：

```
data/calvin-lerobot/
  splitA/                                   # 环境 A
    meta/info.json
    meta/episodes.jsonl
    data/chunk-000/episode_000000.parquet   # 每个 parquet = 一个 episode
    ...
  splitB/  splitC/  splitD/
```

每行 parquet 包含：`image`（200×200，PNG 编码）、`wrist_image`（84×84）、`state`（float32[15]）、`actions`（float32[7]，CALVIN rel_actions）。

**指定数据路径**（优先级从高到低）：
1. 默认路径 `data/calvin-lerobot/`（无需配置）
2. 环境变量 `export CALVIN_DATA_ROOT=/path/to/calvin-lerobot`
3. 命令行参数 `--data-root <path>`

验证数据加载：
```bash
python -m calvin_act info
```

---

## 5. 训练与评测

### 任务一 —— env A 训练

```bash
python -m calvin_act task1 --config configs/act_calvin.json --steps 40000 --num-workers 4
```

### 任务二 —— env A+B+C 联合训练

```bash
python -m calvin_act task2 --config configs/act_calvin.json --steps 40000 --num-workers 4
```

### 任务三 —— env D 零样本评测

```bash
python -m calvin_act task3 \
  --ckpt-a   outputs/task1_envA_40k/model_final.pt \
  --ckpt-abc outputs/task2_envABC_40k/model_final.pt \
  --test-env D
```

输出：`outputs/task3_zeroshot_D/eval_*_on_D.json`、`per_chunk_step_D.png`。

### chunk size 消融

```bash
python -m calvin_act train --envs ABC --chunk-size 5  --out outputs/task2_envABC_H5_40k  --steps 40000
python -m calvin_act train --envs ABC --chunk-size 20 --out outputs/task2_envABC_H20_40k --steps 40000
```

### 常用参数

| 参数 | 含义 |
| --- | --- |
| `--data-root PATH` | 数据根目录 |
| `--config FILE` | ACT 配置 JSON |
| `--steps N` / `--batch-size N` / `--chunk-size H` | 覆盖配置项 |
| `--device cuda\|cpu` | 计算设备（默认自动） |
| `--num-workers N` | DataLoader 进程数（建议 4–8） |
| `--logger swanlab\|wandb\|none` | 实验记录后端 |

---

## 6. 深入分析

完整结论（含全部数字）见 [`analyze/FINDINGS.md`](analyze/FINDINGS.md)，一键复现：

```bash
cd analyze
python run_evals.py   # 先生成 results/arrays/（需要权重）
python run_all.py     # 生成全套分析图
```

### 分析图说明

| 图 | 内容 | 关键结论 |
| --- | --- | --- |
| `fig_generalization_matrix.png` | 模型 × 环境的 held-out Action L1 | ACT-A 在 env B 误差 0.198（域内 0.090 的 2.2×）；ACT-ABC 全环境持平 0.102–0.119 |
| `fig_training_curves.png` | 训练 L1 + KL 曲线（含全集重测点） | CVAE KL 在 5k 步内坍缩；联合模型 train-L1 更高但泛化更好 |
| `fig_convergence_speed.png` | 达到 L1 阈值的步数对比 | ACT-ABC 收敛略慢但域外误差显著更低 |
| `fig_chunk_size_ablation.png` | H=5/10/20 的精度与逐步误差 | 开环下短分块误差更低；首步误差随 H 小幅上升（0.097→0.103） |
| `fig_per_dim_error.png` | 7 维动作逐维 L1 | 视觉偏移下 yaw 与夹爪误差最大（域内的 2.8–3.0×） |
| `fig_error_groups.png` | 平移/旋转误差 + 夹爪错误率 | ACT-A 夹爪错误率在 env B 升至 0.162（域内的 3× ） |
| `fig_temporal_error.png` | 误差随 episode 内位置变化 | 呈 U 形：起点（接近阶段）与终点（精细抓取）误差最高 |
| `fig_visual_shift.png` | 各环境 RGB 均值、亮度分布、视觉距离矩阵 | A→B 视觉距离（0.080）最大，与 ACT-A 在 B 上误差最高高度一致 |
| `fig_image_montage.png` | A/B/C 静态相机样例帧 | 直观展示桌面颜色差异（偏移纯来自视觉） |
| `fig_action_shift.png` | 各环境动作分布直方图 | 三环境动作分布几乎重合——偏移为纯视觉，非动作分布偏移 |
| `fig_policy_cases.png` | env A/B/D 各一帧的预测轨迹 vs GT | ACT-A 在 env B 预测轨迹严重偏离（L1=0.225），ACT-ABC 保持对齐（L1=0.123） |

---

## 7. 模型权重

6 个模型权重（每个约 89 MB）不入库，请从网盘下载后放回对应 `outputs/<run>/model_final.pt`：

```
链接: https://pan.baidu.com/s/14BRCGqMP-1-cDcouDP7lrg?pwd=2026   提取码: 2026
```

---

## 8. 超参数

两个主模型（ACT-A / ACT-ABC）配置完全一致，仅训练数据不同：

| 项 | 取值 |
| --- | --- |
| 视觉骨干 | ResNet-18（ImageNet 预训练），96×96，双路（静态+腕部） |
| chunk size H / CVAE 潜变量维度 | 10 / 32 |
| hidden dim / 注意力头 | 256 / 8 |
| Transformer 编码器/解码器层数 | 4 / 6 |
| KL 权重 β / Dropout | 10 / 0.1 |
| 优化器 | AdamW，CosineAnnealingLR |
| lr / lr_backbone / weight_decay | 1e-4 / 1e-5 / 1e-4 |
| batch size / steps | 32 / 40000 |
| 动作空间 | rel_actions（7 维），min-max 归一化至 [−1, 1] |
| 状态空间 | 15 维，mean-std 归一化 |
| 验证集划分 | val_fraction=0.05，split_seed=42 |
