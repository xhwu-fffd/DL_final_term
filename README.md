# HW3 题目二：基于 ACT 的跨环境操作策略学习与分布迁移分析

复旦大学 CS60003 深度学习与空间智能 · 期末作业 题目二

在 CALVIN 机器人操作基准上训练 **ACT**（Action Chunking with Transformers）视觉-动作策略，系统研究视觉分布偏移对开环动作分块策略的影响，并通过多环境联合训练提升跨域泛化能力。

| 模型 | 训练环境 | env A | env B | env C | env D（零样本）|
|------|----------|-------|-------|-------|----------------|
| ACT-A  | A only  | **0.090** | 0.198 | 0.163 | 0.161 |
| ACT-ABC | A+B+C  | 0.102 | **0.119** | **0.105** | **0.135** |

> 指标：开环 Action L1（raw，越小越好）。ACT-ABC 将跨环境误差降低 36–40%，零样本 env D 改善 16%。

**模型权重下载（百度网盘）：** <https://pan.baidu.com/s/14BRCGqMP-1-cDcouDP7lrg?pwd=2026>  提取码：`2026`
（含 6 个模型权重，每个约 89 MB，下载后放回对应 `outputs/<run>/model_final.pt`）

**SwanLab 训练记录：** <https://swanlab.cn/@xhwu/HW3-Task2-ACT/overview>

---

## 目录结构

```
DL_final_term/
├── calvin_act/              # 核心代码包（python -m calvin_act <command>）
│   ├── cli.py               # 命令行入口（task1/task2/task3/train/evaluate）
│   ├── calvin_data.py       # LeRobot v2.x 数据读取
│   ├── dataset.py           # 动作分块窗口 PyTorch Dataset
│   ├── act_model.py         # ACT 策略（ResNet-18 + CVAE + Transformer）
│   ├── normalize.py         # 动作/状态归一化
│   ├── train.py             # 训练循环（SwanLab 日志）
│   └── evaluate.py          # 开环误差评测
├── configs/
│   └── act_calvin.json      # 主配置（ResNet-18, H=10，两模型共用）
├── outputs/                 # 训练与评测结果（*.pt 权重不入库）
│   ├── task1_envA_40k/      # config.json / history.json / train_summary.json
│   ├── task2_envABC_40k/
│   ├── task2_envABC_H5_40k/
│   ├── task2_envABC_H20_40k/
│   └── task3_zeroshot_D/    # eval_*_on_D.json / per_chunk_step_D.png
├── analyze/                 # 深入分析脚本与图表
│   ├── common.py
│   ├── run_evals.py         # 批量重评测（生成 results/arrays/）
│   ├── run_all.py           # 一键复现全套分析图
│   ├── fig_training.py / fig_generalization.py / fig_chunking.py / ...
│   ├── fig_policy_cases.py  # 策略预测轨迹 case study
│   ├── figures/             # 生成的分析图（PNG）
│   └── results/             # 分析指标 JSON（eval_summary / error_breakdown / ...）
├── requirements.txt
└── README.md
```

> 模型权重 `*.pt`、数据集、SwanLab 日志不入库，见网盘与 `.gitignore`。

---

## 环境配置

```bash
conda create -n calvin_act python=3.9 -y && conda activate calvin_act
pip install torch torchvision        # 选择与 GPU 匹配的 CUDA 版本
pip install -e ".[plot,logging]"     # 或：pip install -r requirements.txt
# 核心依赖：torch, torchvision, numpy, Pillow, pandas, pyarrow, swanlab
```

Conda 用户：`conda env create -f environment.yml`

---

## 数据准备

数据为 LeRobot v2.x 格式，默认根目录 `data/calvin-lerobot/`（可用 `--data-root` 或环境变量 `CALVIN_DATA_ROOT` 覆盖）：

```
data/calvin-lerobot/
  splitA/  splitB/  splitC/  splitD/
    meta/info.json
    meta/episodes.jsonl
    data/chunk-000/episode_000000.parquet   # image(200×200) + wrist_image(84×84) + state(15) + actions(7)
```

验证数据加载：`python -m calvin_act info`

---

## 运行说明

### 一、训练（Train）

```bash
# 任务一：env A 单环境训练（ACT-A）
python -m calvin_act task1 --config configs/act_calvin.json --steps 40000 --num-workers 4

# 任务二：env A+B+C 联合训练（ACT-ABC）
python -m calvin_act task2 --config configs/act_calvin.json --steps 40000 --num-workers 4

# chunk size 消融（H=5 / H=20，均为 A+B+C，40k 步）
python -m calvin_act train --envs ABC --chunk-size 5  --out outputs/task2_envABC_H5_40k  --steps 40000
python -m calvin_act train --envs ABC --chunk-size 20 --out outputs/task2_envABC_H20_40k --steps 40000
```

训练日志（loss / action_l1 / val_action_l1 / kl）由 SwanLab 自动记录（`--logger swanlab`，需提前 `swanlab login`）。

### 二、评测（Evaluate）

```bash
# 任务三：零样本迁移至 env D
python -m calvin_act task3 \
  --ckpt-a   outputs/task1_envA_40k/model_final.pt \
  --ckpt-abc outputs/task2_envABC_40k/model_final.pt \
  --test-env D
# 输出：outputs/task3_zeroshot_D/eval_*_on_D.json、per_chunk_step_D.png
```

### 三、深入分析

```bash
cd analyze
python run_evals.py   # 先生成 results/arrays/（需权重文件）
python run_all.py     # 生成全套分析图（fig_*.png）
```

完整分析结论（含全部数字）见 [`analyze/FINDINGS.md`](analyze/FINDINGS.md)。

---

## 方法要点

**ACT 架构**：ResNet-18（双路相机，96×96）提取视觉特征 → CVAE 编码动作块为潜变量 $z$ → Transformer 编码器融合视觉/状态/$z$ → Transformer 解码器输出 $H$ 步预测动作。推断时 $z=0$（先验均值），实现单帧观测驱动的开环动作分块。

**核心发现**：

1. **视觉偏移放大失效**：开环分块将误读的单帧感知锁定到整个序列。域外误差增长斜率是域内的 **6 倍**（env B 末步—首步差 +0.074 vs 域内 +0.012），集中在视觉敏感的偏航角（yaw）和夹爪（gripper）维度。
2. **多环境联合训练**：无结构变更，仅扩大训练数据覆盖，即可将跨环境误差降低 36–40%，零样本 env D 改善 16%，代价是域内误差小幅上升 13%（泛化-特化权衡）。
3. **分块大小消融**：开环 L1 随 $H$ 单调升高（H=5/10/20 对应 0.097/0.102/0.113），短分块在开环评估上更优；闭环效益需另行验证。

---

## 实验跟踪（SwanLab）

训练时默认用 SwanLab 记录所有指标（`step / loss / action_l1 / val_action_l1 / kl / lr`），每 500 步评估验证集一次：

```bash
swanlab login          # 首次登录（填入 API key）
python -m calvin_act task2 --logger swanlab --project HW3-Task2-ACT
```

本项目的 6 次训练运行（ACT-A/ABC × 20k/40k + H=5/20 消融）均已上传至：
<https://swanlab.cn/@xhwu/HW3-Task2-ACT/overview>

---

## 第三方代码

- [tonyzhaozh/act](https://github.com/tonyzhaozh/act)（ACT 原始实现，ResNet-18 + CVAE + Transformer 架构）
- [huggingface/lerobot](https://github.com/huggingface/lerobot)（CALVIN LeRobot v2.x 数据格式）
- [mees/calvin](https://github.com/mees/calvin)（CALVIN 基准，桌面机械臂操作环境 A/B/C/D）

`calvin_act/` 下的训练/评测/分析代码为本作业自研实现。
