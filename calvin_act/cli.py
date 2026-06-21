# -*- coding: utf-8 -*-
"""``python -m calvin_act <command>`` - the Task 2 pipeline driver.

The three assignment sub-tasks are each runnable on their own:

  task1    train ACT on environment A only            (-> outputs/task1_envA)
  task2    train ACT on environments A+B+C (joint)    (-> outputs/task2_envABC)
  task3    zero-shot eval BOTH models on unseen env D (-> outputs/task3_zeroshot_D)

Lower-level / utility commands:

  train    train ACT on any chosen envs (advanced; e.g. chunk-size sweeps)
  eval     zero-shot open-loop action error on an env
  info     list the environments present under the data root and their sizes
  compare  build the model-comparison table   |  plot   plot Action-L1 curves
  lerobot  describe how to train with the official LeRobot ACT on this data

All paths default to **repo-relative** locations (data at ``task2/data/calvin-lerobot``,
outputs under ``outputs/``) so the same commands run unchanged on any machine/GPU.
Override the data location with ``--data-root`` or the ``CALVIN_DATA_ROOT`` env var.

``python -m calvin_act <command> -h`` for per-command options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo-relative default locations -------------------------------------------
TASK1_OUT = "outputs/task1_envA_40k"
TASK2_OUT = "outputs/task2_envABC_40k"
TASK3_OUT = "outputs/task3_zeroshot_D"


def default_config_path(name: str = "act_calvin.json") -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / name


def parse_envs(s: str) -> list[str]:
    """'A' / 'ABC' / 'A,B,C' -> ['A', ...]."""
    s = s.strip().upper()
    parts = [p for p in s.split(",") if p] if "," in s else list(s)
    envs = [p.strip() for p in parts if p.strip()]
    if not envs:
        raise ValueError(f"could not parse envs from {s!r}")
    return envs


def _load_config(args, default_to_main: bool = False) -> "ACTConfig":
    from .act_config import ACTConfig
    cfg_path = getattr(args, "config", None)
    if cfg_path is None and default_to_main:
        cfg_path = str(default_config_path())
    cfg = ACTConfig.load(cfg_path) if cfg_path else ACTConfig()
    # common overrides
    if getattr(args, "chunk_size", None) is not None:
        cfg.chunk_size = args.chunk_size
    if getattr(args, "steps", None) is not None:
        cfg.steps = args.steps
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size = args.batch_size
    if getattr(args, "backbone", None):
        cfg.vision_backbone = args.backbone
    if getattr(args, "no_pretrained", False):
        cfg.pretrained_backbone = False
    if getattr(args, "image_keys", None):
        cfg.image_keys = tuple(args.image_keys.split(","))
    if getattr(args, "image_size", None) is not None:
        cfg.image_size = args.image_size
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "val_fraction", None) is not None:
        cfg.val_fraction = args.val_fraction
    return cfg


def _data_root(args):
    return getattr(args, "data_root", None)


def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# the three assignment tasks (each independently runnable)
# ---------------------------------------------------------------------------
def cmd_task1(args):
    """Task 1 - base policy: train ACT on environment A only."""
    from .train import train
    cfg = _load_config(args, default_to_main=True)
    out = args.out or TASK1_OUT
    print(f"=== Task 1: train ACT on env A -> {out} ===")
    train(cfg, _data_root(args), ["A"], out, device=args.device,
          logger_backend=args.logger, run_name=args.run_name or "task1_envA",
          num_workers=args.num_workers, max_episodes=args.max_episodes,
          max_train_steps_data=args.max_data)


def cmd_task2(args):
    """Task 2 - joint policy: train ACT on envs A+B+C (identical config)."""
    from .train import train
    cfg = _load_config(args, default_to_main=True)
    out = args.out or TASK2_OUT
    print(f"=== Task 2: train ACT on envs A+B+C -> {out} ===")
    train(cfg, _data_root(args), ["A", "B", "C"], out, device=args.device,
          logger_backend=args.logger, run_name=args.run_name or "task2_envABC",
          num_workers=args.num_workers, max_episodes=args.max_episodes,
          max_train_steps_data=args.max_data)
    # convenience: if Task 1 is also done, plot the convergence comparison
    h1, h2 = Path(TASK1_OUT) / "history.json", Path(out) / "history.json"
    if h1.exists() and h2.exists():
        try:
            _plot_curves([f"envA={h1}", f"envABC={h2}"],
                         "outputs/convergence_A_vs_ABC.png")
        except Exception as e:
            print(f"[task2] (skip convergence plot: {e})")


def cmd_task3(args):
    """Task 3 - zero-shot: eval BOTH models on the unseen env D and compare."""
    from .evaluate import evaluate_offline
    from .metrics import write_comparison_table

    ckpt_a = Path(args.ckpt_a or (Path(TASK1_OUT) / "model_final.pt"))
    ckpt_abc = Path(args.ckpt_abc or (Path(TASK2_OUT) / "model_final.pt"))
    missing = [str(p) for p in (ckpt_a, ckpt_abc) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "missing checkpoint(s): " + ", ".join(missing) +
            ". Run `python -m calvin_act task1` and `task2` first, or pass "
            "--ckpt-a / --ckpt-abc.")

    out = Path(args.out or TASK3_OUT)
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== Task 3: zero-shot eval on env {args.test_env} -> {out} ===")
    test_envs = parse_envs(args.test_env)

    ev_a = evaluate_offline(ckpt_a, _data_root(args), test_envs, device=args.device,
                            max_steps=args.max_steps, batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            out=out / f"eval_A_on_{args.test_env}.json")
    ev_abc = evaluate_offline(ckpt_abc, _data_root(args), test_envs, device=args.device,
                              max_steps=args.max_steps, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              out=out / f"eval_ABC_on_{args.test_env}.json")

    rows = []
    for tag, ckpt, ev, envs in (("ACT-A", ckpt_a, ev_a, "A"),
                                ("ACT-ABC", ckpt_abc, ev_abc, "A+B+C")):
        summ = ckpt.parent / "train_summary.json"
        cfg_path = ckpt.parent / "config.json"
        s = _load_json(summ) if summ.exists() else {}
        action_dim = (_load_json(cfg_path).get("action_dim", 1)
                      if cfg_path.exists() else 1)
        # Divide historical train/val L1 by action_dim so the unit matches
        # evaluate_offline's per-dim mean L1 (action_l1_raw).
        def _scale(v):
            return round(v / action_dim, 6) if v is not None else None
        rows.append({
            "model": tag, "train_envs": envs,
            "final_train_L1_per_dim": _scale(s.get("final_action_l1")),
            "final_val_L1_per_dim": _scale(s.get("final_val_action_l1")),
            "D_action_L1_raw": ev["action_l1_raw"],
            "D_first_step_L1": ev["first_step_l1_raw"],
            "D_action_MSE_raw": ev["action_mse_raw"],
        })
    write_comparison_table(rows, out / "comparison.md", csv_path=out / "comparison.csv")
    print(f"[task3] wrote {out/'comparison.md'} and {out/'comparison.csv'}")

    # per-chunk-step error comparison plot (action-chunking robustness analysis)
    try:
        _plot_per_chunk_step(
            {"ACT-A": ev_a["per_chunk_step_l1_raw"],
             "ACT-ABC": ev_abc["per_chunk_step_l1_raw"]},
            out / f"per_chunk_step_{args.test_env}.png", env=args.test_env)
    except Exception as e:
        print(f"[task3] (skip per-chunk-step plot: {e})")

    print("\n[task3] zero-shot comparison:")
    for r in rows:
        print(f"  {r['model']:8s} train={r['train_envs']:6s}  "
              f"D L1={r['D_action_L1_raw']}  first-step={r['D_first_step_L1']}")


# ---------------------------------------------------------------------------
# data utilities
# ---------------------------------------------------------------------------
def cmd_info(args):
    from .calvin_data import find_data_root, available_envs, env_dir, load_info, load_episodes_meta
    root = find_data_root(_data_root(args))
    envs = parse_envs(args.envs) if args.envs else available_envs(root)
    print(f"data root: {root}")
    if not envs:
        print("  (no environments found - expected split{A,B,C,D}/ with meta/info.json)")
        return {}
    info = {}
    for e in envs:
        try:
            d = env_dir(root, e)
            meta = load_info(d)
            eps = load_episodes_meta(d)
            frames = int(meta.get("total_frames", sum(m.get("length", 0) for m in eps)))
            info[e] = {"dir": d.name, "episodes": len(eps), "frames": frames,
                       "fps": meta.get("fps"), "features": list(meta.get("features", {}))}
            print(f"  env {e}: {len(eps)} episodes, {frames} frames, fps={meta.get('fps')} "
                  f"({d.name})  features={info[e]['features']}")
        except FileNotFoundError as ex:
            print(f"  env {e}: (not present) [{ex}]")
    return info


# ---------------------------------------------------------------------------
# lower-level train / eval
# ---------------------------------------------------------------------------
def cmd_train(args):
    from .train import train
    cfg = _load_config(args)
    train(cfg, _data_root(args), parse_envs(args.envs), args.out,
          device=args.device, logger_backend=args.logger, run_name=args.run_name,
          num_workers=args.num_workers, max_steps=args.steps,
          max_episodes=args.max_episodes, max_train_steps_data=args.max_data)


def cmd_eval(args):
    from .evaluate import evaluate_offline
    evaluate_offline(args.ckpt, _data_root(args), parse_envs(args.envs),
                     device=args.device, max_steps=args.max_steps,
                     batch_size=args.batch_size, num_workers=args.num_workers,
                     max_episodes=args.max_episodes, out=args.out)


# ---------------------------------------------------------------------------
# report: compare table + plots
# ---------------------------------------------------------------------------
def _flatten_row(d: dict) -> dict:
    row = {}
    for k, v in d.items():
        row[k] = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
    return row


def cmd_compare(args):
    from .metrics import write_comparison_table
    rows = []
    for p in args.inputs:
        data = _load_json(p)
        items = data if isinstance(data, list) else [data]
        for it in items:
            row = {"source": Path(p).stem}
            row.update(_flatten_row(it))
            rows.append(row)
    write_comparison_table(rows, args.out, csv_path=args.csv)
    print(f"[compare] wrote {args.out}" + (f" and {args.csv}" if args.csv else ""))


def _plot_curves(histories, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .metrics import moving_average

    plt.figure(figsize=(7, 4.5))
    for spec in histories:
        label, _, path = spec.partition("=")
        if not path:
            path, label = label, Path(label).parent.name
        hist = _load_json(path)
        steps = [h["step"] for h in hist]
        l1 = [h["action_l1"] for h in hist]
        line, = plt.plot(steps, l1, alpha=0.3)
        sm = moving_average(l1, window=max(2, len(l1) // 20))
        plt.plot(steps[-len(sm):], sm, label=f"{label} (train)", linewidth=2,
                 color=line.get_color())
        vsteps = [h["step"] for h in hist if "val_action_l1" in h]
        vl1 = [h["val_action_l1"] for h in hist if "val_action_l1" in h]
        if vsteps:
            plt.plot(vsteps, vl1, "--", label=f"{label} (val)", linewidth=2,
                     color=line.get_color())
    plt.xlabel("training step"); plt.ylabel("Action L1 loss (normalised)")
    plt.title("ACT training convergence"); plt.legend(); plt.grid(alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print(f"[plot] wrote {out}")


def _plot_per_chunk_step(curves: dict, out, env: str = "D"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.5))
    for label, ys in curves.items():
        plt.plot(range(len(ys)), ys, marker="o", label=label, linewidth=2)
    plt.xlabel("position within action chunk (0 = most immediate)")
    plt.ylabel("Action L1 (raw)")
    plt.title(f"Per-chunk-step error on env {env} (action-chunking robustness)")
    plt.legend(); plt.grid(alpha=0.3)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print(f"[plot] wrote {out}")


def cmd_plot(args):
    try:
        _plot_curves(args.histories, args.out)
    except ImportError:
        raise RuntimeError("plotting needs matplotlib: pip install matplotlib")


# ---------------------------------------------------------------------------
# lerobot bridge
# ---------------------------------------------------------------------------
def cmd_lerobot(args):
    from .lerobot_export import describe_lerobot
    describe_lerobot(_data_root(args), parse_envs(args.envs))


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def _add_config_overrides(s):
    s.add_argument("--config", default=None, help="ACTConfig JSON (see configs/)")
    s.add_argument("--chunk-size", type=int, default=None, help="action-chunk horizon H")
    s.add_argument("--steps", type=int, default=None)
    s.add_argument("--batch-size", type=int, default=None)
    s.add_argument("--backbone", default=None, choices=["resnet18", "small"])
    s.add_argument("--no-pretrained", action="store_true")
    s.add_argument("--image-keys", default=None, help="comma list, e.g. image,wrist_image")
    s.add_argument("--image-size", type=int, default=None)
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--val-fraction", type=float, default=None,
                   help="fraction of train episodes held out for the val curve")


def _add_common_train(s, out_required=False):
    s.add_argument("--data-root", default=None,
                   help="root with split{A,B,C,D}/ (default: task2/data/calvin-lerobot "
                        "or $CALVIN_DATA_ROOT)")
    s.add_argument("--out", default=None, required=out_required, help="output run dir")
    s.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    s.add_argument("--logger", default="none", choices=["none", "wandb", "swanlab"])
    s.add_argument("--run-name", default=None)
    s.add_argument("--num-workers", type=int, default=0)
    s.add_argument("--max-episodes", type=int, default=None,
                   help="cap #episodes per env (debug / quick runs)")
    s.add_argument("--max-data", type=int, default=None,
                   help="cap #timesteps loaded for training (debug)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="calvin_act",
        description="HW3 Task 2: ACT cross-environment generalization on CALVIN.")
    sub = p.add_subparsers(dest="command", required=True)

    # -- task1 -------------------------------------------------------------
    s = sub.add_parser("task1", help="train ACT on env A only")
    _add_common_train(s); _add_config_overrides(s)
    s.set_defaults(func=cmd_task1)

    # -- task2 -------------------------------------------------------------
    s = sub.add_parser("task2", help="train ACT on envs A+B+C (joint)")
    _add_common_train(s); _add_config_overrides(s)
    s.set_defaults(func=cmd_task2)

    # -- task3 -------------------------------------------------------------
    s = sub.add_parser("task3", help="zero-shot eval both models on unseen env D")
    s.add_argument("--data-root", default=None)
    s.add_argument("--ckpt-a", default=None,
                   help=f"env-A model (default: {TASK1_OUT}/model_final.pt)")
    s.add_argument("--ckpt-abc", default=None,
                   help=f"env-ABC model (default: {TASK2_OUT}/model_final.pt)")
    s.add_argument("--test-env", default="D", help="held-out env to test on")
    s.add_argument("--out", default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--max-steps", type=int, default=5000)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--num-workers", type=int, default=0)
    s.set_defaults(func=cmd_task3)

    # -- info --------------------------------------------------------------
    s = sub.add_parser("info", help="list envs present under the data root")
    s.add_argument("--data-root", default=None)
    s.add_argument("--envs", default=None, help="restrict to these envs (default: all found)")
    s.set_defaults(func=cmd_info)

    # -- train (advanced) --------------------------------------------------
    s = sub.add_parser("train", help="train ACT on chosen envs (advanced)")
    s.add_argument("--envs", required=True, help="e.g. A   or   ABC")
    _add_common_train(s, out_required=True)
    _add_config_overrides(s)
    s.set_defaults(func=cmd_train)

    # -- eval --------------------------------------------------------------
    s = sub.add_parser("eval", help="zero-shot open-loop action error on an env")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--envs", required=True, help="env to test on, e.g. D")
    s.add_argument("--data-root", default=None)
    s.add_argument("--device", default=None)
    s.add_argument("--max-steps", type=int, default=5000)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--num-workers", type=int, default=0)
    s.add_argument("--max-episodes", type=int, default=None)
    s.add_argument("-o", "--out", default=None)
    s.set_defaults(func=cmd_eval)

    # -- compare / plot ----------------------------------------------------
    s = sub.add_parser("compare", help="build the model-comparison table")
    s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--csv", default=None)
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("plot", help="plot Action-L1 training curves")
    s.add_argument("--histories", nargs="+", required=True,
                   help="label=path/history.json (or just path)")
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(func=cmd_plot)

    # -- lerobot -----------------------------------------------------------
    s = sub.add_parser("lerobot", help="describe the official-LeRobot-ACT training path")
    s.add_argument("--data-root", default=None)
    s.add_argument("--envs", default="A")
    s.set_defaults(func=cmd_lerobot)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, FileNotFoundError, ValueError, OSError) as e:
        print(f"calvin_act: error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
