# -*- coding: utf-8 -*-
"""Bridge to the **official LeRobot ACT** trainer.

The data for this task is *already* in LeRobot v2.x format (one ``split{E}/``
dataset per environment), so **no conversion is needed** — you can point LeRobot's
own ACT training code straight at these folders. This module just validates a
selection of environments and prints ready-to-run guidance; the self-contained
trainer (``python -m calvin_act train ...``) remains the guaranteed-runnable path
and produces the same metrics.
"""

from __future__ import annotations

from pathlib import Path

from .calvin_data import env_dir, find_data_root, load_info, load_episodes_meta


def describe_lerobot(data_root: str | Path | None, envs,
                     image_keys=("image", "wrist_image")) -> dict:
    """Validate the per-environment LeRobot datasets and print usage guidance."""
    root = find_data_root(data_root)
    envs = [e.strip().upper() for e in envs]
    info_per_env = {}
    for e in envs:
        d = env_dir(root, e)
        info = load_info(d)
        eps = load_episodes_meta(d)
        info_per_env[e] = {
            "dir": str(d),
            "episodes": len(eps),
            "frames": int(info.get("total_frames", sum(m.get("length", 0) for m in eps))),
            "fps": info.get("fps"),
            "features": list(info.get("features", {})),
        }
        print(f"[lerobot] env {e}: {info_per_env[e]['episodes']} episodes / "
              f"{info_per_env[e]['frames']} frames  ({d})")

    print(
        "\nThe data is already a LeRobotDataset, so train with LeRobot's ACT directly,\n"
        "e.g. (flags vary by LeRobot version — verify against your install):\n"
        "  pip install lerobot\n"
        f"  lerobot-train --policy.type=act --dataset.root={info_per_env[envs[0]]['dir']} \\\n"
        "                --output_dir=outputs/lerobot/train\n"
        "For multi-environment (A+B+C) training, concatenate the per-env datasets with\n"
        "LeRobot's MultiLeRobotDataset, or use this repo's self-contained trainer:\n"
        "  python -m calvin_act task2\n")
    return info_per_env
