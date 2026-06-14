# -*- coding: utf-8 -*-
"""Optional experiment logging to WandB or SwanLab (assignment §4.1 asks for
loss/metric curves). Degrades to console-only if the backend isn't installed."""

from __future__ import annotations


class RunLogger:
    def __init__(self, backend: str = "none", project: str = "calvin-act",
                 name: str | None = None, config: dict | None = None):
        self.backend = backend
        self.run = None
        if backend in ("none", None):
            return
        try:
            if backend == "wandb":
                import wandb
                self.run = wandb.init(project=project, name=name, config=config or {})
            elif backend == "swanlab":
                import swanlab
                swanlab.init(project=project, experiment_name=name, config=config or {})
                self.run = swanlab
            else:
                raise ValueError(f"unknown logger backend: {backend}")
        except Exception as e:                          # missing dep / offline
            print(f"[logger] '{backend}' unavailable ({e}); logging to console only.")
            self.backend = "none"

    def log(self, metrics: dict, step: int) -> None:
        if self.backend == "wandb" and self.run is not None:
            self.run.log(metrics, step=step)
        elif self.backend == "swanlab" and self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        try:
            if self.backend == "wandb" and self.run is not None:
                self.run.finish()
            elif self.backend == "swanlab" and self.run is not None:
                self.run.finish()
        except Exception:
            pass
