# -*- coding: utf-8 -*-
"""Feature normalisation for states and actions.

The normaliser is **fit on the training split only** and saved alongside the
checkpoint, so the env-A and env-ABC models each carry their own stats and the
zero-shot env-D evaluation applies the *training* stats (no test-time leakage).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Normalizer:
    mode: str                 # "mean_std" | "min_max" | "none"
    loc: np.ndarray           # subtract
    scale: np.ndarray         # divide

    @classmethod
    def fit(cls, x: np.ndarray, mode: str = "mean_std", eps: float = 1e-6
            ) -> "Normalizer":
        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
        if mode == "mean_std":
            loc = x.mean(0)
            scale = x.std(0)
        elif mode == "min_max":                      # -> roughly [-1, 1]
            lo, hi = x.min(0), x.max(0)
            loc = 0.5 * (hi + lo)
            scale = 0.5 * (hi - lo)
        elif mode == "none":
            loc = np.zeros(x.shape[1])
            scale = np.ones(x.shape[1])
        else:
            raise ValueError(f"unknown normalize mode: {mode}")
        scale = np.where(np.abs(scale) < eps, 1.0, scale)
        return cls(mode=mode, loc=loc.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.loc) / self.scale

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.scale + self.loc

    # -- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {"mode": self.mode, "loc": self.loc.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(mode=d["mode"],
                   loc=np.asarray(d["loc"], np.float32),
                   scale=np.asarray(d["scale"], np.float32))


def fit_normalizers(env_data, state_mode: str = "mean_std",
                    action_mode: str = "min_max") -> dict[str, Normalizer]:
    """Fit state + action normalisers from an :class:`EnvData`."""
    return {
        "state": Normalizer.fit(env_data.states, mode=state_mode),
        "action": Normalizer.fit(env_data.actions, mode=action_mode),
    }


def normalizers_to_dict(norms: dict[str, Normalizer]) -> dict:
    return {k: v.to_dict() for k, v in norms.items()}


def normalizers_from_dict(d: dict) -> dict[str, Normalizer]:
    return {k: Normalizer.from_dict(v) for k, v in d.items()}
