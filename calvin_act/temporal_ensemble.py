# -*- coding: utf-8 -*-
"""Action-chunking temporal ensembling for closed-loop inference.

ACT predicts overlapping action chunks every step; temporal ensembling averages
all predictions that target the *current* timestep with exponential weights
``exp(-coeff * age)``. This smooths the open-loop chunks into a stable closed-loop
command and is the mechanism whose robustness under visual shift the report
analyses. Faithful port of LeRobot's ``ACTTemporalEnsembler``.
"""

from __future__ import annotations

import torch


class TemporalEnsembler:
    def __init__(self, chunk_size: int, coeff: float = 0.01):
        self.chunk_size = int(chunk_size)
        self.coeff = float(coeff)
        w = torch.exp(-self.coeff * torch.arange(self.chunk_size).float())
        self.ensemble_weights = w
        self.ensemble_weights_cumsum = torch.cumsum(w, dim=0)
        self.reset()

    def reset(self) -> None:
        self.ensembled_actions = None                 # [B, T, A]
        self.ensembled_actions_count = None           # [T, 1] long

    @torch.no_grad()
    def update(self, actions: torch.Tensor) -> torch.Tensor:
        """Feed the newest predicted chunk ``[B, chunk_size, A]``; return the
        ensembled action ``[B, A]`` to execute this timestep."""
        device = actions.device
        w = self.ensemble_weights.to(device)
        wcs = self.ensemble_weights_cumsum.to(device)

        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=device)
        else:
            # combine the overlap (all but the newest predicted step) with the
            # running ensemble, then append the one genuinely new step.
            self.ensembled_actions *= wcs[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * w[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(
                self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions /= wcs[self.ensembled_actions_count - 1]
            self.ensembled_actions = torch.cat(
                [self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count,
                 torch.ones_like(self.ensembled_actions_count[-1:])], dim=0)

        action = self.ensembled_actions[:, 0]
        self.ensembled_actions = self.ensembled_actions[:, 1:]
        self.ensembled_actions_count = self.ensembled_actions_count[1:]
        return action
