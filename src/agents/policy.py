from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PolicyIO:
    local_state_dim: int
    global_state_dim: int
    action_dim: int


class SharedActor(nn.Module):
    """
    Parameter-shared actor for per-agent local observations.
    """

    def __init__(self, io: PolicyIO, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.io = io

        layers = []
        in_dim = io.local_state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, io.action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, local_obs: torch.Tensor) -> torch.Tensor:
        # local_obs: (batch, n_agents, local_state_dim) -> mean actions (batch, n_agents, action_dim)
        b, n, d = local_obs.shape
        x = local_obs.reshape(b * n, d)
        a = self.mlp(x)
        a = torch.tanh(a)
        return a.reshape(b, n, self.io.action_dim)


class CentralizedCritic(nn.Module):
    def __init__(self, io: PolicyIO, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.io = io

        layers = []
        in_dim = io.global_state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        # global_obs: (batch, global_state_dim) -> (batch, 1)
        return self.mlp(global_obs)

