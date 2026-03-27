from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.agents.policy import CentralizedCritic, PolicyIO, SharedActor


@dataclass
class AgentSpec:
    agent_id: int


class HiveMindAgent:
    """
    Lightweight agent wrapper. Used for completeness of the architecture.
    """

    def __init__(
        self,
        spec: AgentSpec,
        local_state_dim: int,
        global_state_dim: int,
        action_dim: int,
        shared_actor: Optional[SharedActor] = None,
        centralized_critic: Optional[CentralizedCritic] = None,
        device: str = "cpu",
    ):
        self.spec = spec
        self.io = PolicyIO(local_state_dim=local_state_dim, global_state_dim=global_state_dim, action_dim=action_dim)
        self.device = device
        self.shared_actor = shared_actor or SharedActor(self.io).to(self.device)
        self.centralized_critic = centralized_critic or CentralizedCritic(self.io).to(self.device)

    @torch.no_grad()
    def act(self, local_obs: np.ndarray) -> np.ndarray:
        # local_obs: (local_state_dim,)
        x = torch.tensor(local_obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        a = self.shared_actor(x)  # (1,1,action_dim)
        return a.view(-1).cpu().numpy()

