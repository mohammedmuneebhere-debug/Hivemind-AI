from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs


@dataclass(frozen=True)
class CTDEPolicyDims:
    local_state_dim: int
    global_state_dim: int
    n_agents: int
    action_dim: int


class SharedActorPerAgent(nn.Module):
    def __init__(self, local_state_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        in_dim = local_state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, local_obs: th.Tensor) -> th.Tensor:
        # local_obs: (batch, n_agents, local_state_dim)
        b, n, d = local_obs.shape
        x = local_obs.reshape(b * n, d)
        a = self.mlp(x)
        a = th.tanh(a)
        return a.reshape(b, n * a.shape[-1])


class CentralizedCritic(nn.Module):
    def __init__(self, global_state_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        in_dim = global_state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, global_obs_flat: th.Tensor) -> th.Tensor:
        return self.mlp(global_obs_flat)


class CTDEPPOPolicy(ActorCriticPolicy):
    """
    PPO policy with CTDE structure:
      - shared actor MLP applied per-agent from local observations
      - centralized critic from joint/global observation slice

    Observation layout from `HiveMindEnv`:
      obs = concat(local_obs_agent0 (local_state_dim), global_obs_all_agents_flat (global_state_dim))
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *,
        local_state_dim: int,
        global_state_dim: int,
        n_agents: int,
        action_dim: int,
        actor_hidden_sizes: Tuple[int, ...] = (128, 128),
        critic_hidden_sizes: Tuple[int, ...] = (256, 256),
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=None,
            **kwargs,
        )

        self.dims = CTDEPolicyDims(
            local_state_dim=local_state_dim,
            global_state_dim=global_state_dim,
            n_agents=n_agents,
            action_dim=action_dim,
        )

        expected_global = n_agents * local_state_dim
        if expected_global != global_state_dim:
            raise ValueError(f"global_state_dim mismatch: expected {expected_global}, got {global_state_dim}")

        self.shared_actor = SharedActorPerAgent(
            local_state_dim=local_state_dim,
            action_dim=action_dim,
            hidden_sizes=actor_hidden_sizes,
        )
        self.centralized_critic = CentralizedCritic(
            global_state_dim=global_state_dim,
            hidden_sizes=critic_hidden_sizes,
        )

        # Ensure log_std exists and matches action dimension.
        # ActorCriticPolicy calls `_build()` during super init, so log_std is already defined.
        # We re-create it here to make sure the shape matches our action space exactly.
        self.log_std = nn.Parameter(
            th.ones(self.action_space.shape[0]) * self.log_std_init,
            requires_grad=True,
        )
        # Recreate optimizer so it includes our custom networks.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _split_obs(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        # obs: (batch, obs_dim)
        local0 = obs[:, : self.dims.local_state_dim]
        global_flat = obs[:, self.dims.local_state_dim :]
        return local0, global_flat

    def _actor_mean_actions(self, obs: th.Tensor) -> th.Tensor:
        _, global_flat = self._split_obs(obs)
        local_all = global_flat.reshape(-1, self.dims.n_agents, self.dims.local_state_dim)
        return self.shared_actor(local_all)  # (batch, n_agents*action_dim)

    def _critic_values(self, obs: th.Tensor) -> th.Tensor:
        _, global_flat = self._split_obs(obs)
        return self.centralized_critic(global_flat)  # (batch, 1)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        actions_mean = self._actor_mean_actions(obs)
        distribution = self.action_dist.proba_distribution(actions_mean, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self._critic_values(obs)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        if not isinstance(obs, th.Tensor):
            raise TypeError("obs must be a torch.Tensor for CTDEPPOPolicy")

        actions_mean = self._actor_mean_actions(obs)
        distribution = self.action_dist.proba_distribution(actions_mean, self.log_std)
        log_prob = distribution.log_prob(actions)
        values = self._critic_values(obs).squeeze(-1)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs):
        if not isinstance(obs, th.Tensor):
            raise TypeError("obs must be a torch.Tensor for CTDEPPOPolicy")
        actions_mean = self._actor_mean_actions(obs)
        return self.action_dist.proba_distribution(actions_mean, self.log_std)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        if not isinstance(obs, th.Tensor):
            raise TypeError("obs must be a torch.Tensor for CTDEPPOPolicy")
        return self._critic_values(obs).squeeze(-1)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        if not isinstance(observation, th.Tensor):
            raise TypeError("observation must be a torch.Tensor for CTDEPPOPolicy")
        actions_mean = self._actor_mean_actions(observation)
        distribution = self.action_dist.proba_distribution(actions_mean, self.log_std)
        return distribution.get_actions(deterministic=deterministic)

    def _build(self, lr_schedule) -> None:
        """
        Override SB3 network-building.

        We do NOT use ActorCriticPolicy's internal MlpExtractor/action_net/value_net.
        This ensures the saved checkpoint matches this reconstructed policy structure.
        """
        # Only define log_std (for continuous action distributions) and optimizer.
        self.log_std = nn.Parameter(
            th.ones(self.action_space.shape[0]) * self.log_std_init,
            requires_grad=True,
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

