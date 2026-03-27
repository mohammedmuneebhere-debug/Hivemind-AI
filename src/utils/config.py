import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class EnvironmentConfig:
    mode: str = "mock"
    n_agents: int = 2
    local_state_dim: int = 4
    global_state_dim: int = 0  # derived if 0
    action_dim: int = 2
    target_radius: float = 0.6
    target_radius_min: float = 0.25
    target_radius_max: float = 0.9
    collision_distance: float = 0.12
    max_steps: int = 200
    dt: float = 0.1
    accel_scale: float = 2.0


@dataclass(frozen=True)
class RewardConfig:
    step_penalty: float = 0.001
    collision_penalty: float = 4.0
    task_completed_bonus: float = 20.0
    progress_scale: float = 1.0
    per_agent_target_bonus: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    algo: str = "ppo"
    total_timesteps: int = 30000
    seed: int = 42
    save_path: str = "models/ppo_mock.zip"
    tensorboard_log: str = "logs/tb"
    policy: str = "ctde"
    wandb: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 0.0003
    n_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_range: float = 0.2


@dataclass(frozen=True)
class SACConfig:
    learning_rate: float = 0.0003
    buffer_size: int = 1000000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1


@dataclass(frozen=True)
class HiveMindConfig:
    environment: EnvironmentConfig
    reward: RewardConfig
    training: TrainingConfig
    ppo: PPOConfig
    sac: SACConfig


def load_config(config_path: str) -> HiveMindConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    raw_env: Dict[str, Any] = raw.get("environment", {})
    n_agents = int(raw_env.get("n_agents", EnvironmentConfig.n_agents))
    local_state_dim = int(raw_env.get("local_state_dim", EnvironmentConfig.local_state_dim))

    if raw_env.get("global_state_dim", None) in (None, 0, "0"):
        raw_env["global_state_dim"] = n_agents * local_state_dim

    env_cfg = EnvironmentConfig(**raw_env)
    reward_cfg = RewardConfig(**raw.get("reward", {}))
    training_cfg = TrainingConfig(**raw.get("training", {}))
    ppo_cfg = PPOConfig(**raw.get("ppo", {}))
    sac_cfg = SACConfig(**raw.get("sac", {}))

    return HiveMindConfig(
        environment=env_cfg,
        reward=reward_cfg,
        training=training_cfg,
        ppo=ppo_cfg,
        sac=sac_cfg,
    )

