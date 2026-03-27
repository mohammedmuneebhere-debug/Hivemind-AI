from __future__ import annotations

from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.gazebo_env import HiveMindEnv
from src.rl.ctde_ppo_policy import CTDEPPOPolicy
from src.utils.config import HiveMindConfig


def train_ppo(cfg: HiveMindConfig, total_timesteps: Optional[int] = None) -> PPO:
    def make_env():
        env = HiveMindEnv(cfg.environment, cfg.reward, seed=cfg.training.seed)
        return Monitor(env)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy=CTDEPPOPolicy,
        env=vec_env,
        verbose=1,
        tensorboard_log=cfg.training.tensorboard_log,
        learning_rate=cfg.ppo.learning_rate,
        n_steps=cfg.ppo.n_steps,
        batch_size=cfg.ppo.batch_size,
        gamma=cfg.ppo.gamma,
        gae_lambda=cfg.ppo.gae_lambda,
        ent_coef=cfg.ppo.ent_coef,
        vf_coef=cfg.ppo.vf_coef,
        max_grad_norm=cfg.ppo.max_grad_norm,
        clip_range=cfg.ppo.clip_range,
        seed=cfg.training.seed,
        policy_kwargs={
            "local_state_dim": cfg.environment.local_state_dim,
            "global_state_dim": cfg.environment.global_state_dim,
            "n_agents": cfg.environment.n_agents,
            "action_dim": cfg.environment.action_dim,
        },
    )

    model.learn(total_timesteps=total_timesteps or cfg.training.total_timesteps)
    Path(cfg.training.save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(cfg.training.save_path)
    return model

