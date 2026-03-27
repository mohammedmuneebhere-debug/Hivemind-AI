from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from stable_baselines3 import PPO

from src.env.gazebo_env import HiveMindEnv
from src.utils.config import load_config


def _pairwise_mean_distance(positions: np.ndarray) -> float:
    # positions: (n_agents, 2)
    n = positions.shape[0]
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(positions[i] - positions[j])))
    return float(np.mean(dists)) if dists else 0.0


def _coordination_score(agent_distances: np.ndarray) -> float:
    # coord = 1 / (1 + variance(agent_distances))
    var = float(np.var(agent_distances))
    return float(1.0 / (1.0 + var))


def _synchronization_score(agent_distances_end: np.ndarray) -> float:
    # sync = 1 / (1 + std(agent_distances_end))
    std = float(np.std(agent_distances_end))
    return float(1.0 / (1.0 + std))


def _reconstruct_positions_from_obs(
    obs: np.ndarray,
    *,
    local_state_dim: int,
    global_state_dim: int,
    n_agents: int,
) -> np.ndarray:
    # obs = concat(local_obs0 (local_state_dim), global_flat (n_agents*local_state_dim))
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    global_flat = obs[local_state_dim : local_state_dim + global_state_dim]
    global_flat = global_flat[: n_agents * local_state_dim]
    local_all = global_flat.reshape(n_agents, local_state_dim)
    # local entries: [x, y, vx, vy]
    positions = local_all[:, :2]
    return positions.astype(np.float32)


def run_episode(
    *,
    env: HiveMindEnv,
    model: PPO,
    n_agents: int,
    local_state_dim: int,
    global_state_dim: int,
    deterministic: bool,
) -> Tuple[float, int, bool, int, float, float, float]:
    obs, _info = env.reset()

    total_reward = 0.0
    collisions_episode = 0
    completion = False
    steps = 0

    inter_agent_distance_sum = 0.0
    inter_agent_distance_steps = 0

    coordination_sum = 0.0
    coordination_steps = 0

    end_distances = None

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        if bool(info.get("collision", False)):
            collisions_episode = 1
        if bool(info.get("task_completed", False)):
            completion = True

        # coordination based on current distances-to-goal
        distances = np.asarray(info.get("distances", np.zeros(n_agents)), dtype=np.float32).reshape(-1)[:n_agents]
        coordination_sum += _coordination_score(distances)
        coordination_steps += 1
        end_distances = distances

        # inter-agent distance from reconstructed positions
        positions = _reconstruct_positions_from_obs(
            obs,
            local_state_dim=local_state_dim,
            global_state_dim=global_state_dim,
            n_agents=n_agents,
        )
        inter_agent_distance_sum += _pairwise_mean_distance(positions)
        inter_agent_distance_steps += 1

        done = bool(terminated or truncated)

    mean_inter_agent_distance = (
        float(inter_agent_distance_sum / max(1, inter_agent_distance_steps))
        if n_agents >= 2
        else 0.0
    )
    coordination_score = float(coordination_sum / max(1, coordination_steps))
    synchronization_score = _synchronization_score(end_distances) if end_distances is not None else 0.0

    return (
        total_reward,
        collisions_episode,
        completion,
        steps,
        mean_inter_agent_distance,
        coordination_score,
        synchronization_score,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO-only evaluation with CTDE metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to PPO model zip")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to JSON config")
    parser.add_argument("--episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--output-dir", type=str, default="logs/eval", help="Output directory for CSV")
    parser.add_argument("--json-out", type=str, default="", help="Optional JSON output path for aggregated metrics")
    parser.add_argument("--series-out", type=str, default="", help="Optional JSON output path for per-episode series")
    args = parser.parse_args()

    cfg = load_config(args.config)

    env = HiveMindEnv(cfg.environment, cfg.reward, seed=cfg.training.seed)
    model = PPO.load(args.model, device="cpu")

    run_stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_per_episode_{run_stamp}.csv"

    metrics: List[Tuple[float, int, bool, int, float, float, float]] = []
    rewards_series: List[float] = []
    success_series: List[int] = []
    collision_series: List[int] = []
    steps_series: List[int] = []
    coordination_series: List[float] = []
    synchronization_series: List[float] = []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "reward",
                "collisions",
                "completion",
                "steps",
                "mean_inter_agent_distance",
                "coordination_score",
                "synchronization_score",
            ],
        )
        w.writeheader()

        for ep in range(args.episodes):
            (
                reward,
                collisions_episode,
                completion,
                steps,
                mean_inter_agent_distance,
                coordination_score,
                synchronization_score,
            ) = run_episode(
                env=env,
                model=model,
                n_agents=cfg.environment.n_agents,
                local_state_dim=cfg.environment.local_state_dim,
                global_state_dim=cfg.environment.global_state_dim,
                deterministic=args.deterministic,
            )

            metrics.append(
                (reward, collisions_episode, completion, steps, mean_inter_agent_distance, coordination_score, synchronization_score)
            )
            rewards_series.append(float(reward))
            success_series.append(int(completion))
            collision_series.append(int(collisions_episode > 0))
            steps_series.append(int(steps))
            coordination_series.append(float(coordination_score))
            synchronization_series.append(float(synchronization_score))
            w.writerow(
                {
                    "episode": ep + 1,
                    "reward": reward,
                    "collisions": collisions_episode,
                    "completion": int(completion),
                    "steps": steps,
                    "mean_inter_agent_distance": mean_inter_agent_distance,
                    "coordination_score": coordination_score,
                    "synchronization_score": synchronization_score,
                }
            )

    rewards = np.asarray([m[0] for m in metrics], dtype=np.float32)
    collisions = np.asarray([m[1] for m in metrics], dtype=np.float32)
    completions = np.asarray([1.0 if m[2] else 0.0 for m in metrics], dtype=np.float32)
    steps = np.asarray([m[3] for m in metrics], dtype=np.float32)
    inter_agent_distance = np.asarray([m[4] for m in metrics], dtype=np.float32)
    coordination = np.asarray([m[5] for m in metrics], dtype=np.float32)
    synchronization = np.asarray([m[6] for m in metrics], dtype=np.float32)

    success_rate = float(np.mean(completions))
    collision_rate = float(np.mean(collisions > 0))

    print("Evaluation Results")
    print(f"Success Rate: {success_rate * 100:.0f}%")
    print(f"Avg Reward: {float(np.mean(rewards)):.3f}")
    print(f"Collision Rate: {collision_rate:.3f}")
    print(f"Avg Steps: {float(np.mean(steps)):.3f}")
    print(f"Mean Inter-Agent Distance: {float(np.mean(inter_agent_distance)):.3f}")
    print(f"Coordination Score: {float(np.mean(coordination)):.3f}")
    print(f"Synchronization Score: {float(np.mean(synchronization)):.3f}")
    print(f"Wrote CSV: {csv_path}")

    # Optional JSON outputs for UI consumption (no CSV needed).
    if args.json_out:
        import json

        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": "PPO (CTDE)",
            "episodes": int(args.episodes),
            "deterministic": bool(args.deterministic),
            "success_rate": success_rate,
            "avg_reward": float(np.mean(rewards)),
            "collision_rate": collision_rate,
            "avg_steps": float(np.mean(steps)),
            "mean_inter_agent_distance": float(np.mean(inter_agent_distance)),
            "coordination_score": float(np.mean(coordination)),
            "synchronization_score": float(np.mean(synchronization)),
            "timestamp": run_stamp,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.series_out:
        import json

        series_path = Path(args.series_out)
        series_path.parent.mkdir(parents=True, exist_ok=True)

        episodes = list(range(1, int(args.episodes) + 1))
        success_arr = np.asarray(success_series, dtype=np.float32)
        collision_arr = np.asarray(collision_series, dtype=np.float32)
        success_rate_cum = np.cumsum(success_arr) / np.maximum(1, np.arange(1, len(success_arr) + 1))
        collision_rate_cum = np.cumsum(collision_arr) / np.maximum(1, np.arange(1, len(collision_arr) + 1))

        series_payload = {
            "episodes": episodes,
            "reward": [float(x) for x in rewards_series],
            "success": success_series,
            "collision": collision_series,
            "steps": steps_series,
            "success_rate_cum": [float(x) for x in success_rate_cum.tolist()],
            "collision_rate_cum": [float(x) for x in collision_rate_cum.tolist()],
            "coordination_score": [float(x) for x in coordination_series],
            "synchronization_score": [float(x) for x in synchronization_series],
            "timestamp": run_stamp,
        }
        series_path.write_text(json.dumps(series_payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

