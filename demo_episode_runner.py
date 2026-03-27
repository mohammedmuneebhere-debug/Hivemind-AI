"""
UI demo asset generator (GIF + trace JSON).

Same CLI as demo_episode.py, but writes the trace after the GIF and uses
src.utils.trace_io.write_trace_json_resilient to avoid Windows PermissionError
when demo_trace.json is open in an editor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import pyplot as plt
from stable_baselines3 import PPO

from src.env.gazebo_env import HiveMindEnv
from src.utils.config import load_config
from src.utils.trace_io import promote_temp_file, write_trace_json_resilient


def reconstruct_positions_from_obs(
    obs: np.ndarray,
    *,
    local_state_dim: int,
    global_state_dim: int,
    n_agents: int,
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    global_flat = obs[local_state_dim : local_state_dim + global_state_dim]
    global_flat = global_flat[: n_agents * local_state_dim]
    local_all = global_flat.reshape(n_agents, local_state_dim)
    return local_all[:, :2].astype(np.float32)


def distances_to_target(positions: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(positions - target.reshape(1, 2), axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPO multi-agent motion preview GIF + trace (UI)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--gif-out", type=str, default="logs/ui/demo_motion.gif")
    parser.add_argument("--trace-out", type=str, default="logs/ui/demo_trace.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env = HiveMindEnv(cfg.environment, cfg.reward, seed=cfg.training.seed)
    model = PPO.load(args.model, device="cpu")

    gif_out = Path(args.gif_out)
    trace_out = Path(args.trace_out)
    gif_out.parent.mkdir(parents=True, exist_ok=True)
    trace_out.parent.mkdir(parents=True, exist_ok=True)

    obs, info = env.reset()
    target = np.asarray(info["target"], dtype=np.float32)

    local_state_dim = cfg.environment.local_state_dim
    global_state_dim = cfg.environment.global_state_dim
    n_agents = cfg.environment.n_agents

    frames_positions: List[np.ndarray] = []
    step_traces: List[Dict[str, Any]] = []

    for t in range(args.steps):
        positions_in = reconstruct_positions_from_obs(
            obs,
            local_state_dim=local_state_dim,
            global_state_dim=global_state_dim,
            n_agents=n_agents,
        )
        dists_in = distances_to_target(positions_in, target)
        mean_d_in = float(np.mean(dists_in))

        action, _ = model.predict(obs, deterministic=args.deterministic)
        action = np.asarray(action, dtype=np.float32).reshape(n_agents, cfg.environment.action_dim)

        next_obs, reward, terminated, truncated, next_info = env.step(action.reshape(-1))

        positions_out = reconstruct_positions_from_obs(
            next_obs,
            local_state_dim=local_state_dim,
            global_state_dim=global_state_dim,
            n_agents=n_agents,
        )
        dists_out = distances_to_target(positions_out, target)
        mean_d_out = float(np.mean(dists_out))

        collision = bool(next_info.get("collision", False))
        completion = bool(next_info.get("task_completed", False))

        frames_positions.append(positions_out)
        step_traces.append(
            {
                "t": t + 1,
                "input": {
                    "mean_distance": mean_d_in,
                    "distances": dists_in.tolist(),
                },
                "action": action.tolist(),
                "next": {
                    "mean_distance": mean_d_out,
                    "distances": dists_out.tolist(),
                    "reward": float(reward),
                    "collision": collision,
                    "task_completed": completion,
                    "done": bool(terminated or truncated),
                },
            }
        )

        obs = next_obs
        if terminated or truncated:
            break

    trace_payload = {
        "model": "PPO (CTDE)",
        "steps_run": len(frames_positions),
        "target": target.tolist(),
        "trace": step_traces[:20],
    }
    trace_text = json.dumps(trace_payload, indent=2)

    all_positions = np.concatenate(frames_positions, axis=0) if frames_positions else np.zeros((1, 2), dtype=np.float32)
    min_xy = np.min(all_positions, axis=0) - 0.2
    max_xy = np.max(all_positions, axis=0) + 0.2

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Multi-agent motion preview")
    ax.set_xlim(float(min_xy[0]), float(max_xy[0]))
    ax.set_ylim(float(min_xy[1]), float(max_xy[1]))
    ax.grid(True, alpha=0.2)

    ax.scatter(
        [float(target[0])],
        [float(target[1])],
        marker="x",
        c="#fbbf24",
        s=80,
        linewidths=3,
        label="Target",
    )

    agent_colors = ["#60a5fa", "#34d399", "#f472b6", "#f59e0b"][:n_agents]
    agent_scatter = ax.scatter(np.zeros(n_agents, dtype=np.float32), np.zeros(n_agents, dtype=np.float32), s=80, c=agent_colors)

    trails: List[Any] = []
    for _ in range(n_agents):
        (ln,) = ax.plot([], [], linewidth=2, alpha=0.5)
        trails.append(ln)

    history: List[np.ndarray] = []

    def init():
        agent_scatter.set_offsets(np.zeros((n_agents, 2), dtype=np.float32))
        for ln in trails:
            ln.set_data([], [])
        return [agent_scatter, *trails]

    def update(frame_idx: int):
        pos = frames_positions[frame_idx]
        history.append(pos)
        agent_scatter.set_offsets(pos)
        hist_arr = np.stack(history, axis=0)
        for agent_i in range(n_agents):
            trails[agent_i].set_data(hist_arr[:, agent_i, 0], hist_arr[:, agent_i, 1])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        return [agent_scatter, *trails]

    frames = len(frames_positions)
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=200, blit=False)
    tmp_gif = gif_out.parent / f"_tmp_demo_motion_{os.getpid()}.gif"
    anim.save(str(tmp_gif), writer=PillowWriter(fps=5))
    plt.close(fig)

    alt_gif = gif_out.parent / "demo_motion_alt.gif"
    written_gif = promote_temp_file(tmp_gif, gif_out, alt_gif)

    written_trace = write_trace_json_resilient(trace_out, trace_text)

    print(f"Wrote GIF: {written_gif}")
    print(f"Wrote trace JSON: {written_trace}")


if __name__ == "__main__":
    main()
