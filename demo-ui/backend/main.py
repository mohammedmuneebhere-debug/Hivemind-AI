from __future__ import annotations

import json
import subprocess
import sys
import datetime as dt
from pathlib import Path
from typing import Any, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env.gazebo_env import HiveMindEnv  # noqa: E402
from src.utils.config import load_config  # noqa: E402

UI_LOG_DIR = ROOT / "logs" / "ui"
UI_LOG_DIR.mkdir(parents=True, exist_ok=True)

LATEST_METRICS_PATH = UI_LOG_DIR / "latest_metrics.json"
LATEST_SERIES_PATH = UI_LOG_DIR / "latest_series.json"

DEMO_GIF_PATH = UI_LOG_DIR / "demo_motion.gif"
DEMO_GIF_ALT_PATH = UI_LOG_DIR / "demo_motion_alt.gif"
DEMO_MP4_PATH = UI_LOG_DIR / "demo_motion.mp4"
DEMO_TRACE_PATH = UI_LOG_DIR / "demo_trace.json"
# Written when demo_trace.json is locked (e.g. open in an editor on Windows)
DEMO_TRACE_ALT_PATH = UI_LOG_DIR / "demo_trace_alt.json"

app = FastAPI(title="HiveMind AI Demo (PPO UI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(UI_LOG_DIR)), name="ui-static")


class RunEvalRequest(BaseModel):
    episodes: int = Field(default=50, ge=1, le=200)
    deterministic: bool = True
    model: str = Field(default=str(ROOT / "models" / "ppo_mock.zip"))
    config: str = Field(default=str(ROOT / "configs" / "default.json"))


class RunDemoRequest(BaseModel):
    steps: int = Field(default=60, ge=10, le=120)
    deterministic: bool = True
    regen: bool = False
    model: str = Field(default=str(ROOT / "models" / "ppo_mock.zip"))
    config: str = Field(default=str(ROOT / "configs" / "default.json"))


class ManualStepRequest(BaseModel):
    agents: list[list[float]] = Field(
        ...,
        description="Per-agent [accel_x, accel_y]; count must match environment n_agents.",
    )


_manual_env: HiveMindEnv | None = None
_latest_metrics_cache: Dict[str, Any] | None = None
_latest_series_cache: Dict[str, Any] | None = None


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _demo_trace_path() -> Path | None:
    if DEMO_TRACE_PATH.exists():
        return DEMO_TRACE_PATH
    if DEMO_TRACE_ALT_PATH.exists():
        return DEMO_TRACE_ALT_PATH
    return None


def _demo_gif_path() -> Path | None:
    if DEMO_GIF_PATH.exists():
        return DEMO_GIF_PATH
    if DEMO_GIF_ALT_PATH.exists():
        return DEMO_GIF_ALT_PATH
    return None


def _read_demo_trace() -> Dict[str, Any]:
    p = _demo_trace_path()
    if p is None:
        raise FileNotFoundError("no demo trace file")
    return _read_json(p)


def _agent_positions_from_obs(obs: np.ndarray, env: HiveMindEnv) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    ld = env.local_state_dim
    n = env.n_agents
    chunk = obs[ld : ld + n * ld]
    return chunk.reshape(n, ld)[:, :2].astype(np.float32)


def _motion_summary(obs: np.ndarray, info: Dict[str, Any], env: HiveMindEnv) -> Dict[str, Any]:
    target = np.asarray(info["target"], dtype=np.float32)
    pos = _agent_positions_from_obs(obs, env)
    dists = np.linalg.norm(pos - target.reshape(1, 2), axis=1)
    return {
        "positions": pos.astype(float).tolist(),
        "target": target.astype(float).tolist(),
        "mean_distance": float(np.mean(dists)),
        "distances": dists.astype(float).tolist(),
        "n_agents": env.n_agents,
    }


def _demo_response_payload(trace: Dict[str, Any]) -> Dict[str, Any]:
    gif_p = _demo_gif_path()
    gif_name = gif_p.name if gif_p is not None else DEMO_GIF_PATH.name
    out: Dict[str, Any] = {
        "gifUrl": f"/static/{gif_name}",
        "trace": trace,
        "stepsRun": trace.get("steps_run", 0),
    }
    if DEMO_MP4_PATH.exists():
        out["videoUrl"] = f"/static/{DEMO_MP4_PATH.name}"
    return out


def _run_eval_in_memory(req: RunEvalRequest) -> Dict[str, Any]:
    """
    Evaluate PPO directly in-process and return metrics/series.
    This avoids writing latest_metrics.json/latest_series.json, which may be
    locked on Windows when opened in another app.
    """
    cfg = load_config(req.config)
    env = HiveMindEnv(cfg.environment, cfg.reward, seed=cfg.training.seed)
    model = PPO.load(req.model, device="cpu")

    rewards_series: list[float] = []
    success_series: list[int] = []
    collision_series: list[int] = []
    steps_series: list[int] = []
    coordination_series: list[float] = []
    synchronization_series: list[float] = []
    inter_agent_dist_series: list[float] = []

    for _ in range(req.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_collision = False
        ep_success = False
        coord_values: list[float] = []
        end_distances: np.ndarray | None = None
        pairwise_values: list[float] = []

        while not done:
            action, _ = model.predict(obs, deterministic=req.deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            ep_reward += float(reward)
            ep_steps += 1
            ep_collision = ep_collision or bool(info.get("collision", False))
            ep_success = ep_success or bool(info.get("task_completed", False))

            distances = np.asarray(info.get("distances", np.zeros(cfg.environment.n_agents)), dtype=np.float32)
            if distances.size:
                coord_values.append(float(1.0 / (1.0 + float(np.var(distances)))))
                end_distances = distances

            # obs = [local_obs0, global_flat]
            ld = cfg.environment.local_state_dim
            gd = cfg.environment.global_state_dim
            n = cfg.environment.n_agents
            flat = np.asarray(obs, dtype=np.float32).reshape(-1)
            global_flat = flat[ld : ld + gd][: n * ld]
            if global_flat.size == n * ld and n >= 2:
                positions = global_flat.reshape(n, ld)[:, :2]
                dists = []
                for i in range(n):
                    for j in range(i + 1, n):
                        dists.append(float(np.linalg.norm(positions[i] - positions[j])))
                if dists:
                    pairwise_values.append(float(np.mean(dists)))

        rewards_series.append(ep_reward)
        success_series.append(1 if ep_success else 0)
        collision_series.append(1 if ep_collision else 0)
        steps_series.append(ep_steps)
        coordination_series.append(float(np.mean(coord_values)) if coord_values else 0.0)
        if end_distances is None:
            synchronization_series.append(0.0)
        else:
            synchronization_series.append(float(1.0 / (1.0 + float(np.std(end_distances)))))
        inter_agent_dist_series.append(float(np.mean(pairwise_values)) if pairwise_values else 0.0)

    rewards = np.asarray(rewards_series, dtype=np.float32)
    collisions = np.asarray(collision_series, dtype=np.float32)
    completions = np.asarray(success_series, dtype=np.float32)
    steps = np.asarray(steps_series, dtype=np.float32)
    inter_agent = np.asarray(inter_agent_dist_series, dtype=np.float32)
    coordination = np.asarray(coordination_series, dtype=np.float32)
    synchronization = np.asarray(synchronization_series, dtype=np.float32)

    success_rate_cum = np.cumsum(completions) / np.maximum(1, np.arange(1, len(completions) + 1))
    collision_rate_cum = np.cumsum(collisions) / np.maximum(1, np.arange(1, len(collisions) + 1))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    metrics = {
        "model": "PPO (CTDE)",
        "episodes": int(req.episodes),
        "deterministic": bool(req.deterministic),
        "success_rate": float(np.mean(completions)) if len(completions) else 0.0,
        "avg_reward": float(np.mean(rewards)) if len(rewards) else 0.0,
        "collision_rate": float(np.mean(collisions)) if len(collisions) else 0.0,
        "avg_steps": float(np.mean(steps)) if len(steps) else 0.0,
        "mean_inter_agent_distance": float(np.mean(inter_agent)) if len(inter_agent) else 0.0,
        "coordination_score": float(np.mean(coordination)) if len(coordination) else 0.0,
        "synchronization_score": float(np.mean(synchronization)) if len(synchronization) else 0.0,
        "timestamp": stamp,
    }
    series = {
        "episodes": list(range(1, int(req.episodes) + 1)),
        "reward": [float(x) for x in rewards_series],
        "success": [int(x) for x in success_series],
        "collision": [int(x) for x in collision_series],
        "steps": [int(x) for x in steps_series],
        "success_rate_cum": [float(x) for x in success_rate_cum.tolist()],
        "collision_rate_cum": [float(x) for x in collision_rate_cum.tolist()],
        "coordination_score": [float(x) for x in coordination_series],
        "synchronization_score": [float(x) for x in synchronization_series],
        "timestamp": stamp,
    }
    return {"metrics": metrics, "series": series}


@app.get("/api/status")
def status() -> Dict[str, Any]:
    out: Dict[str, Any] = {"running": False}
    if _latest_metrics_cache is not None:
        out["latest_timestamp"] = _latest_metrics_cache.get("timestamp")
    elif LATEST_METRICS_PATH.exists():
        out["latest_timestamp"] = _read_json(LATEST_METRICS_PATH).get("timestamp")
    return out


@app.get("/api/latest-metrics")
def latest_metrics() -> Dict[str, Any]:
    if _latest_metrics_cache is not None:
        return _latest_metrics_cache
    try:
        return _read_json(LATEST_METRICS_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No evaluation has been run yet.")


@app.get("/api/latest-series")
def latest_series() -> Dict[str, Any]:
    if _latest_series_cache is not None:
        return _latest_series_cache
    try:
        return _read_json(LATEST_SERIES_PATH)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No series has been generated yet.")


@app.post("/api/run-eval")
def run_eval(req: RunEvalRequest) -> Dict[str, Any]:
    global _latest_metrics_cache, _latest_series_cache
    model_path = Path(req.model)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing model: {model_path}")

    config_path = Path(req.config)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing config: {config_path}")

    try:
        payload = _run_eval_in_memory(req)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={
                "error": "evaluation_failed",
                "message": str(e),
            },
        )

    _latest_metrics_cache = payload["metrics"]
    _latest_series_cache = payload["series"]
    return payload


@app.get("/api/latest-demo-trace")
def latest_demo_trace() -> Dict[str, Any]:
    try:
        return _read_demo_trace()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No demo episode has been generated yet.")


@app.post("/api/run-demo-episode")
def run_demo(req: RunDemoRequest) -> Dict[str, Any]:
    if not req.regen and _demo_gif_path() is not None and _demo_trace_path() is not None:
        trace = _read_demo_trace()
        return _demo_response_payload(trace)

    model_path = Path(req.model)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing model: {model_path}")

    config_path = Path(req.config)
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing config: {config_path}")

    demo_script = ROOT / "demo_episode_runner.py"
    if not demo_script.exists():
        demo_script = ROOT / "demo_episode.py"
    if not demo_script.exists():
        raise HTTPException(status_code=500, detail="demo_episode_runner.py / demo_episode.py not found at repo root.")

    cmd = [
        sys.executable,
        str(demo_script),
        "--model",
        str(model_path),
        "--config",
        str(config_path),
        "--steps",
        str(req.steps),
        "--gif-out",
        str(DEMO_GIF_PATH),
        "--trace-out",
        str(DEMO_TRACE_PATH),
    ]

    if req.deterministic:
        cmd.append("--deterministic")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "demo_failed",
                "stdout": proc.stdout[-2000:],
                "stderr": proc.stderr[-2000:],
                "returncode": proc.returncode,
            },
        )

    trace = _read_demo_trace()
    return _demo_response_payload(trace)


@app.post("/api/manual-reset")
def manual_reset(config: str = str(ROOT / "configs" / "default.json")) -> Dict[str, Any]:
    global _manual_env
    cfg_path = Path(config)
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing config: {cfg_path}")
    cfg = load_config(str(cfg_path))
    env = HiveMindEnv(cfg.environment, cfg.reward, seed=cfg.training.seed)
    obs, info = env.reset()
    _manual_env = env
    out = _motion_summary(obs, info, env)
    out["message"] = "Environment reset."
    return out


@app.post("/api/manual-step")
def manual_step(req: ManualStepRequest) -> Dict[str, Any]:
    global _manual_env
    if _manual_env is None:
        raise HTTPException(status_code=400, detail="Call POST /api/manual-reset first.")
    env = _manual_env
    if len(req.agents) != env.n_agents:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {env.n_agents} agent rows in `agents`, got {len(req.agents)}.",
        )
    flat: list[float] = []
    for row in req.agents:
        if len(row) != env.action_dim:
            raise HTTPException(status_code=400, detail=f"Each agent needs {env.action_dim} values [ax, ay].")
        flat.extend(float(x) for x in row)
    action = np.asarray(flat, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    out: Dict[str, Any] = {
        "reward": float(reward),
        "collision": bool(info.get("collision", False)),
        "task_completed": bool(info.get("task_completed", False)),
        "done": bool(terminated or truncated),
        "mean_distance": float(info.get("mean_distance", 0.0)),
        "step": int(info.get("step", 0)),
    }
    out.update(_motion_summary(obs, info, env))
    return out
