# HiveMind AI

Multi-agent reinforcement learning project for cooperative control, built around a CTDE (Centralized Training, Decentralized Execution) setup with a polished evaluator-facing demo UI.

---

## Project Summary

HiveMind AI demonstrates how multiple agents can coordinate toward a shared target while avoiding collisions in a physics-inspired mock environment.

The project includes:

- A custom multi-agent environment (`HiveMindEnv`) compatible with Gym/Gymnasium
- A custom CTDE PPO policy implementation (`CTDEPPOPolicy`)
- PPO training and evaluation utilities
- A FastAPI backend + React dashboard for evaluator demos
- Simulation preview generation (GIF, with robust fallback handling)
- Manual action control endpoint/UI for step-by-step scenario demonstration

---

## Phase History (Including SAC in Phase 4)

This repository evolved through multiple experimentation phases:

1. **Phase 1-3 (Core RL setup and sweeps)**
   - Environment design and reward shaping iterations
   - CTDE policy implementation and PPO pipeline stabilization
   - Hyperparameter/reward sweeps (artifact models are still present under `models/`)

2. **Phase 4 (SAC exploration)**
   - SAC was explored as an additional candidate in late-stage evaluation.
   - Artifacts from this stage are still kept in:
     - `logs/phase4/phase4_full_final_20260323_175724`
     - `logs/phase4/phase4_sac_fast_20260323_180712`
     - `models/sac_mock.zip`

3. **Finalization decision**
   - SAC comparison was intentionally removed from the evaluator workflow.
   - The final demo and evaluation path is **PPO-only** for consistency, reproducibility, and simplified presentation.
   - Current backend evaluation endpoint runs PPO and returns metrics/series directly for the dashboard.

In short: **SAC was tested during Phase 4, but the productized demo was finalized with PPO.**

---

## Current Tech Stack

- **RL/Backend**: Python, Stable-Baselines3, FastAPI, NumPy, Matplotlib
- **Frontend**: React + TypeScript + Vite + Recharts
- **Environment**: Custom Gym-style mock env (`src/env/gazebo_env.py`)

---

## Repository Layout

```text
hivemind-ai/
├─ configs/
│  └─ default.json                  # Active config (PPO + env + reward + SAC section retained as history)
├─ models/
│  ├─ ppo_mock.zip                  # Main PPO model used by demo/eval
│  └─ ...                           # Historical experiment artifacts (including SAC)
├─ src/
│  ├─ env/
│  │  ├─ gazebo_env.py              # Mock multi-agent env
│  │  └─ state_encoder.py
│  ├─ rl/
│  │  ├─ ctde_ppo_policy.py         # Custom CTDE PPO policy
│  │  └─ ppo.py                     # PPO training helper
│  └─ utils/
│     ├─ config.py                  # Typed config loader
│     └─ trace_io.py                # Resilient file write helper (Windows lock-safe)
├─ evaluate.py                      # Standalone PPO evaluation script
├─ demo_episode.py                  # Original preview generator
├─ demo_episode_runner.py           # Lock-safe preview generator used by backend
├─ demo-ui/
│  ├─ backend/
│  │  ├─ main.py                    # FastAPI app for evaluation/demo/manual control
│  │  └─ requirements.txt
│  └─ frontend/
│     ├─ src/
│     │  ├─ AppDashboard.tsx        # Main evaluator dashboard
│     │  └─ AppDashboard.css
│     └─ package.json
└─ logs/                            # Evaluation + demo outputs + historical experiment logs
```

---

## Environment & Config Notes

Active config is `configs/default.json`:

- `environment.mode = "mock"`
- `n_agents = 2`
- `local_state_dim = 4`, `action_dim = 2`
- `global_state_dim = 0` in config is auto-derived to `n_agents * local_state_dim` by loader
- Reward includes progress shaping, step penalty, collision penalty, completion bonus
- `training.algo = "ppo"` and `save_path = "models/ppo_mock.zip"`

The `sac` config block is retained for historical traceability but is not part of the active demo/evaluation flow.

---

## Setup

## 1) Python dependencies (root)

From repo root:

```powershell
pip install -r requirements.txt
```

## 2) Demo backend dependencies

```powershell
cd demo-ui/backend
pip install -r requirements.txt
```

## 3) Frontend dependencies

```powershell
cd demo-ui/frontend
npm install
```

---

## Running the Demo (Backend + Frontend)

Open two terminals.

### Terminal A - FastAPI backend

```powershell
cd demo-ui/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal B - React frontend

```powershell
cd demo-ui/frontend
npm run dev -- --host 0.0.0.0 --port 5173
```

Open: `http://localhost:5173`

---

## Dashboard Features for Evaluators

- **Run Evaluation** (PPO only)
  - Returns aggregated metrics and per-episode series
  - Uses backend in-memory evaluation path to avoid JSON lock issues
- **Metrics cards**
  - Success rate, average reward, collision rate, average steps, coordination score
- **Charts**
  - Reward per episode
  - Cumulative success rate
  - Cumulative collision rate
- **Simulation preview**
  - Motion visualization (GIF; can also serve MP4 if present in `logs/ui`)
- **Use-case trace**
  - Step-wise Input -> Action -> Next table for evaluator storytelling
- **Manual action control**
  - Reset environment and apply per-agent acceleration vectors from UI

---

## FastAPI API Reference

Base URL: `http://localhost:8000`

- `GET /api/status`
- `POST /api/run-eval`
  - body: `{ "episodes": 50, "deterministic": true, "model": "...", "config": "..." }`
  - returns: `{ metrics, series }`
- `GET /api/latest-metrics`
- `GET /api/latest-series`
- `POST /api/run-demo-episode`
  - body: `{ "steps": 60, "deterministic": true, "regen": false, "model": "...", "config": "..." }`
  - returns demo asset URLs and trace payload
- `GET /api/latest-demo-trace`
- `POST /api/manual-reset`
- `POST /api/manual-step`
  - body: `{ "agents": [[ax1, ay1], [ax2, ay2], ...] }`

---

## Standalone Evaluation (CLI)

You can still run `evaluate.py` directly:

```powershell
python evaluate.py `
  --model models/ppo_mock.zip `
  --config configs/default.json `
  --episodes 50 `
  --deterministic `
  --output-dir logs/eval_cli
```

Optional outputs:

- `--json-out path/to/metrics.json`
- `--series-out path/to/series.json`

Note: backend demo path currently prefers in-memory eval for reliability on Windows.

---

## Optional PPO Training

If you want to retrain PPO using existing code:

```powershell
python -c "from src.utils.config import load_config; from src.rl.ppo import train_ppo; cfg=load_config('configs/default.json'); train_ppo(cfg)"
```

Model will be saved to `cfg.training.save_path` (default `models/ppo_mock.zip`).

---

## Windows File-Lock Reliability Notes

On Windows, files in `logs/ui` may be locked if open in an editor or preview pane.

Mitigations already added:

- In-memory evaluation cache in backend (avoids mandatory `latest_metrics.json` writes)
- `demo_episode_runner.py` writes assets via temp-file promotion
- `src/utils/trace_io.py` supports resilient fallback writes (e.g., `demo_trace_alt.json`)
- Backend reads alternate demo asset/trace paths when primary files are locked

If preview/eval fails, close open log files in editor tabs first, then retry.

---

## Git / Repo Notes

- A project `.gitignore` is included for Python caches, frontend build artifacts, and local editor files.
- Logs are currently **not fully ignored** by default, so you can choose to push them.
  - If you later want to ignore all runtime logs, uncomment `logs/` in `.gitignore`.

---

## Roadmap (Suggested)

- Add explicit MP4 generation path in `demo_episode_runner.py` with ffmpeg detection
- Add test coverage for backend endpoints (`run-eval`, `run-demo-episode`, manual control)
- Add reproducible benchmark script for PPO checkpoint comparison across seeds
- Add deployment profile (Docker + production API CORS/host config)

---

## License

Add your preferred license file (MIT/Apache-2.0/etc.) if this project will be public.

