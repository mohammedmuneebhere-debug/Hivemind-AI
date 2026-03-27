# HiveMind AI Demo Dashboard

Minimal dashboard for evaluators (PPO CTDE demo).

## 1) FastAPI backend
From the repo root:

```powershell
cd demo-ui/backend
pip install -r requirements.txt
uvicorn main:app --port 8000
```

Endpoints:
- `POST http://localhost:8000/api/run-eval` (runs `evaluate.py` and returns JSON)
- `POST http://localhost:8000/api/run-demo-episode` (generates/serves `demo_motion.gif` + demo trace)
- `GET http://localhost:8000/api/latest-metrics`
- `GET http://localhost:8000/api/latest-series`

## 2) React frontend
In a second terminal:

```powershell
cd demo-ui/frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Then open `http://localhost:5173/`.

The dashboard never shows CSV to evaluators; it uses JSON produced by `evaluate.py`.

