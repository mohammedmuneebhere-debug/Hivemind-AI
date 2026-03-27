"""Apply resilient trace write + GIF-before-trace order to repo-root demo_episode.py."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
target = ROOT / "demo_episode.py"
text = target.read_text(encoding="utf-8")
if "write_trace_json_resilient" in text or "trace_io" in text:
    print("demo_episode.py already includes trace fix")
    raise SystemExit(0)

old_imports = """import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from stable_baselines3 import PPO

from src.env.gazebo_env import HiveMindEnv
from src.utils.config import load_config
"""
new_imports = """import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from stable_baselines3 import PPO

from src.env.gazebo_env import HiveMindEnv
from src.utils.config import load_config
from src.utils.trace_io import write_trace_json_resilient
"""
if old_imports not in text:
    raise SystemExit("demo_episode.py header mismatch; patch manually")
text = text.replace(old_imports, new_imports, 1)

old_loop_vars = """    frames_positions: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    collisions: List[bool] = []
    completions: List[bool] = []
    step_traces: List[Dict[str, Any]] = []
"""
new_loop_vars = """    frames_positions: List[np.ndarray] = []
    step_traces: List[Dict[str, Any]] = []
"""
if old_loop_vars in text:
    text = text.replace(old_loop_vars, new_loop_vars, 1)

old_append = """        frames_positions.append(positions_out)
        actions.append(action)
        rewards.append(float(reward))
        collisions.append(collision)
        completions.append(completion)

        # Compact input/output trace for UI.
"""
new_append = """        frames_positions.append(positions_out)

        # Compact input/output trace for UI.
"""
if old_append in text:
    text = text.replace(old_append, new_append, 1)

old_trace_block = """    # Save trace JSON for UI.
    trace_payload = {
        "model": "PPO (CTDE)",
        "steps_run": len(frames_positions),
        "target": target.tolist(),
        "trace": step_traces[:20],  # keep it readable
    }
    trace_out.write_text(json.dumps(trace_payload, indent=2), encoding="utf-8")

    # Create GIF via matplotlib animation (2D motion preview).
"""
new_trace_block = """    trace_payload = {
        "model": "PPO (CTDE)",
        "steps_run": len(frames_positions),
        "target": target.tolist(),
        "trace": step_traces[:20],  # keep it readable
    }
    trace_text = json.dumps(trace_payload, indent=2)

    # Create GIF via matplotlib animation (2D motion preview).
"""
if old_trace_block not in text:
    raise SystemExit("trace block mismatch")
text = text.replace(old_trace_block, new_trace_block, 1)

old_tail = """    anim.save(str(gif_out), writer=PillowWriter(fps=5))
    plt.close(fig)

    print(f"Wrote GIF: {gif_out}")
    print(f"Wrote trace JSON: {trace_out}")
"""
new_tail = """    anim.save(str(gif_out), writer=PillowWriter(fps=5))
    plt.close(fig)

    written_trace = write_trace_json_resilient(trace_out, trace_text)

    print(f"Wrote GIF: {gif_out}")
    print(f"Wrote trace JSON: {written_trace}")
"""
if old_tail not in text:
    raise SystemExit("tail mismatch")
text = text.replace(old_tail, new_tail, 1)

target.write_text(text, encoding="utf-8")
print("Updated", target)
