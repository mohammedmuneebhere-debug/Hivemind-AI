"""Resilient writes for Windows when logs/ui files are open in another process."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path


def _replace_with_retries(src: str | Path, dst: str | Path, *, attempts: int = 6) -> None:
    last: OSError | None = None
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except OSError as e:
            last = e
            time.sleep(0.12 * (i + 1))
    assert last is not None
    raise last


def promote_temp_file(tmp: Path, primary: Path, alt: Path) -> Path:
    """
    Move a finished temp file onto `primary`; if locked, move onto `alt` instead.
    """
    try:
        _replace_with_retries(tmp, primary)
        return primary
    except OSError:
        try:
            if alt.exists():
                alt.unlink()
        except OSError:
            pass
        _replace_with_retries(tmp, alt)
        print(f"Note: {primary.name} was locked; wrote to {alt.name} instead.")
        return alt


def write_trace_json_resilient(trace_out: Path, text: str) -> Path:
    """
    Avoid PermissionError when the target file is open in another process.
    Temp file + os.replace with retries, then fallback to demo_trace_alt.json.
    """
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    alt = trace_out.parent / "demo_trace_alt.json"
    fd, tmp_path = tempfile.mkstemp(
        suffix=".json.tmp",
        prefix="demo_trace_",
        dir=str(trace_out.parent),
    )
    tmp_p = Path(tmp_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        return promote_temp_file(tmp_p, trace_out, alt)
    except OSError:
        try:
            if tmp_p.exists():
                tmp_p.unlink()
        except OSError:
            pass
        try:
            alt.write_text(text, encoding="utf-8")
        except OSError as e:
            raise RuntimeError(
                f"Could not write {trace_out} or fallback {alt}. "
                "Close demo_trace.json in your editor if it is open, then retry."
            ) from e
        print(f"Note: {trace_out.name} was locked; wrote trace to {alt.name} instead.")
        return alt
