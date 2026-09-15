"""Run one isolated pipeline variant and record user-visible latency milestones."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from netcast_tennisvision.pipeline import runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    milestones: list[dict[str, object]] = []
    original_write_status = runner.write_status

    def capture_status(state: str, progress: int, stage: str, error=None, **extra):
        milestones.append({
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "state": state,
            "progress": progress,
            "stage": stage,
            **extra,
        })
        return original_write_status(state, progress, stage, error, **extra)

    runner.write_status = capture_status
    error = None
    try:
        runner.main()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        payload = {
            "name": args.name,
            "total_seconds": round(time.perf_counter() - started, 3),
            "milestones": milestones,
            "error": error,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
