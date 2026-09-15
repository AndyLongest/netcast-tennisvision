"""PPIO/RunPod-compatible async worker used for reproducible GPU benchmarks."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from netcast_tennisvision.paths import REPOSITORY_ROOT

DATA_DIR = REPOSITORY_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "outputs"
STATUS_PATH = DATA_DIR / "job_status.json"
CURRENT_JOB_PATH = DATA_DIR / "current_job.json"
CLIP_PATH = DATA_DIR / "clip.mp4"
DEMO_PATH = REPOSITORY_ROOT / "assets" / "demo" / "demo.mp4"
RUNNER_LOG_PATH = DATA_DIR / "cloud_runner.log"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _prepare_demo_job() -> str:
    if not DEMO_PATH.is_file():
        raise RuntimeError("容器内缺少基准视频 assets/demo/demo.mp4")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for path in (STATUS_PATH, CURRENT_JOB_PATH, RUNNER_LOG_PATH):
        path.unlink(missing_ok=True)
    shutil.copy2(DEMO_PATH, CLIP_PATH)
    job_id = uuid.uuid4().hex
    CURRENT_JOB_PATH.write_text(
        json.dumps(
            {
                "job_id": job_id,
                "filename": DEMO_PATH.name,
                "file_size": DEMO_PATH.stat().st_size,
                "video_fingerprint": _sha256(DEMO_PATH),
                "fps": 29.914,
                "workload_factor": 1.0,
                "display_correction": {"enabled": False, "strength": 0, "corners": None},
                "started_at": int(time.time()),
                "source": "bundled_server_benchmark",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return job_id


def _scene_summary(path: Path) -> dict[str, Any]:
    scene = _read_json(path)
    frames = scene.get("frames")
    bounces = scene.get("bounces")
    hits = scene.get("hits")
    positioned = 0
    if isinstance(frames, list):
        positioned = sum(
            1 for frame in frames
            if isinstance(frame, dict) and frame.get("b") is not None
        )
    return {
        "fps": scene.get("fps"),
        "frames": scene.get("n_frames"),
        "positioned_ball_frames": positioned,
        "bounces": len(bounces) if isinstance(bounces, list) else None,
        "hits": len(hits) if isinstance(hits, list) else None,
    }


def benchmark_demo(*, timeout_seconds: float = 3600) -> dict[str, Any]:
    """Run the immutable bundled demo and return timing plus compact regression evidence."""
    job_id = _prepare_demo_job()
    started = time.perf_counter()
    report_ready_seconds: float | None = None
    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    with RUNNER_LOG_PATH.open("wb") as log_stream:
        process = subprocess.Popen(
            [sys.executable, "-m", "netcast_tennisvision.pipeline.runner"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            status = _read_json(STATUS_PATH)
            if report_ready_seconds is None and status.get("state") in {"report_ready", "complete"}:
                report_ready_seconds = time.perf_counter() - started
            if time.perf_counter() - started > timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TimeoutError(f"服务器基准超过 {timeout_seconds:.0f} 秒，已终止")
            time.sleep(0.25)
    completed_seconds = time.perf_counter() - started
    status = _read_json(STATUS_PATH)
    if process.returncode != 0 or status.get("state") != "complete":
        tail = RUNNER_LOG_PATH.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"服务器分析失败（exit={process.returncode}, state={status.get('state')}）：{tail}"
        )

    scene_path = OUTPUT_DIR / "scene3d.json"
    video_path = OUTPUT_DIR / "annotated_clip.mp4"
    report_path = OUTPUT_DIR / "rally3d.html"
    missing = [path.name for path in (scene_path, video_path, report_path) if not path.is_file()]
    if missing:
        raise RuntimeError("服务器分析缺少输出：" + ", ".join(missing))

    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        cuda = torch.version.cuda
    except ImportError:
        gpu, cuda = "unknown", None
    return {
        "ok": True,
        "mode": "benchmark_demo",
        "job_id": job_id,
        "runtime": {
            "gpu": gpu,
            "cuda": cuda,
            "report_ready_seconds": round(report_ready_seconds, 3)
            if report_ready_seconds is not None
            else None,
            "complete_seconds": round(completed_seconds, 3),
        },
        "scene": _scene_summary(scene_path),
        "artifacts": {
            "scene3d_sha256": _sha256(scene_path),
            "annotated_video_sha256": _sha256(video_path),
            "annotated_video_bytes": video_path.stat().st_size,
            "report_bytes": report_path.stat().st_size,
        },
    }


def handle_job(job: dict[str, Any]) -> dict[str, Any]:
    """Validate the small async payload before touching the expensive pipeline."""
    payload = job.get("input")
    if not isinstance(payload, dict):
        return {"ok": False, "error": "input 必须是 JSON 对象"}
    mode = payload.get("mode")
    if mode != "benchmark_demo":
        return {"ok": False, "error": "当前测试 Worker 仅支持 benchmark_demo"}
    timeout = payload.get("timeout_seconds", 3600)
    if not isinstance(timeout, int | float) or not 60 <= float(timeout) <= 7200:
        return {"ok": False, "error": "timeout_seconds 必须在 60 到 7200 之间"}
    return benchmark_demo(timeout_seconds=float(timeout))


def main() -> None:
    """Start the RunPod protocol server used by PPIO Async Serverless."""
    import runpod

    runpod.serverless.start({"handler": handle_job})


if __name__ == "__main__":
    main()
