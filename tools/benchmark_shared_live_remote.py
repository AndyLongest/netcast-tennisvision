"""Measure 1/3/5 camera streams sharing one live inference process on a cloud GPU.

Run only on the GPU host with a reachable ZLMediaKit RTMP endpoint and the
``pose_source.mp4``/``pose_calibration.json`` probe assets in ``/app``.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import psutil

from netcast_tennisvision.streaming import live_experiment as live
from netcast_tennisvision.streaming.live_inference import shared_live_inference

ROOT = Path("/app")
RESULT = ROOT / "capacity_shared_results.json"
TERMINAL = {"complete", "error", "stopped"}
os.environ["TENNISVISION_LIVE_SHARED_INFERENCE"] = "1"


def save(path: Path, payload: object) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def gpu_sample() -> list[float] | None:
    try:
        line = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=3,
        ).splitlines()[0]
        return [float(value.strip()) for value in line.split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def publish(stream: str, duration: int) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [live._ffmpeg(), "-hide_banner", "-loglevel", "error", "-re",
         "-i", str(ROOT / "pose_source.mp4"), "-t", str(duration), "-an",
         "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
         "-g", "30", "-bf", "0", "-f", "flv",
         f"rtmp://{live._media_host()}:1935/live/{stream}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run_case(count: int, *, sample_seconds: int = 45) -> dict:
    corners = json.loads((ROOT / "pose_calibration.json").read_text())[
        "court_image_corners"
    ]
    run_id = uuid.uuid4().hex[:10]
    streams = [f"shared-{run_id}-{index}" for index in range(count)]
    sessions: list[live.LiveExperimentSession] = []
    producers: list[subprocess.Popen[bytes]] = []
    external_publishers = os.environ.get("BENCH_EXTERNAL_PUBLISHERS") == "1"
    request_path = ROOT / "capacity_publisher_request.json"
    ready_path = ROOT / f"capacity_publisher_{run_id}.ready"
    samples: list[dict] = []
    started = time.monotonic()
    initial_pool_statistics = shared_live_inference.statistics()
    try:
        for stream in streams:
            session = live.LiveExperimentSession(
                None, external_stream_name=stream, fps_hint=30.00667022912,
                court_corners=corners,
            )
            sessions.append(session)
            session.start()
        while time.monotonic() - started < 150:
            states = [session.snapshot() for session in sessions]
            if any(state.get("state") in TERMINAL for state in states):
                raise RuntimeError(
                    "Stream setup failed: " + repr([
                        (state.get("state"), state.get("error")) for state in states
                    ])
                )
            if all(state.get("state") == "awaiting_stream" for state in states):
                break
            time.sleep(0.25)
        else:
            raise TimeoutError("Shared worker setup timed out")
        warmup_seconds = time.monotonic() - started
        if external_publishers:
            save(request_path, {
                "run_id": run_id, "count": count, "streams": streams,
                "sample_seconds": sample_seconds, "state": "awaiting_publishers",
            })
            until = time.monotonic() + 90
            while not ready_path.exists() and time.monotonic() < until:
                time.sleep(0.25)
            if not ready_path.exists():
                raise TimeoutError("External camera publishers did not start")
        else:
            producers = [publish(stream, sample_seconds + 20) for stream in streams]
        process = psutil.Process()
        process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        sample_started = time.monotonic()
        while time.monotonic() - sample_started < sample_seconds:
            states = [session.snapshot() for session in sessions]
            if any(state.get("state") == "error" for state in states):
                raise RuntimeError(
                    "Stream analysis failed: " + repr([state.get("error") for state in states])
                )
            if any(producer.poll() not in (None, 0) for producer in producers):
                raise RuntimeError("A camera publisher exited unexpectedly")
            sample = {
                "elapsed": round(time.monotonic() - sample_started, 2),
                "gpu": gpu_sample(),
                "cpu_system_percent": psutil.cpu_percent(interval=None),
                "cpu_worker_percent": process.cpu_percent(interval=None),
                "streams": [{key: state.get(key) for key in (
                    "state", "source_time", "analysis_time", "backlog_seconds",
                    "ingested_frames", "processed_frames", "dropped_frames",
                    "detector_frames", "event_cursor", "profile_batches",
                    "profile_total_ms",
                )} for state in states],
            }
            samples.append(sample)
            save(ROOT / "capacity_shared_progress.json", {"count": count, "sample": sample})
            time.sleep(1)
        final = [session.snapshot() for session in sessions]
        final_pool_statistics = shared_live_inference.statistics()
        gpu = [sample["gpu"] for sample in samples if sample["gpu"]]
        return {
            "count": count,
            "external_publishers": external_publishers,
            "warmup_seconds": round(warmup_seconds, 2),
            "sample_seconds": round(time.monotonic() - sample_started, 2),
            "gpu_util_mean": round(statistics.mean(row[0] for row in gpu), 2) if gpu else None,
            "gpu_util_max": max(row[0] for row in gpu) if gpu else None,
            "vram_used_max_mib": max(row[1] for row in gpu) if gpu else None,
            "vram_total_mib": gpu[0][2] if gpu else None,
            "cpu_system_mean_percent": round(statistics.mean(
                sample["cpu_system_percent"] for sample in samples
            ), 2),
            "cpu_worker_mean_percent": round(statistics.mean(
                sample["cpu_worker_percent"] for sample in samples
            ), 2),
            "pool_statistics": {
                key: final_pool_statistics[key] - initial_pool_statistics[key]
                for key in final_pool_statistics
            },
            "streams": final,
            "samples": samples,
        }
    finally:
        if external_publishers:
            save(request_path, {
                "run_id": run_id, "count": count, "streams": streams,
                "sample_seconds": sample_seconds, "state": "finished",
            })
        for session in sessions:
            session.stop()
        for producer in producers:
            if producer.poll() is None:
                producer.terminate()
            try:
                producer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                producer.kill()
        for session in sessions:
            session._thread.join(timeout=10)


def main() -> None:
    results = []
    try:
        counts = tuple(int(value) for value in os.environ.get(
            "BENCH_COUNTS", "1,3,5"
        ).split(","))
        if not counts or any(count < 1 or count > 5 for count in counts):
            raise ValueError("BENCH_COUNTS must contain values from 1 through 5")
        sample_seconds = int(os.environ.get("BENCH_SAMPLE_SECONDS", "45"))
        for count in counts:
            case = run_case(count, sample_seconds=sample_seconds)
            results.append(case)
            save(RESULT, results)
    except Exception as error:
        save(RESULT, {"partial": results, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
