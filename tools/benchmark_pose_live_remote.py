"""Isolated pose A/B hooks around the maintained RTMP live worker (cloud experiment)."""

from __future__ import annotations

import gc
import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from netcast_tennisvision.events.serve_sequence import (
    detect_serve_sequences,
    pose_schedule,
    select_pose,
)
from netcast_tennisvision.streaming import live_experiment as live
from netcast_tennisvision.vision.player_identity import select_side_players


class PoseProbe:
    def __init__(self, enabled):
        self.enabled = enabled
        self.model = YOLO(str(live.ROOT / "pose_model.pt")) if enabled else None
        if self.model is not None:
            self.model.predict(
                [np.zeros((320, 320, 3), np.uint8)] * 8, device=0, imgsz=320, verbose=False
            )
        self.poses = {}
        self.samples = 0
        self.seconds = 0.0
        self.serves = {}
        self.batches = []

    def process(self, frames, base, images, fps, corners):
        if not self.enabled:
            return
        started = time.perf_counter()
        for m in frames:
            players = select_side_players(
                {"court_corners": corners, "person_boxes": m["person_boxes"]}
            )
            m["players_world"] = [
                {"side": s, "xy": p["world"], "box": p["box"]} for s, p in players.items()
            ]
        # Align the scheduler's stride to the stream frame index as the window slides.
        stride = max(1, round(fps / 10))
        padding = base % stride
        scheduled = {(f - padding, s) for f, s in pose_schedule([{}] * padding + frames, fps)}
        crops, identities = [], []
        for absolute, image in images.items():
            f = absolute - base
            if not 0 <= f < len(frames):
                continue
            for p in frames[f]["players_world"]:
                side = p["side"]
                if (f, side) not in scheduled:
                    continue
                left, top, right, bottom = p["box"]
                h = bottom - top
                x0, y0 = max(0, int(left - h * 0.5)), max(0, int(top - h * 0.6))
                x1 = min(image.shape[1], int(right + h * 0.5))
                y1 = min(image.shape[0], int(bottom + h * 0.2))
                if x1 > x0 and y1 > y0:
                    crops.append(image[y0:y1, x0:x1].copy())
                    identities.append((absolute, side, p))
        for k in range(0, len(crops), 8):
            results = self.model.predict(
                crops[k : k + 8], device=0, imgsz=320, conf=0.35, verbose=False
            )
            for r, (f, side, p) in zip(results, identities[k : k + 8], strict=True):
                self.poses[f, side] = select_pose(
                    r.keypoints.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), p, None
                )
        self.samples += len(crops)
        self.poses = {(f, s): kp for (f, s), kp in self.poses.items() if f >= base}
        recent = {(f - base, s): kp for (f, s), kp in self.poses.items()}
        for event in detect_serve_sequences(frames, recent, fps=fps):
            absolute = base + event["frame"]
            if not any(abs(absolute - previous) < fps * 1.5 for previous in self.serves):
                self.serves[absolute] = {
                    **event,
                    "frame": absolute,
                    "decision_frame": base + event["decision_frame"],
                    "preparation_start_frame": base + event["preparation_start_frame"],
                    "release_frame": base + event["release_frame"],
                }
        self.seconds += time.perf_counter() - started

    def record(self, session, producer_started, indices, fps):
        now = time.time()
        state = session.snapshot()
        self.batches.append(
            {
                "first_frame": indices[0],
                "last_frame": indices[-1],
                "delay_ms": (now - producer_started - indices[-1] / fps) * 1000,
                "queued_frames": max(0, state.get("ingested_frames", 0) - indices[-1] - 1),
                "dropped_frames": state.get("dropped_frames", 0),
            }
        )


def install_hook():
    """Patch only this process's module, with exact anchors; deployed files stay untouched."""
    source = Path(live.__file__).read_text(encoding="utf-8")
    replacements = [
        (
            "                pending_frames.clear()",
            "                probe_images = dict(zip(pending_source_indices, pending_frames))\n"
            "                probe_indices = list(pending_source_indices)\n"
            "                pending_frames.clear()",
        ),
        (
            "                fixed_lag = max(10, round(0.35 * fps))",
            "                self.pose_probe.process(tracking_window, window_start, probe_images, fps, corners)\n"
            "                self.pose_probe.record(self, producer_started, probe_indices, fps)\n"
            "                fixed_lag = max(10, round(0.35 * fps))",
        ),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError("Live worker changed; benchmark hook requires review")
        source = source.replace(before, after)
    exec(compile(source, live.__file__, "exec"), live.__dict__)


def main():
    install_hook()
    root = live.ROOT
    corners = json.loads((root / "pose_calibration.json").read_text())["court_image_corners"]
    results = []
    for index, enabled in enumerate((False, True, True, False)):
        probe = PoseProbe(enabled)
        stream = "pose-ab-" + uuid.uuid4().hex[:12]
        session = live.LiveExperimentSession(
            None, external_stream_name=stream, fps_hint=30.00667022912, court_corners=corners
        )
        session.pose_probe = probe
        producer = None
        session.start()
        deadline = time.monotonic() + 180
        try:
            while session._thread.is_alive() and time.monotonic() < deadline:
                state = session.snapshot()
                if state.get("state") == "awaiting_stream" and producer is None:
                    producer = subprocess.Popen(
                        [
                            live._ffmpeg(),
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-re",
                            "-i",
                            str(root / "pose_source.mp4"),
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "ultrafast",
                            "-tune",
                            "zerolatency",
                            "-g",
                            "30",
                            "-bf",
                            "0",
                            "-f",
                            "flv",
                            f"rtmp://{live._media_host()}:1935/live/{stream}",
                        ]
                    )
                if producer is not None and producer.poll() is not None:
                    if producer.returncode != 0:
                        raise RuntimeError("RTMP producer failed")
                    session.finish_source()
                (root / "pose_progress.json").write_text(
                    json.dumps(
                        {
                            "run": index,
                            "pose": enabled,
                            "state": state.get("state"),
                            "analysis_time": state.get("analysis_time"),
                            "backlog_seconds": state.get("backlog_seconds"),
                        }
                    )
                )
                time.sleep(1)
            if session._thread.is_alive():
                raise TimeoutError("Live run exceeded 180 seconds")
            state = session.snapshot()
            result = {
                "pose": enabled,
                "state": state,
                "pose_seconds": probe.seconds,
                "pose_samples": probe.samples,
                "serves": list(probe.serves.values()),
                "batches": probe.batches,
            }
            results.append(result)
            (root / "pose_results.json").write_text(
                json.dumps(
                    {
                        "gpu": torch.cuda.get_device_name(),
                        "runs": results,
                        "complete": index == 3,
                    }
                )
            )
            print(index, enabled, state.get("state"), probe.samples, flush=True)
            if state.get("state") != "complete":
                raise RuntimeError(str(state.get("error", state.get("state"))))
        finally:
            if producer is not None and producer.poll() is None:
                producer.terminate()
                producer.wait(timeout=10)
            session.stop()
            session._thread.join(timeout=10)
        del session, probe
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
