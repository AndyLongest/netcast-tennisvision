"""Local source-paced A/B compute-latency experiment; excludes network/browser delay.

No cached ball/pose inference. Only calibration and static background are prepared
before the stream, like live_experiment. Pose sees arrived frames only. This isolated
adapter is not a deployed live worker and does not test RTMP/WebRTC transport.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from netcast_tennisvision.events.landing_event_detector import detect_landing_impulses
from netcast_tennisvision.events.serve_sequence import (
    detect_serve_sequences,
    pose_schedule,
    select_pose,
)
from netcast_tennisvision.paths import REPOSITORY_ROOT as ROOT
from netcast_tennisvision.streaming.live_experiment import _ffmpeg, _person_boxes
from netcast_tennisvision.tracking.world_tracker import track_ball_persistent
from netcast_tennisvision.vision.player_identity import select_side_players
from netcast_tennisvision.vision.racketvision import (
    DEFAULT_THRESHOLD,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    SEQUENCE_LENGTH,
    _decode_candidates,
    _median_background,
    load_model,
)


def run(args, enabled, iteration):
    metadata = cv2.VideoCapture(str(args.video))
    width, height, fps, count = (
        int(metadata.get(3)),
        int(metadata.get(4)),
        metadata.get(5),
        int(metadata.get(7)),
    )
    metadata.release()
    scene = json.loads(args.calibration.read_text(encoding="utf-8"))
    corners = np.float32(scene["court_image_corners"]) * [width, height]
    corners = corners.astype(np.float32)
    inverse = cv2.getPerspectiveTransform(
        corners, np.float32([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]])
    )
    background = np.moveaxis(_median_background(args.video, count).astype(np.float32) / 255, -1, 0)
    ball = load_model(ROOT / "models/racketvision_balltrack_state_v1.pt", torch.device("cuda"))
    person = YOLO(str(ROOT / "models/yolo11n-seg.pt"))
    pose = YOLO(str(ROOT / "models/yolo11n-pose.pt")) if enabled else None
    zeros = torch.zeros((16, 3 * (SEQUENCE_LENGTH + 1), MODEL_HEIGHT, MODEL_WIDTH), device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        ball(zeros)
    person.predict(np.zeros((height, width, 3), np.uint8), device=0, imgsz=640, verbose=False)
    if pose:
        pose.predict([np.zeros((320, 320, 3), np.uint8)] * 8, device=0, imgsz=320, verbose=False)
    torch.cuda.synchronize()
    del zeros
    ff = subprocess.Popen(
        [
            _ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-ss",
            str(args.start),
            "-i",
            str(args.video),
            "-t",
            str(args.seconds),
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    q = queue.Queue(maxsize=90)
    shared = {"received": 0, "dropped": 0, "first": None}

    def reader():
        size = width * height * 3
        try:
            while True:
                payload = bytearray()
                while len(payload) < size:
                    part = ff.stdout.read(size - len(payload))
                    if not part:
                        break
                    payload.extend(part)
                if len(payload) != size:
                    break
                now = time.perf_counter()
                if shared["first"] is None:
                    shared["first"] = now
                item = (
                    shared["received"],
                    now,
                    np.frombuffer(payload, np.uint8).reshape(height, width, 3),
                )
                shared["received"] += 1
                try:
                    q.put_nowait(item)
                except queue.Full:
                    q.get_nowait()
                    shared["dropped"] += 1
                    q.put_nowait(item)
        finally:
            q.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    window = deque(maxlen=SEQUENCE_LENGTH)
    pending = []
    inputs = []
    history = []
    poses = {}
    records = []
    last_boxes = np.empty((0, 4), np.float32)
    pose_seconds = 0.0
    pose_samples = 0
    serve_frames = set()

    def process():
        nonlocal last_boxes, pose_seconds, pose_samples
        if not pending:
            return
        tensor = torch.from_numpy(np.stack(inputs)).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            maps = ball(tensor).float().cpu().numpy()
        offsets = [i for i, (f, _, _) in enumerate(pending) if f % 4 == 0]
        if not offsets and not len(last_boxes):
            offsets = [0]
        detections = (
            person.predict(
                [pending[i][2] for i in offsets],
                device=0,
                imgsz=640,
                conf=0.25,
                classes=[0],
                verbose=False,
            )
            if offsets
            else []
        )
        boxes = {i: _person_boxes(r) for i, r in zip(offsets, detections, strict=True)}
        for i, ((f, _arrival, _image), heat) in enumerate(zip(pending, maps, strict=True)):
            if i in boxes:
                last_boxes = boxes[i]
            while len(history) <= f:
                history.append(
                    {
                        "candidates": [],
                        "person_boxes": last_boxes.copy(),
                        "corners": corners,
                        "M": np.linalg.inv(inverse),
                        "M_inv": inverse,
                        "is_court": True,
                    }
                )
            history[f].update(
                candidates=_decode_candidates(
                    heat,
                    DEFAULT_THRESHOLD,
                    width / MODEL_WIDTH,
                    height / MODEL_HEIGHT,
                    max_candidates=1,
                )
            )
        base = max(0, len(history) - round(8 * fps))
        tracked = [dict(m) for m in history[base:]]
        track_ball_persistent(
            tracked,
            fps=fps,
            spatial=min(width / 1280, height / 720),
            speed_scale=min(width / 1280, height / 720),
            frame_size=(width, height),
            play_mode="singles",
        )
        detect_landing_impulses(
            tracked, radius=8, min_score=0.52, min_gap_frames=max(10, round(0.38 * fps))
        )
        for j, m in enumerate(tracked):
            selected = select_side_players(
                {"court_corners": corners, "person_boxes": m["person_boxes"]}
            )
            m["players_world"] = [
                {"side": s, "xy": p["world"], "box": p["box"]} for s, p in selected.items()
            ]
            history[base + j] = m
        if pose:
            began_pose = time.perf_counter()
            scheduled = set(pose_schedule(history, fps))
            batch = []
            identities = []
            for f, _, image in pending:
                for p in history[f]["players_world"]:
                    side = p["side"]
                    if (f, side) not in scheduled:
                        continue
                    left, top, right, bottom = p["box"]
                    h = bottom - top
                    x0, y0 = max(0, int(left - h * 0.5)), max(0, int(top - h * 0.6))
                    x1, y1 = min(width, int(right + h * 0.5)), min(height, int(bottom + h * 0.2))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    batch.append(image[y0:y1, x0:x1].copy())
                    identities.append((f, side, p))
            for k in range(0, len(batch), 8):
                results = pose.predict(
                    batch[k : k + 8], device=0, imgsz=320, conf=0.35, verbose=False
                )
                for r, (f, side, p) in zip(results, identities[k : k + 8], strict=True):
                    poses[f, side] = select_pose(
                        r.keypoints.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), p, None
                    )
            pose_samples += len(batch)
            recent = {(f - base, s): kp for (f, s), kp in poses.items() if f >= base}
            for event in detect_serve_sequences(tracked, recent, fps=fps):
                serve_frames.add(event["frame"] + base)
            pose_seconds += time.perf_counter() - began_pose
        finished = time.perf_counter()
        for f, arrival, _ in pending:
            records.append(
                {
                    "frame": f,
                    "input_to_result_ms": 1000 * (finished - arrival),
                    "source_to_result_ms": 1000 * (finished - shared["first"] - f / fps),
                    "backlog_seconds": max(0, (shared["received"] - pending[-1][0] - 1) / fps),
                }
            )
        if len(records) % 160 == 0:
            print(
                iteration,
                enabled,
                len(records),
                round(records[-1]["source_to_result_ms"]),
                flush=True,
            )
        pending.clear()
        inputs.clear()

    try:
        while True:
            item = q.get()
            if item is None:
                break
            f, arrival, image = item
            small = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))
            window.append(np.moveaxis(small.astype(np.float32) / 255, -1, 0))
            seq = list(window)
            while len(seq) < SEQUENCE_LENGTH:
                seq.insert(0, seq[0])
            inputs.append(np.concatenate([background, *seq], axis=0))
            pending.append(item)
            if len(pending) == 16:
                process()
        process()
        thread.join(timeout=5)
        ff.wait(timeout=5)
    finally:
        if ff.poll() is None:
            ff.kill()
            ff.wait()
    steady = [r for r in records if r["frame"] >= round(3 * fps)]

    def metric(name):
        values = [r[name] for r in steady]
        return {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": max(values),
        }

    result = {
        "enabled": enabled,
        "iteration": iteration,
        "fps": fps,
        **shared,
        "processed": len(records),
        "input_to_result_ms": metric("input_to_result_ms"),
        "source_to_result_ms": metric("source_to_result_ms"),
        "backlog_seconds": metric("backlog_seconds"),
        "pose_seconds": pose_seconds,
        "pose_samples": pose_samples,
        "serve_frames": sorted(serve_frames),
        "records": records,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"run_{iteration}_{enabled}.json").write_text(json.dumps(result), encoding="utf-8")
    print(
        "DONE",
        iteration,
        enabled,
        {k: v for k, v in result.items() if k not in ("records", "first")},
        flush=True,
    )
    ball = person = pose = None
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--start", type=float, default=50)
    p.add_argument("--seconds", type=float, default=25)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--order", nargs="+", choices=("off", "on"), default=["off", "on", "on", "off"])
    args = p.parse_args()
    for iteration, mode in enumerate(args.order):
        run(args, mode == "on", iteration)


if __name__ == "__main__":
    main()
