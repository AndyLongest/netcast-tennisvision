"""Compare serial and concurrent frozen ball/person branches on one native-rate clip."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from netcast_tennisvision.vision.inference import (
    iter_batched_person_detections,
    recommended_batch_size,
)
from netcast_tennisvision.vision.racketvision import detect_video_candidates


def hash_ball_rows(rows: list[list[tuple[float, ...]]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(np.asarray(row, dtype=np.float64).tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_ball(video: Path, cache: Path) -> dict[str, object]:
    started = time.perf_counter()
    rows = detect_video_candidates(
        video,
        Path("models/racketvision_balltrack_state_v1.pt"),
        cache,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold=0.5,
        batch_size=4,
        max_candidates=8,
        alternative_threshold=0.30,
    )
    elapsed = time.perf_counter() - started
    return {
        "seconds": round(elapsed, 3),
        "frames": len(rows),
        "sha256": hash_ball_rows(rows),
    }


def run_people(video: Path, model: YOLO) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    digest = hashlib.sha256()
    frames = 0
    started = time.perf_counter()
    for _frame, result in iter_batched_person_detections(
        capture,
        model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=recommended_batch_size("cuda" if torch.cuda.is_available() else "cpu"),
        person_kwargs={"conf": 0.35, "imgsz": 960, "classes": [0]},
        prefetch=True,
    ):
        boxes = (
            result.boxes.xyxy.cpu().numpy().astype(np.float32)
            if result.boxes is not None and len(result.boxes)
            else np.zeros((0, 4), np.float32)
        )
        person = np.zeros((height, width), np.uint8)
        if result.masks is not None:
            for raw_mask in result.masks.data.cpu().numpy():
                resized = cv2.resize(
                    raw_mask, (width, height), interpolation=cv2.INTER_NEAREST
                )
                person |= (resized > 0.5).astype(np.uint8)
        person = cv2.dilate(person, np.ones((3, 3), np.uint8))
        ok, encoded = cv2.imencode(".png", person * 255)
        if not ok:
            raise RuntimeError("could not encode person mask")
        digest.update(boxes.tobytes())
        digest.update(encoded.tobytes())
        frames += 1
    capture.release()
    elapsed = time.perf_counter() - started
    return {
        "seconds": round(elapsed, 3),
        "frames": frames,
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)

    sequential_people_model = YOLO("models/yolo11n-seg.pt")
    sequential_started = time.perf_counter()
    sequential_ball = run_ball(args.video, args.directory / "serial_ball_cache")
    sequential_people = run_people(args.video, sequential_people_model)
    sequential_wall = time.perf_counter() - sequential_started

    del sequential_people_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    concurrent_people_model = YOLO("models/yolo11n-seg.pt")
    concurrent_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="detector-branch") as pool:
        ball_future = pool.submit(
            run_ball, args.video, args.directory / "parallel_ball_cache"
        )
        people_future = pool.submit(run_people, args.video, concurrent_people_model)
        concurrent_ball = ball_future.result()
        concurrent_people = people_future.result()
    concurrent_wall = time.perf_counter() - concurrent_started

    payload = {
        "video": str(args.video),
        "sequential": {
            "wall_seconds": round(sequential_wall, 3),
            "ball": sequential_ball,
            "people": sequential_people,
        },
        "concurrent": {
            "wall_seconds": round(concurrent_wall, 3),
            "ball": concurrent_ball,
            "people": concurrent_people,
        },
        "exact": {
            "ball": sequential_ball["sha256"] == concurrent_ball["sha256"],
            "people": sequential_people["sha256"] == concurrent_people["sha256"],
        },
    }
    (args.directory / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
