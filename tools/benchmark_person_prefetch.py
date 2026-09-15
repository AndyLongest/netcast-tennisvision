"""Benchmark full-rate person inference plus production mask materialization."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from netcast_tennisvision.vision.inference import (
    iter_batched_person_detections,
    recommended_batch_size,
)


def run_once(
    video: Path, model: YOLO, *, device: str, prefetch: bool, imgsz: int,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    batch_size = recommended_batch_size(device)
    digest = hashlib.sha256()
    frames = 0
    started = time.perf_counter()
    for _frame, result in iter_batched_person_detections(
        capture,
        model,
        device=device,
        batch_size=batch_size,
        person_kwargs={"conf": 0.35, "imgsz": imgsz, "classes": [0]},
        prefetch=prefetch,
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
        "prefetch": prefetch,
        "imgsz": imgsz,
        "frames": frames,
        "seconds": round(elapsed, 3),
        "fps": round(frames / elapsed, 3),
        "result_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--runs", type=int, default=4)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("models/yolo11n-seg.pt")
    runs = [
        run_once(args.video, model, device=device, prefetch=True, imgsz=args.imgsz)
        for _ in range(args.runs)
    ]
    payload = {"device": device, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
