"""Blind evaluation of the public RacketVision BallTrack checkpoint on videos.

This utility deliberately performs no fine-tuning and uses one fixed threshold for
every input video. It creates an annotated side-by-side video plus JSON/CSV data.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "racketvision_balltrack_state_v1.pt"
MODEL_WIDTH = 512
MODEL_HEIGHT = 288
SEQUENCE_LENGTH = 4


def load_model(checkpoint: Path, device: torch.device):
    from racketvision_runtime import RacketVisionBallTrack

    model = RacketVisionBallTrack()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    state = {(key[7:] if key.startswith("module.") else key): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def video_metadata(video: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return width, height, fps, frames


def compute_background(video: Path, total_frames: int, samples: int = 180) -> np.ndarray:
    """Match the public pipeline: pixel-wise median of uniformly sampled frames."""
    capture = cv2.VideoCapture(str(video))
    indices = np.linspace(0, max(total_frames - 1, 0), min(samples, total_frames), dtype=int)
    resized: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            resized.append(cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT)))
    capture.release()
    if not resized:
        raise RuntimeError(f"Could not sample frames from {video}")
    return np.median(np.stack(resized), axis=0).astype(np.uint8)


def decode_heatmap(heatmap: np.ndarray, threshold: float) -> tuple[float, float, float]:
    mask = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    confidence = float(np.mean(heatmap[y : y + height, x : x + width]))
    return x + width / 2.0, y + height / 2.0, confidence


def prepare_input(frames: deque[np.ndarray], background: np.ndarray, device: torch.device) -> torch.Tensor:
    sequence = list(frames)
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.insert(0, sequence[0])
    inputs = [background, *sequence[-SEQUENCE_LENGTH:]]
    channels = [np.moveaxis(frame.astype(np.float32) / 255.0, -1, 0) for frame in inputs]
    tensor = torch.from_numpy(np.concatenate(channels, axis=0)[None])
    return tensor.to(device, non_blocking=True)


def draw_result(frame: np.ndarray, point: tuple[int, int] | None, confidence: float, trail: deque):
    canvas = frame.copy()
    if len(trail) > 1:
        for index in range(1, len(trail)):
            alpha = index / len(trail)
            color = (int(120 + 90 * alpha), int(40 + 80 * alpha), 255)
            cv2.line(canvas, trail[index - 1], trail[index], color, max(1, int(1 + 2 * alpha)), cv2.LINE_AA)
    if point is not None:
        cv2.circle(canvas, point, 11, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.circle(canvas, point, 8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, point, 4, (85, 255, 185), -1, cv2.LINE_AA)
        status = f"BALL  {confidence:.3f}"
        color = (85, 255, 185)
    else:
        status = "NO DETECTION"
        color = (150, 150, 160)
    cv2.rectangle(canvas, (20, 20), (285, 68), (18, 17, 28), -1)
    cv2.putText(canvas, "RacketVision  MS-TrackNetV3", (34, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 242), 1, cv2.LINE_AA)
    cv2.putText(canvas, status, (34, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return canvas


def make_comparison(left: np.ndarray, right: np.ndarray, output_width: int = 1920) -> np.ndarray:
    panel_width = output_width // 2
    panel_height = round(panel_width * left.shape[0] / left.shape[1])
    left_panel = cv2.resize(left, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    right_panel = cv2.resize(right, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(left_panel, (20, 20), (160, 56), (18, 17, 28), -1)
    cv2.putText(left_panel, "ORIGINAL", (34, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([left_panel, right_panel], axis=1)


def evaluate_video(
    model,
    device: torch.device,
    video: Path,
    output_dir: Path,
    threshold: float,
    max_frames: int | None,
) -> dict:
    width, height, fps, total_frames = video_metadata(video)
    if max_frames:
        total_frames = min(total_frames, max_frames)
    background = compute_background(video, total_frames)
    background_model = cv2.resize(background, (MODEL_WIDTH, MODEL_HEIGHT))

    capture = cv2.VideoCapture(str(video))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_video = output_dir / f"{video.stem}_racketvision_raw.mp4"
    comparison_video = output_dir / f"{video.stem}_racketvision_comparison_raw.mp4"
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    panel_height = round(960 * height / width)
    comparison_writer = cv2.VideoWriter(
        str(comparison_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, panel_height)
    )

    window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    trail: deque[tuple[int, int]] = deque(maxlen=10)
    records: list[dict] = []
    started = time.perf_counter()
    frame_index = 0

    while frame_index < total_frames:
        ok, original = capture.read()
        if not ok:
            break
        resized = cv2.resize(original, (MODEL_WIDTH, MODEL_HEIGHT))
        window.append(resized)
        tensor = prepare_input(window, background_model, device)
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    prediction, _, _ = model.forward(frames=tensor)
            else:
                prediction, _, _ = model.forward(frames=tensor)
        heatmap = prediction[0, 0].float().cpu().numpy()
        model_x, model_y, confidence = decode_heatmap(heatmap, threshold)
        point = None
        if confidence > 0:
            point = (int(round(model_x * width / MODEL_WIDTH)), int(round(model_y * height / MODEL_HEIGHT)))
            trail.append(point)
        annotated = draw_result(original, point, confidence, trail)
        writer.write(annotated)
        comparison_writer.write(make_comparison(original, annotated))
        records.append(
            {
                "frame": frame_index,
                "time_seconds": round(frame_index / fps, 4),
                "x": point[0] if point else None,
                "y": point[1] if point else None,
                "confidence": round(confidence, 6),
                "detected": point is not None,
            }
        )
        frame_index += 1
        if frame_index % 100 == 0:
            elapsed = time.perf_counter() - started
            print(f"{video.name}: {frame_index}/{total_frames} ({frame_index / elapsed:.1f} FPS)", flush=True)

    capture.release()
    writer.release()
    comparison_writer.release()
    elapsed = time.perf_counter() - started

    csv_path = output_dir / f"{video.stem}_racketvision.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer_csv.writeheader()
        writer_csv.writerows(records)

    detected = [record for record in records if record["detected"]]
    summary = {
        "video": str(video.resolve()),
        "architecture": "RacketVision MS-TrackNetV3 (4 frames + median background)",
        "checkpoint": str(DEFAULT_CHECKPOINT.resolve()),
        "threshold": threshold,
        "input_fps": fps,
        "frames": len(records),
        "detected_frames": len(detected),
        "detection_coverage": round(len(detected) / max(len(records), 1), 6),
        "confidence_mean_detected": round(float(np.mean([r["confidence"] for r in detected])), 6) if detected else 0.0,
        "confidence_median_detected": round(float(np.median([r["confidence"] for r in detected])), 6) if detected else 0.0,
        "processing_fps": round(len(records) / max(elapsed, 1e-9), 3),
        "raw_video": str(raw_video.resolve()),
        "comparison_video": str(comparison_video.resolve()),
        "csv": str(csv_path.resolve()),
    }
    summary_path = output_dir / f"{video.stem}_racketvision_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "evaluations" / "racketvision"
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    summaries = [
        evaluate_video(model, device, Path(video), args.output_dir, args.threshold, args.max_frames)
        for video in args.videos
    ]
    (args.output_dir / "racketvision_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
