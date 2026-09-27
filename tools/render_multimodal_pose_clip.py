"""Render existing ball/court overlays plus human and racket keypoints."""

from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

RACKET_EDGES = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
    ("top", "bottom"),
    ("left", "right"),
    ("bottom", "handle"),
)
COCO_EDGES = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--rackets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, default=Path("models/yolo11n-pose.pt"))
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    timeline = json.loads(args.rackets.read_text(encoding="utf-8"))
    tracks = build_racket_tracks(timeline["frames"])
    pose_model = YOLO(str(args.pose_model))

    source = cv2.VideoCapture(str(args.source))
    annotated = cv2.VideoCapture(str(args.annotated))
    fps = float(source.get(cv2.CAP_PROP_FPS))
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    annotated.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="racket-review-", dir=args.output.parent) as temp_dir:
        silent_path = Path(temp_dir) / "silent.mp4"
        writer = cv2.VideoWriter(
            str(silent_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            for frame_index in range(args.start_frame, args.end_frame):
                source_ok, source_frame = source.read()
                annotated_ok, canvas = annotated.read()
                if not source_ok or not annotated_ok:
                    break
                result = pose_model.predict(
                    source_frame,
                    classes=[0],
                    conf=0.2,
                    imgsz=640,
                    device=args.device,
                    verbose=False,
                )[0]
                draw_human_poses(canvas, result)
                far_left, far_top, far_right, far_bottom = 530, 40, 760, 190
                far_crop = source_frame[far_top:far_bottom, far_left:far_right]
                far_result = pose_model.predict(
                    far_crop,
                    classes=[0],
                    conf=0.05,
                    imgsz=640,
                    device=args.device,
                    verbose=False,
                )[0]
                draw_human_poses(canvas, far_result, offset=(far_left, far_top))
                for side, color in (("far", (255, 180, 0)), ("near", (0, 180, 255))):
                    racket = interpolate_track(tracks[side], frame_index)
                    if racket is not None:
                        draw_racket(canvas, racket, color, side)
                draw_legend(canvas, frame_index, fps)
                writer.write(canvas)
        finally:
            writer.release()
            source.release()
            annotated.release()

        duration = (args.end_frame - args.start_frame) / fps
        start_s = args.start_frame / fps
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(silent_path),
            "-ss",
            f"{start_s:.6f}",
            "-t",
            f"{duration:.6f}",
            "-i",
            str(args.source),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            str(args.output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            details = error.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg mux failed:\n{details}") from error
    print(f"rendered {args.output}")
    return 0


def build_racket_tracks(frames: list[dict[str, Any]]) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    tracks: dict[str, list[tuple[int, dict[str, Any]]]] = {"near": [], "far": []}
    for frame in frames:
        frame_index = int(frame["frame_index"])
        buckets: dict[str, list[dict[str, Any]]] = {"near": [], "far": []}
        for racket in frame["rackets"]:
            x1, y1, x2, y2 = racket["bbox_xyxy"]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            if center_x > 1220:
                continue
            side = "near" if center_y > 250 else "far"
            buckets[side].append(racket)
        for side, candidates in buckets.items():
            if not candidates:
                continue
            best = max(candidates, key=lambda item: float(item["bbox_confidence"]))
            threshold = 0.1 if side == "near" else 0.12
            if float(best["bbox_confidence"]) >= threshold:
                tracks[side].append((frame_index, best))
    return tracks


def interpolate_track(
    track: list[tuple[int, dict[str, Any]]], frame_index: int
) -> dict[str, Any] | None:
    if not track:
        return None
    indexes = [item[0] for item in track]
    position = bisect.bisect_left(indexes, frame_index)
    if position < len(track) and track[position][0] == frame_index:
        return track[position][1]
    if position == 0 or position == len(track):
        return None
    before_index, before = track[position - 1]
    after_index, after = track[position]
    if after_index - before_index > 20:
        return None
    amount = (frame_index - before_index) / (after_index - before_index)
    return interpolate_racket(before, after, amount)


def interpolate_racket(
    before: dict[str, Any], after: dict[str, Any], amount: float
) -> dict[str, Any]:
    def blend(left: float, right: float) -> float:
        return float(left) + amount * (float(right) - float(left))

    return {
        "bbox_xyxy": [blend(a, b) for a, b in zip(before["bbox_xyxy"], after["bbox_xyxy"], strict=True)],
        "bbox_confidence": min(before["bbox_confidence"], after["bbox_confidence"]),
        "keypoints": {
            name: {
                key: blend(before["keypoints"][name][key], after["keypoints"][name][key])
                for key in ("x", "y", "confidence")
            }
            for name in before["keypoints"]
        },
    }


def draw_human_poses(
    canvas: np.ndarray, result: Any, offset: tuple[int, int] = (0, 0)
) -> None:
    if result.keypoints is None:
        return
    points = result.keypoints.xy.detach().cpu().numpy()
    confidences = result.keypoints.conf.detach().cpu().numpy()
    for person_points, person_confidences in zip(points, confidences, strict=True):
        person_points = person_points + np.asarray(offset)
        for start, end in COCO_EDGES:
            if person_confidences[start] < 0.25 or person_confidences[end] < 0.25:
                continue
            start_point = tuple(np.rint(person_points[start]).astype(int))
            end_point = tuple(np.rint(person_points[end]).astype(int))
            cv2.line(canvas, start_point, end_point, (80, 255, 80), 2, cv2.LINE_AA)
        for point, confidence in zip(person_points, person_confidences, strict=True):
            if confidence >= 0.25:
                cv2.circle(canvas, tuple(np.rint(point).astype(int)), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_racket(canvas: np.ndarray, racket: dict[str, Any], color: tuple[int, int, int], side: str) -> None:
    points = racket["keypoints"]
    for start, end in RACKET_EDGES:
        if min(points[start]["confidence"], points[end]["confidence"]) < 0.2:
            continue
        first = (round(points[start]["x"]), round(points[start]["y"]))
        second = (round(points[end]["x"]), round(points[end]["y"]))
        edge_color = (40, 40, 255) if {start, end} == {"bottom", "handle"} else color
        cv2.line(canvas, first, second, edge_color, 3, cv2.LINE_AA)
    for name, point in points.items():
        if point["confidence"] >= 0.2:
            center = (round(point["x"]), round(point["y"]))
            cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)
            if name == "handle":
                cv2.circle(canvas, center, 7, (40, 40, 255), 2, cv2.LINE_AA)
    x1, y1, x2, _ = (round(value) for value in racket["bbox_xyxy"])
    cv2.rectangle(canvas, (x1, y1), (x2, round(racket["bbox_xyxy"][3])), color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"RACKET {side.upper()}", (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_legend(canvas: np.ndarray, frame_index: int, fps: float) -> None:
    overlay = canvas.copy()
    cv2.rectangle(overlay, (18, 18), (355, 112), (12, 18, 20), -1)
    cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
    cv2.putText(canvas, "BALL: magenta  |  HUMAN: green", (32, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "RACKET: orange/blue  |  HANDLE: red", (32, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"frame {frame_index}   t={frame_index / fps:.2f}s", (32, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 220, 225), 1, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
