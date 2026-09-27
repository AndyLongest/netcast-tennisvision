"""Extract a framework-neutral RacketPose timeline from a tennis video."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from netcast_tennisvision.vision.racket_pose import (
    KEYPOINT_NAMES,
    OpenMMLabRacketPose,
    RacketFrame,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--bbox-threshold", type=float, default=0.3)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least one")
    if args.start_frame < 0:
        parser.error("--start-frame cannot be negative")
    if args.end_frame is not None and args.end_frame <= args.start_frame:
        parser.error("--end-frame must be greater than --start-frame")

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise SystemExit("video reports no valid frame rate")

    estimator = OpenMMLabRacketPose(
        device=args.device,
        bbox_threshold=args.bbox_threshold,
    )
    load_started = time.perf_counter()
    estimator.load()
    load_seconds = time.perf_counter() - load_started
    frames: list[dict[str, object]] = []
    inference_seconds = 0.0
    frame_index = args.start_frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    try:
        while True:
            if args.end_frame is not None and frame_index >= args.end_frame:
                break
            ok, image = capture.read()
            if not ok:
                break
            if (frame_index - args.start_frame) % args.stride == 0:
                inference_started = time.perf_counter()
                rackets = estimator.infer(image)
                inference_seconds += time.perf_counter() - inference_started
                frame = RacketFrame(
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps,
                    rackets=rackets,
                )
                frames.append(frame.to_dict())
            frame_index += 1
    finally:
        capture.release()

    payload = {
        "schema_version": 1,
        "source_video": str(args.video.resolve()),
        "fps": fps,
        "source_frame_range": [args.start_frame, frame_index],
        "stride": args.stride,
        "coordinate_system": "original_frame_pixels_xy",
        "keypoint_order": list(KEYPOINT_NAMES),
        "timing": {
            "model_load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "sampled_frames": len(frames),
        },
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(
        f"wrote {len(frames)} sampled frames to {args.output}; "
        f"load={load_seconds:.3f}s inference={inference_seconds:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
