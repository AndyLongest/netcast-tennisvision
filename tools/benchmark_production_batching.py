"""Compare sequential and batched production ball inference without annotations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netcast_tennisvision.vision.racketvision import detect_video_candidates  # noqa: E402

CHECKPOINT = ROOT / "models" / "racketvision_balltrack_state_v1.pt"


def compare(reference, candidate) -> dict[str, float | int | bool]:
    if len(reference) != len(candidate):
        return {
            "same_frames": False,
            "reference_frames": len(reference),
            "candidate_frames": len(candidate),
        }
    detection_mismatches = 0
    max_xy_delta = 0.0
    max_confidence_delta = 0.0
    for left, right in zip(reference, candidate, strict=True):
        if bool(left) != bool(right):
            detection_mismatches += 1
            continue
        if not left:
            continue
        max_xy_delta = max(
            max_xy_delta, abs(left[0][0] - right[0][0]), abs(left[0][1] - right[0][1])
        )
        max_confidence_delta = max(max_confidence_delta, abs(left[0][2] - right[0][2]))
    return {
        "same_frames": True,
        "detection_mismatches": detection_mismatches,
        "max_xy_delta_px": max_xy_delta,
        "max_confidence_delta": max_confidence_delta,
        "equivalent": detection_mismatches == 0
        and max_xy_delta == 0.0
        and max_confidence_delta <= 1e-6,
    }


def run(video: Path, batch_size: int, cache_dir: Path):
    started = time.perf_counter()
    result = detect_video_candidates(
        video, CHECKPOINT, cache_dir, device="cuda", batch_size=batch_size
    )
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for video in args.videos:
        print(f"\n基准逐帧模式：{video.name}", flush=True)
        sequential, sequential_seconds = run(video, 1, args.output / "cache_batch1")
        print(f"加速批处理模式：{video.name}", flush=True)
        batched, batched_seconds = run(
            video, args.batch_size, args.output / f"cache_batch{args.batch_size}"
        )
        audit = compare(sequential, batched)
        row = {
            "video": str(video.resolve()),
            "frames": len(sequential),
            "batch_size": args.batch_size,
            "sequential_seconds": round(sequential_seconds, 3),
            "batched_seconds": round(batched_seconds, 3),
            "speedup": round(sequential_seconds / max(batched_seconds, 1e-9), 3),
            **audit,
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    destination = args.output / "summary.json"
    destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果：{destination}", flush=True)


if __name__ == "__main__":
    main()
