"""Compare the current demo outputs and runtime with the frozen cleanup baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests" / "fixtures" / "cleanup_e2e_baseline.json"
SCENE = ROOT / "data" / "outputs" / "scene3d.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="also require byte-identical scene/viewer/video artifacts",
    )
    args = parser.parse_args()
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    scene = json.loads(SCENE.read_text(encoding="utf-8"))
    current = {
        "frames": len(scene["frames"]),
        "positioned_frames": sum(frame.get("b") is not None for frame in scene["frames"]),
        "observed_frames": sum(frame.get("s") == "observed" for frame in scene["frames"]),
        "predicted_frames": sum(
            frame.get("s") == "occluded_predicted" for frame in scene["frames"]
        ),
        "bounces": len(scene["bounces"]),
        "hits": len(scene["hits"]),
        "scene_sha256": sha256(SCENE),
        "annotated_sha256": sha256(ROOT / "data" / "outputs" / "annotated_clip.mp4"),
        "viewer_sha256": sha256(ROOT / "data" / "outputs" / "rally3d.html"),
        "bounce_times": [event.get("t") for event in scene["bounces"]],
        "hit_times": [event.get("t") for event in scene["hits"]],
    }
    failures = []
    artifact_keys = {"scene_sha256", "annotated_sha256", "viewer_sha256"}
    semantic_keys = set(current) - artifact_keys
    for key in semantic_keys:
        value = current[key]
        if value != baseline[key]:
            failures.append(f"{key}: expected {baseline[key]!r}, got {value!r}")
    artifact_differences = [
        f"{key}: expected {baseline[key]!r}, got {current[key]!r}"
        for key in sorted(artifact_keys)
        if current[key] != baseline[key]
    ]
    if args.strict_artifacts:
        failures.extend(artifact_differences)
    speed_ratio = args.elapsed_seconds / float(baseline["elapsed_seconds"])
    allowed = float(baseline["maximum_allowed_regression_ratio"])
    if speed_ratio > allowed:
        failures.append(f"runtime ratio {speed_ratio:.3f} exceeds allowed {allowed:.3f}")
    print(json.dumps({
        "elapsed_seconds": args.elapsed_seconds,
        "baseline_seconds": baseline["elapsed_seconds"],
        "runtime_ratio": round(speed_ratio, 4),
        **current,
    }, ensure_ascii=False, indent=2))
    if failures:
        print("E2E REGRESSION FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if artifact_differences:
        print("ARTIFACT NOTE: semantic counters/timestamps match, but bytes differ because fresh ")
        print("court calibration, player crops or video encoding are environment-sensitive:")
        for difference in artifact_differences:
            print(f"  - {difference}")
    else:
        print("Artifact hashes are byte-identical to the frozen environment.")
    print("E2E REGRESSION PASSED: semantic outputs and runtime are within the frozen envelope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
