"""Compare optional pose off/shadow overhead using an existing isolated analysis cache."""
import argparse
import json
import pickle
from pathlib import Path

from netcast_tennisvision.vision.serve_pose import verify_serve_pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True, help="trusted local analysis.pkl only")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.analysis.open("rb") as stream:
        frames = pickle.load(stream)["frames_meta"]
    report = {}
    for mode in ("off", "shadow"):
        _, report[mode] = verify_serve_pose(args.video, frames, fps=args.fps, mode=mode)
    report["video_seconds"] = len(frames) / args.fps
    report["overhead_fraction"] = report["shadow"]["elapsed_seconds"] / report["video_seconds"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
