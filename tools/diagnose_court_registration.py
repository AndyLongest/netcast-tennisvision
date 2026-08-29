"""Run only the notebook's court-registration stage on a clip.

This avoids loading the ball/person models and is useful when a new camera view fails
before tracking begins.  It intentionally executes the notebook definitions so the
diagnostic cannot drift into a second implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--skip-global-fit", action="store_true")
    parser.add_argument("--frame-seconds", type=float)
    parser.add_argument("--sample-step", type=int, default=20)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    root = args.project.resolve()
    sys.path.insert(0, str(root / "src"))
    clip = args.clip.resolve()
    target = root / "data" / "clip.mp4"
    if not clip.exists():
        raise FileNotFoundError(clip)

    old_clip = None
    if target.exists() and not os.path.samefile(clip, target):
        fd, name = tempfile.mkstemp(suffix=".mp4", dir=os.environ.get("TEMP"))
        os.close(fd)
        old_clip = Path(name)
        shutil.copy2(target, old_clip)
        shutil.copy2(clip, target)

    try:
        notebook_path = args.notebook.resolve() if args.notebook else root / "notebooks" / "tennis_detection.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        namespace: dict[str, object] = {"__name__": "__court_diagnostic__"}
        for index in (2, 6):
            source = "".join(notebook["cells"][index]["source"])
            exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)

        if args.frame_seconds is not None:
            cv2 = namespace["cv2"]
            cap = cv2.VideoCapture(str(target))
            cap.set(cv2.CAP_PROP_POS_MSEC, args.frame_seconds * 1000)
            ok, frame = cap.read()
            cap.release()
            if not ok:
                raise RuntimeError(f"could not read frame at {args.frame_seconds}s")
            quad, info = namespace["detect_court"](frame)
            print("single-frame info:", info)
            np = namespace["np"]
            print("single-frame corners:", None if quad is None else np.round(quad, 1).tolist())
            if args.skip_global_fit:
                return

        source = "".join(notebook["cells"][8]["source"])
        if args.sample_step != 20:
            source = source.replace("frame_idx % 20 == 0", f"frame_idx % {args.sample_step} == 0")
        if args.max_frames is not None:
            source = source.replace(
                "    ok, frame = cap.read()",
                f"    if frame_idx >= {args.max_frames}:\n        break\n    ok, frame = cap.read()",
            )
        if args.start_frame:
            source = source.replace(
                "frame_idx = 0",
                f"frame_idx = {args.start_frame}\ncap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)",
            )
        if args.skip_global_fit:
            marker = "# The consensus is only as good"
            report = (
                'print("detections:", len(detections))\n'
                'print("consensus inliers:", len(inliers))\n'
                'print("seed corners:", np.round(SEED_CORNERS, 1).tolist())\n'
                "raise SystemExit(0)\n"
            )
            source = source.replace(marker, report + marker)
        exec(compile(source, "notebook-cell-8", "exec"), namespace)
        if not args.skip_global_fit:
            source = "".join(notebook["cells"][10]["source"])
            exec(compile(source, "notebook-cell-10", "exec"), namespace)
    finally:
        if old_clip is not None:
            shutil.copy2(old_clip, target)
            old_clip.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
