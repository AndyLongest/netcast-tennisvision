"""Isolated full-rate far-court crop experiment; never changes product data."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from netcast_tennisvision.vision.racketvision import detect_video_candidates

ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(video, reference, destination, cache, roi_video=None, roi_box=None, *, low_view=False):
    destination.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    ok, first = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read source")
    h, w = first.shape[:2]
    corners = np.float32(reference["court_image_corners"]) * [w, h]

    def calibration(*_):
        return dict(
            corners=corners,
            frame=first,
            plate=cv2.cvtColor(first, cv2.COLOR_BGR2GRAY),
            profile_id="experiment-locked-report",
            inliers=0,
        )

    ns = {
        "__name__": "__main__",
        "_pipeline_find_camera_calibration": calibration,
        "_pipeline_manual_court_calibration": lambda *args, **kwargs: corners.copy(),
        "_pipeline_video_output_args": lambda: [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
        ],
        "_racketvision_inference_progress": lambda done, total: print(
            f"BALL {done}/{total}", flush=True
        ),
    }
    notebook = json.loads((ROOT / "notebooks/tennis_detection.ipynb").read_text(encoding="utf-8"))
    metrics = {}
    started = time.perf_counter()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or cell["id"] == "3aa72724":
            continue
        source = "".join(cell["source"])
        if cell["id"] == "0e85d582":
            source = source.replace(
                'DATA_DIR = ROOT / "data"', f"DATA_DIR = Path({str(destination / 'runtime')!r})"
            )
            source = source.replace(
                'CLIP_PATH = DATA_DIR / "clip.mp4"', f"CLIP_PATH = Path({str(video)!r})"
            )
            source = source.replace(
                'OUTPUT_DIR = DATA_DIR / "outputs"', f"OUTPUT_DIR = Path({str(destination)!r})"
            )
            source = source.replace(
                'CACHE_DIR = DATA_DIR / "cache"', f"CACHE_DIR = Path({str(cache)!r})"
            )
        if cell["id"] == "1d6d5185":
            start = source.index("    # Stable display trail:")
            end = source.index('    if meta["ball_px"]:', start)
            source = source[:start] + source[end:]
            if low_view:
                source = source.replace('    encoder.stdin.write(frame.tobytes())', '''    active_inferred = [e for e in occluded_landings if e['decision_frame'] <= idx < e['decision_frame'] + FPS*.7]
    if active_inferred:
        cv2.rectangle(frame, (8, 38), (min(W-8, 650), 72), (30, 30, 30), -1)
        cv2.putText(frame, 'INFERRED BOUNCE - location unknown', (16, 62), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'LOW VIEW EXPERIMENT | far crop + occlusion evidence', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
    encoder.stdin.write(frame.tobytes())''')
        print("CELL", cell["id"], flush=True)
        exec(compile(source, cell["id"], "exec"), ns)
        if cell["id"] == "1551a712" and roi_video:
            t = time.perf_counter()
            rows = detect_video_candidates(
                roi_video,
                ROOT / "models/racketvision_balltrack_state_v1.pt",
                cache / "roi",
                device="cuda",
                max_candidates=1,
                progress=ns["_racketvision_inference_progress"],
            )
            assert len(rows) == len(ns["frames_meta"]), (len(rows), len(ns["frames_meta"]))
            world = np.float32([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]])
            inverse = cv2.getPerspectiveTransform(corners.astype(np.float32), world)

            def far(point, inverse=inverse):
                v = inverse @ np.array([point[0], point[1], 1.0])
                xy = v[:2] / v[2]
                return -1 <= xy[0] <= 11.97 and 11.885 <= xy[1] <= 26

            changed = 0
            for meta, row in zip(ns["frames_meta"], rows, strict=False):
                primary = [c[:5] for c in meta["candidates"] if len(c) < 6 or c[5] >= 0.5]
                if not row:
                    continue
                x, y, confidence, bw, bh = row[0]
                candidate = (x + roi_box[0], y + roi_box[1], confidence, bw, bh)
                if low_view:
                    # Crop proposals supplement missing full-frame detections. Keep
                    # detector-backed full-frame observations when they disagree.
                    accept = not primary or (
                        np.linalg.norm(np.asarray(candidate[:2])-primary[0][:2]) <= w*.012
                        and candidate[2] > primary[0][2])
                else:
                    accept = far(candidate) and (not primary or far(primary[0]))
                if accept:
                    meta["candidates"] = [candidate]
                    changed += 1
            metrics.update(
                roi_seconds=time.perf_counter() - t,
                replaced_candidate_frames=changed,
                roi_box=roi_box,
            )
            print("ROI", metrics, flush=True)
        if cell["id"] == "9338f9cf" and low_view:
            from netcast_tennisvision.events.occluded_landing import infer_occluded_landings
            ns["occluded_landings"] = infer_occluded_landings(
                ns["frames_meta"], ns["events"], ns["bounces"], fps=ns["FPS"], width=w, roi_box=roi_box)
            metrics["occluded_events"] = ns["occluded_landings"]
            print("OCCLUDED", len(ns["occluded_landings"]), flush=True)
        if cell["id"] == "72c069b4" and low_view:
            ns["scene"]["experimental_occluded_landings"] = ns["occluded_landings"]
            (destination / "scene3d.json").write_text(json.dumps(ns["scene"]), encoding="utf-8")
        if cell["id"] == "a782a51e":
            with (destination / "analysis.pkl").open("wb") as f:
                pickle.dump({k: ns[k] for k in ("frames_meta", "events", "bounces", "rallies")}, f)
        if cell["id"] == "1d6d5185":
            break
    metrics["total_seconds"] = time.perf_counter() - started
    (destination / "timing.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--low-view", action="store_true", help="Isolated low-view fusion and unpositioned bounce inference")
    a = p.parse_args()
    os.chdir(ROOT)
    os.environ["MPLBACKEND"] = "Agg"
    ff = sorted(
        (Path.home() / "AppData/Local/Microsoft/WinGet/Packages").glob(
            "Gyan.FFmpeg.Shared_*/ffmpeg-*/bin/ffmpeg.exe"
        )
    )
    if ff:
        os.environ["PATH"] = str(ff[-1].parent) + os.pathsep + os.environ["PATH"]
    import IPython.display

    IPython.display.display = lambda *args, **kwargs: None
    random.seed(20260824)
    np.random.seed(20260824)
    torch.manual_seed(20260824)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    a.out.mkdir(parents=True, exist_ok=True)
    reference = json.loads(a.reference.read_text(encoding="utf-8"))
    if a.low_view:
        from netcast_tennisvision.vision.viewpoint import classify_viewpoint
        probe = cv2.VideoCapture(str(a.video))
        category = classify_viewpoint(reference["court_image_corners"], probe.get(3), probe.get(4))["category"]
        probe.release()
        if category != "low":
            raise ValueError("Low-view experiment requires baseline height ratio <= 30%")
    run_pipeline(a.video, reference, a.out / "baseline", a.out / "cache")
    cap = cv2.VideoCapture(str(a.video))
    w = int(cap.get(3))
    h = int(cap.get(4))
    cap.release()
    corners = np.float32(reference["court_image_corners"]) * [w, h]
    H = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]]), corners.astype(np.float32)
    )
    points = cv2.perspectiveTransform(
        np.float32([[[0, 11.885], [10.97, 11.885], [10.97, 23.77], [0, 23.77]]]), H
    )[0]
    cw = min(w, int((points[:, 0].max() - points[:, 0].min()) * 1.16) // 16 * 16)
    ch = (cw * 9 // 16) // 2 * 2
    x = max(0, min(w - cw, int(points[:, 0].mean() - cw / 2))) // 2 * 2
    y = max(0, min(h - ch, int((points[:, 1].min() + points[:, 1].max()) / 2 - ch / 2))) // 2 * 2
    roi = a.out / "far_crop.mkv"
    if not roi.exists():
        subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-y",
                "-i",
                str(a.video),
                "-vf",
                f"crop={cw}:{ch}:{x}:{y}",
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-fps_mode",
                "passthrough",
                str(roi),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    run_pipeline(a.video, reference, a.out / "roi", a.out / "cache", roi, [x, y, cw, ch], low_view=a.low_view)
    print("EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
