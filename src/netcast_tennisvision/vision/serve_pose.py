"""Optional, bounded offline pose verification; off has no model/import cost.

This is not a live-stream worker. Sparse source-frame sampling only affects pose,
never the native-rate ball trajectory. Shadow results cannot change rally IDs.
"""
from __future__ import annotations

import hashlib
import os
import time

MODEL_SHA256 = "869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0"


def raised_arm_sequence(samples):
    """Require opposite arms in ordered toss/contact poses, with visible shoulders."""
    toss_arms = set()
    for offset, points in sorted(samples, key=lambda row: row[0]):
        if len(points) != 17:
            continue
        for shoulder, elbow, wrist, other in ((5, 7, 9, 10), (6, 8, 10, 9)):
            if min(points[k][2] for k in (shoulder, elbow, wrist, other, 11, 12)) < .5:
                continue
            torso = abs((points[11][1] + points[12][1]) / 2 - points[shoulder][1])
            if torso < 8:
                continue
            overhead = points[wrist][1] < points[shoulder][1] - .35 * torso
            if offset < -.15 and overhead and points[other][1] > points[shoulder][1]:
                toss_arms.add(wrist)
            if -.1 <= offset <= .25 and overhead and other in toss_arms:
                return True
    return False


def verify_serve_pose(video, frames, *, fps, mode=None, budget_seconds=None):
    mode = mode if mode is not None else os.environ.get("TENNISVISION_SERVE_POSE", "off")
    report = {"mode": mode, "status": "disabled", "elapsed_seconds": 0.0,
              "inference_seconds": 0.0, "samples": 0, "candidates": 0, "confirmed": []}
    if mode == "off":
        return [], report
    if mode not in {"shadow", "on"}:
        raise ValueError("TENNISVISION_SERVE_POSE must be off, shadow or on")
    start = time.perf_counter()
    budget = float(budget_seconds if budget_seconds is not None else
                   os.environ.get("TENNISVISION_SERVE_POSE_BUDGET_SECONDS", "15"))
    if not 0 < budget <= 120:
        raise ValueError("pose budget must be in (0, 120] seconds")
    model = capture = None
    try:
        from netcast_tennisvision.events.serve_sequence import (
            detect_serve_sequences,
            pose_schedule,
            select_pose,
        )
        schedule = pose_schedule(frames, fps)
        report["planned_samples"] = len(schedule)
        if not schedule:
            report["status"] = "no_candidates"
            return [], report
        if time.perf_counter() - start >= budget:
            report["status"] = "budget_exceeded"
            return [], report
        from netcast_tennisvision.paths import REPOSITORY_ROOT
        path = REPOSITORY_ROOT / "models" / "yolo11n-pose.pt"
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
            report["status"] = "missing_or_invalid_weights"
            return [], report
        import cv2
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            report["status"] = "cuda_unavailable"
            return [], report
        model = YOLO(str(path), task="pose")
        capture = cv2.VideoCapture(str(video))
        decoded_index = -1
        poses, batch, identities = {}, [], []
        inference_times = []

        def flush():
            if not batch:
                return
            began = time.perf_counter()
            results = model.predict(batch, imgsz=320, device=0, conf=.35, verbose=False)
            for result, (f, side, player, shape) in zip(results, identities, strict=True):
                poses[(f, side)] = select_pose(result.keypoints.data.cpu().numpy(),
                                              result.boxes.xyxy.cpu().numpy(), player, shape)
            elapsed = time.perf_counter()-began
            inference_times.append(elapsed)
            report["inference_seconds"] += elapsed
            report["samples"] += len(batch)
            batch.clear()
            identities.clear()

        for f, side in schedule:
            if time.perf_counter() - start >= budget:
                report["status"] = "budget_exceeded"
                return [], report
            player = next(p for p in frames[f]["players_world"] if p["side"] == side)
            ok = True
            while decoded_index < f:
                if time.perf_counter() - start >= budget:
                    report["status"] = "budget_exceeded"
                    return [], report
                ok = capture.grab()
                decoded_index += 1
                if not ok:
                    break
            if not ok:
                report["status"] = "decode_failed"
                return [], report
            ok, image = capture.retrieve()
            if not ok:
                report["status"] = "decode_failed"
                return [], report
            left, top, right, bottom = player["box"]
            height = bottom-top
            x0, y0 = max(0, int(left-height*.5)), max(0, int(top-height*.6))
            x1, y1 = min(image.shape[1], int(right+height*.5)), min(image.shape[0], int(bottom+height*.2))
            if x1 <= x0 or y1 <= y0:
                continue
            batch.append(image[y0:y1,x0:x1].copy())
            identities.append((f,side,player,image.shape))
            if len(batch) == 8:
                flush()
        flush()
        confirmed = detect_serve_sequences(frames, poses, fps=fps)
        if time.perf_counter() - start >= budget:
            report["status"] = "budget_exceeded"
            return [], report
        report.update(status="complete", confirmed=confirmed, candidates=len(confirmed))
        if inference_times:
            report["first_batch_seconds"] = inference_times[0]
            report["batch_count"] = len(inference_times)
        return confirmed if mode == "on" else [], report
    except Exception as exc:
        # This optional stage must also isolate backend-specific errors (e.g. cv2.error).
        # KeyboardInterrupt/SystemExit still propagate; diagnostics remain visible.
        report.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        return [], report
    finally:
        if capture is not None:
            capture.release()
        if model is not None:
            del model
            import gc
            gc.collect()
            # Do not retain an optional model's cache after this analysis stage.
            import torch
            torch.cuda.empty_cache()
        report["elapsed_seconds"] = round(time.perf_counter() - start, 4)
