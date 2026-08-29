"""Landing-time refinement for a fixed 30 fps tennis track.

This module deliberately consumes an already-finished ball track.  It never proposes,
removes, or moves ball detections.  Its only job is to infer the hidden instant at which
the two observed flight branches meet the court plane.
"""
from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np


def _robust_polynomial(samples: list[tuple[float, np.ndarray, float]], degree: int):
    t = np.asarray([s[0] for s in samples], dtype=float)
    points = np.vstack([s[1] for s in samples]).astype(float)
    confidence = np.asarray([s[2] for s in samples], dtype=float)
    origin = float(np.mean(t))
    design = np.vander(t - origin, degree + 1, increasing=True)
    weights = np.clip(confidence, 0.15, 1.0)
    coeff = None
    residual = np.zeros(len(samples), dtype=float)
    for _ in range(4):
        root_w = np.sqrt(weights)[:, None]
        coeff, *_ = np.linalg.lstsq(design * root_w, points * root_w, rcond=None)
        residual = np.linalg.norm(design @ coeff - points, axis=1)
        scale = max(1.0, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        huber = np.minimum(1.0, 2.5 * scale / np.maximum(residual, 1e-6))
        weights = np.clip(confidence, 0.15, 1.0) * huber

    def evaluate(frame_f: float) -> np.ndarray:
        row = np.asarray([(frame_f - origin) ** power for power in range(degree + 1)])
        return row @ coeff

    rms = float(np.sqrt(np.average(residual ** 2, weights=np.maximum(weights, 1e-6))))
    return evaluate, rms


def estimate_landing_subframe(
    frame: int,
    frames_meta: list[dict[str, Any]],
    *,
    radius: int = 8,
    confirmation_frames: int = 2,
) -> dict[str, Any]:
    """Fit the incoming/outgoing branches and locate their sub-frame intersection.

    Only detector-backed samples from the same persistent track are used.  Predicted
    positions may be displayed by the UI, but cannot decide a touchdown.
    """
    if not 0 <= frame < len(frames_meta):
        raise IndexError(frame)
    centre = frames_meta[frame]
    track_id = centre.get("ball_track_id")
    fallback = centre.get("ball_px")
    lo, hi = max(0, frame - radius), min(len(frames_meta), frame + radius + 1)
    before: list[tuple[float, np.ndarray, float]] = []
    after: list[tuple[float, np.ndarray, float]] = []
    for index in range(lo, hi):
        meta = frames_meta[index]
        point = meta.get("ball_px")
        if (point is None or not meta.get("ball_seen", False)
                or (track_id is not None and meta.get("ball_track_id") != track_id)):
            continue
        sample = (float(index), np.asarray(point, dtype=float),
                  float(meta.get("ball_confidence", 0.5)))
        (before if index <= frame else after).append(sample)

    support = len(before) + len(after)
    if len(before) < 3 or len(after) < 3 or fallback is None:
        return {
            "frame_f": float(frame), "px": fallback, "confidence": "frame",
            "support": support, "decision_frame": min(len(frames_meta) - 1,
                                                         frame + confirmation_frames),
            "branch_disagreement_px": None, "uncertainty_px": None,
        }

    degree_before = 2 if len(before) >= 5 else 1
    degree_after = 2 if len(after) >= 5 else 1
    incoming, residual_before = _robust_polynomial(before, degree_before)
    outgoing, residual_after = _robust_polynomial(after, degree_after)

    # 0.02 frame is about 0.67 ms at 30 fps.  This estimates time more finely without
    # pretending that the underlying image observations themselves have that precision.
    candidates = np.linspace(frame - 1.0, frame + 1.0, 101)
    disagreements = np.asarray([
        np.linalg.norm(incoming(t) - outgoing(t)) for t in candidates
    ])
    best = int(np.argmin(disagreements))
    frame_f = float(candidates[best])
    p_in, p_out = incoming(frame_f), outgoing(frame_f)
    point = 0.5 * (p_in + p_out)
    disagreement = float(disagreements[best])
    uncertainty = float(np.hypot(residual_before + residual_after, disagreement))
    quality = "subframe" if disagreement <= 7.0 and uncertainty <= 12.0 else "frame"
    if quality == "frame":
        frame_f = float(frame)
        point = np.asarray(fallback, dtype=float)

    return {
        "frame_f": frame_f,
        "px": (float(point[0]), float(point[1])),
        "confidence": quality,
        "support": support,
        "residual_before_px": residual_before,
        "residual_after_px": residual_after,
        "branch_disagreement_px": disagreement,
        "uncertainty_px": uncertainty,
        # A landing is announced only after post-contact evidence exists.
        "decision_frame": min(len(frames_meta) - 1,
                              max(frame + 1, ceil(frame_f) + confirmation_frames)),
    }
