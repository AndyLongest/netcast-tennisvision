"""Opt-in low-view experiment: temporal bounce evidence without a claimed location.

Never fills ball observations or changes production events. A bounded post-contact
window supplies evidence; inferred events remain separate from positioned landings.
"""
from __future__ import annotations

import numpy as np

from .landing_event_detector import score_landing_impulse


def infer_occluded_landings(frames, events, bounces, *, fps, width, roi_box):
    radius = max(4, round(fps * .30))
    max_missing = max(1, round(fps * .20))
    real = [i for i, m in enumerate(frames) if m.get("ball_seen") and m.get("ball_px") is not None]
    scored = []
    x, y, rw, rh = roi_box
    for left, right in zip(real, real[1:], strict=False):
        if not 1 <= right-left-1 <= max_missing:
            continue
        track = frames[left].get("ball_track_id")
        if track is None or frames[right].get("ball_track_id") != track:
            continue
        points = np.asarray([frames[j]["ball_px"] for j in (left, right)])
        if not all(x <= px <= x+rw and y <= py <= y+rh for px, py in points):
            continue
        # A camera cut must not masquerade as a ball impulse.
        a, b = frames[left].get("corners"), frames[right].get("corners")
        if a is not None and b is not None and np.max(np.abs(np.asarray(a)-b)) > width*.01:
            continue
        for f in range(left+1, right):
            if f+radius >= len(frames):
                continue
            if any(abs(int(e["frame"])-f) <= max(2, round(fps*.16)) for e in events if e.get("kind") == "hit"):
                continue
            if any(abs(int(e["frame"])-f) <= round(fps*.35) for e in bounces):
                continue
            view = list(frames)
            view[f] = dict(frames[f], ball_track_id=track)
            impulse = score_landing_impulse(f, view, radius=radius)
            if (impulse is None or impulse.score < .72 or impulse.bic_gain < 6
                    or impulse.impulse_error_px > max(1.5, width/480)
                    or impulse.impulse_y_px_frame >= -width/1920):
                continue
            scored.append({"frame": f, "time_s": f/fps, "decision_frame": f+radius,
                           "score": impulse.score, "missing_interval": [left+1, right-1],
                           "position": None, "status": "inferred",
                           "evidence": "observed-before-after-upward-impulse",
                           "support_before": impulse.support_before, "support_after": impulse.support_after})
    kept = []
    for item in sorted(scored, key=lambda e: e["score"], reverse=True):
        if all(abs(item["frame"]-e["frame"]) > round(fps*.4) for e in kept):
            kept.append(item)
    return sorted(kept, key=lambda e: e["frame"])

