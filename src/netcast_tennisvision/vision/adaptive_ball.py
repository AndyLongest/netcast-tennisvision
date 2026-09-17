"""Plan event-preserving BallTrack inference frames.

The public BallTrack model still receives four consecutive native video frames for every
inferred timestamp.  This module only decides *which output timestamps* need a model
call.  Stable flight may stay on the scout cadence; suspected births, losses, racket
contacts and motion impulses are expanded back to native-rate windows.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdaptiveBallPlan:
    scout_frames: frozenset[int]
    dense_frames: frozenset[int]
    reasons: dict[str, int]

    @property
    def inference_frames(self) -> frozenset[int]:
        return self.scout_frames | self.dense_frames


def scout_frame_indices(total_frames: int, stride: int = 2) -> frozenset[int]:
    """Return a bounded native-timeline scout schedule."""
    if total_frames <= 0:
        return frozenset()
    stride = max(1, int(stride))
    indices = set(range(0, total_frames, stride))
    indices.add(total_frames - 1)
    return frozenset(indices)


def _primary(row: Sequence[Sequence[float]]) -> np.ndarray | None:
    if not row or len(row[0]) < 3:
        return None
    return np.asarray(row[0][:3], dtype=float)


def _near_player(point: np.ndarray, boxes: Any) -> bool:
    if boxes is None:
        return False
    for raw in np.asarray(boxes, dtype=float).reshape(-1, 4):
        x1, y1, x2, y2 = raw
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        if (x1 - 0.55 * width <= point[0] <= x2 + 0.55 * width
                and y1 - 0.45 * height <= point[1] <= y2 + 0.30 * height):
            return True
    return False


def plan_event_preserving_frames(
    scout_candidates: Sequence[Sequence[Sequence[float]]],
    frame_metadata: Sequence[dict[str, Any]],
    *,
    fps: float,
    stride: int = 2,
    guard_seconds: float = 0.30,
) -> AdaptiveBallPlan:
    """Expand sparse scout evidence into native-rate event guard windows.

    The planner is deliberately recall-biased.  It does not classify a hit or landing;
    it merely asks the frozen detector to revisit frames where downstream contact logic
    could otherwise lose temporal support.
    """
    total = len(scout_candidates)
    scout = scout_frame_indices(total, stride)
    if total == 0 or stride <= 1:
        return AdaptiveBallPlan(scout, scout, {"full_rate": total})

    guard = max(4, int(round(max(float(fps), 1.0) * guard_seconds)))
    dense: set[int] = set()
    reason_centres: dict[str, set[int]] = {
        "clip_boundary": set(),
        "birth_or_loss": set(),
        "brief_gap": set(),
        "player_contact": set(),
        "motion_impulse": set(),
        "weak_candidate": set(),
    }

    def mark(frame: int, reason: str, radius: int = guard) -> None:
        reason_centres[reason].add(int(frame))
        dense.update(range(max(0, frame - radius), min(total, frame + radius + 1)))

    mark(0, "clip_boundary", radius=min(guard, total - 1))
    mark(total - 1, "clip_boundary", radius=min(guard, total - 1))

    ordered = sorted(scout)
    raw_observations: list[tuple[int, np.ndarray]] = []
    for frame in ordered:
        point = _primary(scout_candidates[frame])
        present = point is not None
        if present:
            raw_observations.append((frame, point))

    # Raw heatmap peaks can jump to shoes, lights or a loose ball.  Event scheduling must
    # not turn every such jump into a full-rate window.  First form permissive but
    # physically continuous scout runs, allowing one missing scout sample.
    runs: list[list[tuple[int, np.ndarray]]] = []
    current: list[tuple[int, np.ndarray]] = []
    for frame, point in raw_observations:
        if current:
            prior_frame, prior_point = current[-1]
            gap = frame - prior_frame
            speed = float(np.linalg.norm(point[:2] - prior_point[:2])) / max(gap, 1)
            if gap > 2 * stride or speed > 58.0:
                if len(current) >= 4:
                    runs.append(current)
                current = []
        current.append((frame, point))
    if len(current) >= 4:
        runs.append(current)

    for run in runs:
        mark(run[0][0], "birth_or_loss")
        mark(run[-1][0], "birth_or_loss")
        for (previous_frame, _), (frame, point) in zip(run, run[1:], strict=False):
            if frame - previous_frame > stride:
                mark((frame + previous_frame) // 2, "brief_gap")
            boxes = frame_metadata[frame].get("person_boxes") if frame < len(frame_metadata) else None
            if _near_player(point, boxes):
                mark(frame, "player_contact")

    # Sparse samples are enough to nominate a possible impulse.  The actual event is
    # always decided later from the recovered native-rate observations.
    for run in runs:
        for (fa, pa), (fb, pb), (fc, pc) in zip(
                run, run[1:], run[2:], strict=False):
            va = (pb[:2] - pa[:2]) / max(1, fb - fa)
            vb = (pc[:2] - pb[:2]) / max(1, fc - fb)
            speed_a, speed_b = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            if min(speed_a, speed_b) < 0.35:
                continue
            cosine = float(np.dot(va, vb) / max(speed_a * speed_b, 1e-6))
            acceleration = float(np.linalg.norm(vb - va))
            relative_acceleration = acceleration / max(speed_a, speed_b, 1.0)
            vertical_reversal = va[1] * vb[1] < 0 and abs(va[1] - vb[1]) > 0.7
            if cosine < 0.55 or relative_acceleration > 0.82 or vertical_reversal:
                mark(fb, "motion_impulse")

    reasons = {name: len(frames) for name, frames in reason_centres.items() if frames}
    return AdaptiveBallPlan(scout, frozenset(dense), reasons)


def plan_coarse_track_recovery(
    frame_metadata: Sequence[dict[str, Any]],
    *,
    fps: float,
    scout_stride: int = 2,
    guard_seconds: float = 0.34,
) -> AdaptiveBallPlan:
    """Plan native-rate recovery from an already associated sparse ball track."""
    total = len(frame_metadata)
    scout = scout_frame_indices(total, scout_stride)
    guard = max(5, int(round(max(float(fps), 1.0) * guard_seconds)))
    dense: set[int] = set()
    centres: dict[str, set[int]] = {
        "track_boundary": set(),
        "player_contact": set(),
        "tracker_contact": set(),
        "motion_impulse": set(),
    }

    def mark(frame: int, reason: str, radius: int = guard) -> None:
        centres[reason].add(int(frame))
        dense.update(range(max(0, frame - radius), min(total, frame + radius + 1)))

    tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame, meta in enumerate(frame_metadata):
        point = meta.get("ball_px")
        track_id = meta.get("ball_track_id")
        if point is None or track_id is None:
            continue
        value = np.asarray(point, dtype=float)
        tracks.setdefault(int(track_id), []).append((frame, value))
        if _near_player(value, meta.get("person_boxes")):
            mark(frame, "player_contact")
        if meta.get("ball_motion_mode") in {"player_hit", "bounce"}:
            mark(frame, "tracker_contact")

    for observations in tracks.values():
        mark(observations[0][0], "track_boundary")
        mark(observations[-1][0], "track_boundary")
        # Five-frame central velocities suppress the alternating coast/measurement ripple
        # created by a stride-two scout while retaining genuine contact impulses.
        if len(observations) < 7:
            continue
        frames = np.asarray([item[0] for item in observations], dtype=int)
        points = np.asarray([item[1] for item in observations], dtype=float)
        for index in range(3, len(points) - 3):
            va = (points[index] - points[index - 3]) / max(1, frames[index] - frames[index - 3])
            vb = (points[index + 3] - points[index]) / max(1, frames[index + 3] - frames[index])
            speed_a, speed_b = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            if min(speed_a, speed_b) < 0.45:
                continue
            cosine = float(np.dot(va, vb) / max(speed_a * speed_b, 1e-6))
            relative = float(np.linalg.norm(vb - va)) / max(speed_a, speed_b, 1.0)
            vertical_reversal = va[1] * vb[1] < 0 and abs(va[1] - vb[1]) > 0.8
            if cosine < 0.72 or relative > 0.66 or vertical_reversal:
                mark(int(frames[index]), "motion_impulse")

    reasons = {name: len(values) for name, values in centres.items() if values}
    return AdaptiveBallPlan(scout, frozenset(dense), reasons)
