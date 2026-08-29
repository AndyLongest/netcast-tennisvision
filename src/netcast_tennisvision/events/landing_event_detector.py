"""Trajectory-only landing event proposals for a frozen tennis-ball track.

The detector compares two local explanations of the *same, already accepted* ball
positions: a smooth flight and a flight with one velocity impulse.  A ground bounce
introduces an upward impulse even when perspective means the image-space y velocity
does not actually change sign.  This module never creates, removes, or moves a ball
observation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from math import exp, log
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LandingImpulse:
    frame: int
    score: float
    smooth_error_px: float
    impulse_error_px: float
    error_reduction: float
    bic_gain: float
    impulse_x_px_frame: float
    impulse_y_px_frame: float
    support_before: int
    support_after: int
    track_id: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _robust_fit(design: np.ndarray, points: np.ndarray, confidence: np.ndarray):
    weights = np.clip(confidence.astype(float), 0.15, 1.0)
    coeff = np.zeros((design.shape[1], 2), dtype=float)
    residual = np.zeros(len(points), dtype=float)
    for _ in range(4):
        root = np.sqrt(weights)[:, None]
        coeff, *_ = np.linalg.lstsq(design * root, points * root, rcond=None)
        residual = np.linalg.norm(design @ coeff - points, axis=1)
        median = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - median)))
        scale = max(0.75, mad)
        huber = np.minimum(1.0, 2.5 * scale / np.maximum(residual, 1e-6))
        weights = np.clip(confidence, 0.15, 1.0) * huber
    sse = float(np.sum(weights * residual * residual))
    rms = float(np.sqrt(sse / max(float(np.sum(weights)), 1e-9)))
    return coeff, sse, rms


def score_landing_impulse(
    frame: int,
    frames_meta: list[dict[str, Any]],
    *,
    radius: int = 8,
    min_side_support: int = 3,
) -> LandingImpulse | None:
    """Score one possible bounce without changing the supplied trajectory.

    A quadratic is the no-contact model.  The alternative adds ``max(0, t)``;
    its coefficient is the instantaneous image velocity change at the candidate.
    BIC penalises the two extra parameters so a split cannot win merely by being
    more flexible.
    """
    if not 0 <= frame < len(frames_meta):
        return None
    centre = frames_meta[frame]
    track_id = centre.get("ball_track_id")
    samples = []
    for index in range(max(0, frame - radius), min(len(frames_meta), frame + radius + 1)):
        meta = frames_meta[index]
        point = meta.get("ball_px")
        if (point is None or not meta.get("ball_seen", False)
                or (track_id is not None and meta.get("ball_track_id") != track_id)):
            continue
        samples.append((index - frame, point, float(meta.get("ball_confidence", 0.5))))
    before = sum(1 for t, _, _ in samples if t < 0)
    after = sum(1 for t, _, _ in samples if t > 0)
    if before < min_side_support or after < min_side_support:
        return None

    t = np.asarray([row[0] for row in samples], dtype=float)
    points = np.asarray([row[1] for row in samples], dtype=float)
    confidence = np.asarray([row[2] for row in samples], dtype=float)
    smooth_design = np.column_stack((np.ones_like(t), t, t * t))
    impulse_design = np.column_stack((smooth_design, np.maximum(t, 0.0)))
    _, smooth_sse, smooth_rms = _robust_fit(smooth_design, points, confidence)
    coeff, impulse_sse, impulse_rms = _robust_fit(impulse_design, points, confidence)

    n_scalar = max(2 * len(samples), 1)
    smooth_bic = n_scalar * log(max(smooth_sse / n_scalar, 1e-9)) + 6 * log(n_scalar)
    impulse_bic = n_scalar * log(max(impulse_sse / n_scalar, 1e-9)) + 8 * log(n_scalar)
    bic_gain = float(smooth_bic - impulse_bic)
    bic_support = 1.0 / (1.0 + exp(-np.clip((bic_gain - 2.0) / 4.0, -30.0, 30.0)))
    reduction = float(np.clip(1.0 - impulse_sse / max(smooth_sse, 1e-9), 0.0, 1.0))

    impulse = coeff[-1]
    ordered = points[np.argsort(t)]
    typical_step = float(np.median(np.linalg.norm(np.diff(ordered, axis=0), axis=1)))
    impulse_ratio = float(np.linalg.norm(impulse) / max(typical_step, 1.0))
    strength = float(np.clip(impulse_ratio / 1.8, 0.0, 1.0))
    # Image y points down.  Ground contact contributes an upward (negative-y)
    # velocity impulse even if the total post-bounce motion still points down.
    upward = 1.0 / (1.0 + exp(np.clip((float(impulse[1]) + 0.15 * typical_step)
                                      / max(0.35 * typical_step, 1.0), -30.0, 30.0)))
    balance = min(before, after) / max(before, after)
    score = float(np.clip((0.48 * bic_support + 0.30 * reduction + 0.22 * strength)
                          * upward * (0.70 + 0.30 * balance), 0.0, 1.0))
    return LandingImpulse(
        frame=int(frame), score=score,
        smooth_error_px=smooth_rms, impulse_error_px=impulse_rms,
        error_reduction=reduction, bic_gain=bic_gain,
        impulse_x_px_frame=float(impulse[0]),
        impulse_y_px_frame=float(impulse[1]),
        support_before=before, support_after=after, track_id=track_id,
    )


def detect_landing_impulses(
    frames_meta: list[dict[str, Any]],
    *,
    radius: int = 8,
    min_score: float = 0.48,
    min_gap_frames: int = 10,
    candidate_filter: Callable[[LandingImpulse], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return context-filtered, non-maximum-suppressed proposals over a track.

    Filtering must happen before non-maximum suppression. Otherwise a racket impulse can
    suppress a slightly weaker nearby court bounce and then be rejected by the caller,
    permanently losing the real landing.
    """
    scored = []
    for frame in range(len(frames_meta)):
        result = score_landing_impulse(frame, frames_meta, radius=radius)
        if (result is not None and result.score >= min_score
                and (candidate_filter is None or candidate_filter(result))):
            scored.append(result)
    selected: list[LandingImpulse] = []
    for candidate in sorted(scored, key=lambda item: item.score, reverse=True):
        if any(abs(candidate.frame - kept.frame) < min_gap_frames for kept in selected):
            continue
        selected.append(candidate)
    return [item.to_dict() for item in sorted(selected, key=lambda item: item.frame)]


def enforce_one_landing_between_hits(
    bounces: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    fps: float,
    rally_reset_seconds: float,
) -> tuple[list[dict[str, Any]], set[int]]:
    """Keep at most one displayable landing in each uninterrupted ball flight.

    A physical second bounce ends the point; it is not a new landing to highlight.
    Another landing is accepted only after a detected racket hit, or after a long quiet
    gap starts a new rally. This rule filters events and never changes the ball track.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")

    ordered = sorted(bounces, key=lambda item: int(item["frame"]))
    hit_frames = sorted(
        int(event["frame"])
        for event in events
        if event.get("kind") == "hit"
    )
    reset_frames = max(0.0, float(rally_reset_seconds) * float(fps))
    kept: list[dict[str, Any]] = []
    rejected: set[int] = set()
    last_accepted_frame: int | None = None

    def latest_hit(frame: int) -> dict[str, Any] | None:
        candidates = [event for event in events
                      if event.get("kind") == "hit" and int(event["frame"]) < frame]
        return max(candidates, key=lambda event: int(event["frame"])) if candidates else None

    def follows_expected_half(bounce: dict[str, Any], hit: dict[str, Any] | None) -> bool:
        """A strike's first landing belongs on the opposite half of the court."""
        if hit is None:
            return False
        hit_side = hit.get("contact_side")
        bounce_side = bounce.get("court_side")
        return (hit_side in {"near", "far"} and bounce_side in {"near", "far"}
                and hit_side != bounce_side)

    def weak_single_model(bounce: dict[str, Any]) -> bool:
        """A location proposal with neither temporal nor measured impulse support."""
        return (float(bounce.get("sequence_p") or 0.0) < 0.08
                and float(bounce.get("impulse_score") or 0.0) < 0.20
                and bounce.get("evidence") not in {
                    "serve-first-service-box-touch",
                    "terminal-track-end+audio-window+court",
                })

    def strongly_confirmed(bounce: dict[str, Any]) -> bool:
        return (float(bounce.get("sequence_p") or 0.0) >= 0.18
                and float(bounce.get("impulse_score") or 0.0) >= 0.62)

    for bounce in ordered:
        frame = int(bounce["frame"])
        if last_accepted_frame is None:
            allowed = True
        else:
            new_rally = frame - last_accepted_frame > reset_frames
            racket_hit_between = any(
                last_accepted_frame < hit_frame < frame for hit_frame in hit_frames
            )
            allowed = new_rally or racket_hit_between

        if allowed:
            # If a very weak candidate and a later multi-signal touchdown occupy the same
            # half, an intervening same-side "hit" cannot make both legal: after that hit
            # the ball must travel to the opposite court. Keep the physically supported
            # touchdown and retire the premature flash.
            if kept and kept[-1].get("court_side") == bounce.get("court_side"):
                between = [event for event in events
                           if event.get("kind") == "hit"
                           and int(kept[-1]["frame"]) < int(event["frame"]) < frame]
                same_side_hits = (between and all(
                    event.get("contact_side") == bounce.get("court_side") for event in between
                ))
                if (same_side_hits and weak_single_model(kept[-1])
                        and strongly_confirmed(bounce)):
                    rejected.add(int(kept[-1]["frame"]))
                    kept[-1] = bounce
                    last_accepted_frame = frame
                    continue
            kept.append(bounce)
            last_accepted_frame = frame
        else:
            # A premature same-half candidate must not steal the one display slot from a
            # later candidate on the half the previous racket strike actually targeted.
            # This replacement is deliberately narrow: both candidates must belong to the
            # same uninterrupted flight, and only court-side consistency can replace the
            # earlier physical touchdown. It never chooses a later second bounce merely
            # because its image impulse happened to be stronger.
            previous = kept[-1]
            context_hit = latest_hit(frame)
            same_flight = (context_hit is not None
                           and int(context_hit["frame"]) < int(previous["frame"])
                           and not any(int(previous["frame"]) < h < frame for h in hit_frames))
            expected_half_replacement = (follows_expected_half(bounce, context_hit)
                                         and not follows_expected_half(previous, context_hit))
            stronger_same_half_replacement = (
                previous.get("court_side") == bounce.get("court_side")
                and weak_single_model(previous) and strongly_confirmed(bounce)
            )
            if same_flight and (expected_half_replacement or stronger_same_half_replacement):
                rejected.add(int(previous["frame"]))
                kept[-1] = bounce
                last_accepted_frame = frame
            else:
                rejected.add(frame)

    return kept, rejected
