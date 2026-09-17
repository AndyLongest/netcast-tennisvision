"""Uncertainty-aware tennis line calls on a calibrated court.

This module classifies an already confirmed touchdown.  It never moves the touchdown
and never decides whether a bounce happened.  Court dimensions are measured to the
outside edge of the lines, matching the ITF convention, and a ball whose footprint can
touch that boundary is not called out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

LineCall = Literal["in", "out", "review"]


@dataclass(frozen=True)
class LineCallResult:
    """Auditable result of comparing one touchdown with the legal court boundary."""

    call: LineCall
    signed_margin_m: float
    uncertainty_m: float
    ball_radius_m: float
    nearest_boundary: str
    confidence: float


def _signed_rectangle_margin(
    x: float,
    y: float,
    *,
    left: float,
    right: float,
    near: float,
    far: float,
) -> tuple[float, str]:
    """Return positive distance inside and negative distance outside a rectangle."""
    boundary_distances = {
        "left_sideline": x - left,
        "right_sideline": right - x,
        "near_baseline": y - near,
        "far_baseline": far - y,
    }
    if left <= x <= right and near <= y <= far:
        boundary = min(boundary_distances, key=boundary_distances.get)  # type: ignore[arg-type]
        return float(boundary_distances[boundary]), boundary

    dx_left, dx_right = max(left - x, 0.0), max(x - right, 0.0)
    dy_near, dy_far = max(near - y, 0.0), max(y - far, 0.0)
    dx, dy = dx_left or dx_right, dy_near or dy_far
    if dx and dy:
        boundary = (
            ("left_sideline" if dx_left else "right_sideline") + "+" +
            ("near_baseline" if dy_near else "far_baseline")
        )
    elif dx:
        boundary = "left_sideline" if dx_left else "right_sideline"
    else:
        boundary = "near_baseline" if dy_near else "far_baseline"
    return -float(np.hypot(dx, dy)), boundary


def classify_line_call(
    x: float,
    y: float,
    *,
    court_width: float = 10.97,
    court_length: float = 23.77,
    singles_inset: float = 1.37,
    match_format: str = "singles",
    uncertainty_m: float = 0.0,
    ball_radius_m: float = 0.0335,
) -> LineCallResult:
    """Classify a touchdown without pretending uncertain footage is centimetre-exact.

    ``uncertainty_m`` is the radial world-space error bound propagated from image-space
    touchdown and court-calibration uncertainty.  A call is automatic only when the
    complete uncertainty interval agrees.  Otherwise it is sent to ``review``.
    """
    if match_format not in {"singles", "doubles"}:
        raise ValueError(f"Unsupported match format: {match_format!r}")
    if not np.isfinite([x, y, uncertainty_m, ball_radius_m]).all():
        raise ValueError("Line-call inputs must be finite")
    if uncertainty_m < 0 or ball_radius_m <= 0:
        raise ValueError("Line-call uncertainty must be non-negative and ball radius positive")

    left = singles_inset if match_format == "singles" else 0.0
    right = court_width - singles_inset if match_format == "singles" else court_width
    signed_margin, boundary = _signed_rectangle_margin(
        float(x), float(y), left=left, right=right, near=0.0, far=court_length,
    )
    # Positive means even the ball centre is inside.  A slightly negative centre remains
    # in when the physical ball footprint overlaps the outside edge of painted line.
    overlap_margin = signed_margin + ball_radius_m
    if overlap_margin >= uncertainty_m:
        call: LineCall = "in"
    elif overlap_margin < -uncertainty_m:
        call = "out"
    else:
        call = "review"

    # Confidence measures how far a decided call sits beyond the review band.  Review
    # itself is deliberately zero-confidence, rather than a misleading high score for
    # being very close to the boundary.
    scale = max(uncertainty_m + ball_radius_m, 1e-6)
    confidence = (
        min(1.0, max(0.0, (abs(overlap_margin) - uncertainty_m) / scale))
        if call != "review" else 0.0
    )
    return LineCallResult(
        call=call,
        signed_margin_m=signed_margin,
        uncertainty_m=float(uncertainty_m),
        ball_radius_m=float(ball_radius_m),
        nearest_boundary=boundary,
        confidence=float(confidence),
    )


def world_uncertainty_from_homography(
    pixel: tuple[float, float],
    inverse_homography: np.ndarray,
    *,
    touchdown_uncertainty_px: float | None,
    calibration_uncertainty_px: float,
    sigma_multiplier: float = 2.0,
    minimum_uncertainty_m: float = 0.025,
) -> float:
    """Propagate local pixel uncertainty through the ground-plane homography.

    Perspective makes one pixel represent very different distances at the near and far
    baselines.  Sampling the local Jacobian avoids a single metre tolerance for both.
    """
    matrix = np.asarray(inverse_homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("inverse_homography must be a finite 3x3 matrix")
    measurement_px = max(0.0, float(touchdown_uncertainty_px or 0.0))
    calibration_px = max(0.0, float(calibration_uncertainty_px))
    radius_px = max(0.5, sigma_multiplier * float(np.hypot(measurement_px, calibration_px)))
    x, y = map(float, pixel)
    samples = np.asarray([
        [[x, y]], [[x + radius_px, y]], [[x - radius_px, y]],
        [[x, y + radius_px]], [[x, y - radius_px]],
    ], dtype=np.float32)
    world = cv2.perspectiveTransform(samples, matrix).reshape(-1, 2)
    if not np.isfinite(world).all():
        return float(minimum_uncertainty_m)
    local = np.linalg.norm(world[1:] - world[0], axis=1)
    return max(float(np.max(local)), float(minimum_uncertainty_m))
