"""Descriptive camera-view metadata; never selects or rejects an analysis path."""
from __future__ import annotations

import math

import numpy as np

from .court_calibration import validate_manual_calibration


def classify_viewpoint(corners, width, height) -> dict:
    """Use source-normalized NL, NR, FR, FL corners, without display gutters.

    The user-defined split is <= 30% low, > 30% high.
    This is image occupancy, not a physical camera angle.
    Existing geometry validation remains independent of these descriptive thresholds.
    """
    result = {"version": 2, "category": "unavailable", "analysis_path": "existing",
              "height_ratio": None, "depth_width_ratio": None, "reason": "invalid_geometry"}
    try:
        width, height = float(width), float(height)
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            return result
        points = np.asarray(corners, dtype=float)
        if points.shape != (4, 2):
            return result
        points = points * [width, height]
        world = np.array([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]], dtype=np.float32)
        validate_manual_calibration(points, (height, width), world)
    except (ValueError, TypeError, OverflowError):
        return result
    near, far = (points[0] + points[1]) / 2, (points[2] + points[3]) / 2
    near_edge, far_edge = points[1] - points[0], points[2] - points[3]
    mean_width = (np.linalg.norm(near_edge) + np.linalg.norm(far_edge)) / 2
    tangent = near_edge / np.linalg.norm(near_edge) + far_edge / np.linalg.norm(far_edge)
    tangent /= np.linalg.norm(tangent)
    delta = near - far
    depth = abs(float(delta @ np.array([-tangent[1], tangent[0]])))
    ratio = abs(float(delta[1])) / height
    shape = depth / mean_width
    result.update(height_ratio=round(ratio, 6), depth_width_ratio=round(float(shape), 6))
    # Suppress arithmetic noise at exactly 30%, not a transition band.
    if round(ratio, 15) <= 0.30:
        result.update(category="low", reason="height_at_most_30_percent")
    else:
        result.update(category="high", reason="height_above_30_percent")
    return result
