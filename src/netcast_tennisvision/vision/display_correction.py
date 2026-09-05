"""Display-only perspective correction for a fixed tennis camera.

Tracking and landing inference always consume the original frames. This module is used
only after annotations have been rendered, so a user's aesthetic choice cannot alter any
ball observation or touchdown decision.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import cv2
import numpy as np


def parse_display_correction(
    strength_value: str | None,
    corners_value: str | None,
) -> dict[str, Any]:
    """Validate browser headers and return a durable, normalized configuration."""
    if strength_value in (None, "", "0"):
        return {"enabled": False, "strength": 0, "corners": None}
    try:
        strength = int(strength_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("画面矫正比例无效") from exc
    if not 0 <= strength <= 100:
        raise ValueError("画面矫正比例必须在 0% 到 100% 之间")
    if strength == 0:
        return {"enabled": False, "strength": 0, "corners": None}

    import json

    try:
        corners = np.asarray(json.loads(corners_value or ""), dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("请确认球场的四个角点") from exc
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise ValueError("球场角点必须是四组有效坐标")
    if np.any(corners < 0) or np.any(corners > 1):
        raise ValueError("球场角点超出了预览画面")
    area = abs(float(cv2.contourArea(corners.astype(np.float32))))
    if area < 0.02:
        raise ValueError("球场四角围成的区域过小，请重新标记")
    return {
        "enabled": True,
        "strength": strength,
        "corners": [[round(float(x), 6), round(float(y), 6)] for x, y in corners],
    }


def display_homography(
    image_shape: Sequence[int],
    corners_normalized: Sequence[Sequence[float]],
    strength_percent: float,
) -> np.ndarray:
    """Map a calibrated court trapezoid toward an equal-width display rectangle."""
    height, width = int(image_shape[0]), int(image_shape[1])
    source = np.asarray(corners_normalized, dtype=np.float32).reshape(4, 2).copy()
    source[:, 0] *= width
    source[:, 1] *= height
    near_left, near_right, far_right, far_left = source
    near_width = float(near_right[0] - near_left[0])
    far_width = float(far_right[0] - far_left[0])
    if near_width <= 1 or far_width <= 1:
        raise ValueError("球场左右角点顺序不正确")
    strength = float(np.clip(strength_percent, 0, 100)) / 100.0
    equal_width = (near_width + far_width) / 2.0
    target_near_width = near_width + strength * (equal_width - near_width)
    target_far_width = far_width + strength * (equal_width - far_width)
    centre_x = float(np.mean(source[:, 0]))
    target = source.copy()
    target[0, 0] = centre_x - target_near_width / 2.0
    target[1, 0] = centre_x + target_near_width / 2.0
    target[2, 0] = centre_x + target_far_width / 2.0
    target[3, 0] = centre_x - target_far_width / 2.0
    court_matrix = cv2.getPerspectiveTransform(source, target)
    return fit_homography_to_frame(
        court_matrix, height=height, width=width, court_corners=source,
    )


def fit_homography_to_frame(
    matrix: np.ndarray,
    *,
    height: int,
    width: int,
    court_corners: np.ndarray,
) -> np.ndarray:
    """Keep the court and both baseline-player bands visible after correction."""
    near_left, near_right, far_right, far_left = np.asarray(court_corners, np.float32)
    near_width = float(near_right[0] - near_left[0])
    far_width = float(far_right[0] - far_left[0])
    depth = float((near_left[1] + near_right[1] - far_right[1] - far_left[1]) / 2.0)
    protected = np.float32([[
        near_left + (-0.08 * near_width, 0.18 * depth),
        near_right + (0.08 * near_width, 0.18 * depth),
        far_right + (0.30 * far_width, -0.28 * depth),
        far_left + (-0.30 * far_width, -0.28 * depth),
    ]])
    warped = cv2.perspectiveTransform(protected, matrix)[0]
    if not np.all(np.isfinite(warped)):
        raise ValueError("矫正后的画面边界无效，请降低比例或重新标记球场")
    minimum = warped.min(axis=0)
    maximum = warped.max(axis=0)
    span = maximum - minimum
    if np.any(span <= 1):
        raise ValueError("矫正后的画面范围过小，请重新标记球场")
    scale = min(1.0, width / float(span[0]), height / float(span[1]))

    def minimal_translation(low: float, high: float, limit: int) -> float:
        low *= scale
        high *= scale
        if scale < 0.999999:
            return (limit - (high - low)) / 2.0 - low
        if low < 0:
            return -low
        if high > limit:
            return limit - high
        return 0.0

    translate_x = minimal_translation(float(minimum[0]), float(maximum[0]), width)
    translate_y = minimal_translation(float(minimum[1]), float(maximum[1]), height)
    fit = np.array(
        [[scale, 0.0, translate_x], [0.0, scale, translate_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return fit @ matrix


def apply_display_correction(frame: np.ndarray, config: Mapping[str, Any] | None) -> np.ndarray:
    """Warp one rendered frame; return the input unchanged when correction is disabled."""
    if not config or not config.get("enabled") or not config.get("corners"):
        return frame
    corners_key = tuple(float(value) for point in config["corners"] for value in point)
    matrix = _cached_display_homography(
        frame.shape[0], frame.shape[1], corners_key, float(config["strength"]),
    )
    return cv2.warpPerspective(
        frame,
        matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


@lru_cache(maxsize=16)
def _cached_display_homography(
    height: int,
    width: int,
    corners_key: tuple[float, ...],
    strength: float,
) -> np.ndarray:
    """Compute one matrix per fixed camera/configuration and reuse it for every frame."""
    corners = np.asarray(corners_key, dtype=np.float64).reshape(4, 2)
    return display_homography((height, width), corners, strength)
