"""Confidence policy and validation for automatic/manual court calibration."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CalibrationConfidence:
    score: float
    automatic: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibration_preview_score(frame: np.ndarray, *, court_detected: bool) -> float:
    """Rank sampled frames for the manual four-corner preview.

    Videos commonly begin with a black transition.  A preview must therefore be chosen
    from the sampled sequence, not hard-wired to frame zero.  A frame in which the court
    detector already found a plausible quadrilateral always outranks a merely bright
    frame; contrast and spatial detail break ties between otherwise valid samples.
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        return float("-inf")
    gray = image.astype(np.float32).mean(axis=2)
    low, high = np.percentile(gray, (5, 95))
    contrast = float(high - low)
    detail = float(
        np.mean(np.abs(np.diff(gray, axis=0)))
        + np.mean(np.abs(np.diff(gray, axis=1)))
    )
    visible = float(high) >= 12.0 and (contrast >= 5.0 or detail >= 1.0)
    if not visible:
        return -1_000_000.0 + float(high) + contrast + detail
    return (1_000_000.0 if court_detected else 0.0) + float(high) + contrast + detail


def automatic_calibration_confidence(
    *, sampled_frames: int, detected_frames: int, consensus_support: int,
    fit_error_px: float, fit_note: str = "",
) -> CalibrationConfidence:
    """Return a conservative automatic-acceptance score for one fixed-camera clip."""
    sample_support = min(1.0, detected_frames / max(4.0, 0.25 * sampled_frames))
    consensus = min(1.0, consensus_support / max(3.0, 0.12 * sampled_frames))
    error = max(0.0, min(1.0, (4.0 - float(fit_error_px)) / 3.25))
    score = 0.34 * sample_support + 0.28 * consensus + 0.38 * error
    reasons: list[str] = []
    if detected_frames == 0:
        reasons.append("未在采样画面中找到完整球场")
    elif sample_support < 0.65:
        reasons.append("自动识别只得到少量稳定画面")
    if consensus < 0.65:
        reasons.append("不同画面的球场位置不够一致")
    if error < 0.60:
        reasons.append("九条场线与画面贴合度不足")
    if "fallback" in fit_note.lower() or "failed" in fit_note.lower():
        score *= 0.82
        reasons.append("自动精修使用了保守回退")
    score = float(max(0.0, min(1.0, score)))
    return CalibrationConfidence(score, score >= 0.72, tuple(reasons))


def validate_manual_calibration(
    corners: Sequence[Sequence[float]], image_shape: Sequence[int], world_quad: np.ndarray,
) -> np.ndarray:
    """Validate mapping geometry only; manual clicks need no camera-view priors.

    The user supplies near-L, near-R, far-R, far-L. Do not infer those semantic
    labels from screen height, size, perspective ratio or camera alignment.
    """
    try:
        points = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("请标记四个有效的球场角点") from exc
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("请标记四个有效的球场角点")

    # Normalization avoids an image-resolution-dependent degeneracy threshold.
    span = np.ptp(points, axis=0)
    if np.any(span == 0):
        raise ValueError("四个角点不能重合或共线")
    normalized = (points - points.min(axis=0)) / span
    edges = np.roll(normalized, -1, axis=0) - normalized
    next_edges = np.roll(edges, -1, axis=0)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    if not (np.all(crosses > 0) or np.all(crosses < 0)):
        raise ValueError("请沿球场边界依次标点，四角不能重合、共线、交叉或凹陷")
    homography = cv2.getPerspectiveTransform(
        np.asarray(world_quad, np.float32), normalized.astype(np.float32),
    )
    if not np.isfinite(homography).all() or np.linalg.matrix_rank(homography) < 3:
        raise ValueError("四个角点无法建立有效的透视映射，请重新标记")
    return points
