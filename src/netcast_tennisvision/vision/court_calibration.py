"""Confidence policy and validation for automatic/manual court calibration."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .court_registration import validate_court_corners


@dataclass(frozen=True)
class CalibrationConfidence:
    score: float
    automatic: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    """Validate user clicks ordered near-L, near-R, far-R, far-L."""
    points = np.asarray(corners, dtype=np.float64)
    check = validate_court_corners(points, image_shape, world_quad=world_quad)
    if not check.valid:
        raise ValueError("；".join(check.reasons))
    return points
