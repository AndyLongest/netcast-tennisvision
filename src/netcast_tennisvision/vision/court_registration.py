"""Geometry and scoring primitives for robust tennis-court registration.

The notebook's detector deliberately remains lightweight.  This module supplies the
part that its original implementation was missing: a candidate must explain a real,
non-degenerate tennis-court quadrilateral and must receive support from *each* family
of painted lines, not merely obtain a good global average by collapsing several model
lines onto one bright image edge.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryCheck:
    valid: bool
    reasons: tuple[str, ...]
    metrics: dict[str, float]


def validate_court_corners(
    corners: np.ndarray,
    image_shape: Sequence[int],
    *,
    world_quad: np.ndarray | None = None,
) -> GeometryCheck:
    """Validate ``[near-L, near-R, far-R, far-L]`` image corners.

    Bounds are intentionally broad enough for ordinary elevated behind-baseline views,
    but reject the collapsed trapezoids produced by accidental line combinations.
    """

    q = np.asarray(corners, dtype=np.float64)
    h, w = int(image_shape[0]), int(image_shape[1])
    reasons: list[str] = []
    metrics: dict[str, float] = {}

    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return GeometryCheck(False, ("corners are not four finite image points",), metrics)

    diag = float(np.hypot(w, h))
    near_w = float(np.linalg.norm(q[1] - q[0]))
    far_w = float(np.linalg.norm(q[2] - q[3]))
    left_d = float(np.linalg.norm(q[3] - q[0]))
    right_d = float(np.linalg.norm(q[2] - q[1]))
    near_y = float((q[0, 1] + q[1, 1]) / 2)
    far_y = float((q[2, 1] + q[3, 1]) / 2)
    near_x = float((q[0, 0] + q[1, 0]) / 2)
    far_x = float((q[2, 0] + q[3, 0]) / 2)
    depth = near_y - far_y
    area = float(abs(cv2.contourArea(q.astype(np.float32))))
    pairwise = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
    min_pair = float(pairwise[np.triu_indices(4, 1)].min())
    crosses = []
    for i in range(4):
        a = q[(i + 1) % 4] - q[i]
        b = q[(i + 2) % 4] - q[(i + 1) % 4]
        crosses.append(float(a[0] * b[1] - a[1] * b[0]))

    metrics.update(
        near_width_ratio=near_w / w,
        far_width_ratio=far_w / w,
        far_near_ratio=far_w / max(near_w, 1e-9),
        depth_ratio=depth / h,
        area_ratio=area / (w * h),
        min_corner_separation_ratio=min_pair / diag,
        left_right_depth_ratio=min(left_d, right_d) / max(max(left_d, right_d), 1e-9),
        centre_drift_ratio=abs(far_x - near_x) / max(near_w, 1e-9),
        near_baseline_y_ratio=near_y / h,
    )

    if near_y <= far_y:
        reasons.append("near baseline is not below far baseline")
    if near_y < 0.70 * h:
        reasons.append("near baseline is too high for the supported wide-court view")
    if near_w < 0.25 * w:
        reasons.append("near baseline is too narrow")
    if far_w < 0.08 * w:
        reasons.append("far baseline is too narrow")
    # In a complete behind-baseline view the far baseline remains a substantial span.
    # Much smaller ratios are produced by venue edges converging near the vanishing
    # point (the demo failure locked onto the wall/floor seam at ratio 0.11).
    if not 0.22 <= metrics["far_near_ratio"] <= 0.75:
        reasons.append("far/near baseline ratio is implausible")
    if depth < 0.15 * h:
        reasons.append("court depth is too small")
    if not 0.07 <= metrics["area_ratio"] <= 0.88:
        reasons.append("court area is implausible")
    if min_pair < 0.04 * diag:
        reasons.append("two court corners have collapsed together")
    if min(crosses) < -1e-6 and max(crosses) > 1e-6:
        reasons.append("court quadrilateral is non-convex or self-crossing")
    if metrics["left_right_depth_ratio"] < 0.20:
        reasons.append("one sideline has collapsed")
    # The supported camera contract is an elevated view from behind the baseline,
    # approximately on the centre line. Large centre drift is the signature of mixing
    # one main-court sideline with a line from an adjacent court or venue structure.
    if metrics["centre_drift_ratio"] > 0.14:
        reasons.append("court centre drifts implausibly across depth")
    if np.any(q[:, 0] < -0.20 * w) or np.any(q[:, 0] > 1.20 * w):
        reasons.append("court lies too far outside the image horizontally")
    if np.any(q[:, 1] < -0.20 * h) or np.any(q[:, 1] > 1.20 * h):
        reasons.append("court lies too far outside the image vertically")

    if world_quad is not None and not reasons:
        H = cv2.getPerspectiveTransform(
            np.asarray(world_quad, np.float32), q.astype(np.float32)
        )
        if not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-8:
            reasons.append("homography is singular")

    return GeometryCheck(not reasons, tuple(reasons), metrics)


def per_line_coverage(
    projected_points: np.ndarray,
    line_mask: np.ndarray,
    *,
    n_lines: int,
) -> tuple[np.ndarray, float]:
    """Return coverage for every model line and the on-screen sample fraction."""

    p = np.asarray(projected_points, dtype=np.float64).reshape(n_lines, -1, 2)
    h, w = line_mask.shape[:2]
    scores = np.zeros(n_lines, dtype=np.float64)
    visible = np.zeros(n_lines, dtype=np.float64)
    for i, line in enumerate(p):
        x = np.rint(line[:, 0]).astype(int)
        y = np.rint(line[:, 1]).astype(int)
        on = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        visible[i] = float(on.mean())
        if on.any():
            scores[i] = float((line_mask[y[on], x[on]] > 0).mean())
    return scores, float(visible.mean())


def court_template_score(line_coverages: Iterable[float]) -> float:
    """Balanced full-template score resistant to one-line/collapsed solutions.

    The two baselines, two doubles sidelines, two singles sidelines and three service
    lines each have to contribute.  The lower quartile prevents a few perfect lines
    from hiding unsupported parts of the court.
    """

    c = np.asarray(tuple(line_coverages), dtype=np.float64)
    if c.shape != (9,):
        raise ValueError("expected coverage for the nine ITF template lines")
    groups = (
        c[0:2],  # baselines
        c[2:4],  # doubles sidelines
        c[4:6],  # singles sidelines
        c[6:9],  # service-box paint
    )
    group_support = np.array([np.mean(g) for g in groups])
    return float(
        0.35 * np.quantile(c, 0.25)
        + 0.35 * np.min(group_support)
        + 0.30 * np.mean(c)
    )


def court_surface_candidate(frame: np.ndarray) -> tuple[np.ndarray | None, dict[str, float]]:
    """Estimate the outer court from its dominant playing-surface component.

    This is a proposal generator, not an unconditional acceptance rule.  It is useful
    when the far baseline is only one or two pixels wide and Hough misses one of its
    supporting sidelines.  The colour is learned from the lower-centre of the current
    frame, so blue, purple, green and clay courts require no named colour threshold.

    Returns corners in ``[near-L, near-R, far-R, far-L]`` order.  Downstream template
    paint scoring and geometry validation must still approve the proposal.
    """

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    # The lower-centre *inside* the near court is stable across full-court views.  A
    # lower band would sample the run-off apron when the near baseline is high in frame.
    roi = hsv[int(0.52 * h):int(0.66 * h), int(0.34 * w):int(0.66 * w)]
    if roi.size == 0:
        return None, {"reason": "empty surface seed region"}

    # Pick the dominant saturated hue.  Hue is substantially more stable than Lab
    # distance across the severe near/far illumination gradient in indoor footage.
    flat = roi.reshape(-1, 3)
    colourful = flat[flat[:, 1] >= 35]
    if len(colourful) < 100:
        return None, {"reason": "playing surface has insufficient colour support"}
    histogram = np.bincount(colourful[:, 0].astype(np.int32), minlength=180).astype(float)
    histogram = np.convolve(np.r_[histogram[-3:], histogram, histogram[:3]],
                            np.ones(7), mode="valid")
    target_hue = float(np.argmax(histogram))
    hue = hsv[:, :, 0]
    hue_distance = np.minimum(abs(hue - target_hue), 180.0 - abs(hue - target_hue))
    threshold = 25.0
    mask = ((hue_distance <= threshold) & (hsv[:, :, 1] >= 25) &
            (hsv[:, :, 2] >= 25)).astype(np.uint8) * 255
    mask[: int(0.25 * h)] = 0
    # The net, white paint and players split the surface into bands at 480p.  Bridge
    # those narrow occlusions before selecting a component; adjacent courts remain
    # separated by much wider apron/structure gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, {"reason": "no playing-surface component", "threshold": threshold}

    probe_y = int(0.78 * h)
    probe_x = int(0.50 * w)
    label = int(labels[probe_y, probe_x])
    if label == 0:
        candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_TOP] < 0.80 * h]
        if not candidates:
            return None, {"reason": "no central playing surface", "threshold": threshold}
        label = max(candidates, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    component = labels == label

    rows: list[tuple[float, float, float]] = []
    for y in range(int(0.30 * h), min(h, int(0.98 * h))):
        xs = np.flatnonzero(component[y])
        if len(xs) >= 0.12 * w:
            rows.append((float(y), float(xs[0]), float(xs[-1])))
    if len(rows) < 0.18 * h:
        return None, {"reason": "surface component is too short", "threshold": threshold}

    rows_np = np.asarray(rows)
    # Remove the few widest/narrowest scanlines caused by players or line gaps.
    widths = rows_np[:, 2] - rows_np[:, 1]
    keep = (widths >= np.quantile(widths, 0.05)) & (widths <= np.quantile(widths, 0.98))
    rows_fit = rows_np[keep]
    if len(rows_fit) < 20:
        return None, {"reason": "insufficient clean surface rows", "threshold": threshold}

    left_fit = np.polyfit(rows_fit[:, 0], rows_fit[:, 1], 1)
    right_fit = np.polyfit(rows_fit[:, 0], rows_fit[:, 2], 1)
    top_y = float(np.quantile(rows_np[:, 0], 0.015))
    bottom_y = float(np.quantile(rows_np[:, 0], 0.985))
    quad = np.array(
        [
            [np.polyval(left_fit, bottom_y), bottom_y],
            [np.polyval(right_fit, bottom_y), bottom_y],
            [np.polyval(right_fit, top_y), top_y],
            [np.polyval(left_fit, top_y), top_y],
        ],
        dtype=np.float64,
    )
    return quad, {
        "hue": target_hue,
        "threshold": threshold,
        "top_y": top_y,
        "bottom_y": bottom_y,
        "component_area_ratio": float(component.mean()),
    }


def select_court_medoid(
    candidates: Sequence[np.ndarray], image_shape: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a temporally supported candidate instead of a coordinate-wise median.

    A coordinate median can splice corners from different hypotheses.  The medoid is an
    actually observed court, chosen by maximum neighbour support and then minimum
    within-support distance.  Returns the chosen court and the supporting indices.
    """

    if len(candidates) == 0:
        raise ValueError("no court candidates")
    q = np.asarray(candidates, dtype=np.float64)
    diag = float(np.hypot(image_shape[1], image_shape[0]))
    distance = np.sqrt(np.mean((q[:, None] - q[None, :]) ** 2, axis=(2, 3))) / diag
    radius = 0.055
    support = distance <= radius
    counts = support.sum(axis=1)
    med = np.array(
        [np.median(distance[i, support[i]]) for i in range(len(q))], dtype=np.float64
    )
    best = int(np.lexsort((med, -counts))[0])
    return q[best].copy(), np.flatnonzero(support[best])


def propagate_court_corners(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    corners: np.ndarray,
) -> tuple[np.ndarray, bool, dict[str, float]]:
    """Track a reliable court fit across one adjacent frame with sparse optical flow.

    This is deliberately a one-frame bridge, never an accumulating replacement for the
    nine-line detector.  The next scheduled keyframe performs a full paint fit again.
    """
    previous = np.asarray(previous_gray, dtype=np.uint8)
    current = np.asarray(current_gray, dtype=np.uint8)
    q = np.asarray(corners, dtype=np.float64)
    if previous.shape != current.shape or previous.ndim != 2:
        return q.copy(), False, {"reason": 1.0}

    mask = np.zeros(previous.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(q).astype(np.int32), 255)
    features = cv2.goodFeaturesToTrack(
        previous, maxCorners=120, qualityLevel=0.015, minDistance=8,
        mask=mask, blockSize=5,
    )
    if features is None or len(features) < 12:
        return q.copy(), False, {"features": 0.0}
    moved, status, error = cv2.calcOpticalFlowPyrLK(
        previous, current, features, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
    )
    valid = status.reshape(-1).astype(bool)
    src = features.reshape(-1, 2)[valid]
    dst = moved.reshape(-1, 2)[valid]
    if len(src) < 10:
        return q.copy(), False, {"features": float(len(src))}
    transform, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0,
        maxIters=500, confidence=0.99,
    )
    if transform is None or inliers is None or int(inliers.sum()) < 8:
        return q.copy(), False, {"features": float(len(src)), "inliers": 0.0}
    projected = cv2.transform(q.astype(np.float32)[None, :, :], transform)[0].astype(float)
    check = validate_court_corners(projected, current.shape)
    residual = np.linalg.norm(
        cv2.transform(src[None, :, :].astype(np.float32), transform)[0] - dst, axis=1)
    inlier_fraction = float(inliers.mean())
    median_error = float(np.median(residual[inliers.reshape(-1).astype(bool)]))
    ok = check.valid and inlier_fraction >= 0.55 and median_error <= 1.6
    return (projected if ok else q.copy()), ok, {
        "features": float(len(src)), "inlier_fraction": inlier_fraction,
        "median_error_px": median_error,
    }
