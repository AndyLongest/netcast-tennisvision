"""Camera, court, and player geometry used by trajectory association."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


def candidate_xy(candidate: Iterable[float]) -> tuple[float, float]:
    values = tuple(candidate)
    return float(values[0]), float(values[1])


def homography_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    """Apply a 3x3 homography and reject points at infinity."""
    q = np.asarray(matrix, dtype=float) @ np.array([point[0], point[1], 1.0], dtype=float)
    if not np.all(np.isfinite(q)) or abs(float(q[2])) < 1e-9:
        return None
    return q[:2] / q[2]


def pixel_to_world_local(
    meta: dict[str, Any], point: np.ndarray
) -> tuple[np.ndarray, float, float] | None:
    """Return ground point and local metres/pixel in image x/y directions."""
    matrix = meta.get("M_inv")
    if matrix is None:
        return None
    matrix = np.asarray(matrix, dtype=float)

    try:
        centre = homography_point(matrix, point)
        right = homography_point(matrix, point + (1.0, 0.0))
        down = homography_point(matrix, point + (0.0, 1.0))
    except (FloatingPointError, ZeroDivisionError):
        return None
    if centre is None or right is None or down is None:
        return None
    scale_x = float(np.linalg.norm(right - centre))
    scale_y = float(np.linalg.norm(down - centre))
    if not (1e-5 < scale_x < 2.0 and 1e-5 < scale_y < 2.0):
        return None
    return centre, scale_x, scale_y


def inside_court_projection(meta: dict[str, Any], point: np.ndarray) -> bool:
    corners = meta.get("corners")
    if corners is None:
        return False
    polygon = np.asarray(corners, dtype=float)
    edges = np.roll(polygon, -1, axis=0) - polygon
    offsets = point - polygon
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(cross >= -1e-6) or np.all(cross <= 1e-6))


def near_player(meta: dict[str, Any], point: np.ndarray, spatial: float) -> bool:
    """Whether a point is inside a player's body plus plausible racket reach."""
    boxes = meta.get("person_boxes")
    if boxes is None:
        return False
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        width, height = x2 - x1, y2 - y1
        if height < 10.0 * spatial:
            continue
        pad_x = max(14.0 * spatial, 0.65 * width)
        pad_y = max(10.0 * spatial, 0.30 * height)
        if x1 - pad_x <= point[0] <= x2 + pad_x and y1 - pad_y <= point[1] <= y2 + pad_y:
            return True
    return False


def inside_player_body(meta: dict[str, Any], point: np.ndarray, spatial: float) -> bool:
    """Stricter torso/legs overlap, excluding the wider racket-reach envelope."""
    boxes = meta.get("person_boxes")
    if boxes is None:
        return False
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
        if y2 - y1 < 10.0 * spatial:
            continue
        pad = 1.5 * spatial
        if x1 - pad <= point[0] <= x2 + pad and y1 - pad <= point[1] <= y2 + pad:
            return True
    return False


def launches_toward_opponent(
    frames_meta: list[dict[str, Any]],
    observations: list[tuple[int, np.ndarray, float]],
    spatial: float,
    fps: float = 30.0,
) -> bool:
    """Certify a three-point outgoing flight born beside a player's racket reach."""
    if len(observations) < 3:
        return False
    first_frame, first_point, _ = observations[0]
    _, last_point, _ = observations[-1]
    meta = frames_meta[first_frame]
    net_y, boxes = meta.get("net_y_px"), meta.get("person_boxes")
    if net_y is None or boxes is None or not near_player(meta, first_point, spatial):
        return False
    nearest = min(
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        key=lambda box: abs(0.5 * (box[0] + box[2]) - first_point[0])
        + abs(0.5 * (box[1] + box[3]) - first_point[1]),
    )
    player_y = 0.5 * (nearest[1] + nearest[3])
    toward = -1.0 if player_y > float(net_y) else 1.0
    displacement = last_point - first_point
    velocities = [
        (observations[index][1] - observations[index - 1][1])
        / max(1, observations[index][0] - observations[index - 1][0])
        for index in range(1, len(observations))
    ]
    speeds = [float(np.linalg.norm(velocity)) for velocity in velocities]
    agreement = (
        float(velocities[0] @ velocities[1]) / max(1e-6, speeds[0] * speeds[1])
        if len(velocities) >= 2
        else 1.0
    )
    physically_plausible = all(
        plausible_ball_step(
            frames_meta,
            observations[index - 1][0], observations[index - 1][1],
            observations[index][0], observations[index][1],
            fps=fps, spatial=spatial,
        )
        for index in range(1, len(observations))
    )
    return bool(
        toward * displacement[1] >= 2.0 * spatial
        and np.linalg.norm(displacement) >= 5.0 * spatial
        and np.mean([observation[2] for observation in observations]) >= 0.34
        and physically_plausible
        and agreement >= 0.42
        # A near-baseline serve or groundstroke changes pixel scale very rapidly as it
        # leaves the camera.  World-space plausibility above is authoritative when a
        # court calibration exists; this ratio only rejects truly incoherent blobs.
        and (min(speeds) <= 1e-6 or max(speeds) / max(min(speeds), 1e-6) <= 5.0)
        and (
            len(velocities) < 2
            or np.linalg.norm(velocities[-1] - velocities[-2]) <= 50.0 * spatial
        )
    )


def plausible_ball_step(
    frames_meta: list[dict[str, Any]],
    first_frame: int,
    first_point: np.ndarray,
    second_frame: int,
    second_point: np.ndarray,
    *,
    fps: float,
    spatial: float,
    max_ground_speed_m_s: float = 90.0,
) -> bool:
    """Validate a fast step in court metres, falling back to perspective-safe pixels.

    The ground projection is imperfect while the ball is airborne, so the ceiling is
    intentionally above the fastest recorded tennis serves.  Its purpose is to reject a
    teleport, not to estimate shot speed.  Unlike one global pixel threshold, it does not
    punish a near-camera launch merely because perspective makes it cover many pixels.
    """
    gap = max(1, int(second_frame) - int(first_frame))
    first_meta = frames_meta[int(first_frame)]
    second_meta = frames_meta[int(second_frame)]
    first_inverse = first_meta.get("M_inv")
    second_inverse = second_meta.get("M_inv")
    if first_inverse is not None and second_inverse is not None:
        first_world = homography_point(
            np.asarray(first_inverse, float), np.asarray(first_point, float)
        )
        second_world = homography_point(
            np.asarray(second_inverse, float), np.asarray(second_point, float)
        )
        if first_world is not None and second_world is not None:
            speed_m_s = float(np.linalg.norm(second_world - first_world)) * float(fps) / gap
            # Height-to-ground projection can stretch a real airborne displacement, but
            # values far beyond this ceiling cannot belong to one tennis ball.
            return speed_m_s <= max_ground_speed_m_s
    pixel_speed = float(np.linalg.norm(np.asarray(second_point) - np.asarray(first_point))) / gap
    return pixel_speed <= 60.0 * spatial


def camera_centre_from_homography(
    matrix: np.ndarray, frame_size: tuple[int, int], *, min_height: float = 1.5
) -> tuple[np.ndarray, float] | None:
    """Recover the calibrated camera centre from the court-plane homography."""
    width, height = frame_size
    u0, v0 = width / 2.0, height / 2.0
    centred = np.array([[1, 0, -u0], [0, 1, -v0], [0, 0, 1]], float) @ matrix
    a, b, c = centred[:, 0]
    d, e, f = centred[:, 1]
    focal_sq: list[float] = []
    if abs(c * f) > 1e-12:
        focal_sq.append(float(-(a * d + b * e) / (c * f)))
    if abs(c * c - f * f) > 1e-12:
        focal_sq.append(float(-(a * a + b * b - d * d - e * e) / (c * c - f * f)))
    focal_sq = [value for value in focal_sq if np.isfinite(value) and value > 0.0]
    if not focal_sq:
        return None
    focal = float(np.sqrt(np.mean(focal_sq)))
    intrinsic_inv = np.linalg.inv(
        np.array([[focal, 0, u0], [0, focal, v0], [0, 0, 1.0]], float)
    )
    plane_pose = intrinsic_inv @ np.asarray(matrix, dtype=float)
    scale = 2.0 / (
        np.linalg.norm(plane_pose[:, 0]) + np.linalg.norm(plane_pose[:, 1])
    )
    r1, r2, translation = (
        scale * plane_pose[:, 0],
        scale * plane_pose[:, 1],
        scale * plane_pose[:, 2],
    )
    rotation = np.column_stack([r1, r2, np.cross(r1, r2)])
    u, _, vt = np.linalg.svd(rotation)
    centre = -(u @ vt).T @ translation
    if not np.all(np.isfinite(centre)) or not min_height <= float(centre[2]) <= 30.0:
        return None
    return centre[:2], float(centre[2])
