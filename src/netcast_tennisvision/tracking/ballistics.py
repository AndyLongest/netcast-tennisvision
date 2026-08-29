"""Gravity-constrained 3D prediction for short monocular occlusions.

The detector and association stages remain authoritative.  This module only replaces
synthetic positions in already-confirmed gaps when a calibrated 3D fit is trustworthy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import camera_centre_from_homography, homography_point, near_player

GRAVITY_M_S2 = 9.81
MIN_OBSERVATIONS = 5
MAX_FIT_OBSERVATIONS = 9
FIT_RADIUS_FRAMES = 10


def predict_ballistic_pixel(
    frames_meta: list[dict[str, Any]],
    observations: list[tuple[int, tuple[float, ...]]],
    target_frame: int,
    fps: float,
    spatial: float,
    frame_size: tuple[int, int],
) -> tuple[np.ndarray, float] | None:
    """Fit a gravity-constrained 3D flight arc to calibrated monocular rays.

    Each 2D observation defines a camera ray.  The unknown 3D trajectory obeys
    X/Y constant velocity and ``Z(t) = Z0 + Vz*t - g*t²/2``.  The returned scalar is
    horizontal metres per frame, useful to perspective-aware callers.
    """
    if len(observations) < MIN_OBSERVATIONS:
        return None
    observations = sorted(
        sorted(observations, key=lambda item: abs(item[0] - target_frame))[
            :MAX_FIT_OBSERVATIONS
        ],
        key=lambda item: item[0],
    )
    reference_frame = observations[-1][0]
    if (
        target_frame < observations[0][0] - MAX_FIT_OBSERVATIONS
        or target_frame > reference_frame + MAX_FIT_OBSERVATIONS
    ):
        return None

    rows: list[np.ndarray] = []
    right_hand_side: list[float] = []
    samples: list[tuple[int, np.ndarray, np.ndarray, float, np.ndarray]] = []
    for frame, value in observations:
        meta = frames_meta[frame]
        matrix, inverse = meta.get("M"), meta.get("M_inv")
        if matrix is None or inverse is None:
            continue
        pose = camera_centre_from_homography(np.asarray(matrix, float), frame_size)
        ground = homography_point(np.asarray(inverse, float), np.asarray(value[:2], float))
        if pose is None or ground is None:
            continue
        camera_xy, camera_z = pose
        time_s = (frame - reference_frame) / fps
        ray_x = float((ground[0] - camera_xy[0]) / camera_z)
        ray_y = float((ground[1] - camera_xy[1]) / camera_z)
        rows.extend(
            (
                np.array([1.0, time_s, 0.0, 0.0, ray_x, ray_x * time_s]),
                np.array([0.0, 0.0, 1.0, time_s, ray_y, ray_y * time_s]),
            )
        )
        half_gt2 = 0.5 * GRAVITY_M_S2 * time_s * time_s
        right_hand_side.extend(
            (float(ground[0] + ray_x * half_gt2), float(ground[1] + ray_y * half_gt2))
        )
        samples.append(
            (
                frame,
                np.asarray(value[:2], float),
                camera_xy,
                camera_z,
                np.asarray(matrix, float),
            )
        )
    if len(samples) < MIN_OBSERVATIONS:
        return None

    solution = _robust_gravity_fit(np.vstack(rows), np.asarray(right_hand_side), len(samples))
    if solution is None:
        return None
    x0, velocity_x, y0, velocity_y, z0, velocity_z = map(float, solution)
    horizontal_speed = float(np.hypot(velocity_x, velocity_y))
    if not 0.5 <= horizontal_speed <= 65.0:
        return None

    reprojection_errors: list[float] = []
    heights: list[float] = []
    for frame, measured, camera_xy, camera_z, matrix in samples:
        time_s = (frame - reference_frame) / fps
        projected = _project_state(
            x0,
            velocity_x,
            y0,
            velocity_y,
            z0,
            velocity_z,
            time_s,
            camera_xy,
            camera_z,
            matrix,
        )
        if projected is None:
            return None
        pixel, height_m = projected
        heights.append(height_m)
        reprojection_errors.append(float(np.linalg.norm(pixel - measured)))
    if min(heights) < -0.35 or max(heights) > 5.5:
        return None
    if (
        float(np.median(reprojection_errors)) > 3.0 * spatial
        or float(np.percentile(reprojection_errors, 90)) > 7.0 * spatial
    ):
        return None

    target_meta = frames_meta[target_frame]
    matrix = target_meta.get("M")
    if matrix is None:
        return None
    pose = camera_centre_from_homography(np.asarray(matrix, float), frame_size)
    if pose is None:
        return None
    time_s = (target_frame - reference_frame) / fps
    projected = _project_state(
        x0,
        velocity_x,
        y0,
        velocity_y,
        z0,
        velocity_z,
        time_s,
        pose[0],
        pose[1],
        np.asarray(matrix, float),
    )
    if projected is None:
        return None
    pixel, height_m = projected
    width, height = frame_size
    if not -0.35 <= height_m <= 5.5:
        return None
    if not (
        -0.15 * width <= pixel[0] <= 1.15 * width
        and -0.15 * height <= pixel[1] <= 1.15 * height
    ):
        return None
    return pixel, horizontal_speed / fps


def repair_occluded_flight(
    segment: dict[str, Any],
    frames_meta: list[dict[str, Any]],
    fps: float,
    spatial: float,
    frame_size: tuple[int, int],
) -> int:
    """Replace trustworthy missing-frame predictions without touching detections."""
    repaired = 0
    count = len(segment["frames"])
    event_indices = [
        index
        for index, mode in enumerate(segment["motion_mode"])
        if mode in {"player_hit", "bounce"}
    ]
    for index in range(count):
        if segment["seen"][index]:
            continue
        frame = segment["frames"][index]
        current = np.asarray(segment["smooth"][index][:2], dtype=float)
        if near_player(frames_meta[frame], current, spatial):
            continue
        left_event = max((event for event in event_indices if event < index), default=-1)
        right_event = min((event for event in event_indices if event > index), default=count)
        observations = _flight_observations(
            segment, index, left_event + 1, right_event
        )
        prediction = predict_ballistic_pixel(
            frames_meta, observations, frame, fps, spatial, frame_size
        )
        if prediction is None:
            continue
        pixel, _ = prediction
        if float(np.linalg.norm(pixel - current)) > 30.0 * spatial:
            continue
        segment["smooth"][index][:2] = pixel
        repaired += 1
    return repaired


def _robust_gravity_fit(
    design: np.ndarray, target: np.ndarray, sample_count: int
) -> np.ndarray | None:
    """Three-pass Huber-like weighted least squares with rank checking."""
    weights = np.ones(sample_count, dtype=float)
    solution = None
    for _ in range(3):
        row_weights = np.repeat(np.sqrt(weights), 2)
        solution, _, rank, singular = np.linalg.lstsq(
            design * row_weights[:, None], target * row_weights, rcond=None
        )
        if rank < 6 or singular[-1] <= 1e-8 * singular[0]:
            return None
        pair_error = (design @ solution - target).reshape(-1, 2)
        error = np.linalg.norm(pair_error, axis=1)
        robust_scale = max(
            0.015, 1.4826 * float(np.median(np.abs(error - np.median(error))))
        )
        weights = np.minimum(1.0, (2.5 * robust_scale) / np.maximum(error, 1e-9))
    return solution


def _project_state(
    x0: float,
    velocity_x: float,
    y0: float,
    velocity_y: float,
    z0: float,
    velocity_z: float,
    time_s: float,
    camera_xy: np.ndarray,
    camera_z: float,
    homography: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    height_m = z0 + velocity_z * time_s - 0.5 * GRAVITY_M_S2 * time_s * time_s
    depth_scale = 1.0 - height_m / camera_z
    if depth_scale <= 0.08:
        return None
    world_xy = np.array([x0 + velocity_x * time_s, y0 + velocity_y * time_s])
    ground = camera_xy + (world_xy - camera_xy) / depth_scale
    pixel = homography_point(homography, ground)
    return None if pixel is None else (pixel, height_m)


def _flight_observations(
    segment: dict[str, Any], index: int, left: int, right: int
) -> list[tuple[int, tuple[float, ...]]]:
    observations = []
    for other in range(max(left, index - FIT_RADIUS_FRAMES), min(right, index + FIT_RADIUS_FRAMES + 1)):
        measurement = segment["meas"][other]
        if measurement is None or segment["motion_mode"][other] != "flight":
            continue
        observations.append(
            (
                segment["frames"][other],
                (
                    float(measurement[0]),
                    float(measurement[1]),
                    float(segment["confidence"][other]),
                ),
            )
        )
    return observations
