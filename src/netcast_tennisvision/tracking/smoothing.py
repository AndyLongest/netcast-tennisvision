"""Kalman/RTS segment smoothing and conservative offline trajectory cleanup."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import near_player


def build_segment(
    start: int,
    end: int,
    observations: dict[int, tuple[float, ...]],
    transition: np.ndarray,
    observation: np.ndarray,
    process_noise: np.ndarray,
    measurement_noise: np.ndarray,
    initial_covariance: np.ndarray,
    track_id: int,
    termination: str,
) -> dict[str, Any]:
    """Run the forward filter and backward RTS pass for one confirmed track."""
    first = observations[min(observations)]
    state = np.array([first[0], first[1], 0.0, 0.0], dtype=float)
    covariance = initial_covariance.copy()
    segment: dict[str, Any] = {
        "frames": [],
        "x_pred": [],
        "P_pred": [],
        "x_post": [],
        "P_post": [],
        "seen": [],
        "meas": [],
        "confidence": [],
        "track_id": track_id,
        "motion_mode": [],
        "termination": termination,
    }
    coast = 0
    last_confidence = 0.0
    inferred_spatial = max(1.0, float(np.sqrt(initial_covariance[0, 0]) / 2.0))
    max_prediction_speed = 18.0 * inferred_spatial

    def clamp_velocity(value: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(value[2:]))
        if speed <= max_prediction_speed:
            return value
        value = value.copy()
        value[2:] *= max_prediction_speed / speed
        return value

    for frame in range(start, end + 1):
        if frame == start:
            predicted, predicted_covariance = state.copy(), covariance.copy()
        else:
            predicted = transition @ state
            predicted_covariance = transition @ covariance @ transition.T + process_noise
        value = observations.get(frame)
        if value is not None:
            measurement = np.asarray(value[:2], dtype=float)
            innovation_covariance = (
                observation @ predicted_covariance @ observation.T + measurement_noise
            )
            innovation = measurement - observation @ predicted
            mahalanobis = float(
                innovation @ np.linalg.pinv(innovation_covariance) @ innovation
            )
            gain = (
                predicted_covariance
                @ observation.T
                @ np.linalg.pinv(innovation_covariance)
            )
            state = clamp_velocity(predicted + gain @ innovation)
            covariance = (np.eye(4) - gain @ observation) @ predicted_covariance
            coast = 0
            detector_confidence = float(value[2]) if len(value) >= 3 else 1.0
            agreement = float(np.exp(-0.08 * min(mahalanobis, 25.0)))
            last_confidence = float(
                np.clip((0.55 + 0.45 * detector_confidence) * agreement, 0.05, 1.0)
            )
        else:
            state, covariance = clamp_velocity(predicted), predicted_covariance
            coast += 1
        segment["frames"].append(frame)
        segment["x_pred"].append(predicted.copy())
        segment["P_pred"].append(predicted_covariance.copy())
        segment["x_post"].append(state.copy())
        segment["P_post"].append(covariance.copy())
        segment["seen"].append(value is not None)
        segment["meas"].append(tuple(value[:2]) if value is not None else None)
        segment["motion_mode"].append(
            str(value[3]) if value is not None and len(value) >= 4 else "flight"
        )
        confidence = (
            last_confidence * float(np.exp(-coast / 10.0))
            if value is None
            else last_confidence
        )
        segment["confidence"].append(confidence)

    segment["smooth"] = rts_smooth(segment, transition)
    _anchor_real_measurements(segment)
    _cap_synthetic_speed(segment, max_prediction_speed)
    _support_occlusion_confidence_from_both_sides(segment)
    return segment


def rts_smooth(segment: dict[str, Any], transition: np.ndarray) -> list[np.ndarray]:
    """Backward Rauch–Tung–Striebel smoothing pass."""
    count = len(segment["x_post"])
    states: list[np.ndarray] = [None] * count  # type: ignore[list-item]
    covariances: list[np.ndarray] = [None] * count  # type: ignore[list-item]
    states[-1], covariances[-1] = segment["x_post"][-1], segment["P_post"][-1]
    for index in range(count - 2, -1, -1):
        gain = (
            segment["P_post"][index]
            @ transition.T
            @ np.linalg.pinv(segment["P_pred"][index + 1])
        )
        states[index] = segment["x_post"][index] + gain @ (
            states[index + 1] - segment["x_pred"][index + 1]
        )
        covariances[index] = segment["P_post"][index] + gain @ (
            covariances[index + 1] - segment["P_pred"][index + 1]
        ) @ gain.T
    return states


def repair_isolated_midflight_backtracks(
    segment: dict[str, Any],
    frames_meta: list[dict[str, Any]],
    spatial: float,
) -> int:
    """Repair a 1–2 frame backwards spike only when both sides prove continuation."""
    measurements = segment["meas"]
    modes = segment["motion_mode"]
    repaired: set[int] = set()
    threshold = 6.0 * spatial
    min_progress = 1.25 * spatial

    for island_length in (2, 1):
        for start in range(2, len(measurements) - island_length - 1):
            island = range(start, start + island_length)
            if any(index in repaired or measurements[index] is None for index in island):
                continue
            left2, left = start - 2, start - 1
            right, right2 = start + island_length, start + island_length + 1
            if any(measurements[index] is None for index in (left2, left, right, right2)):
                continue
            if any(
                modes[index] in {"player_hit", "bounce"}
                for index in range(left2, right2 + 1)
            ):
                continue
            points = {
                index: np.asarray(measurements[index], dtype=float)
                for index in range(left2, right2 + 1)
                if measurements[index] is not None
            }
            if any(
                near_player(frames_meta[segment["frames"][index]], point, spatial)
                for index, point in points.items()
            ):
                continue
            before_dy = float(points[left][1] - points[left2][1])
            after_dy = float(points[right2][1] - points[right][1])
            if abs(before_dy) < min_progress or abs(after_dy) < min_progress:
                continue
            if before_dy * after_dy <= 0.0:
                continue
            direction = float(np.sign(before_dy + after_dy))
            if direction * float(points[right][1] - points[left][1]) < min_progress:
                continue

            expected: dict[int, np.ndarray] = {}
            worst_backstep = 0.0
            for offset, index in enumerate(island, start=1):
                fraction = offset / float(island_length + 1)
                expected[index] = points[left] + fraction * (points[right] - points[left])
                deviation = direction * float(points[index][1] - expected[index][1])
                worst_backstep = min(worst_backstep, deviation)
            if worst_backstep > -threshold:
                continue
            for index in island:
                segment["smooth"][index][:2] = expected[index]
                repaired.add(index)
    return len(repaired)


def _anchor_real_measurements(segment: dict[str, Any]) -> None:
    for index, value in enumerate(segment["meas"]):
        if value is not None:
            segment["smooth"][index][:2] = np.asarray(value, dtype=float)


def _cap_synthetic_speed(segment: dict[str, Any], maximum: float) -> None:
    passes = (
        (range(1, len(segment["smooth"])), True),
        (range(len(segment["smooth"]) - 2, -1, -1), False),
    )
    for indices, forward in passes:
        for index in indices:
            if segment["seen"][index]:
                continue
            neighbour = index - 1 if forward else index + 1
            delta = segment["smooth"][index][:2] - segment["smooth"][neighbour][:2]
            distance = float(np.linalg.norm(delta))
            if distance > maximum:
                segment["smooth"][index][:2] = (
                    segment["smooth"][neighbour][:2] + delta * (maximum / distance)
                )


def _support_occlusion_confidence_from_both_sides(segment: dict[str, Any]) -> None:
    next_seen: int | None = None
    next_confidence = 0.0
    for index in range(len(segment["frames"]) - 1, -1, -1):
        if segment["seen"][index]:
            next_seen = index
            next_confidence = segment["confidence"][index]
            continue
        if next_seen is None:
            continue
        previous_seen = next(
            (other for other in range(index - 1, -1, -1) if segment["seen"][other]),
            None,
        )
        if previous_seen is None:
            continue
        evidence_distance = max(index - previous_seen, next_seen - index)
        supported = min(segment["confidence"][previous_seen], next_confidence)
        supported *= float(np.exp(-evidence_distance / 18.0))
        segment["confidence"][index] = max(segment["confidence"][index], supported)
