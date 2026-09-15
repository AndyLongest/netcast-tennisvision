"""Persistent fixed-camera court calibration with visual verification.

Profiles are runtime data, never model parameters. A stored quadrilateral is reused only
after local-feature matching proves that the current clip comes from the same static
camera. Failure to match simply falls back to the full calibration path.
"""
from __future__ import annotations

import base64
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from netcast_tennisvision.paths import REPOSITORY_ROOT

PROFILE_PATH = REPOSITORY_ROOT / "data" / "camera_profiles.json"
PROFILE_VERSION = 1
THUMB_WIDTH = 640


def _thumbnail(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, THUMB_WIDTH / max(1, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized


def _encode_thumbnail(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".jpg", _thumbnail(frame), [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not ok:
        raise RuntimeError("无法保存固定机位参考画面")
    return base64.b64encode(encoded).decode("ascii")


def _decode_thumbnail(value: str) -> np.ndarray | None:
    try:
        payload = np.frombuffer(base64.b64decode(value), np.uint8)
        return cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
    except (ValueError, TypeError):
        return None


def _encode_plate(plate: np.ndarray) -> str:
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise RuntimeError("无法保存固定机位球场底图")
    return base64.b64encode(encoded).decode("ascii")


def _decode_plate(value: str) -> np.ndarray | None:
    try:
        payload = np.frombuffer(base64.b64decode(value), np.uint8)
        return cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
    except (ValueError, TypeError):
        return None


def _read_profiles(path: Path = PROFILE_PATH) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, dict) or payload.get("version") != PROFILE_VERSION:
        return []
    profiles = payload.get("profiles")
    return profiles if isinstance(profiles, list) else []


def _write_profiles(profiles: list[dict[str, Any]], path: Path = PROFILE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({
        "version": PROFILE_VERSION,
        "profiles": profiles,
    }, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def match_reference_frame(
    reference: np.ndarray,
    current: np.ndarray,
    normalized_corners: list[list[float]],
) -> dict[str, Any] | None:
    """Transform stored court corners when two frames show the same static camera."""
    reference_gray, current_gray = _thumbnail(reference), _thumbnail(current)
    if abs(reference_gray.shape[1] / reference_gray.shape[0]
           - current_gray.shape[1] / current_gray.shape[0]) > 0.02:
        return None
    orb = cv2.ORB_create(nfeatures=1400, fastThreshold=12)
    key_ref, desc_ref = orb.detectAndCompute(reference_gray, None)
    key_cur, desc_cur = orb.detectAndCompute(current_gray, None)
    if desc_ref is None or desc_cur is None or len(key_ref) < 30 or len(key_cur) < 30:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_ref, desc_cur, k=2)
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < 0.72 * pair[1].distance
    ]
    if len(good) < 24:
        return None
    source = np.float32([key_ref[item.queryIdx].pt for item in good])
    target = np.float32([key_cur[item.trainIdx].pt for item in good])
    transform, mask = cv2.findHomography(source, target, cv2.RANSAC, 2.5)
    if transform is None or mask is None:
        return None
    inliers = int(mask.sum())
    inlier_ratio = inliers / len(good)
    if inliers < 20 or inlier_ratio < 0.55:
        return None

    image_quad = np.float32([
        [0, 0], [reference_gray.shape[1] - 1, 0],
        [reference_gray.shape[1] - 1, reference_gray.shape[0] - 1],
        [0, reference_gray.shape[0] - 1],
    ]).reshape(-1, 1, 2)
    aligned_quad = cv2.perspectiveTransform(image_quad, transform).reshape(-1, 2)
    alignment_px = float(np.mean(np.linalg.norm(aligned_quad - image_quad.reshape(-1, 2), axis=1)))
    # A saved profile means the mount did not move. If feature matching estimates a
    # material camera shift, recalibration is safer than adapting trusted court corners.
    if alignment_px > 3.5:
        return None

    normalized = [[float(x), float(y)] for x, y in normalized_corners]
    if any(not (-0.03 <= x <= 1.03 and -0.03 <= y <= 1.03) for x, y in normalized):
        return None
    return {
        "corners": normalized,
        "inliers": inliers,
        "inlier_ratio": float(inlier_ratio),
        "alignment_px": alignment_px,
        "score": float(min(1.0, inlier_ratio * min(1.0, inliers / 80))),
    }


def _sample_frames(video: Path, count: int = 5) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    total = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = sorted({min(total - 1, int(total * ratio)) for ratio in (0.0, 0.02, 0.05, 0.1, 0.18)})
    frames: list[np.ndarray] = []
    for index in indices[:count]:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok and frame is not None and float(frame.mean()) > 5:
            frames.append(frame)
    capture.release()
    return frames


def find_camera_calibration(
    video: Path,
    frame_shape: tuple[int, int],
    *,
    profile_path: Path = PROFILE_PATH,
) -> dict[str, Any] | None:
    """Return a visually verified prior calibration, or ``None`` for full detection."""
    height, width = frame_shape
    best: dict[str, Any] | None = None
    for frame in _sample_frames(video):
        for profile in _read_profiles(profile_path):
            reference = _decode_thumbnail(str(profile.get("reference_jpeg", "")))
            plate = _decode_plate(str(profile.get("plate_png", "")))
            normalized = profile.get("corners")
            if (reference is None or plate is None
                    or not isinstance(normalized, list) or len(normalized) != 4):
                continue
            match = match_reference_frame(reference, frame, normalized)
            if match is None or (best is not None and match["score"] <= best["score"]):
                continue
            match["corners"] = np.float32([
                [x * width, y * height] for x, y in match["corners"]
            ])
            match["frame"] = frame
            match["plate"] = plate
            match["profile_id"] = profile.get("id")
            best = match
    return best


def remember_camera_calibration(
    frame: np.ndarray,
    corners: np.ndarray,
    *,
    profile_id: str | None = None,
    calibration_plate: np.ndarray | None = None,
    profile_path: Path = PROFILE_PATH,
) -> None:
    """Save one validated fixed-camera calibration for later clips."""
    profiles = _read_profiles(profile_path)
    height, width = frame.shape[:2]
    normalized = [[float(x / width), float(y / height)] for x, y in np.asarray(corners)]
    now = int(time.time())
    if profile_id:
        for profile in profiles:
            if profile.get("id") == profile_id:
                profile["last_used_at"] = now
                if calibration_plate is not None and not profile.get("plate_png"):
                    profile["plate_png"] = _encode_plate(calibration_plate)
                _write_profiles(profiles, profile_path)
                return
    profiles.append({
        "id": uuid.uuid4().hex,
        "created_at": now,
        "last_used_at": now,
        "corners": normalized,
        "reference_jpeg": _encode_thumbnail(frame),
        "plate_png": _encode_plate(calibration_plate if calibration_plate is not None else frame),
    })
    _write_profiles(profiles[-12:], profile_path)
