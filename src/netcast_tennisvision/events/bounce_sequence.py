"""Temporal tennis-bounce classifier built from an open trajectory dataset.

The ArtLabss tennis-tracking project labels individual bounce frames and uses the
previous 20 ball positions and velocities with a time-series forest.  This module
keeps that useful idea, but centres a 21-frame window on the decision frame and
normalises coordinates by image size so the model is not tied to 1920x1080 video.

The classifier is deliberately only one signal in the final event detector.  It
cannot distinguish a racket hit from a bounce by itself because the public dataset
has binary bounce labels; player/racket proximity and court geometry remain separate
features in the main pipeline.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

REFERENCE_URL = (
    "https://raw.githubusercontent.com/ArtLabss/tennis-tracking/"
    "3a633be55f13667649c385ff76055176bf074c6b/bigDF.csv"
)
REFERENCE_SHA256 = "722182de1279cd51accf3444388150e13e26908ec9718993f1d61417236e267a"
WINDOW_RADIUS = 10


def ensure_reference_csv(path: Path) -> Path:
    """Download the public-domain reference trajectory once and verify its hash."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = urlopen(REFERENCE_URL, timeout=30).read()
        if hashlib.sha256(payload).hexdigest() != REFERENCE_SHA256:
            raise RuntimeError("The open bounce reference changed; refusing unverified data")
        path.write_bytes(payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        raise RuntimeError(f"Invalid bounce reference checksum: {path}")
    return path


def _interpolate(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=float)
    valid = np.isfinite(coords).all(axis=1)
    out = coords.copy()
    frame = np.arange(len(coords))
    for axis in range(2):
        good = valid & np.isfinite(coords[:, axis])
        if good.sum() >= 2:
            out[:, axis] = np.interp(frame, frame[good], coords[good, axis])
        elif good.sum() == 1:
            out[:, axis] = coords[good, axis][0]
        else:
            out[:, axis] = 0.0
    return out, valid


def _feature_matrix(coords: np.ndarray, width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """One camera-normalised motion descriptor for every centred 21-frame window."""
    xy, visible = _interpolate(coords)
    scale = np.array([max(float(width), 1.0), max(float(height), 1.0)])
    xy = xy / scale
    vx, vy = np.gradient(xy[:, 0]), np.gradient(xy[:, 1])
    ax, ay = np.gradient(vx), np.gradient(vy)
    speed = np.hypot(vx, vy)
    curvature = vx * ay - vy * ax
    channels = np.column_stack((xy[:, 0], xy[:, 1], vx, vy, ax, ay, speed, curvature,
                                visible.astype(float)))
    radius = WINDOW_RADIUS
    padded = np.pad(channels, ((radius, radius), (0, 0)), mode="edge")
    rows = []
    for i in range(len(coords)):
        window = padded[i:i + 2 * radius + 1].copy()
        # Local offsets describe trajectory shape independently of where the camera put it.
        window[:, 0] -= window[radius, 0]
        window[:, 1] -= window[radius, 1]
        rows.append(np.r_[window.T.ravel(), channels[i, 0:2]])
    return np.asarray(rows, np.float32), visible


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xs, ys, labels = [], [], []
    with ensure_reference_csv(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            labels.append(int(row["bounce"]))
    return np.column_stack((xs, ys)), np.asarray(labels, dtype=np.uint8)


def train_open_classifier(reference_csv: Path) -> ExtraTreesClassifier:
    coords, exact = load_reference(reference_csv)
    features, _ = _feature_matrix(coords, 1920.0, 1080.0)
    # A human label can be one frame either side of the actual compressed-video contact.
    target = np.maximum.reduce((exact, np.roll(exact, 1), np.roll(exact, -1)))
    target[:WINDOW_RADIUS] = 0
    target[-WINDOW_RADIUS:] = 0
    model = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=29,
        n_jobs=-1,
    )
    model.fit(features, target)
    model.reference_rows_ = len(target)
    model.reference_bounces_ = int(exact.sum())
    return model


def bounce_probabilities(
    model: ExtraTreesClassifier,
    coords: np.ndarray,
    width: float,
    height: float,
    min_visible: int = 8,
) -> np.ndarray:
    """Return a bounce probability at every frame; unreliable windows are NaN."""
    features, visible = _feature_matrix(coords, width, height)
    probability = model.predict_proba(features)[:, 1]
    support = np.convolve(visible.astype(int), np.ones(2 * WINDOW_RADIUS + 1, int), mode="same")
    probability[support < min_visible] = np.nan
    return probability
