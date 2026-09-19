"""Optional visual point boundaries from a persistent broadcast score panel.

This reads no player identity or score value. Unsupported panel styles return no
boundaries, leaving the contact timeline authoritative.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def score_panel(frame):
    small = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Recognize the common opaque blue two-row score panel, not the whole
    # broadcast image. A speed badge on the opposite side cannot split points.
    mask = cv2.inRange(hsv, np.array([85, 180, 25]), np.array([135, 255, 220]))
    mask[:300] = 0
    mask[:,180:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 21), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(contour)
        if (x <= 12 and 70 <= w <= 175 and 18 <= h <= 50 and w/h >= 2
                and cv2.contourArea(contour)/(w*h) >= .70):
            return (x / 640, y / 360, w / 640, h / 360)
    return None


class ScoreboardChanges:
    def __init__(self, fps):
        self.confirmation = max(3, round(fps * .35))
        self.region = None
        self.reference = self.pending = None
        self.pending_frame = self.stable = 0
        self.boundaries = []

    def update(self, frame, index):
        if self.region is None:
            # Discovery is occasional; event timing remains native-rate once found.
            if index % self.confirmation:
                return
            self.region = score_panel(frame)
            if self.region is None:
                return
        x, y, w, h = self.region
        height, width = frame.shape[:2]
        # Only the numeric columns. Player names and flags cannot reset a rally.
        crop = frame[round(y*height):round((y+h)*height),
                     round((x+.60*w)*width):round((x+w)*width)]
        if not crop.size:
            return
        patch = cv2.resize(crop, (96, 64), interpolation=cv2.INTER_AREA)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        ink = np.min(patch, axis=2) > 155
        density = float(ink.mean())
        if not .01 <= density <= .40:
            self.pending = None
            self.stable = 0

            return
        if self.reference is None:
            self.reference = ink
            return
        if np.mean(ink != self.reference) < .006:
            self.pending = None
            self.stable = 0
            return
        if self.pending is None or np.mean(ink != self.pending) > .003:
            self.pending = ink
            self.pending_frame = index
            self.stable = 1
            return
        self.stable += 1
        if self.stable >= self.confirmation:
            self.boundaries.append(self.pending_frame)
            self.reference = ink
            self.pending = None
            self.stable = 0


def video_score_boundaries(video, *, count, fps, cache_dir):
    path = Path(video)
    signature = hashlib.sha256(
        f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}:{count}:v2".encode()
    ).hexdigest()[:24]
    cache = Path(cache_dir) / f"score_boundaries_{signature}.json"
    if cache.exists():
        try:
            value = json.loads(cache.read_text())
            if isinstance(value, list) and all(isinstance(i, int) and 0 <= i < count for i in value):
                return value
        except (OSError, ValueError):
            pass
    detector = ScoreboardChanges(fps)
    capture = cv2.VideoCapture(str(path))
    try:
        for i in range(count):
            ok, frame = capture.read()
            if not ok:
                break
            detector.update(frame, i)
    finally:
        capture.release()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(detector.boundaries))
    return detector.boundaries
