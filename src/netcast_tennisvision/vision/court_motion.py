"""Carry a confirmed court through small camera reframings, without refitting lines."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


class ConfirmedCourtMotion:
    """Register each image to the original calibration; never accumulate drift."""

    def __init__(self, reference: np.ndarray, corners: np.ndarray):
        self.corners = np.asarray(corners, float).copy()
        self.height, self.width = reference.shape[:2]
        self.size = (640, max(1, round(640 * self.height / self.width)))
        self.orb = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
        self.mask = np.zeros(self.size[::-1], np.uint8)
        h, w = self.mask.shape
        # Ignore fixed broadcast graphics and the scoreboard: they do not move
        # with the photographed court when the editor changes the crop.
        self.mask[int(.125*h):int(.83*h), int(.04*w):int(.95*w)] = 255
        self.keys, self.descriptors = self.orb.detectAndCompute(self.gray(reference), self.mask)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.previous = self.corners.copy()

    def gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, self.size, interpolation=cv2.INTER_AREA)

    def update(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        keys, descriptors = self.orb.detectAndCompute(self.gray(frame), self.mask)
        if self.descriptors is None or descriptors is None:
            return self.previous.copy(), False
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < .72*p[1].distance]
        if len(good) < 30:
            return self.previous.copy(), False
        source = np.float32([self.keys[p.queryIdx].pt for p in good])
        target = np.float32([keys[p.trainIdx].pt for p in good])
        transform, mask = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if transform is None or mask is None or mask.sum() < 25 or mask.mean() < .50:
            return self.previous.copy(), False
        support = source[mask.ravel().astype(bool)]
        if np.ptp(support[:, 0]) < .25*self.size[0] or np.ptp(support[:, 1]) < .20*self.size[1]:
            return self.previous.copy(), False
        scale = np.asarray([self.size[0]/self.width, self.size[1]/self.height])
        small = self.corners * scale
        projected = cv2.transform(small.astype(np.float32)[None], transform)[0]
        movement = np.linalg.norm(projected-small, axis=1)
        # Preserve exact trusted coordinates for an unchanged camera (including
        # the frozen demo); only measured reframing changes the homography.
        candidate = self.corners if movement.max() <= 3.5 else projected / scale
        if np.max(np.linalg.norm((candidate-self.previous)*scale, axis=1)) > .75:
            self.previous = candidate.copy()
        if candidate is self.corners:
            self.previous = self.corners.copy()
        return self.previous.copy(), True


def registered_video_courts(video, reference, corners, count, *, cache_dir=None):
    """Return geometry per native frame, independently of ball observations."""
    cache = None
    if cache_dir is not None:
        path = Path(video)
        identity = f"{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}:{count}:v2"
        signature = hashlib.sha256(identity.encode() + reference.tobytes()
                                   + np.asarray(corners, float).tobytes()).hexdigest()[:24]
        cache = Path(cache_dir) / f"court_motion_{signature}.npz"
        if cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as saved:
                    q, valid, times = saved["corners"], saved["verified"], saved["times"]
                if (q.shape == (count, 4, 2) and valid.shape == times.shape == (count,)
                        and np.isfinite(q).all() and np.isfinite(times).all()):
                    return q, valid, times
            except (OSError, ValueError, KeyError):
                pass
    registration = ConfirmedCourtMotion(reference, corners)
    capture = cv2.VideoCapture(str(video))
    result, valid, times = [], [], []
    try:
        for _ in range(count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("球场画面对齐未能读取完整视频")
            q, matched = registration.update(frame)
            result.append(q)
            valid.append(matched)
            times.append(capture.get(cv2.CAP_PROP_POS_MSEC) / 1000)
    finally:
        capture.release()
    result, valid, times = np.asarray(result), np.asarray(valid, bool), np.asarray(times)
    if not np.isfinite(times).all() or (count > 1 and np.any(np.diff(times) <= 0)):
        raise RuntimeError("视频帧时间戳无效，无法可靠同步落点与原视频")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, corners=result, verified=valid, times=times)
    return result, valid, times
