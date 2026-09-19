import cv2
import numpy as np

from netcast_tennisvision.vision.court_motion import (
    ConfirmedCourtMotion,
    PeriodicCourtMotion,
    registered_video_courts,
)


def test_five_minute_corrections_hold_geometry_without_per_frame_retry(monkeypatch):
    calls = []
    corners = np.array([[0, 9], [9, 9], [7, 1], [2, 1]], float)

    def register(self, frame):
        calls.append(frame)
        return corners + [0, len(calls)], len(calls) != 2

    monkeypatch.setattr(ConfirmedCourtMotion, "update", register)
    motion = PeriodicCourtMotion(np.zeros((36, 64), np.uint8), corners)
    first, _ = motion.update("start", .1)
    for t in [.2, 10, 299.99, 300.09]:
        held, ok = motion.update("skip", t)
        np.testing.assert_array_equal(held, first)
        assert ok
    _, ok = motion.update("five minutes", 300.1)
    assert not ok
    for t in [301, 500, 600.09]:
        _, ok = motion.update("skip failed retry", t)
        assert not ok
    motion.update("ten minutes", 600.1)
    assert calls == ["start", "five minutes", "ten minutes"]


def test_confirmed_corners_follow_reframing_without_accumulating_drift():
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 256, (360, 640), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    corners = np.array([[50, 290], [590, 290], [440, 90], [200, 90]], float)
    tracker = ConfirmedCourtMotion(frame, corners)
    still, ok = tracker.update(frame)
    assert ok
    np.testing.assert_array_equal(still, corners)
    moved = cv2.warpAffine(frame, np.float32([[1, 0, 0], [0, 1, -12]]), (640, 360))
    shifted, ok = tracker.update(moved)
    assert ok
    np.testing.assert_allclose(shifted, corners + [0, -12], atol=.5)
    restored, ok = tracker.update(frame)
    assert ok
    np.testing.assert_array_equal(restored, corners)


def test_unmatched_frame_does_not_invent_new_corners():
    frame = np.zeros((360, 640), np.uint8)
    corners = np.array([[50, 290], [590, 290], [440, 90], [200, 90]], float)
    result, ok = ConfirmedCourtMotion(frame, corners).update(frame)
    assert not ok
    np.testing.assert_array_equal(result, corners)


def test_native_presentation_times_survive_missing_initial_frames(monkeypatch, tmp_path):
    class Capture:
        def __init__(self, *_):
            self.index = -1

        def read(self):
            self.index += 1
            return True, np.zeros((36, 64, 3), np.uint8)

        def get(self, _):
            return [100, 133.313, 200][self.index]

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    video = tmp_path / "source.mp4"
    video.write_bytes(b"test identity")
    reference = np.zeros((36, 64, 3), np.uint8)
    corners = np.asarray([[0, 35], [63, 35], [45, 10], [20, 10]])
    _, _, times = registered_video_courts(video, reference, corners, 3, cache_dir=tmp_path)
    np.testing.assert_allclose(times, [.100, .133313, .200])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: (_ for _ in ()).throw(AssertionError("cache missed")))
    _, _, cached = registered_video_courts(video, reference, corners, 3, cache_dir=tmp_path)
    np.testing.assert_array_equal(times, cached)
