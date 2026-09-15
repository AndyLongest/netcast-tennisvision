import cv2
import numpy as np

from netcast_tennisvision.vision.camera_profiles import (
    match_reference_frame,
    remember_camera_calibration,
)


def feature_frame(width=640, height=360):
    frame = np.zeros((height, width, 3), np.uint8)
    rng = np.random.default_rng(7)
    for x, y in rng.integers([20, 20], [width - 20, height - 20], size=(180, 2)):
        cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)
    cv2.rectangle(frame, (120, 55), (520, 315), (70, 190, 70), 3)
    return frame


def test_same_camera_transforms_normalized_corners():
    reference = feature_frame()
    matrix = np.float32([[1, 0, 2], [0, 1, 1]])
    current = cv2.warpAffine(reference, matrix, (640, 360))
    corners = [[0.2, 0.85], [0.8, 0.85], [0.65, 0.2], [0.35, 0.2]]

    match = match_reference_frame(reference, current, corners)

    assert match is not None
    assert match["inliers"] >= 20
    assert match["alignment_px"] < 3.5
    assert np.allclose(match["corners"], corners)


def test_unrelated_camera_is_rejected():
    reference = feature_frame()
    unrelated = np.random.default_rng(19).integers(0, 256, reference.shape, np.uint8)
    corners = [[0.2, 0.85], [0.8, 0.85], [0.65, 0.2], [0.35, 0.2]]
    assert match_reference_frame(reference, unrelated, corners) is None


def test_remembered_profile_is_runtime_json(tmp_path):
    path = tmp_path / "camera_profiles.json"
    frame = feature_frame()
    corners = np.float32([[128, 306], [512, 306], [416, 72], [224, 72]])
    remember_camera_calibration(frame, corners, profile_path=path)
    body = path.read_text(encoding="utf-8")
    assert '"version": 1' in body
    assert '"reference_jpeg"' in body
    assert '"plate_png"' in body
