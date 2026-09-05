import cv2
import numpy as np
import pytest

from netcast_tennisvision.vision.display_correction import (
    apply_display_correction,
    display_homography,
    parse_display_correction,
)

CORNERS = [[0.14, 0.76], [0.88, 0.74], [0.61, 0.32], [0.39, 0.32]]


def test_disabled_display_correction_never_copies_or_changes_a_frame():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    assert apply_display_correction(frame, {"enabled": False}) is frame


def test_full_correction_maps_both_baselines_to_the_same_width():
    matrix = display_homography((1000, 2000, 3), CORNERS, 100)
    source = np.asarray(CORNERS, np.float32) * np.array([2000, 1000], np.float32)
    target = cv2.perspectiveTransform(source[None], matrix)[0]
    near_width = target[1, 0] - target[0, 0]
    far_width = target[2, 0] - target[3, 0]
    assert near_width == pytest.approx(far_width, abs=1e-3)


def test_court_and_far_player_band_are_fitted_inside_the_output_canvas():
    height, width = 1000, 2000
    matrix = display_homography((height, width, 3), CORNERS, 70)
    source = np.asarray(CORNERS, np.float32) * np.array([width, height], np.float32)
    near_left, near_right, far_right, far_left = source
    near_width = near_right[0] - near_left[0]
    far_width = far_right[0] - far_left[0]
    depth = (near_left[1] + near_right[1] - far_right[1] - far_left[1]) / 2
    protected = np.float32([[
        near_left + (-0.08 * near_width, 0.18 * depth),
        near_right + (0.08 * near_width, 0.18 * depth),
        far_right + (0.30 * far_width, -0.28 * depth),
        far_left + (-0.30 * far_width, -0.28 * depth),
    ]])
    target = cv2.perspectiveTransform(protected, matrix)[0]
    assert target[:, 0].min() >= -1e-3
    assert target[:, 1].min() >= -1e-3
    assert target[:, 0].max() <= width + 1e-3
    assert target[:, 1].max() <= height + 1e-3


def test_browser_configuration_is_normalized_and_bounded():
    config = parse_display_correction("30", "[[0.14,0.76],[0.88,0.74],[0.61,0.32],[0.39,0.32]]")
    assert config["enabled"] is True
    assert config["strength"] == 30
    with pytest.raises(ValueError, match="0% 到 100%"):
        parse_display_correction("110", "[]")
