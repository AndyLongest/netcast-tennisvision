import cv2
import numpy as np

from netcast_tennisvision.vision.play_mode import detect_play_mode

WORLD = np.float32([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]])


def _metric_frame(near, far):
    # Build realistic image boxes by projecting their feet from world to a 1280x720 court.
    image = np.float32([[120, 650], [1160, 650], [790, 150], [490, 150]])
    matrix = cv2.getPerspectiveTransform(WORLD, image)
    inverse = np.linalg.inv(matrix)
    boxes = []
    for y, count in ((4.0, near), (19.0, far)):
        for index in range(count):
            x = 3.0 + index * 4.5
            foot = cv2.perspectiveTransform(np.float32([[[x, y]]]), matrix)[0, 0]
            height = 150 if y < 11.885 else 70
            boxes.append([foot[0] - 18, foot[1] - height, foot[0] + 18, foot[1]])
    return {"M_inv": inverse, "person_boxes": np.asarray(boxes, np.float32)}


def test_one_player_on_each_side_is_singles_despite_short_misses():
    frames = [_metric_frame(1, 1) for _ in range(80)] + [_metric_frame(1, 0) for _ in range(20)]
    result = detect_play_mode(frames, sample_stride=1)
    assert result.mode == "singles"
    assert (result.near_players, result.far_players) == (1, 1)


def test_two_players_on_each_side_is_doubles_despite_intermittent_occlusion():
    frames = [_metric_frame(2, 2) for _ in range(70)] + [_metric_frame(1, 2) for _ in range(30)]
    result = detect_play_mode(frames, sample_stride=1)
    assert result.mode == "doubles"
    assert (result.near_players, result.far_players) == (2, 2)


def test_asymmetric_coach_feed_is_training_mode():
    frames = [_metric_frame(1, 2) for _ in range(70)] + [_metric_frame(1, 1) for _ in range(30)]
    result = detect_play_mode(frames, sample_stride=1)
    assert result.mode == "training"
    assert result.label == "训练模式"
    assert (result.near_players, result.far_players) == (1, 2)
