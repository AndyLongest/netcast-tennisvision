import cv2
import numpy as np

from netcast_tennisvision.vision.scoreboard import ScoreboardChanges, score_panel


def panel(score):
    frame = np.zeros((360, 640, 3), np.uint8)
    cv2.rectangle(frame, (0, 328), (120, 359), (180, 70, 20), -1)
    cv2.putText(frame, str(score), (87, 341), 0, .4, (255, 255, 255), 1)
    cv2.putText(frame, "0", (87, 355), 0, .4, (255, 255, 255), 1)
    return frame


def test_persistent_numeric_change_detects_clipped_point_boundary():
    detector = ScoreboardChanges(30)
    assert score_panel(panel(0)) is not None
    for i in range(15):
        detector.update(panel(0), i)
    for i in range(15, 30):
        detector.update(panel(15), i)
    assert detector.boundaries == [15]


def test_flash_and_opposite_speed_badge_do_not_split_rallies():
    detector = ScoreboardChanges(30)
    for i in range(15):
        detector.update(panel(0), i)
    detector.update(panel(15), 15)
    for i in range(16, 40):
        frame = panel(0)
        cv2.putText(frame, str(i), (550, 340), 0, .5, (255, 255, 255), 1)
        detector.update(frame, i)
    assert detector.boundaries == []
    assert score_panel(np.zeros((360, 640, 3), np.uint8)) is None
