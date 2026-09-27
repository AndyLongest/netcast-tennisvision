from __future__ import annotations

import sys

import pytest

from netcast_tennisvision.vision.racket_pose import (
    OpenMMLabRacketPose,
    RacketPosePrediction,
    RacketPoseUnavailable,
    choose_player_racket,
)


def prediction(x: float, confidence: float = 0.9) -> RacketPosePrediction:
    return RacketPosePrediction.from_upstream(
        {
            "bbox": [[x, 10, x + 20, 50]],
            "bbox_score": confidence,
            "keypoints": [
                [x + 10, 10],
                [x + 10, 35],
                [x + 10, 50],
                [x, 22],
                [x + 20, 22],
            ],
            "keypoint_scores": [0.9, 0.8, 0.95, 0.7, 0.75],
        }
    )


def test_upstream_prediction_is_normalized_to_named_keypoints() -> None:
    racket = prediction(100)
    assert racket.bbox_xyxy == (100.0, 10.0, 120.0, 50.0)
    assert tuple(racket.keypoints) == ("top", "bottom", "handle", "left", "right")
    assert racket.keypoints["handle"].x == 110.0
    assert racket.to_dict()["keypoints"]["handle"]["confidence"] == 0.95


def test_invalid_prediction_is_rejected_at_the_boundary() -> None:
    with pytest.raises(ValueError, match="five canonical"):
        RacketPosePrediction.from_upstream(
            {
                "bbox": [[0, 0, 10, 10]],
                "bbox_score": 0.9,
                "keypoints": [[1, 1]],
                "keypoint_scores": [0.9],
            }
        )


def test_player_association_prefers_handle_near_wrist() -> None:
    near = prediction(100, confidence=0.75)
    far = prediction(300, confidence=0.99)
    assert choose_player_racket([far, near], wrist_points=[(111, 51)]) is near


def test_importing_contract_does_not_import_openmmlab() -> None:
    assert "mmdet" not in sys.modules
    assert "mmpose" not in sys.modules


def test_backend_fails_closed_before_optional_imports_when_models_are_missing(tmp_path) -> None:
    backend = OpenMMLabRacketPose(
        detector_path=tmp_path / "detector.pth",
        pose_path=tmp_path / "pose.pth",
    )
    with pytest.raises(RacketPoseUnavailable, match="missing RacketPose model"):
        backend.load()
