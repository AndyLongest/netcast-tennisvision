import json
from pathlib import Path

import numpy as np
import pytest
import torch

from netcast_tennisvision.pipeline.runner import (
    RACKETVISION_BALLTRACK,
    RACKETVISION_BALLTRACK_SHA256,
    sha256,
    verify_racketvision_balltrack,
)
from netcast_tennisvision.vision.racketvision import _decode, _decode_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_training_decoder_retains_alternatives_without_changing_legacy_primary():
    heatmap = np.zeros((40, 60), dtype=np.float32)
    heatmap[5:10, 8:14] = 0.72
    heatmap[22:30, 35:45] = 0.61  # largest component remains the legacy winner

    legacy = _decode(heatmap, 0.5, 2.0, 3.0)
    candidates = _decode_candidates(
        heatmap, 0.5, 2.0, 3.0, max_candidates=8,
    )

    assert candidates[0] == legacy
    assert len(candidates) == 2
    assert candidates[1][:2] == pytest.approx((22.0, 22.5))


def test_training_decoder_marks_subthreshold_alternatives_without_inventing_match_primary():
    heatmap = np.zeros((40, 60), dtype=np.float32)
    heatmap[5:9, 8:12] = 0.42
    heatmap[20:25, 35:40] = 0.36

    candidates = _decode_candidates(
        heatmap, 0.5, 1.0, 1.0,
        max_candidates=8, alternative_threshold=0.30,
    )

    assert _decode(heatmap, 0.5, 1.0, 1.0) is None
    assert len(candidates) == 2
    assert all(candidate[5] == 0.0 for candidate in candidates)


@pytest.mark.assets
def test_public_checkpoint_is_immutable_and_loadable():
    assert verify_racketvision_balltrack() == RACKETVISION_BALLTRACK_SHA256
    assert sha256(RACKETVISION_BALLTRACK) == RACKETVISION_BALLTRACK_SHA256
    state = torch.load(RACKETVISION_BALLTRACK, map_location="cpu", weights_only=True)
    assert sum(tensor.numel() for tensor in state.values()) == 11_341_525
    assert state["down_block_1.conv_1.conv.weight"].shape == (64, 15, 3, 3)
    assert state["predictor.weight"].shape == (4, 64, 1, 1)


def test_notebook_routes_candidates_from_racketvision():
    notebook = json.loads((ROOT / "notebooks/tennis_detection.ipynb").read_text("utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "detect_video_candidates" in source
    assert "racketvision_balltrack_state_v1.pt" in source
    assert "iter_sparse_person_detections" in source
    assert "ball_model = YOLO" not in source


def test_minimap_is_not_suppressed_when_court_fit_is_unavailable():
    notebook = json.loads((ROOT / "notebooks/tennis_detection.ipynb").read_text("utf-8"))
    render_cell = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if "draw_minimap(frame, render_minimap(" in "".join(cell.get("source", []))
        and "encoder.stdin.write(frame.tobytes())" in "".join(cell.get("source", []))
    )
    minimap_block = render_cell[render_cell.rfind("# The minimap is a fixed ITF-court UI layer"):]
    assert "draw_minimap(frame, render_minimap(" in minimap_block
    assert 'if meta["is_court"]' not in minimap_block.split("encoder.stdin.write", 1)[0]


def test_fixed_camera_manual_calibration_cannot_be_revoked_by_weak_line_score():
    notebook = json.loads((ROOT / "notebooks/tennis_detection.ipynb").read_text("utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "calibration_signature = hashlib.sha256(" in source
    assert 'f"cal{calibration_signature}"' in source
    assert "smooth = np.repeat(np.asarray(CALIB_CORNERS, float)[None]" in source
    assert 'm["is_court"] = True if FIXED_COURT else' in source


def test_notebook_exports_temporal_play_mode_and_uses_its_player_counts():
    notebook = json.loads((ROOT / "notebooks/tennis_detection.ipynb").read_text("utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "play_mode_result = detect_play_mode(frames_meta, sample_stride=5)" in source
    assert "expected_players =" in source
    assert '"play_mode": play_mode_result.as_dict()' in source


def test_landing_feedback_contract_is_preserved():
    notebook = json.loads((ROOT / "notebooks/tennis_detection.ipynb").read_text("utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'flash_zones = [b for b in bounces if b["zone"] != "Out"]' in source
    assert 'b["decision_frame"] + k' in source
    assert 'b["rally_id"] == active_rally' in source
    assert 'b["zone"] == "Out"' in source
