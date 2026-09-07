import json
from pathlib import Path

import pytest
import torch

from netcast_tennisvision.pipeline.runner import (
    RACKETVISION_BALLTRACK,
    RACKETVISION_BALLTRACK_SHA256,
    sha256,
    verify_racketvision_balltrack,
)

ROOT = Path(__file__).resolve().parents[1]


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


def test_landing_feedback_contract_is_preserved():
    notebook = json.loads((ROOT / "notebooks/tennis_detection.ipynb").read_text("utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert 'flash_zones = [b for b in bounces if b["zone"] != "Out"]' in source
    assert 'b["decision_frame"] + k' in source
    assert 'b["rally_id"] == active_rally' in source
    assert 'b["zone"] == "Out"' in source
