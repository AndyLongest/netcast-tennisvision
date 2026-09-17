from __future__ import annotations

import numpy as np
import pytest

from netcast_tennisvision.events.line_call import (
    classify_line_call,
    world_uncertainty_from_homography,
)


def test_clear_singles_calls_use_outer_line_edges() -> None:
    assert classify_line_call(5.0, 10.0, uncertainty_m=0.02).call == "in"
    assert classify_line_call(1.10, 10.0, uncertainty_m=0.02).call == "out"


def test_ball_whose_footprint_definitely_touches_the_line_is_in() -> None:
    result = classify_line_call(1.35, 10.0, uncertainty_m=0.01)
    assert result.call == "in"
    assert result.nearest_boundary == "left_sideline"
    assert result.signed_margin_m == pytest.approx(-0.02)


def test_low_quality_close_call_is_sent_to_review() -> None:
    result = classify_line_call(1.28, 10.0, uncertainty_m=0.08)
    assert result.call == "review"
    assert result.confidence == 0.0


def test_doubles_uses_the_full_court_width() -> None:
    assert classify_line_call(0.20, 10.0, match_format="doubles").call == "in"
    assert classify_line_call(0.20, 10.0, match_format="singles").call == "out"


def test_pixel_uncertainty_is_projected_locally_in_world_metres() -> None:
    # One image pixel equals 0.1m in this simple affine homography.
    inverse = np.asarray([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=float)
    uncertainty = world_uncertainty_from_homography(
        (100.0, 50.0), inverse,
        touchdown_uncertainty_px=2.0,
        calibration_uncertainty_px=1.0,
        sigma_multiplier=2.0,
    )
    assert uncertainty == pytest.approx(2 * np.hypot(2.0, 1.0) * 0.1, rel=1e-5)
