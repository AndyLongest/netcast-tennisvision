import numpy as np
import pytest

from netcast_tennisvision.vision.court_calibration import (
    automatic_calibration_confidence,
    calibration_preview_score,
    validate_manual_calibration,
)

WORLD = np.array([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]], np.float32)


def test_strong_automatic_calibration_skips_manual_step():
    quality = automatic_calibration_confidence(
        sampled_frames=80, detected_frames=55, consensus_support=42,
        fit_error_px=0.7, fit_note="global refit",
    )
    assert quality.automatic
    assert quality.score > 0.9


def test_weak_automatic_calibration_requests_help():
    quality = automatic_calibration_confidence(
        sampled_frames=80, detected_frames=3, consensus_support=1,
        fit_error_px=3.8, fit_note="fallback",
    )
    assert not quality.automatic
    assert quality.reasons


def test_manual_corners_must_form_supported_trapezoid():
    valid = [[105, 421], [733, 427], [512, 237], [331, 235]]
    assert validate_manual_calibration(valid, (480, 854), WORLD).shape == (4, 2)
    with pytest.raises(ValueError):
        validate_manual_calibration([[10, 10]] * 4, (480, 854), WORLD)


def test_calibration_preview_rejects_black_intro_frame():
    black = np.zeros((120, 200, 3), np.uint8)
    visible = np.zeros_like(black)
    visible[25:100, 20:180] = 80
    visible[50:54, 20:180] = 230

    assert calibration_preview_score(
        visible, court_detected=False,
    ) > calibration_preview_score(black, court_detected=False)


def test_detected_court_preview_outranks_unrelated_bright_frame():
    court = np.full((120, 200, 3), 45, np.uint8)
    court[30:34, 20:180] = 220
    title_card = np.full_like(court, 245)
    title_card[:, ::20] = 20

    assert calibration_preview_score(
        court, court_detected=True,
    ) > calibration_preview_score(title_card, court_detected=False)
