import numpy as np
import pytest

from netcast_tennisvision.vision.court_calibration import (
    automatic_calibration_confidence,
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
