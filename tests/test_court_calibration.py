import numpy as np
import pytest

from netcast_tennisvision.vision.court_calibration import (
    automatic_calibration_confidence,
    calibration_preview_score,
    validate_manual_calibration,
)

WORLD = np.array([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]], np.float32)


def test_manual_court_accepts_extra_space_below_near_baseline():
    from netcast_tennisvision.vision.court_registration import validate_court_corners

    corners = np.array([[105, 300], [733, 306], [512, 116], [331, 114]])
    np.testing.assert_array_equal(
        validate_manual_calibration(corners, (480, 854), WORLD), corners,
    )
    automatic = validate_court_corners(corners, (480, 854))
    assert automatic.reasons == (
        "near baseline is too high for the supported wide-court view",
    )


def test_live_api_accepts_confirmed_court_high_in_frame():
    from netcast_tennisvision.api.server import parse_live_court_corners

    corners = [[0.1, 0.6], [0.9, 0.6], [0.65, 0.2], [0.35, 0.2]]
    assert parse_live_court_corners(corners) == corners


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


def test_manual_corners_must_form_valid_geometry():
    valid = [[105, 421], [733, 427], [512, 237], [331, 235]]
    assert validate_manual_calibration(valid, (480, 854), WORLD).shape == (4, 2)
    with pytest.raises(ValueError):
        validate_manual_calibration([[10, 10]] * 4, (480, 854), WORLD)


@pytest.mark.parametrize("corners", [
    [[0.49, 0.5], [0.51, 0.5], [0.51, 0.49], [0.49, 0.49]],  # tiny rectangle
    [[0, 1], [1, 1], [1, 0], [0, 0]],  # whole image; equal baseline widths
    [[0.1, 0.8], [0.9, 0.8], [0.51, 0.2], [0.49, 0.2]],  # strong perspective
    [[0, 0.9], [0.9, 0.9], [0.95, 0.8], [0.8, 0.1]],  # asymmetric sides/drift
    [[0, 0], [1, 0], [1, 1], [0, 1]],  # near/far labels are user supplied
    [[-0.5, 1.5], [1.5, 1.5], [1.5, -0.5], [-0.5, -0.5]],  # outside frame
])
def test_manual_and_live_accept_geometry_without_view_priors(corners):
    from netcast_tennisvision.api.server import parse_live_court_corners
    from netcast_tennisvision.streaming.live_experiment import _camera_corners

    points = np.asarray(corners)
    np.testing.assert_array_equal(validate_manual_calibration(points, (1, 1), WORLD), points)
    assert parse_live_court_corners(corners) == corners
    np.testing.assert_allclose(_camera_corners(1280, 720, corners), points * [1280, 720])


@pytest.mark.parametrize("corners", [
    [[0, 0], [1, 1], [1, 0], [0, 1]],  # crossing
    [[0, 0], [1, 0], [0.2, 0.2], [0, 1]],  # concave
    [[0, 0], [0, 0], [1, 1], [0, 1]],  # repeated
    [[0, 0], [0.5, 0], [1, 0], [0, 1]],  # three collinear
    [[0, 0], [1, 0], [1, float("inf")], [0, 1]],
    [[0, 0], [1, 0], [1, float("nan")], [0, 1]],
])
def test_manual_and_live_reject_invalid_geometry(corners):
    from netcast_tennisvision.api.server import parse_live_court_corners

    with pytest.raises(ValueError):
        validate_manual_calibration(corners, (1, 1), WORLD)
    with pytest.raises(ValueError):
        parse_live_court_corners(corners)


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
