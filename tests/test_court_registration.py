import cv2
import numpy as np

from court_registration import (
    court_surface_candidate,
    court_template_score,
    propagate_court_corners,
    select_court_medoid,
    validate_court_corners,
)

WORLD = np.array([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]], np.float32)


def test_valid_behind_baseline_court_passes():
    q = np.array([[105, 421], [733, 427], [512, 237], [331, 235]], float)
    check = validate_court_corners(q, (480, 854), world_quad=WORLD)
    assert check.valid, check.reasons


def test_collapsed_far_baseline_is_rejected():
    q = np.array([[22, 322], [569, 328], [399, 135], [21, 313]], float)
    check = validate_court_corners(q, (480, 854), world_quad=WORLD)
    assert not check.valid
    assert any("collapsed" in reason or "far/near" in reason for reason in check.reasons)


def test_wall_floor_seam_cannot_masquerade_as_far_baseline():
    q = np.array([[104, 423], [731, 429], [458, 188], [389, 187]], float)
    check = validate_court_corners(q, (480, 854), world_quad=WORLD)
    assert not check.valid
    assert "far/near baseline ratio is implausible" in check.reasons


def test_surface_proposal_recovers_coloured_court_not_wall_edge():
    frame = np.full((480, 854, 3), (75, 135, 75), np.uint8)
    truth = np.array([[105, 423], [733, 423], [512, 235], [331, 235]], np.int32)
    cv2.fillConvexPoly(frame, truth, (130, 72, 58))
    cv2.polylines(frame, [truth], True, (235, 235, 235), 3, cv2.LINE_AA)
    candidate, _ = court_surface_candidate(frame)
    assert candidate is not None
    assert abs(candidate[2, 1] - 235) < 15
    assert abs(candidate[0, 1] - 423) < 15
    assert validate_court_corners(candidate, frame.shape, world_quad=WORLD).valid


def test_balanced_score_penalizes_template_collapse():
    balanced = court_template_score([0.72] * 9)
    collapsed = court_template_score([1, 1, 1, 1, 0, 0, 0, 0, 0])
    assert balanced > collapsed


def test_medoid_returns_an_observed_consensus_member():
    base = np.array([[105, 421], [733, 427], [512, 237], [331, 235]], float)
    candidates = [base, base + 2, base - 2, np.array([[20, 322], [569, 327], [390, 125], [19, 314]])]
    medoid, support = select_court_medoid(candidates, (480, 854))
    assert any(np.array_equal(medoid, candidate) for candidate in candidates)
    assert len(support) == 3


def test_one_frame_optical_flow_propagates_small_camera_motion():
    rng = np.random.default_rng(7)
    previous = np.zeros((480, 854), np.uint8)
    for x, y in rng.integers([110, 230], [730, 425], size=(80, 2)):
        cv2.circle(previous, (int(x), int(y)), 2, 255, -1)
    transform = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], np.float32)
    current = cv2.warpAffine(previous, transform, (854, 480))
    corners = np.array([[105, 421], [733, 427], [512, 237], [331, 235]], float)

    propagated, ok, diagnostics = propagate_court_corners(previous, current, corners)

    assert ok, diagnostics
    assert np.max(np.abs(propagated - (corners + [3, -2]))) < 0.8
