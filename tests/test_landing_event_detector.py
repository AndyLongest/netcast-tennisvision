import numpy as np

from netcast_tennisvision.events.landing_event_detector import (
    detect_landing_impulses,
    enforce_one_landing_between_hits,
    score_landing_impulse,
)


def _frames(points, *, track_id=2):
    return [
        {"ball_px": tuple(point), "ball_seen": True, "ball_track_id": track_id,
         "ball_confidence": 0.9}
        for point in points
    ]


def test_smooth_flight_is_not_a_landing():
    t = np.arange(31, dtype=float) - 15
    points = np.column_stack((300 + 4 * t, 200 + 2 * t + 0.08 * t * t))
    result = score_landing_impulse(15, _frames(points), radius=8)
    assert result is not None
    assert result.score < 0.25


def test_upward_velocity_impulse_is_detected_without_sign_reversal():
    # The ball still moves down the image after contact (dy remains positive), but its
    # downward speed drops sharply.  A sign-change-only detector misses this case.
    t = np.arange(31, dtype=float) - 15
    y = 180 + 8 * t + 0.05 * t * t - 5 * np.maximum(t, 0)
    x = 400 + 3 * t
    result = score_landing_impulse(15, _frames(np.column_stack((x, y))), radius=8)
    assert result is not None
    assert result.score > 0.55
    assert result.impulse_y_px_frame < -4.0


def test_predicted_points_and_other_track_do_not_create_support():
    t = np.arange(25, dtype=float) - 12
    points = np.column_stack((300 + 3 * t, 200 + 7 * t - 6 * np.maximum(t, 0)))
    frames = _frames(points)
    for index in range(13, 18):
        frames[index]["ball_seen"] = False
    for index in range(18, len(frames)):
        frames[index]["ball_track_id"] = 9
    assert score_landing_impulse(12, frames, radius=8) is None


def test_non_maximum_suppression_keeps_one_contact():
    t = np.arange(45, dtype=float) - 22
    points = np.column_stack((300 + 3 * t, 180 + 8 * t - 6 * np.maximum(t, 0)))
    events = detect_landing_impulses(_frames(points), radius=8, min_score=0.40,
                                     min_gap_frames=10)
    assert len(events) == 1
    assert abs(events[0]["frame"] - 22) <= 1


def test_context_filter_runs_before_non_maximum_suppression():
    t = np.arange(45, dtype=float)
    y = 180 + 9 * t - 8 * np.maximum(t - 16, 0) - 5 * np.maximum(t - 22, 0)
    points = np.column_stack((300 + 3 * t, y))
    frames = _frames(points)

    unrestricted = detect_landing_impulses(
        frames, radius=6, min_score=0.25, min_gap_frames=10
    )
    assert unrestricted
    rejected_frame = unrestricted[0]["frame"]
    filtered = detect_landing_impulses(
        frames,
        radius=6,
        min_score=0.25,
        min_gap_frames=10,
        candidate_filter=lambda candidate: abs(candidate.frame - rejected_frame) >= 3,
    )

    assert filtered
    assert all(event["frame"] != rejected_frame for event in filtered)


def test_second_landing_without_racket_hit_is_suppressed():
    bounces = [{"frame": 10}, {"frame": 20}]
    events = [{"frame": 10, "kind": "bounce"}, {"frame": 20, "kind": "bounce"}]

    kept, rejected = enforce_one_landing_between_hits(
        bounces, events, fps=30, rally_reset_seconds=2.0,
    )

    assert [item["frame"] for item in kept] == [10]
    assert rejected == {20}


def test_racket_hit_rearms_landing_highlight():
    bounces = [{"frame": 10}, {"frame": 20}]
    events = [
        {"frame": 10, "kind": "bounce"},
        {"frame": 15, "kind": "hit"},
        {"frame": 20, "kind": "bounce"},
    ]

    kept, rejected = enforce_one_landing_between_hits(
        bounces, events, fps=30, rally_reset_seconds=2.0,
    )

    assert [item["frame"] for item in kept] == [10, 20]
    assert rejected == set()


def test_new_rally_rearms_landing_without_visible_hit():
    bounces = [{"frame": 10}, {"frame": 80}]
    events = [{"frame": 10, "kind": "bounce"}, {"frame": 80, "kind": "bounce"}]

    kept, rejected = enforce_one_landing_between_hits(
        bounces, events, fps=30, rally_reset_seconds=2.0,
    )

    assert [item["frame"] for item in kept] == [10, 80]
    assert rejected == set()


def test_expected_half_replaces_premature_same_half_candidate():
    bounces = [
        {"frame": 20, "court_side": "near"},
        {"frame": 30, "court_side": "far"},
    ]
    events = [
        {"frame": 10, "kind": "hit", "contact_side": "near"},
        {"frame": 20, "kind": "bounce"},
        {"frame": 30, "kind": "bounce"},
    ]

    kept, rejected = enforce_one_landing_between_hits(
        bounces, events, fps=30, rally_reset_seconds=2.0,
    )

    assert [item["frame"] for item in kept] == [30]
    assert rejected == {20}


def test_strong_same_half_touchdown_replaces_weak_premature_candidate():
    bounces = [
        {"frame": 20, "court_side": "far", "sequence_p": 0.02, "impulse_score": 0.03},
        {"frame": 30, "court_side": "far", "sequence_p": 0.30, "impulse_score": 0.85},
    ]
    events = [
        {"frame": 10, "kind": "hit", "contact_side": "near"},
        {"frame": 20, "kind": "bounce"},
        {"frame": 30, "kind": "bounce"},
    ]

    kept, rejected = enforce_one_landing_between_hits(
        bounces, events, fps=30, rally_reset_seconds=2.0,
    )

    assert [item["frame"] for item in kept] == [30]
    assert rejected == {20}


def test_same_side_hit_cannot_validate_two_same_half_landings():
    bounces = [
        {"frame": 20, "court_side": "far", "sequence_p": 0.02, "impulse_score": 0.03},
        {"frame": 40, "court_side": "far", "sequence_p": 0.28, "impulse_score": 0.86},
    ]
    events = [
        {"frame": 10, "kind": "hit", "contact_side": "near"},
        {"frame": 20, "kind": "bounce"},
        {"frame": 30, "kind": "hit", "contact_side": "far"},
        {"frame": 40, "kind": "bounce"},
    ]

    kept, rejected = enforce_one_landing_between_hits(
        bounces, events, fps=30, rally_reset_seconds=2.0,
    )

    assert [item["frame"] for item in kept] == [40]
    assert rejected == {20}
