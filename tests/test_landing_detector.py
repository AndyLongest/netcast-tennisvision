from netcast_tennisvision.events.landing_detector import estimate_landing_subframe


def _track(points, contact=5):
    frames = []
    for point in points:
        frames.append({
            "ball_px": tuple(point), "ball_seen": True, "ball_track_id": 3,
            "ball_confidence": 0.9,
        })
    return frames


def test_landing_is_subframe_and_waits_for_post_contact_evidence():
    # Two image-space branches meet at frame 5.4; no captured frame is the touchdown.
    points = []
    for frame in range(12):
        if frame <= 5:
            point = (30 + 5 * frame, 25 + 7 * frame)
        else:
            point = (57 + 4 * (frame - 5.4), 62.8 - 5 * (frame - 5.4))
        points.append(point)
    result = estimate_landing_subframe(5, _track(points), radius=5)
    assert result["confidence"] == "subframe"
    assert 5.2 <= result["frame_f"] <= 5.6
    assert result["decision_frame"] >= 8


def test_predicted_points_cannot_decide_a_landing():
    points = [(20 + i, 40 + i) for i in range(10)]
    frames = _track(points)
    for index in range(6, 10):
        frames[index]["ball_seen"] = False
    result = estimate_landing_subframe(5, frames, radius=4)
    assert result["confidence"] == "frame"
    assert result["support"] == 5


def test_other_track_is_not_used_as_future_evidence():
    points = [(20 + i, 40 + i) for i in range(10)]
    frames = _track(points)
    for index in range(6, 10):
        frames[index]["ball_track_id"] = 4
    result = estimate_landing_subframe(5, frames, radius=4)
    assert result["confidence"] == "frame"


def test_delayed_candidate_recovers_true_contact_without_moving_track():
    import copy
    points = [(20 + 3*t, 50 + 4*t + .08*t*t - 9*max(t-8.4, 0)) for t in range(20)]
    frames = _track(points)
    original = copy.deepcopy(frames)
    result = estimate_landing_subframe(10, frames, radius=8)
    assert abs(result["frame_f"] - 8.4) < .15
    assert result["impulse_bic_gain"] > 10
    assert frames == original


def test_camera_translation_does_not_change_inferred_contact():
    import numpy as np
    points = [(20 + 3*t, 50 + 4*t + .08*t*t - 9*max(t-8.4, 0)) for t in range(20)]
    frames = _track(points)
    for i, meta in enumerate(frames):
        shift = 32 if i >= 9 else 0
        meta["M"] = np.asarray([[1, 0, 0], [0, 1, shift], [0, 0, 1]], float)
        meta["M_inv"] = np.linalg.inv(meta["M"])
        meta["ball_px"] = (points[i][0], points[i][1] + shift)
    result = estimate_landing_subframe(10, frames, radius=8)
    assert abs(result["frame_f"] - 8.4) < .15
