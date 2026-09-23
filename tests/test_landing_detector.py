import copy

import numpy as np
import pytest

from netcast_tennisvision.events.landing_detector import (
    estimate_landing_subframe,
    landing_candidate_meta,
)
from netcast_tennisvision.events.landing_event_detector import detect_landing_impulses


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



@pytest.mark.parametrize("vertical_scale", [.3, 1.0])
@pytest.mark.parametrize("missing", [(), (10,), (9, 10, 11)])
def test_contact_without_captured_touchdown_for_both_view_scales(vertical_scale, missing):
    # Contact at 10.4, between native frames, with and without a detection hole.
    points = [(200+3*t, 150+vertical_scale*(6*t+.05*t*t-12*max(t-10.4, 0)))
              for t in range(24)]
    frames = _track(points)
    for i in missing:
        frames[i].update(ball_px=None, ball_seen=False, ball_track_id=None)
    before = copy.deepcopy(frames)
    proposals = detect_landing_impulses(
        frames, radius=7, min_score=.62,
        candidate_filter=lambda p: landing_candidate_meta(p.frame, frames, radius=7) is not None)
    assert len(proposals) == 1
    fit = estimate_landing_subframe(proposals[0]['frame'], frames, radius=7)
    assert abs(fit['frame_f']-10.4) < .15
    expected = (231.2, 150+vertical_scale*(6*10.4+.05*10.4**2))
    assert np.linalg.norm(np.array(fit['px'])-expected) < .3
    assert fit['decision_frame'] >= proposals[0]['frame']+7
    assert frames == before


def test_missing_centre_cannot_join_tracks_or_use_predicted_tail():
    frames = _track([(200+3*t, 150+6*t-12*max(t-10.4, 0)) for t in range(24)])
    frames[10].update(ball_px=None, ball_seen=False, ball_track_id=None)
    for m in frames[11:]:
        m['ball_track_id'] = 5
    assert landing_candidate_meta(10, frames, radius=7) is None
    for m in frames[11:]:
        m.update(ball_track_id=3, ball_seen=False)
    assert landing_candidate_meta(10, frames, radius=7) is None


def test_missing_point_smooth_flight_does_not_get_a_fabricated_location():
    frames = _track([(200+3*t, 150+6*t+.05*t*t) for t in range(24)])
    frames[10].update(ball_px=(999, 999), ball_seen=False)
    assert landing_candidate_meta(10, frames, radius=7) is None
