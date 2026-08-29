import numpy as np

from temporal_world_tracker import (
    _ballistic_pixel_prediction,
    refine_touchdown_subframe,
    track_ball_persistent,
)


def _frames(count, observations):
    result = []
    for frame in range(count):
        candidates = observations.get(frame, [])
        result.append({"is_court": True, "candidates": candidates})
    return result


def _candidate(x, y, confidence=0.9):
    return (float(x), float(y), float(confidence), 0.0, 0.0)


def test_tracker_accepts_any_positive_native_frame_rate():
    for fps in (1.0, 23.976, 25.17, 29.1, 29.97, 30.0, 59.94, 60.0, 120.0):
        frames = _frames(4, {})
        segments, diagnostics = track_ball_persistent(
            frames, fps=fps, spatial=1.0, speed_scale=25.0 / fps,
            frame_size=(640, 360),
        )
        assert segments == []
        assert diagnostics.confirmed_births == 0


def test_single_detection_cannot_create_a_ball():
    frames = _frames(12, {4: [_candidate(100, 100)]})
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0, frame_size=(640, 360)
    )
    assert segments == []
    assert all(frame["ball_px"] is None for frame in frames)
    assert diagnostics.confirmed_births == 0


def test_ball_survives_short_occlusion_and_rejects_teleport():
    observations = {}
    for frame in list(range(0, 6)) + list(range(10, 16)):
        observations[frame] = [_candidate(80 + 7 * frame, 100 + 3 * frame)]
    observations[7] = [_candidate(600, 20)]  # bright object, physically unreachable
    frames = _frames(16, observations)
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0, frame_size=(640, 360), hard_cap=45
    )
    assert len(segments) == 1
    assert all(frames[i]["ball_px"] is not None for i in range(0, 16))
    assert all(frames[i]["ball_state"] == "occluded_predicted" for i in range(6, 10))
    assert len({frames[i]["ball_track_id"] for i in range(16)}) == 1
    assert diagnostics.rejected_teleports >= 1


def test_future_observation_reconnects_a_one_second_occlusion_at_30fps():
    observations = {}
    for frame in list(range(0, 7)) + list(range(38, 46)):
        observations[frame] = [_candidate(80 + 4 * frame, 90 + 2 * frame, 0.82)]
    frames = _frames(46, observations)
    segments, _ = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=45,
    )
    assert len(segments) == 1
    assert all(frames[i]["ball_px"] is not None for i in range(46))
    assert all(frames[i]["ball_state"] == "occluded_predicted" for i in range(7, 38))
    assert frames[22]["ball_confidence"] < frames[7]["ball_confidence"]
    assert frames[37]["ball_confidence"] > 0.0


def test_observed_confidence_uses_detector_and_motion_evidence():
    observations = {
        frame: [_candidate(60 + 5 * frame, 80 + 2 * frame, 0.62)]
        for frame in range(8)
    }
    frames = _frames(8, observations)
    track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0, frame_size=(640, 360)
    )
    assert 0.55 < frames[6]["ball_confidence"] < 1.0


def test_search_expands_from_previous_position_until_a_confident_ball_is_found():
    observations = {
        0: [_candidate(10, 100, 0.8)],
        1: [_candidate(15, 100, 0.8)],
        2: [_candidate(20, 100, 0.8)],
        3: [_candidate(45, 100, 0.8)],
    }
    frames = _frames(4, observations)
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100,
    )
    assert len(segments) == 1
    assert segments[0]["meas"][-1] == (45.0, 100.0)
    assert frames[3]["ball_search_radius_px"] == 36.0
    assert diagnostics.adaptive_search_recoveries >= 1


def test_local_candidate_wins_before_search_expands_to_distant_distractor():
    observations = {
        0: [_candidate(10, 100, 0.8)],
        1: [_candidate(15, 100, 0.8)],
        2: [_candidate(20, 100, 0.8)],
        3: [_candidate(27, 100, 0.30), _candidate(90, 100, 0.99)],
    }
    frames = _frames(4, observations)
    segments, _ = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100,
    )
    assert segments[0]["meas"][-1] == (27.0, 100.0)
    assert frames[3]["ball_search_radius_px"] == 18.0


def test_fast_ball_increases_the_first_search_radius_from_recent_speed():
    observations = {
        0: [_candidate(10, 100, 0.8)],
        1: [_candidate(30, 100, 0.8)],
        2: [_candidate(50, 100, 0.8)],
        3: [_candidate(90, 100, 0.8)],
    }
    frames = _frames(4, observations)
    segments, _ = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100,
    )
    assert segments[0]["meas"][-1] == (90.0, 100.0)
    assert frames[3]["ball_estimated_speed_px_frame"] >= 18.0
    assert frames[3]["ball_search_radius_px"] > 18.0


def test_extreme_one_frame_speed_jump_is_rejected():
    observations = {
        0: [_candidate(100, 100, 0.8)],
        1: [_candidate(105, 100, 0.8)],
        2: [_candidate(110, 100, 0.8)],
        3: [_candidate(190, 100, 0.99)],
        4: [_candidate(120, 100, 0.8)],
    }
    frames = _frames(5, observations)
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100,
    )
    assert segments[0]["meas"][3] is None
    assert segments[0]["meas"][4] == (120.0, 100.0)
    assert diagnostics.rejected_teleports >= 1


def test_player_proximity_allows_direction_reset_instead_of_straight_coast():
    observations = {
        0: [_candidate(10, 100)], 1: [_candidate(20, 100)],
        2: [_candidate(30, 100)], 3: [_candidate(20, 100)],
        4: [_candidate(10, 100)],
    }
    frames = _frames(5, observations)
    for frame in frames:
        frame["person_boxes"] = np.array([[22, 70, 42, 130]], dtype=float)
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100, chi2_gate=0.5,
    )
    assert segments[0]["meas"][3] == (20.0, 100.0)
    assert frames[3]["ball_motion_mode"] == "player_hit"
    assert diagnostics.player_hit_velocity_resets >= 1


def test_downward_ball_can_take_a_bounce_branch_and_reset_velocity():
    observations = {
        0: [_candidate(100, 10)], 1: [_candidate(100, 20)],
        2: [_candidate(100, 30)], 3: [_candidate(100, 22)],
        4: [_candidate(100, 15)],
    }
    frames = _frames(5, observations)
    for frame in frames:
        frame["corners"] = np.array([[0, 200], [200, 200], [200, 0], [0, 0]], dtype=float)
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100, chi2_gate=0.5,
    )
    assert segments[0]["meas"][3] == (100.0, 22.0)
    assert frames[3]["ball_motion_mode"] == "bounce"
    assert diagnostics.bounce_velocity_resets >= 1


def test_midflight_backstep_is_repaired_without_discarding_detector_hit():
    observations = {
        0: [_candidate(100, 150)], 1: [_candidate(100, 140)],
        2: [_candidate(100, 130)],
        3: [_candidate(100, 145, 0.99)],  # impossible return toward the near hitter
        4: [_candidate(100, 110)], 5: [_candidate(100, 100)],
    }
    frames = _frames(6, observations)
    for index, frame in enumerate(frames):
        frame["net_y_px"] = 90.0
        frame["person_boxes"] = (np.array([[80, 125, 120, 175]], dtype=float)
                                  if index == 0 else np.empty((0, 4), dtype=float))
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100, chi2_gate=100.0,
    )
    assert segments[0]["meas"][3] == (100.0, 145.0)
    assert abs(float(segments[0]["smooth"][3][1]) - 120.0) < 1e-6
    assert segments[0]["meas"][4] == (100.0, 110.0)
    assert diagnostics.midflight_backtrack_repairs >= 1


def test_backtrack_repair_does_not_flatten_certified_player_hit_launch():
    observations = {
        0: [_candidate(100, 110)], 1: [_candidate(100, 120)],
        2: [_candidate(100, 130)], 3: [_candidate(100, 120)],
        4: [_candidate(100, 110)], 5: [_candidate(100, 100)],
    }
    frames = _frames(6, observations)
    for frame in frames:
        frame["net_y_px"] = 90.0
        frame["person_boxes"] = np.array([[80, 105, 120, 155]], dtype=float)
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100, chi2_gate=0.5,
    )
    assert segments[0]["meas"][5] == (100.0, 100.0)
    assert diagnostics.player_hit_velocity_resets >= 1


def test_calibrated_ballistic_model_does_not_treat_lob_height_as_depth_speed():
    width, height, fps = 1280, 720, 30.0
    camera = np.array([5.5, -8.6, 6.2])
    target = np.array([5.5, 11.9, 0.0])
    forward = target - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ camera
    intrinsic = np.array(
        [[1100.0, 0.0, width / 2], [0.0, 1100.0, height / 2], [0.0, 0.0, 1.0]]
    )
    homography = intrinsic @ np.column_stack(
        [rotation[:, 0], rotation[:, 1], translation]
    )
    inverse = np.linalg.inv(homography)

    def project(point):
        q = intrinsic @ (rotation @ point + translation)
        return q[:2] / q[2]

    start = np.array([5.5, 2.0, 1.2])
    velocity = np.array([1.5, 18.0, 9.0])
    frames, observations = [], []
    for frame in range(8):
        time_s = frame / fps
        point = start + velocity * time_s + np.array([0.0, 0.0, -4.905 * time_s ** 2])
        pixel = project(point)
        frames.append({"M": homography, "M_inv": inverse})
        observations.append((frame, (float(pixel[0]), float(pixel[1]), 0.9)))

    prediction = _ballistic_pixel_prediction(
        frames, observations[:7], 7, fps, 1.0, (width, height)
    )
    assert prediction is not None
    true_pixel = np.asarray(observations[7][1][:2])
    linear_pixel = 2 * np.asarray(observations[6][1][:2]) - np.asarray(observations[5][1][:2])
    assert np.linalg.norm(prediction[0] - true_pixel) < 0.1
    assert np.linalg.norm(prediction[0] - true_pixel) < np.linalg.norm(linear_pixel - true_pixel)


def test_perspective_search_exports_an_ellipse_not_a_circle():
    observations = {i: [_candidate(100 + 5 * i, 100 + 2 * i)] for i in range(5)}
    frames = _frames(5, observations)
    for frame in frames:
        frame["M_inv"] = np.array([[0.02, 0, 0], [0, 0.08, 0], [0, 0, 1]], dtype=float)
    track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=100,
    )
    axes = frames[3]["ball_search_axes_px"]
    assert axes is not None
    assert axes[0] > axes[1]




def test_missing_ball_at_border_with_outward_motion_terminates_as_frame_exit():
    observations = {
        0: [_candidate(600, 100, 0.9)],
        1: [_candidate(615, 100, 0.9)],
        2: [_candidate(630, 100, 0.9)],
    }
    frames = _frames(30, observations)
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=80,
    )
    assert diagnostics.frame_exit_terminations == 1
    assert segments[0]["termination"] == "out_of_frame"
    assert frames[2]["ball_terminal_reason"] == "out_of_frame"


def test_missing_ball_in_net_band_requires_confirmation_then_terminates():
    observations = {
        0: [_candidate(200, 100, 0.9)],
        1: [_candidate(200, 110, 0.9)],
        2: [_candidate(200, 120, 0.9)],
    }
    frames = _frames(30, observations)
    for frame in frames:
        frame["net_y_px"] = 130.0
    segments, diagnostics = track_ball_persistent(
        frames, fps=30.0, spatial=1.0, speed_scale=1.0,
        frame_size=(640, 360), hard_cap=80,
    )
    assert diagnostics.net_terminations == 1
    assert segments[0]["termination"] == "net_hit"
    assert frames[2]["ball_terminal_reason"] == "net_hit"


def test_subframe_touchdown_is_between_samples():
    frames = []
    # Two flight branches meet at frame 5.4, between captured 30 fps samples.
    for frame in range(11):
        if frame <= 5:
            point = (20 + 4 * frame, 40 + 6 * frame)
        else:
            point = (41.6 + 3 * (frame - 5.4), 72.4 - 5 * (frame - 5.4))
        frames.append({"ball_px": point, "ball_seen": True})
    result = refine_touchdown_subframe(5, frames, radius=4)
    assert result["confidence"] == "subframe"
    assert 5.1 < result["frame_f"] < 5.8
    assert result["decision_frame"] == 6
