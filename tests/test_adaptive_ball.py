from netcast_tennisvision.vision.adaptive_ball import (
    plan_event_preserving_frames,
    scout_frame_indices,
)


def _candidate(x: float, y: float, confidence: float = 0.7):
    return [(x, y, confidence, 3.0, 3.0)]


def test_scout_schedule_keeps_native_timeline_endpoints():
    assert scout_frame_indices(8, 3) == frozenset({0, 3, 6, 7})


def test_stable_flight_remains_sparse_away_from_boundaries():
    total = 90
    rows = [[] for _ in range(total)]
    for frame in scout_frame_indices(total, 2):
        rows[frame] = _candidate(frame * 2.0, 80.0 - frame * 0.3)
    plan = plan_event_preserving_frames(rows, [{} for _ in rows], fps=30, stride=2)
    assert 45 not in plan.inference_frames
    assert len(plan.inference_frames) < 0.72 * total


def test_player_contact_is_recovered_at_native_rate():
    total = 80
    rows = [[] for _ in range(total)]
    metadata = [{} for _ in rows]
    for frame in scout_frame_indices(total, 2):
        rows[frame] = _candidate(float(frame), 50.0)
    metadata[40]["person_boxes"] = [[35.0, 35.0, 45.0, 65.0]]
    plan = plan_event_preserving_frames(rows, metadata, fps=30, stride=2)
    assert set(range(31, 50)).issubset(plan.inference_frames)
    assert plan.reasons["player_contact"] == 1


def test_sparse_direction_change_opens_dense_event_window():
    total = 80
    rows = [[] for _ in range(total)]
    for frame in scout_frame_indices(total, 2):
        x = frame if frame <= 40 else 80 - frame
        rows[frame] = _candidate(float(x), 50.0)
    plan = plan_event_preserving_frames(rows, [{} for _ in rows], fps=30, stride=2)
    assert set(range(31, 50)).issubset(plan.inference_frames)
    assert plan.reasons["motion_impulse"] >= 1


def test_one_missing_scout_sample_is_backfilled():
    total = 70
    rows = [[] for _ in range(total)]
    for frame in scout_frame_indices(total, 2):
        if frame != 34:
            rows[frame] = _candidate(float(frame), 50.0)
    plan = plan_event_preserving_frames(rows, [{} for _ in rows], fps=30, stride=2)
    assert set(range(26, 43)).issubset(plan.inference_frames)
    assert plan.reasons["brief_gap"] >= 1
