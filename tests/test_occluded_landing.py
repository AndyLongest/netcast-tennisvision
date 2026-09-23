import copy

from netcast_tennisvision.events.occluded_landing import infer_occluded_landings


def sample(impulse=8):
    frames = []
    for i in range(45):
        t = i-22
        frames.append(dict(ball_px=(400+2*t, 230+3*t+.05*t*t-impulse*max(t, 0)),
                           ball_seen=True, ball_confidence=.9, ball_track_id=1))
    for i in range(21, 24):
        frames[i].update(ball_px=None, ball_seen=False)
    return frames


def infer(frames, hits=(), bounces=()):
    return infer_occluded_landings(frames, hits, bounces, fps=30, width=960, roi_box=[0, 0, 960, 544])


def test_hidden_bounce_has_bounded_decision_and_no_invented_position():
    frames = sample()
    before = copy.deepcopy(frames)
    events = infer(frames)
    assert len(events) == 1
    assert events[0]['frame'] == 22
    assert events[0]['decision_frame'] == 31
    assert events[0]['position'] is None
    assert frames == before


def test_smooth_flight_and_nearby_racket_do_not_become_bounces():
    assert infer(sample(0)) == []
    assert infer(sample(), [{'kind': 'hit', 'frame': 23}]) == []
    assert infer(sample(), bounces=[{'frame': 22}]) == []


def test_other_track_or_camera_cut_cannot_bridge_gap():
    frames = sample()
    for m in frames[24:]:
        m['ball_track_id'] = 2
    assert infer(frames) == []
    frames = sample()
    frames[20]['corners'] = [[0, 0]]*4
    frames[24]['corners'] = [[100, 0]]*4
    assert infer(frames) == []


def test_predicted_tail_is_not_real_evidence():
    frames = sample()
    for m in frames[24:]:
        m['ball_seen'] = False
    assert infer(frames) == []
