from netcast_tennisvision.events.rallies import assign_rallies


def test_terminal_point_restarts_at_hit_before_any_new_bounce():
    hits = [{"kind": "hit", "frame": f} for f in (0, 30, 55)]
    bounces = [{"frame": 15}, {"frame": 40, "line_call": "out"}, {"frame": 65}]
    rallies = assign_rallies(hits, bounces, [], fps=30)
    assert [h["rally_id"] for h in hits] == [0, 0, 1]
    assert [b["rally_id"] for b in bounces] == [0, 0, 1]
    assert rallies[0]["display_end_frame"] == 55
    assert rallies[1]["start_frame"] == 55


def test_long_gap_does_not_split_point():
    events = [{"kind": "hit", "frame": 0}, {"kind": "bounce", "frame": 40},
              {"kind": "bounce", "frame": 80}, {"kind": "hit", "frame": 110}]
    rallies = assign_rallies(events, [], [], fps=30, total_frames=300)
    assert len(rallies) == 1
    assert rallies[0]["display_end_frame"] == 300


def test_net_and_ball_collection_do_not_prolong_finished_point():
    hits = [{"kind": "hit", "frame": 0}, {"kind": "hit", "frame": 100}]
    nets = [{"frame": 20, "decision_frame": 25}]
    bounces = [{"frame": 40}, {"frame": 70}]
    rallies = assign_rallies(hits, bounces, nets, fps=30)
    assert len(rallies) == 2
    assert rallies[0]["end_frame"] == 25
    assert nets[0]["rally_id"] == 0


def test_second_bounce_closes_point_until_next_hit():
    hits = [{"kind": "hit", "frame": f} for f in (0, 900)]
    bounces = [{"frame": 20, "outcome": "second_bounce_point_over"}, {"frame": 700}]
    rallies = assign_rallies(hits, bounces, [], fps=30, total_frames=1000)
    assert len(rallies) == 2
    assert bounces[1]["rally_id"] == 0
    assert rallies[0]["display_end_frame"] == 900
    assert rallies[1]["display_end_frame"] == 1000


def test_ten_second_gap_restarts_only_at_hit():
    hits = [{"kind": "hit", "frame": f} for f in (0, 300, 601)]
    rallies = assign_rallies(hits, [], [], fps=30, total_frames=700)
    assert [h["rally_id"] for h in hits] == [0, 0, 1]
    assert rallies[1]["start_reason"] == "inactivity"


def test_confirmed_serve_can_split_without_terminal_event():
    hits = [{"kind": "hit", "frame": f} for f in (0, 30, 100)]
    serves = [{"frame": 90, "decision_frame": 96}]
    rallies = assign_rallies(hits, [], [], fps=30, total_frames=200, serves=serves)
    assert len(rallies) == 2
    assert rallies[1]["start_reason"] == "serve"
    assert hits[-1]["rally_id"] == 1
