from netcast_tennisvision.events.rallies import assign_rallies


def test_terminal_point_restarts_at_hit_before_any_new_bounce():
    hits = [{"kind": "hit", "frame": f} for f in (0, 30, 55)]
    bounces = [{"frame": 15}, {"frame": 40, "line_call": "out"}, {"frame": 65}]
    rallies = assign_rallies(hits, bounces, [], fps=30)
    assert [h["rally_id"] for h in hits] == [0, 0, 1]
    assert [b["rally_id"] for b in bounces] == [0, 0, 1]
    assert rallies[0]["display_end_frame"] == 55
    assert rallies[1]["start_frame"] == 55


def test_rejected_candidates_do_not_bridge_inactivity():
    events = [{"kind": "hit", "frame": 0}, {"kind": "bounce", "frame": 40},
              {"kind": "bounce", "frame": 80}, {"kind": "hit", "frame": 110}]
    rallies = assign_rallies(events, [], [], fps=30)
    assert len(rallies) == 2
    assert rallies[0]["display_end_frame"] == 60


def test_net_and_ball_collection_do_not_prolong_finished_point():
    hits = [{"kind": "hit", "frame": 0}, {"kind": "hit", "frame": 100}]
    nets = [{"frame": 20, "decision_frame": 25}]
    bounces = [{"frame": 40}, {"frame": 70}]
    rallies = assign_rallies(hits, bounces, nets, fps=30)
    assert len(rallies) == 2
    assert rallies[0]["end_frame"] == 25
    assert nets[0]["rally_id"] == 0


def test_score_change_splits_edited_points_without_a_contact_gap():
    hits = [{"kind": "hit", "frame": f} for f in (0, 30, 60, 90)]
    bounces = [{"frame": f} for f in (15, 45, 75)]
    rallies = assign_rallies(hits, bounces, [], fps=30, boundary_frames=[50])
    assert len(rallies) == 2
    assert rallies[0]["display_end_frame"] == 50
    assert rallies[1]["start_frame"] == 50
    assert hits[2]["rally_id"] == 1
