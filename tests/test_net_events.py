from netcast_tennisvision.events.net_events import collect_net_hits


def test_net_event_uses_last_trusted_x_and_confirmation_time():
    frames = [
        {},
        {
            "ball_terminal_reason": "net_hit",
            "ball_terminal_decision_frame": 7,
            "world": (4.25, 10.8),
            "ball_confidence": 0.74,
        },
    ]
    contacts = [{"frame": 0, "kind": "hit", "rally_id": 3, "player_id": "B"}]

    events = collect_net_hits(
        frames, contacts, fps=30.0, court_width=10.97, net_y=11.885
    )

    assert events == [
        {
            "frame": 1,
            "decision_frame": 7,
            "t": 0.233,
            "x": 4.25,
            "y": 11.885,
            "rally_id": 3,
            "player_id": "B",
            "confidence": 0.74,
            "outcome": "net",
        }
    ]


def test_low_confidence_tracker_terminal_stays_internal_only():
    frames = [
        {
            "ball_terminal_reason": "net_hit",
            "ball_terminal_decision_frame": 10,
            "world": (5.1, 11.2),
            "ball_confidence": 0.27,
        }
    ]

    assert collect_net_hits(
        frames, [], fps=30.0, court_width=10.97, net_y=11.885
    ) == []
