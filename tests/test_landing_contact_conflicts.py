from netcast_tennisvision.events.landing_event_detector import racket_confounded_landing


def test_racket_impulse_is_not_also_a_ground_contact():
    hits = [{"kind": "hit", "frame": 100, "contact_hypothesis": {"confidence": .9}}]
    landing = {"frame": 99, "sequence_p": .025, "impulse_score": .95,
               "arc": {"reverses": False, "disagree": .7}}
    assert racket_confounded_landing(landing, hits, fps=30)
    assert not racket_confounded_landing({**landing, "frame": 95}, hits, fps=30)
    assert not racket_confounded_landing({**landing, "arc": {"reverses": True, "disagree": .5}}, hits, fps=30)
