from contact_hypothesis import classify_contact_hypotheses


def test_close_racket_and_direction_reversal_is_a_hit():
    decision = classify_contact_hypotheses(
        existing_kind="bounce", racket_gap=0.25, player_gap=0.2,
        horizontal_reversal=True, ground_turn=True, ground_arc=0.1,
    )
    assert decision.kind == "hit"
    assert decision.changed
    assert "racket-proximity" in decision.evidence


def test_two_sided_ground_arc_beats_player_overlap():
    decision = classify_contact_hypotheses(
        existing_kind="hit", racket_gap=1.4, player_gap=0.35,
        horizontal_reversal=False, ground_turn=True, ground_arc=0.95,
        ground_impulse=0.8, sequence_probability=0.3,
    )
    assert decision.kind == "bounce"
    assert decision.changed


def test_ambiguous_evidence_preserves_existing_label():
    decision = classify_contact_hypotheses(
        existing_kind="bounce", racket_gap=1.1, player_gap=0.8,
        horizontal_reversal=False, ground_turn=True, ground_arc=0.1,
    )
    assert decision.kind == "bounce"
    assert not decision.changed


def test_player_body_overlap_alone_cannot_create_hit():
    decision = classify_contact_hypotheses(
        existing_kind=None, racket_gap=2.0, player_gap=0.1,
        horizontal_reversal=False, ground_turn=False,
    )
    assert decision.kind == "ambiguous"


def test_proven_racket_gate_is_preserved_without_ground_arc():
    decision = classify_contact_hypotheses(
        existing_kind="bounce", racket_gap=0.95, player_gap=0.6,
        horizontal_reversal=False, ground_turn=True, ground_arc=0.0,
    )
    assert decision.kind == "hit"


def test_strong_ground_arc_can_override_racket_projection_overlap():
    decision = classify_contact_hypotheses(
        existing_kind="bounce", racket_gap=0.7, player_gap=0.4,
        horizontal_reversal=False, ground_turn=True, ground_arc=1.0,
        ground_impulse=0.9, sequence_probability=0.35,
    )
    assert decision.kind == "bounce"
