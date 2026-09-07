import numpy as np

from netcast_tennisvision.vision.player_identity import (
    IdentityResult,
    attach_identity_to_frames,
    attribute_landings_to_hitters,
    classify_observations,
    make_dense_decisions,
)


def _observation(feature):
    return {"feature": np.asarray(feature, dtype=np.float32)}


def _positioned_observation(feature, x, y):
    return {
        "feature": np.asarray(feature, dtype=np.float32),
        "world": np.asarray([x, y], dtype=np.float32),
    }


def test_identity_assignment_requires_three_consistent_samples_before_switching_sides():
    direct = {"near": _observation([1, 0]), "far": _observation([0, 1])}
    swapped = {"near": _observation([0, 1]), "far": _observation([1, 0])}
    observations = {0: direct, 1: direct, 2: swapped, 3: swapped, 4: swapped}

    decisions, metrics = classify_observations(
        observations,
        {"A": np.asarray([1, 0]), "B": np.asarray([0, 1])},
        enrol_end=-1,
    )

    assert decisions[3]["mapping"] == {"near": "A", "far": "B"}
    assert decisions[4]["mapping"] == {"near": "B", "far": "A"}
    assert metrics["identity_side_switches"] == 1


def test_ambiguous_samples_do_not_trigger_a_side_switch():
    observations = {
        index: {"near": _observation([0.49, 0.51]), "far": _observation([0.51, 0.49])}
        for index in range(4)
    }
    decisions, metrics = classify_observations(
        observations,
        {"A": np.asarray([1, 0]), "B": np.asarray([0, 1])},
        enrol_end=-1,
    )
    assert decisions[3]["mapping"] == {"near": "A", "far": "B"}
    assert metrics["identity_side_switches"] == 0
    assert metrics["ambiguous_pair_samples"] == 4


def test_appearance_cannot_teleport_players_across_the_court():
    direct = {
        "near": _positioned_observation([1, 0], 4.0, 2.0),
        "far": _positioned_observation([0, 1], 7.0, 22.0),
    }
    false_swap = {
        "near": _positioned_observation([0, 1], 4.0, 2.0),
        "far": _positioned_observation([1, 0], 7.0, 22.0),
    }
    observations = {0: direct, 5: false_swap, 10: false_swap, 15: false_swap}

    decisions, metrics = classify_observations(
        observations,
        {"A": np.asarray([1, 0]), "B": np.asarray([0, 1])},
        enrol_end=-1,
        fps=30.0,
    )

    assert decisions[15]["mapping"] == {"near": "A", "far": "B"}
    assert metrics["identity_side_switches"] == 0
    assert metrics["motion_rejected_switch_samples"] == 1


def test_persistent_changeover_is_accepted_after_enough_travel_time():
    direct = {
        "near": _positioned_observation([1, 0], 4.0, 2.0),
        "far": _positioned_observation([0, 1], 7.0, 22.0),
    }
    swapped = {
        "near": _positioned_observation([0, 1], 4.0, 2.0),
        "far": _positioned_observation([1, 0], 7.0, 22.0),
    }
    observations = {0: direct, 5: swapped, 10: swapped, 15: swapped, 60: swapped}

    decisions, metrics = classify_observations(
        observations,
        {"A": np.asarray([1, 0]), "B": np.asarray([0, 1])},
        enrol_end=-1,
        fps=30.0,
    )

    assert decisions[15]["mapping"] == {"near": "A", "far": "B"}
    assert decisions[60]["mapping"] == {"near": "B", "far": "A"}
    assert metrics["identity_side_switches"] == 1


def test_landing_colour_uses_temporally_stable_identity_after_false_swap():
    direct = {
        "near": _positioned_observation([1, 0], 4.0, 2.0),
        "far": _positioned_observation([0, 1], 7.0, 22.0),
    }
    false_swap = {
        "near": _positioned_observation([0, 1], 4.0, 2.0),
        "far": _positioned_observation([1, 0], 7.0, 22.0),
    }
    sparse, _ = classify_observations(
        {0: direct, 5: false_swap, 10: false_swap, 15: false_swap},
        {"A": np.asarray([1, 0]), "B": np.asarray([0, 1])},
        enrol_end=-1,
        fps=30.0,
    )
    frames = [{"players_world": []} for _ in range(20)]
    attach_identity_to_frames(frames, IdentityResult(make_dense_decisions(sparse, 20), {}))
    events = [{"kind": "hit", "frame": 16, "rally_id": 0, "contact_side": "near"}]
    bounces = [{"frame": 18, "rally_id": 0, "world": (5.0, 20.0)}]

    attribute_landings_to_hitters(events, bounces, frames)

    assert bounces[0]["player_id"] == "A"
    assert bounces[0]["identity_source"] == "preceding_hit"


def test_landing_inherits_the_previous_hitter_identity_within_its_rally():
    frames = [{"players_world": []} for _ in range(12)]
    dense = make_dense_decisions({0: {
        "mapping": {"near": "A", "far": "B"},
        "pair_advantage": 0.31,
    }}, len(frames))
    attach_identity_to_frames(frames, IdentityResult(dense, {}))
    events = [
        {"kind": "hit", "frame": 3, "rally_id": 0, "contact_side": "near"},
        {"kind": "hit", "frame": 8, "rally_id": 0, "contact_side": "far"},
    ]
    bounces = [
        {"frame": 6, "rally_id": 0},
        {"frame": 10, "rally_id": 0},
        {"frame": 11, "rally_id": 1, "world": (4.0, 4.0)},
    ]
    attribute_landings_to_hitters(events, bounces, frames)
    assert [bounce["player_id"] for bounce in bounces] == ["A", "B", "B"]
    assert bounces[0]["identity_confidence"] == 0.31
    assert bounces[2]["identity_source"] == "opposite_landing_half"
