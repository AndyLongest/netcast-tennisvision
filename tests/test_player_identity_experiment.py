import numpy as np

from netcast_tennisvision.vision.player_identity import (
    IdentityResult,
    attach_identity_to_frames,
    attribute_landings_to_hitters,
    classify_observations,
    distinguish_player_colors,
    dominant_player_color,
    identity_palette,
    make_dense_decisions,
    stable_player_color,
)


def _observation(feature):
    return {"feature": np.asarray(feature, dtype=np.float32)}


def _positioned_observation(feature, x, y):
    return {
        "feature": np.asarray(feature, dtype=np.float32),
        "world": np.asarray([x, y], dtype=np.float32),
    }


def test_player_colour_comes_from_the_central_torso_not_the_box_background():
    crop = np.full((120, 80, 3), (180, 90, 30), dtype=np.uint8)
    crop[22:75, 16:64] = (20, 30, 220)

    colour = dominant_player_color(crop)

    assert colour is not None
    displayed = stable_player_color([colour])
    assert int(displayed[1:3], 16) > 180
    assert int(displayed[1:3], 16) > int(displayed[5:7], 16)


def test_white_and_black_shirts_are_not_removed_as_border_background():
    for colour in ((240, 240, 240), (5, 5, 5)):
        crop = np.full((120, 80, 3), colour, dtype=np.uint8)
        crop[70:] = (180, 90, 30)
        assert np.max(np.abs(dominant_player_color(crop) - colour)) < 2
    assert stable_player_color([np.array([25, 8, 4])]) == '#696969'


def test_similar_uniforms_receive_distinguishable_display_colours():
    palette = distinguish_player_colors({'A': '#777777', 'B': '#787878'})
    assert abs(int(palette['A'][1:3], 16) - int(palette['B'][1:3], 16)) >= 100
    assert distinguish_player_colors({'A': '#e83020', 'B': '#20a0f0'}) == {'A': '#e83020', 'B': '#20a0f0'}


def test_single_visible_shirt_can_confirm_swap_without_tiny_far_player():
    def person(feature, colour):
        return {**_observation(feature), 'box': np.array([0, 0, 60, 150]),
                'color_bgr': np.array(colour)}
    direct = {'near': person([1, 0], [8, 8, 8]), 'far': person([0, 1], [240, 240, 240])}
    observations = {f: direct for f in range(6)}
    # ReID alone is biased towards the near-side prototype; shirt evidence is clear.
    observations.update({f: {'near': person([.8, .6], [220, 215, 210])} for f in (30, 40, 55, 65)})
    decisions, _ = classify_observations(observations, {'A': np.array([1, 0]), 'B': np.array([0, 1])},
                                        enrol_end=5, fps=30)
    assert decisions[30]['mapping']['near'] == 'A'
    assert decisions[65]['mapping']['near'] == 'B'


def test_sustained_changeover_is_not_erased_by_one_long_rally():
    frames = [{'player_identity_by_side': {'near': 'A', 'far': 'B'}} for _ in range(90)]
    frames += [{'player_identity_by_side': {'near': 'B', 'far': 'A'}} for _ in range(60)]
    bounces = [{'frame': 10, 'rally_id': 0, 'y': 20}, {'frame': 130, 'rally_id': 0, 'y': 20}]
    attribute_landings_to_hitters([], bounces, frames, fps=30)
    assert [b['player_id'] for b in bounces] == ['A', 'B']


def test_identity_palette_follows_the_person_across_a_side_change():
    red = np.asarray([20, 30, 220], dtype=np.float32)
    blue = np.asarray([220, 80, 20], dtype=np.float32)
    observations = {
        0: {"near": {"color_bgr": red}, "far": {"color_bgr": blue}},
        1: {"near": {"color_bgr": blue}, "far": {"color_bgr": red}},
    }
    decisions = [
        {"mapping": {"near": "A", "far": "B"}},
        {"mapping": {"near": "B", "far": "A"}},
    ]

    palette = identity_palette(observations, decisions)

    assert int(palette["A"][1:3], 16) > int(palette["A"][5:7], 16)
    assert int(palette["B"][5:7], 16) > int(palette["B"][1:3], 16)


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
    assert bounces[0]["identity_source"] == "rally_consensus+landing_half"


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
    assert bounces[0]["identity_rally_consensus"] == 1.0
    assert bounces[2]["identity_source"] == "clipped_rally+landing_half"


def test_landing_half_repairs_a_single_frame_wrong_hitter_side():
    frames = [{"players_world": []} for _ in range(20)]
    dense = make_dense_decisions({0: {
        "mapping": {"near": "A", "far": "B"},
        "pair_advantage": 0.4,
    }}, len(frames))
    attach_identity_to_frames(frames, IdentityResult(dense, {}))
    events = [
        # The single-frame racket test was wrong: a shot landing near was struck far.
        {"kind": "hit", "frame": 5, "rally_id": 2, "contact_side": "near"},
    ]
    bounces = [
        {"frame": 12, "rally_id": 2, "world": (4.0, 6.0), "court_side": "near"},
    ]

    attribute_landings_to_hitters(events, bounces, frames)

    assert bounces[0]["player_id"] == "B"
    assert bounces[0]["identity_source"] == "rally_consensus+landing_half"
    assert events[0]["player_id"] == "B"
    assert events[0]["contact_side_corrected"] is True


def test_rally_consensus_prevents_mid_rally_identity_colour_swap():
    frames = [{"players_world": []} for _ in range(30)]
    decisions = {
        0: {"mapping": {"near": "A", "far": "B"}, "pair_advantage": 0.3},
        18: {"mapping": {"near": "B", "far": "A"}, "pair_advantage": 0.12},
    }
    attach_identity_to_frames(
        frames,
        IdentityResult(make_dense_decisions(decisions, len(frames)), {}),
    )
    events = [
        {"kind": "hit", "frame": 3, "rally_id": 0, "contact_side": "far"},
        {"kind": "hit", "frame": 22, "rally_id": 0, "contact_side": "near"},
    ]
    bounces = [
        {"frame": 10, "rally_id": 0, "world": (5.0, 5.0), "court_side": "near"},
        {"frame": 27, "rally_id": 0, "world": (5.0, 20.0), "court_side": "far"},
    ]

    attribute_landings_to_hitters(events, bounces, frames)

    assert [bounce["player_id"] for bounce in bounces] == ["B", "A"]
    assert all(bounce["identity_rally_consensus"] > 0.5 for bounce in bounces)


def test_clipped_landing_uses_the_current_mapping_after_a_changeover():
    frames = [{"players_world": []} for _ in range(20)]
    dense = make_dense_decisions({0: {
        "mapping": {"near": "B", "far": "A"},
        "pair_advantage": 0.26,
    }}, len(frames))
    attach_identity_to_frames(frames, IdentityResult(dense, {}))
    bounce = {"frame": 12, "world": (4.0, 6.0), "court_side": "near"}

    attribute_landings_to_hitters([], [bounce], frames)

    assert bounce["player_id"] == "A"
    assert bounce["identity_confidence"] == 0.26
    assert bounce["identity_source"] == "clipped_rally+landing_half"
