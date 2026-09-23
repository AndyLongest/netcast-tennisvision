import copy
import json
from pathlib import Path

import numpy as np

from netcast_tennisvision.events.serve_sequence import detect_serve_sequences


def clips():
    return json.loads((Path(__file__).parent / "fixtures/serve_sequence_review_v1.json").read_text())


def test_reviewed_serves_and_overhead_negative_windows():
    data = clips()
    for clip in data["clips"]:
        poses = {(f, side): np.array(kp) for f, side, kp in clip["poses"]}
        found = detect_serve_sequences(clip["frames"], poses, fps=data["fps"])
        assert bool(found) == clip["expected_serve"], clip["start_seconds"]


def test_missing_ball_or_release_without_strike_cannot_confirm():
    data = clips()
    clip = data["clips"][0]
    poses = {(f, side): np.array(kp) for f, side, kp in clip["poses"]}
    frames = copy.deepcopy(clip["frames"])
    for m in frames:
        m["ball_seen"] = False
    assert detect_serve_sequences(frames, poses, fps=data["fps"]) == []
    cutoff = round(2.8*data["fps"])
    assert detect_serve_sequences(clip["frames"][:cutoff],
        {key: value for key,value in poses.items() if key[0] < cutoff}, fps=data["fps"]) == []


def test_image_scaling_does_not_change_boundaries():
    data = clips()
    clip = data["clips"][0]
    poses = {(f, side): np.array(kp) for f, side, kp in clip["poses"]}
    expected = detect_serve_sequences(clip["frames"], poses, fps=data["fps"])
    frames = copy.deepcopy(clip["frames"])
    for m in frames:
        if m["ball_px"] is not None:
            m["ball_px"] = [v*.5 for v in m["ball_px"]]
        for p in m["players_world"]:
            p["box"] = [v*.5 for v in p["box"]]
    for kp in poses.values():
        kp[:,:2] *= .5
    assert detect_serve_sequences(frames, poses, fps=data["fps"]) == expected


def test_preparation_contacts_do_not_bridge_or_flash_as_landings():
    from netcast_tennisvision.events.serve_sequence import exclude_serve_preparation
    events = [{"kind":"hit","frame":f} for f in (1,10,15,20,25)]
    bounces = [{"frame":12}, {"frame":24}]
    original = copy.deepcopy((events,bounces))
    kept_events, kept_bounces = exclude_serve_preparation(events,bounces,
        [{"preparation_start_frame":10,"frame":20}])
    assert [e["frame"] for e in kept_events] == [1,20,25]
    assert [e["frame"] for e in kept_bounces] == [24]
    assert (events,bounces) == original
