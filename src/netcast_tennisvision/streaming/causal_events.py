"""Causal delivery rules for simulated live landing events.

The production report is generated offline and can use centred temporal windows.  A live
consumer must not reveal an event until every future frame required by that window has
actually arrived.  This module makes that availability boundary explicit without changing
the frozen landing coordinates or classifications.
"""
from __future__ import annotations

from math import ceil
from typing import Any


def schedule_landing_events(
    landings: list[dict[str, Any]],
    *,
    fps: float,
    future_support_frames: int = 10,
) -> list[dict[str, Any]]:
    """Return immutable landing results with an honest live availability timestamp.

    ``future_support_frames`` is ten for the current centred 21-frame bounce descriptor.
    An event may have a later explicit ``decision_frame``; in that case the later boundary
    wins.  Input dictionaries are copied and never mutated.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if future_support_frames < 0:
        raise ValueError("future_support_frames must be non-negative")

    scheduled: list[dict[str, Any]] = []
    for event in landings:
        touchdown = float(event.get("touchdown_frame_f", event["frame"]))
        explicit_decision = int(event.get("decision_frame", ceil(touchdown)))
        available_frame = max(explicit_decision, ceil(touchdown) + future_support_frames)
        scheduled.append({
            **event,
            "available_frame": available_frame,
            "available_t": available_frame / fps,
            "causal_delay_ms": 1000.0 * (available_frame - touchdown) / fps,
        })
    return sorted(scheduled, key=lambda item: (item["available_frame"], item["frame"]))
