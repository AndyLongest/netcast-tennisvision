"""Convert hard tracker terminations at the net into product-facing events."""

from __future__ import annotations

from typing import Any


def collect_net_hits(
    frames_meta: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    *,
    fps: float,
    court_width: float,
    net_y: float,
) -> list[dict[str, Any]]:
    """Return one red-cross event for every confirmed net termination.

    The tracker stores the terminal on the last observed ball frame and separately
    records when the missing-ball grace window confirmed it. The marker therefore stays
    at the final trustworthy lateral coordinate but appears only after confirmation.
    """
    if fps <= 0:
        raise ValueError("net events require a positive frame rate")
    ordered_contacts = sorted(
        (item for item in contacts if "frame" in item), key=lambda item: int(item["frame"])
    )
    result: list[dict[str, Any]] = []
    for frame, meta in enumerate(frames_meta):
        if meta.get("ball_terminal_reason") != "net_hit":
            continue
        confidence = float(meta.get("ball_confidence", 0.0))
        # A red cross is a product-level claim, not a tracker diagnostic. Low-confidence
        # terminations remain available in frames_meta for audit, but are too ambiguous
        # to interrupt the visible rally as a confirmed net contact.
        if confidence < 0.50:
            continue
        world = meta.get("world") or meta.get("world_ground")
        x = court_width / 2.0
        if world is not None and len(world) >= 1:
            x = max(0.0, min(court_width, float(world[0])))
        preceding = next(
            (item for item in reversed(ordered_contacts) if int(item["frame"]) <= frame),
            None,
        )
        decision_frame = max(frame, int(meta.get("ball_terminal_decision_frame") or frame))
        result.append(
            {
                "frame": frame,
                "decision_frame": decision_frame,
                "t": round(decision_frame / fps, 3),
                "x": round(x, 2),
                "y": round(float(net_y), 3),
                "rally_id": preceding.get("rally_id") if preceding else None,
                "player_id": preceding.get("player_id") if preceding else None,
                "confidence": round(confidence, 2),
                "outcome": "net",
            }
        )
    return result
