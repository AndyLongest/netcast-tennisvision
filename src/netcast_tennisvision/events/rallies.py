"""Point boundaries from confirmed contacts, terminal outcomes and inactivity."""
from __future__ import annotations


def assign_rallies(events, bounces, net_hits, *, fps, reset_seconds=2.0, boundary_frames=()):
    """Assign shared IDs in place and return a replay timeline.

    Only confirmed landings participate: rejected bounce candidates and ball
    collection must not bridge otherwise separate points. A terminal outcome
    closes a point; only a racket hit can start the next one after that.
    """
    contacts = [(int(e["frame"]), "hit", e) for e in events if e["kind"] == "hit"]
    contacts += [(int(b["frame"]), "bounce", b) for b in bounces]
    contacts += [(int(n["frame"]), "net", n) for n in net_hits]
    contacts.sort(key=lambda row: (row[0], row[1] != "hit"))
    rallies, current, previous, closed = [], None, None, False
    boundaries = sorted(set(boundary_frames))
    for frame, kind, item in contacts:
        gap = previous is not None and frame - previous > fps * reset_seconds
        boundary = next((b for b in reversed(boundaries)
                         if (previous if previous is not None else -1) < b <= frame
                         and (current is None or b > current["start_frame"])), None)
        if boundary is not None and current is not None:
            current["end_reason"] = "scoreboard_change"
        if current is None or boundary is not None or (kind == "hit" and (closed or gap)) or (gap and not closed):
            current = {"rally_id": len(rallies), "start_frame": boundary if boundary is not None else frame,
                       "end_frame": frame, "end_reason": "inactivity"}
            rallies.append(current)
            closed = False
        item["rally_id"] = current["rally_id"]
        if closed:
            # Post-point bounces retain their attribution but cannot prolong
            # the active replay interval or postpone the next rally reset.
            continue
        previous = frame
        current["end_frame"] = max(frame, int(item.get("decision_frame", frame)))
        terminal = (kind == "net" or item.get("outcome") in ("out", "second_bounce_point_over")
                    or item.get("line_call") == "out")
        if terminal:
            closed = True
            current["end_reason"] = "net" if kind == "net" else item.get("outcome", "out")
    # Keep each last marker briefly visible; expire it during dead time even
    # when no next point was detected. This is a presentation boundary only.
    for i, rally in enumerate(rallies):
        rally["display_end_frame"] = rally["end_frame"] + round(fps * reset_seconds)
        if i + 1 < len(rallies):
            rally["display_end_frame"] = min(rally["display_end_frame"], rallies[i + 1]["start_frame"])
        following_boundary = next((b for b in boundaries if b > rally["start_frame"]), None)
        if following_boundary is not None:
            rally["display_end_frame"] = min(rally["display_end_frame"], following_boundary)
    lookup = {b["frame"]: b["rally_id"] for b in bounces}
    for event in events:
        if event["kind"] == "bounce" and event["frame"] in lookup:
            event["rally_id"] = lookup[event["frame"]]
    return rallies
