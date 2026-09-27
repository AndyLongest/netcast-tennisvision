"""Point boundaries from confirmed contacts, terminal outcomes."""
from __future__ import annotations


def assign_rallies(events, bounces, net_hits, *, fps, total_frames=None, serves=(), reset_seconds=10.0):
    """Assign shared IDs in place and return a replay timeline.

    Only confirmed landings participate: rejected bounce candidates and ball
    collection must not bridge otherwise separate points. A terminal outcome
    closes a point; only a racket hit can start the next one after that.
    """
    contacts = [(int(e["frame"]), "hit", e) for e in events if e["kind"] == "hit"]
    contacts += [(int(b["frame"]), "bounce", b) for b in bounces]
    contacts += [(int(n["frame"]), "net", n) for n in net_hits]
    contacts += [(int(s["frame"]), "serve", s) for s in serves]
    contacts.sort(key=lambda row: (row[0], {"serve": 0, "hit": 1}.get(row[1], 2)))
    rallies, current, closed = [], None, False
    previous = None
    for frame, kind, item in contacts:
        gap = previous is not None and frame - previous > fps * reset_seconds
        start = kind == "serve" or (kind == "hit" and (closed or gap))
        if current is None or start:
            reason = "serve" if kind == "serve" else "terminal" if closed else "inactivity" if gap else "first_contact"
            if current is not None and not closed:
                current["end_reason"] = reason
            current = {"rally_id": len(rallies), "start_frame": frame,
                       "end_frame": frame, "end_reason": "open", "start_reason": reason}
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
    # No inactivity expiry: keep this point until the next confirmed point starts.
    replay_end = total_frames if total_frames is not None else max(
        (r["end_frame"] + 1 for r in rallies), default=0)
    for i, rally in enumerate(rallies):
        rally["display_end_frame"] = (
            rallies[i + 1]["start_frame"] if i + 1 < len(rallies) else replay_end)
    lookup = {b["frame"]: b["rally_id"] for b in bounces}
    for event in events:
        if event["kind"] == "bounce" and event["frame"] in lookup:
            event["rally_id"] = lookup[event["frame"]]
    return rallies
