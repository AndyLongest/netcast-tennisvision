"""Serve recognition from observed wrist/ball co-motion, release and racket-side reach.

Image distances are normalized by player height. Wrist regions approximate hands;
no claim of palm or racket tracking is made. Missing ball observations never prove contact.
"""
from __future__ import annotations

import numpy as np


def pose_schedule(frames, fps):
    active = {}
    schedule = []
    stride = max(1, round(fps / 10))
    for f, m in enumerate(frames):
        for p in m.get("players_world", []):
            side = p.get("side")
            if side not in ("near", "far") or abs(p["xy"][1] - (0 if side == "near" else 23.77)) > 2.5:
                continue
            left, t, r, b = p["box"]
            h = b - t
            ball = m.get("ball_px")
            if h <= 0:
                continue
            if m.get("ball_seen") and ball is not None and left-.2*h < ball[0] < r+.2*h and t+.2*h < ball[1] < b-.1*h:
                active[side] = f + round(2.5 * fps)
            if f % stride == 0 and f <= active.get(side, -1):
                schedule.append((f, side))
    return schedule


def select_pose(points, boxes, player, image_shape):
    left, t, r, b = player["box"]
    h = b-t
    x0, y0 = max(0, int(left-h*.5)), max(0, int(t-h*.6))
    if not len(points):
        return None
    cx, cy = (left+r)/2-x0, (t+b)/2-y0
    i = min(range(len(boxes)), key=lambda i: abs((boxes[i][0]+boxes[i][2])/2-cx)+abs((boxes[i][1]+boxes[i][3])/2-cy))
    kp = points[i].copy()
    kp[:, :2] += [x0, y0]
    return kp


def detect_serve_sequences(frames, poses, *, fps):
    serves = []
    for side in ("near", "far"):
        rows = []
        for (f, s), kp in sorted(poses.items()):
            if s != side or kp is None:
                continue
            m = frames[f]
            p = next((p for p in m.get("players_world", []) if p.get("side") == side), None)
            if p is None or abs(p["xy"][1] - (0 if side == "near" else 23.77)) > 2.5:
                continue
            ball = m.get("ball_px")
            if not m.get("ball_seen") or ball is None:
                continue
            h = p["box"][3]-p["box"][1]
            if h > 0:
                rows.append((f, np.asarray(ball, dtype=float), kp, h))
        last = -fps*10
        for i, (f, ball, kp, h) in enumerate(rows):
            if f-last < 1.5*fps:
                continue
            for wrist, other in ((9, 10), (10, 9)):
                if kp[wrist,2] < .5 or np.linalg.norm(ball-kp[wrist,:2])/h > .16:
                    continue
                held = [r for r in rows[max(0,i-12):i] if .08*fps <= f-r[0] <= .65*fps
                        and r[2][wrist,2] >= .5 and np.linalg.norm(r[1]-r[2][wrist,:2])/r[3] < .16]
                if not held:
                    continue
                prev = held[-1]
                db, dw = ball-prev[1], kp[wrist,:2]-prev[2][wrist,:2]
                if np.linalg.norm(db) < .06*h or dw[1] > -.035*h or db[1] > -.035*h:
                    continue
                if np.dot(db,dw) <= .6*np.linalg.norm(db)*np.linalg.norm(dw) or np.linalg.norm(db-dw) > .15*h:
                    continue
                released = [r for r in rows[i+1:] if 0 < r[0]-f <= .5*fps and r[2][wrist,2] >= .5
                            and r[1][1] < ball[1]-.15*h
                            and np.linalg.norm(r[1]-r[2][wrist,:2]) > .22*h]
                if len(released) < 2:
                    continue
                release = released[0][0]
                # Find the opposite wrist reaching overhead near the ball after the toss.
                contacts = [r for r in rows[i+1:] if .25*fps <= r[0]-release <= 1.6*fps
                            and min(r[2][other,2],r[2][5,2],r[2][6,2]) >= .5
                            and r[2][other,1] < (r[2][5,1]+r[2][6,1])/2
                            and np.linalg.norm(r[1]-r[2][other,:2])/r[3] < .65]
                if not contacts:
                    continue
                contact = min(contacts,key=lambda r:np.linalg.norm(r[1]-r[2][other,:2])/r[3])
                cf, cb, _, ch = contact
                after = [r for r in rows if .1*fps <= r[0]-cf <= .65*fps]
                direction = -1 if side == "near" else 1
                outgoing = [r for r in after if direction*(r[1][1]-cb[1]) > .15*ch and np.linalg.norm(r[1]-cb) > .25*ch]
                if len(outgoing) < 2:
                    continue
                # Toss changes to driven flight: require a nonvertical outgoing path or
                # reversal from descending toss to the net, rather than a caught toss.
                before_contact = [r for r in rows if .08*fps <= cf-r[0] <= .3*fps]
                if not before_contact:
                    continue
                vin = cb-before_contact[-1][1]
                vout = outgoing[0][1]-cb
                norm = np.linalg.norm(vin)*np.linalg.norm(vout)
                if norm <= 0 or np.dot(vin,vout)/norm > .94:
                    continue
                serves.append({"frame": int(cf), "decision_frame": int(outgoing[1][0]),
                               "preparation_start_frame": int(prev[0]), "release_frame": int(release),
                               "side": side, "pose_verified": True,
                               "evidence": "observed-ball+wrist-comotion+release+opposite-overhead-wrist+flight-change"})
                last = cf
                break
    return sorted(serves, key=lambda s:s["frame"])


def exclude_serve_preparation(events, bounces, serves):
    """Exclude contacts during demonstrated hold/toss, without touching ball observations."""
    def eligible(item):
        return not any(s["preparation_start_frame"] <= item["frame"] < s["frame"] for s in serves)
    return [e for e in events if eligible(e)], [b for b in bounces if eligible(b)]
