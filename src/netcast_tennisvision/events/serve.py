"""Conservative serve sequence proposals from observed ball and player geometry.

No pose model: these are geometric serve hypotheses, not measured racket actions.
Requires a toss and outgoing flight; never modifies ball observations.
"""
from __future__ import annotations

import numpy as np


def propose_serves(frames, *, fps):
    serves = []
    n = len(frames)
    for f, meta in enumerate(frames):
        if serves and f - serves[-1]["frame"] < fps * 2:
            continue
        ball = meta.get("ball_px")
        if ball is None or not meta.get("ball_seen"):
            continue
        for player in meta.get("players_world", []):
            x, y = player["xy"]
            side = player.get("side")
            baseline = 0 if side == "near" else 23.77
            if side not in ("near", "far") or abs(y - baseline) > 2.5:
                continue
            left, top, right, bottom = player["box"]
            height = bottom - top
            center = (left + right) / 2
            if height <= 0 or abs(ball[0] - center) > .7 * height:
                continue
            if not top - .9 * height <= ball[1] <= top + .2 * height:
                continue
            before = []
            positions = []
            for j in range(max(0, f - round(1.5 * fps)), f):
                m = frames[j]
                ps = [p for p in m.get("players_world", []) if p.get("side") == side]
                if ps:
                    positions.append(ps[0]["xy"])
                b = m.get("ball_px")
                if m.get("ball_seen") and b is not None and abs(b[0] - center) < .7 * height:
                    before.append((j, b[1]))
            if len(before) < 5 or len(positions) < fps * .6:
                continue
            if np.max(np.ptp(np.asarray(positions), axis=0)) > 1.2:
                continue
            # Ordered low-to-high observations demonstrate a toss rather than a caught lob.
            low = [(j, py) for j, py in before if top + .25 * height < py < bottom]
            if not low:
                continue
            launch, _ = low[-1]
            rise = [(j, py) for j, py in before if j > launch]
            if len(rise) < 3 or not .15 * fps <= f - launch <= 1.3 * fps:
                continue
            if sum(b[1] < a[1] for a, b in zip(rise, rise[1:], strict=False)) < .65 * (len(rise)-1):
                continue
            after = []
            for j in range(f + 1, min(n, f + round(.8 * fps) + 1)):
                m = frames[j]
                b = m.get("ball_px")
                if b is not None and m.get("ball_seen"):
                    after.append((j, b))
            if len(after) < 5:
                continue
            # Far-side serves travel down-screen, near-side serves up-screen toward the net.
            sign = -1 if side == "near" else 1
            departure = [(j, b) for j, b in after
                         if sign * (b[1] - ball[1]) > .7 * height
                         and abs(b[0] - center) < 2.5 * height]
            if not departure:
                continue
            # Near-side toss alone also rises: require a trajectory direction change at contact.
            post = np.array(after[min(4,len(after)-1)][1]) - np.array(ball)
            if side == "near" and abs(post[0]) < .12 * height and post[1] <= 0:
                continue
            serves.append({"frame": f, "decision_frame": departure[0][0],
                           "preparation_start_frame": launch, "side": side,
                           "evidence": "baseline+stable-player+observed-toss+overhead+outgoing-flight"})
            break
    return serves
