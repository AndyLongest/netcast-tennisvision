"""Opt-in contact evidence from visible shoulder/elbow/wrist sequences.

Consumes cached poses and accepted ball observations. Missing joints are unknown,
never proof of absent swing. No model loading, tracking mutation or hidden I/O.
"""
from __future__ import annotations

import numpy as np

from .landing_detector import estimate_landing_subframe
from .landing_event_detector import score_landing_impulse


def pose_contact_evidence(frame, frames, poses, *, fps):
    radius = max(3, round(.20*fps))
    ball = frames[frame].get('ball_px')
    if ball is None:
        return dict(status='unknown', reason='no_ball', decision_frame=frame)
    rows = [r for r in poses if abs(r['frame']-frame) <= radius]
    details = []
    for side in sorted({r['side'] for r in rows}):
        arms = []
        for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
            valid = []
            for r in rows:
                if r['side'] != side or r['points'] is None or r['height'] <= 0:
                    continue
                kp = np.asarray(r['points'], float)
                if kp.shape != (17, 3) or not np.isfinite(kp).all():
                    continue
                if min(kp[k, 2] for k in (shoulder, elbow, wrist)) < .5:
                    continue
                valid.append((r, kp))
            valid.sort(key=lambda item: item[0]['frame'])
            if (len(valid) < 3 or valid[0][0]['frame'] > frame-.08*fps
                    or valid[-1][0]['frame'] < frame+.08*fps):
                arms.append(dict(status='unknown'))
                continue
            closest, kp = min(valid, key=lambda item: abs(item[0]['frame']-frame))
            if abs(closest['frame']-frame) > max(1, round(.07*fps)):
                arms.append(dict(status='unknown'))
                continue
            rel = np.asarray([(p[wrist, :2]-p[shoulder, :2])/r['height'] for r, p in valid])
            motion = float(np.max(np.linalg.norm(rel[:, None]-rel[None, :], axis=2)))
            distance = float(np.linalg.norm(kp[wrist, :2]-ball)/closest['height'])
            status = ('swing' if motion >= .18 and distance <= .55 else
                      'quiet' if motion < .12 or (distance > .70 and motion < .18) else 'unknown')
            arms.append(dict(status=status, motion=motion, distance=distance,
                             decision_frame=valid[-1][0]['frame']))
        status = ('swing' if any(a['status']=='swing' for a in arms) else
                  'quiet' if all(a['status']=='quiet' for a in arms) else 'unknown')
        details.append(dict(side=side, status=status, arms=arms))
    status = ('swing' if any(d['status']=='swing' for d in details) else
              'quiet' if details and all(d['status']=='quiet' for d in details) else 'unknown')
    return dict(status=status, players=details,
                decision_frame=max([frame]+[a.get('decision_frame', frame)
                                            for d in details for a in d['arms']]))


def propose_pose_contact_corrections(events, frames, poses, *, fps, enabled=False):
    """Return event-only proposals; quiet arms alone can never declare a bounce."""
    if not enabled:
        return []
    corrections = []
    radius = max(4, round(7*fps/30))
    for event in events:
        if event.get('kind') != 'hit':
            continue
        f = int(event['frame'])
        pose = pose_contact_evidence(f, frames, poses, fps=fps)
        if pose['status'] != 'quiet':
            continue
        # An upstream contact proposal can precede the actual image impulse.
        # Search a bounded +/-67 ms, then check the arms again at that candidate.
        search = max(1, round(fps/15))
        scored = [score_landing_impulse(j, frames, radius=radius)
                  for j in range(max(0, f-search), min(len(frames), f+search+1))]
        impulse = max((s for s in scored if s is not None),
                      key=lambda s: s.score, default=None)
        candidate = impulse.frame if impulse is not None else f
        pose = pose_contact_evidence(candidate, frames, poses, fps=fps)
        fit = estimate_landing_subframe(candidate, frames, radius=radius)
        if (impulse is None or impulse.score < .75 or impulse.bic_gain < 10
                or pose['status'] != 'quiet'
                or fit.get('confidence') != 'subframe' or fit.get('px') is None):
            continue
        corrections.append(dict(frame=f, candidate_frame=candidate, kind='bounce', pose=pose,
                                impulse_score=impulse.score, fit=fit,
                                decision_frame=max(pose['decision_frame'], fit['decision_frame']),
                                evidence='visible-quiet-arms+measured-ground-impulse'))
    return corrections
