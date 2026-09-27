"""Bounded causal stationary-candidate memory for opt-in tracking experiments."""
from __future__ import annotations

import numpy as np


class StationaryPrior:
    def __init__(self, fps, spatial):
        self.fps = fps
        self.radius = max(2.0, 2.5 * spatial)
        self.anchors = []

    def reset(self):
        self.anchors.clear()

    def update(self, frame, points):
        # Recent repeated dwell, not whole-video density. An abandoned location expires.
        self.anchors = [a for a in self.anchors if frame-a['last'] <= self.fps]
        result = []
        for point in points:
            p = np.asarray(point, float)
            matches = [a for a in self.anchors if np.linalg.norm(p-a['point']) <= self.radius]
            anchor = min(matches, key=lambda a: np.linalg.norm(p-a['point'])) if matches else None
            if anchor is None:
                anchor = dict(point=p.copy(), first=frame, last=frame, samples=[])
                self.anchors.append(anchor)
            if not anchor['samples'] or anchor['samples'][-1] != frame:
                anchor['samples'].append(frame)
            anchor['samples'] = [f for f in anchor['samples'] if frame-f <= self.fps]
            anchor['last'] = frame
            samples = anchor['samples']
            result.append(len(samples) >= max(6, round(.5*self.fps))
                          and frame-samples[0] >= .65*self.fps)
        self.anchors = self.anchors[-128:]
        return result


def coherent_birth(observations, minimum_span):
    points = np.asarray([row[1] for row in observations], float)
    if len(points) < 3:
        return False
    steps = np.diff(points, axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    displacement = float(np.linalg.norm(points[-1]-points[0]))
    # Reject back-and-forth jitter while allowing a curved launch or toss.
    return displacement >= minimum_span and displacement >= .65 * float(lengths.sum())
