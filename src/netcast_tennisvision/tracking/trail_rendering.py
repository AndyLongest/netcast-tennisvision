"""Conservative screen-space stabilization for the presentation trail only.

The detector-backed current ball position remains exact.  Historical points may move only
a few pixels to reduce frame-to-frame acceleration noise; this module never changes the
tracker, event detector, landing coordinates, or exported confidence.
"""

from __future__ import annotations

import numpy as np


def stabilize_screen_trail(
    points,
    *,
    strength: float = 14.0,
    max_shift_px: float = 8.0,
) -> list[tuple[float, float]]:
    """Return a low-curvature trail whose newest point is exactly the observation.

    This is a second-difference regularized least-squares fit.  Unlike a global ballistic
    refit it stays close to every observed point.  The hard endpoint preserves the accurate
    purple ball overlay, while the curvature penalty spreads its small residual error over
    the recent tail instead of producing a one-frame hook.
    """
    observed = np.asarray(points, dtype=float)
    if observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("points must be an N x 2 sequence")
    count = len(observed)
    if count < 4:
        return [tuple(map(float, point)) for point in observed]

    second_difference = np.zeros((count - 2, count), dtype=float)
    rows = np.arange(count - 2)
    second_difference[rows, rows] = 1.0
    second_difference[rows, rows + 1] = -2.0
    second_difference[rows, rows + 2] = 1.0
    curvature = second_difference.T @ second_difference

    # Recent history receives more measurement weight. The current point is a hard anchor;
    # the penultimate points are still allowed to absorb a few pixels of endpoint noise.
    base_weights = np.linspace(0.75, 2.5, count)
    base_weights[-1] = 1_000_000.0
    weights = base_weights.copy()
    fitted = observed.copy()
    for _ in range(3):
        system = np.diag(weights) + float(strength) * curvature
        fitted = np.column_stack(
            [np.linalg.solve(system, weights * observed[:, axis]) for axis in (0, 1)]
        )
        residual = np.linalg.norm(fitted - observed, axis=1)
        scale = max(0.75, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        robust = np.minimum(1.0, (2.5 * scale) / np.maximum(residual, 1e-9))
        weights = base_weights * robust
        weights[-1] = 1_000_000.0

    displacement = fitted - observed
    distance = np.linalg.norm(displacement, axis=1)
    over = distance > max_shift_px
    displacement[over] *= (max_shift_px / distance[over])[:, None]
    fitted = observed + displacement
    fitted[-1] = observed[-1]
    return [tuple(map(float, point)) for point in fitted]


def contact_bounded_start(index: int, lower_bound: int, contact_frames) -> int:
    """Start at the newest completed contact, but show the incoming trail on its frame."""
    completed = [int(frame) for frame in contact_frames if lower_bound < frame < index]
    return max(lower_bound, max(completed)) if completed else lower_bound
