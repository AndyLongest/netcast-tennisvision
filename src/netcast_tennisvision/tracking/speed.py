"""Read-only monocular flight-speed estimates, never ground-projected speeds.

Calibration assumes square pixels, a centred principal point and negligible lens
distortion. Results are model estimates, not radar-validated measurements.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .geometry import camera_centre_from_homography

# Tennis prior: k = rho * Cd * pi * radius**2 / (2 * mass), in 1/m.
# Independent implementation of standard aerodynamic dynamics; see research note.
TENNIS_DRAG = 1.21 * .55 * np.pi * .033**2 / (2 * .057)


def flight_states(state, times, drag):
    """Integrate [x,vx,y,vy,z,vz] from the window centre in both directions."""
    times = np.asarray(times, float)
    if drag == 0:
        result = np.tile(state, (len(times), 1))
        result[:, [0, 2, 4]] += times[:, None] * state[[1, 3, 5]]
        result[:, 4] -= 4.905 * times**2
        result[:, 5] -= 9.81 * times
        return result

    def dynamics(_, value):
        velocity = value[[1, 3, 5]]
        acceleration = -drag * np.linalg.norm(velocity) * velocity + [0, 0, -9.81]
        return np.column_stack((velocity, acceleration)).ravel()

    result = np.tile(state, (len(times), 1))
    for sign in (-1, 1):
        indices = np.flatnonzero(sign * times > 0)
        indices = indices[np.argsort(sign * times[indices])]
        if not len(indices):
            continue
        integration = solve_ivp(dynamics, (0, times[indices[-1]]), state,
                                t_eval=times[indices], rtol=1e-6, atol=1e-8)
        if not integration.success:
            raise ValueError('flight integration failed')
        result[indices] = integration.y.T
    return result


def fit_flight_speed(pixels, times, matrix, frame_size, *, drag=TENNIS_DRAG, diagnostics=None, linear_only=False):
    """Estimate 3-D speed at the centre of one uninterrupted flight window.

    Reject short, poorly conditioned or non-ballistic tracks. No input is mutated.
    The interval describes pixel-noise sensitivity only, not total uncertainty.
    """
    def reject(reason):
        if diagnostics is not None:
            diagnostics[reason] = diagnostics.get(reason, 0) + 1
        return None

    if linear_only:
        drag = 0.0
    pixels, times = np.asarray(pixels, float), np.asarray(times, float)
    matrix = np.asarray(matrix, float)
    if (len(times) < 10 or pixels.shape != (len(times), 2)
            or matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(pixels))):
        return reject("invalid_or_insufficient_samples")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        return reject("invalid_timestamps")
    if not .3 - 1e-9 <= np.ptp(times) <= .85 or np.max(np.diff(times)) > .12:
        return reject("time_span_or_gap")
    # Homographies have arbitrary nonzero scale, including a negative sign.
    # Live inversion and offline getPerspectiveTransform choose different gauges.
    if abs(matrix[2, 2]) < 1e-12:
        return reject("calibration_gauge_unavailable")
    matrix = matrix / matrix[2, 2]
    try:
        camera = camera_centre_from_homography(matrix, frame_size, min_height=.05)
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return reject("singular_calibration")
    if camera is None:
        return reject("camera_unavailable")
    xy, z = camera
    ground_h = np.c_[pixels, np.ones(len(pixels))] @ inverse.T
    if np.min(np.abs(ground_h[:, 2])) < 1e-9:
        return reject("ray_at_horizon")
    ground = ground_h[:, :2] / ground_h[:, 2:]
    rays = (ground - xy) / z
    t = times - np.mean(times)
    a = np.zeros((2 * len(t), 6))
    a[::2, :2] = np.c_[np.ones(len(t)), t]
    a[1::2, 2:4] = np.c_[np.ones(len(t)), t]
    a[::2, 4:] = rays[:, :1] * np.c_[np.ones(len(t)), t]
    a[1::2, 4:] = rays[:, 1:] * np.c_[np.ones(len(t)), t]
    b = (ground + rays * (4.905 * t[:, None] ** 2)).ravel()
    initial, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < 6:
        return reject("rank_deficient")

    def project(state):
        states = flight_states(state, t, drag)
        height = states[:, 4]
        denominator = z - height
        denominator = np.where(abs(denominator) < 1e-8, 1e-8, denominator)
        world = states[:, [0, 2]]
        # Homogeneous projection stays well-defined at ball height == camera height.
        floor_h = np.c_[z * world - height[:, None] * xy, denominator]
        screen = floor_h @ matrix.T
        return screen[:, :2] / screen[:, 2:]

    def residual(state):
        return (project(state) - pixels).ravel()

    if linear_only:
        # Experimental fast path: no nonlinear search and no ODE integration.
        # Numerical pixel Jacobian is for uncertainty screening only.
        columns = []
        for axis in range(6):
            step = np.zeros(6)
            step[axis] = 1e-5 * max(1.0, abs(initial[axis]))
            columns.append((residual(initial + step) - residual(initial - step)) / (2 * step[axis]))
        fit = SimpleNamespace(x=initial, success=True, jac=np.column_stack(columns))
    else:
        fit = least_squares(residual, initial, loss='soft_l1', f_scale=2, max_nfev=100)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return reject("optimizer_failed")
    error = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    scale = frame_size[0] / 1920
    if np.median(error) > 2.5 * scale or np.percentile(error, 90) > 6 * scale:
        return reject("reprojection_error")
    heights = flight_states(fit.x, t, drag)[:, 4]
    velocity = fit.x[[1, 3, 5]]
    speed = float(np.linalg.norm(velocity))
    if min(heights) < -.1 or max(heights) > 8 or not 1 <= speed <= 85:
        return reject("nonphysical_flight")
    # Linearized noise propagation; reject weak depth observability.
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    if singular[-1] < 1e-6 or singular[0] / singular[-1] > 1e6:
        return reject("ill_conditioned")
    covariance = np.linalg.inv(fit.jac.T @ fit.jac) * max(scale, float(np.median(error)))**2
    gradient = np.zeros(6)
    gradient[[1, 3, 5]] = velocity / speed
    sensitivity = float(2 * np.sqrt(max(0, gradient @ covariance @ gradient)))
    if sensitivity > .2 * speed:
        return reject("pixel_sensitivity")
    drag_sensitivity = 0.0
    nominal_drag = drag
    if drag > 0:
        # Probe the material prior rather than presenting one Cd as measured truth.
        for candidate_drag in (nominal_drag * .75, nominal_drag * 1.25):
            drag = candidate_drag
            alternative = least_squares(residual, fit.x, loss='soft_l1', f_scale=2,
                                        max_nfev=40)
            if not alternative.success:
                return reject("drag_optimizer_failed")
            other_speed = float(np.linalg.norm(alternative.x[[1, 3, 5]]))
            drag_sensitivity = max(drag_sensitivity, abs(other_speed - speed))
        drag = nominal_drag
        if drag_sensitivity > .2 * speed:
            return reject("drag_sensitivity")
    return {
        'time_s': round(float(np.mean(times)), 4),
        'drag_per_m': round(float(drag), 6),
        'drag_sensitivity_kmh': round(drag_sensitivity * 3.6, 1),
        'speed_kmh': round(speed * 3.6, 1),
        'pixel_sensitivity_kmh': round(sensitivity * 3.6, 1),
        'reprojection_median_px': round(float(np.median(error)), 2),
        'observations': len(t), 'camera_height_m': round(z, 3),
        'camera_elevation_to_court_centre_deg': round(float(np.degrees(
            np.arctan2(z, np.linalg.norm(xy - [5.485, 11.885])))), 2),
        'start_time_s': round(float(times[0]), 4),
        'end_time_s': round(float(times[-1]), 4),
    }


def analyze_speeds(frames, events, times, frame_size, *, method="drag", diagnostics=None):
    if method not in {"drag", "gravity", "linear", "off"}:
        raise ValueError("unknown speed method")
    if method == "off":
        return {"method": "off", "status": "disabled", "estimates": []}
    """Split at contacts, track/camera changes and gaps; fit bounded windows."""
    contacts = {int(e['frame']) for e in events}
    estimates, samples = [], []
    previous = None

    def finish():
        # Non-overlapping windows avoid counting one flight repeatedly.
        for start in range(0, len(samples), 20):
            indices = samples[start:start + 20]
            if len(indices) < 10:
                if diagnostics is not None:
                    diagnostics['short_segment'] = diagnostics.get('short_segment', 0) + 1
                continue
            try:
                result = fit_flight_speed(
                    [frames[i]['ball_px'][:2] for i in indices],
                    [times[i] for i in indices], frames[indices[0]]['M'], frame_size,
                    drag=TENNIS_DRAG if method == 'drag' else 0, diagnostics=diagnostics,
                    linear_only=method == 'linear',
                )
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                # Optional reporting must not abort a valid detection report.
                result = None
            if result is not None:
                result.update(start_frame=indices[0], end_frame=indices[-1])
                estimates.append(result)
        samples.clear()

    for i, meta in enumerate(frames):
        boundary = i in contacts or not meta.get('is_court', True)
        if previous is not None:
            old = frames[previous]
            boundary |= (meta.get('ball_track_id') != old.get('ball_track_id')
                         or times[i] - times[previous] > .12
                         or not np.array_equal(meta.get('M'), old.get('M')))
        if boundary:
            finish()
            previous = None
        if i in contacts:
            continue
        if meta.get('ball_seen') and meta.get('ball_px') is not None and meta.get('M') is not None:
            samples.append(i)
            previous = i
    finish()
    return {'method': {'drag': 'court-camera-drag-v2', 'gravity': 'court-camera-gravity-v1', 'linear': 'court-camera-linear-experimental'}[method], 'status': 'estimated' if estimates else 'unavailable',
            'assumptions': ('centred principal point; square pixels; no lens distortion; '
                            + ('gravity + tennis quadratic drag; spin and wind unmodelled'
                               if method == 'drag' else 'gravity only; drag, spin and wind unmodelled')),
            'metric': '3D speed at window midpoint, not racket exit speed',
            'estimates': estimates}
