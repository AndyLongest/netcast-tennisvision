import numpy as np
import pytest

from netcast_tennisvision.tracking.speed import analyze_speeds, fit_flight_speed


def flight(height):
    camera = np.array([5.5, -8.6, height])
    forward = np.array([5.5, 11.9, 0]) - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(forward, right), forward])
    intrinsic = np.array([[1100, 0, 960], [0, 1100, 540], [0, 0, 1]])
    projection = intrinsic @ np.column_stack([rotation, -rotation @ camera])
    matrix = projection[:, [0, 1, 3]]
    times = np.arange(20) / 30
    t = times - times.mean()
    velocity = np.array([3., 18., 1.])
    world = np.array([5.5, 10, 2]) + t[:, None] * velocity
    world[:, 2] -= 4.905 * t**2
    screen = np.c_[world, np.ones(len(t))] @ projection.T
    return screen[:, :2] / screen[:, 2:], times, matrix, np.linalg.norm(velocity) * 3.6


@pytest.mark.parametrize('height', [1.43, 6.2])
def test_known_3d_speed_both_views(height):
    pixels, times, matrix, expected = flight(height)
    result = fit_flight_speed(pixels, times, matrix, (1920, 1080), drag=0)
    assert result is not None
    assert abs(result['speed_kmh'] - expected) < .2
    # Ground-plane projection of this airborne flight is not the true speed.
    assert result['camera_height_m'] == pytest.approx(height, abs=.001)


def test_no_stationary_or_short_speed():
    pixels, times, matrix, _ = flight(6.2)
    assert fit_flight_speed(pixels[:4], times[:4], matrix, (1920, 1080), drag=0) is None
    assert fit_flight_speed(np.repeat(pixels[:1], 20, axis=0), times, matrix, (1920, 1080), drag=0) is None


def test_contacts_split_and_synthetic_positions_excluded():
    pixels, times, matrix, _ = flight(6.2)
    frames = [dict(ball_px=p, ball_seen=True, M=matrix, ball_track_id=1) for p in pixels]
    assert analyze_speeds(frames, [], times, (1920, 1080))['estimates']
    assert not analyze_speeds(frames, [{'frame': 9}], times, (1920, 1080))['estimates']
    for frame in frames:
        frame['ball_seen'] = False
    assert not analyze_speeds(frames, [], times, (1920, 1080))['estimates']


@pytest.mark.parametrize('height', [1.43, 6.2])
def test_drag_flight_known_speed(height):
    from scipy.integrate import solve_ivp

    from netcast_tennisvision.tracking.speed import TENNIS_DRAG
    # Independent forward integration from the start, with a different state layout.
    camera = np.array([5.5, -8.6, height])
    forward = np.array([5.5, 11.9, 0]) - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(forward, right), forward])
    projection = np.array([[1100, 0, 960], [0, 1100, 540], [0, 0, 1]]) @ np.column_stack([rotation, -rotation @ camera])
    times = np.arange(20) / 30
    def derivative(t, state):
        return np.r_[state[3:], [0, 0, -9.81] - TENNIS_DRAG * np.linalg.norm(state[3:]) * state[3:]]
    truth = solve_ivp(derivative, (0, times[-1]), [4, 4, 1.8, 3, 28, 3],
                      dense_output=True, rtol=1e-11, atol=1e-12)
    world = truth.sol(times)[:3].T
    screen = np.c_[world, np.ones(len(times))] @ projection.T
    pixels = screen[:, :2] / screen[:, 2:]
    expected = np.linalg.norm(truth.sol(times.mean())[3:]) * 3.6
    result = fit_flight_speed(pixels, times, projection[:, [0, 1, 3]], (1920, 1080))
    assert result is not None
    assert abs(result['speed_kmh'] - expected) < .3
    assert result['drag_sensitivity_kmh'] >= 0


def test_speed_can_be_disabled():
    assert analyze_speeds([], [], [], (1920, 1080), method='off')['status'] == 'disabled'


@pytest.mark.parametrize('factor', [-3., -.01, .05, 8.])
def test_speed_homography_scale_invariance(factor):
    pixels, times, matrix, expected = flight(1.43)
    original = matrix.copy()
    result = fit_flight_speed(pixels, times, matrix * factor, (1920, 1080), drag=0)
    assert result is not None
    assert abs(result['speed_kmh'] - expected) < .2
    np.testing.assert_array_equal(matrix, original)


def test_speed_reports_rejection_reason():
    pixels, times, matrix, _ = flight(6.2)
    reasons = {}
    assert fit_flight_speed(pixels[:4], times[:4], matrix, (1920, 1080), diagnostics=reasons) is None
    assert reasons == {'invalid_or_insufficient_samples': 1}


@pytest.mark.parametrize('height', [1.43, 6.2])
def test_linear_speed_without_nonlinear_solver(height, monkeypatch):
    import netcast_tennisvision.tracking.speed as module
    def forbidden(*args, **kwargs):
        raise AssertionError('heavy solver called in linear mode')
    monkeypatch.setattr(module, 'least_squares', forbidden)
    monkeypatch.setattr(module, 'solve_ivp', forbidden)
    pixels, times, matrix, expected = flight(height)
    result = fit_flight_speed(pixels, times, matrix, (1920,1080), linear_only=True)
    assert result is not None
    # Compare velocity at the known flight midpoint.
    vz = 1 - 9.81 * (times.mean() - times.mean())
    assert abs(result['speed_kmh'] - np.linalg.norm([3,18,vz])*3.6) < .2
