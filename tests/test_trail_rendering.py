import numpy as np

from netcast_tennisvision.tracking.trail_rendering import (
    contact_bounded_start,
    stabilize_screen_trail,
)


def _acceleration_energy(points):
    return float(np.sum(np.diff(np.asarray(points), n=2, axis=0) ** 2))


def test_stabilizer_reduces_wave_but_keeps_current_ball_exact():
    time = np.arange(25, dtype=float)
    clean = np.column_stack((100 + 8 * time, 300 - 5 * time + 0.12 * time**2))
    noisy = clean + np.column_stack((3.5 * (-1.0) ** time, 2.5 * np.sin(time * 2.2)))

    stable = np.asarray(stabilize_screen_trail(noisy, strength=14.0, max_shift_px=8.0))

    assert np.array_equal(stable[-1], noisy[-1])
    assert _acceleration_energy(stable) < 0.18 * _acceleration_energy(noisy)
    assert np.max(np.linalg.norm(stable - noisy, axis=1)) <= 8.0 + 1e-9


def test_short_trail_is_left_unchanged():
    points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    assert stabilize_screen_trail(points) == points


def test_contact_boundary_is_applied_after_contact_not_on_contact_frame():
    contacts = [10, 20]
    assert contact_bounded_start(20, 0, contacts) == 10
    assert contact_bounded_start(21, 0, contacts) == 20
    assert contact_bounded_start(30, 24, contacts) == 24
