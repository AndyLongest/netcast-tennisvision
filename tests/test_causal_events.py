import pytest

from netcast_tennisvision.streaming.causal_events import schedule_landing_events


def test_live_event_waits_for_every_required_future_frame():
    result = schedule_landing_events(
        [{"frame": 100, "touchdown_frame_f": 100.2, "decision_frame": 103}],
        fps=30.0,
        future_support_frames=10,
    )[0]

    assert result["available_frame"] == 111
    assert result["available_t"] == pytest.approx(3.7)
    assert result["causal_delay_ms"] == pytest.approx(360.0)


def test_later_explicit_decision_boundary_is_preserved():
    result = schedule_landing_events(
        [{"frame": 20, "touchdown_frame_f": 20.0, "decision_frame": 35}],
        fps=25.0,
        future_support_frames=10,
    )[0]

    assert result["available_frame"] == 35


@pytest.mark.parametrize("fps,future", [(0.0, 10), (30.0, -1)])
def test_invalid_live_schedule_is_rejected(fps, future):
    with pytest.raises(ValueError):
        schedule_landing_events([], fps=fps, future_support_frames=future)
