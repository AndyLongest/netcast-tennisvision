from __future__ import annotations

from netcast_tennisvision.streaming.live_experiment import (
    LiveExperimentManager,
    _bounded_int_env,
    _camera_corners,
    _webrtc_origin,
    compare_landing_events,
)


def test_live_integer_tuning_is_bounded(monkeypatch) -> None:
    monkeypatch.setenv("NETCAST_TEST_INTEGER", "999")
    assert _bounded_int_env("NETCAST_TEST_INTEGER", 4, 1, 16) == 16
    monkeypatch.setenv("NETCAST_TEST_INTEGER", "invalid")
    assert _bounded_int_env("NETCAST_TEST_INTEGER", 4, 1, 16) == 4


def test_live_manager_reuses_only_an_active_session(monkeypatch) -> None:
    started = []

    class FakeSession:
        def __init__(self, _source=None, **_options) -> None:
            self.id = str(len(started))
            self.state = "preparing"

        def start(self) -> None:
            started.append(self)

        def snapshot(self) -> dict[str, str]:
            return {"state": self.state}

    monkeypatch.setattr(
        "netcast_tennisvision.streaming.live_experiment.LiveExperimentSession", FakeSession
    )
    manager = LiveExperimentManager()
    first = manager.start_demo()
    assert manager.start_demo() is first
    first.state = "complete"
    second = manager.start_demo()
    assert second is not first
    assert len(started) == 2


def test_live_manager_rejects_unknown_session() -> None:
    manager = LiveExperimentManager()
    assert manager.get("missing") is None
    assert manager.stop("missing") is False
    assert manager.finish("missing") is False


def test_live_manager_accepts_an_explicit_uploaded_source(tmp_path, monkeypatch) -> None:
    source = tmp_path / "uploaded.mp4"
    source.write_bytes(b"video")
    started = []

    class FakeSession:
        def __init__(self, selected_source, **options) -> None:
            self.id = "uploaded"
            self.source = selected_source
            self.options = options

        def start(self) -> None:
            started.append(self.source)

        def snapshot(self) -> dict[str, str]:
            return {"state": "running"}

    monkeypatch.setattr(
        "netcast_tennisvision.streaming.live_experiment.LiveExperimentSession", FakeSession
    )
    manager = LiveExperimentManager()

    corners = [[0.1, 0.8], [0.9, 0.8], [0.7, 0.2], [0.3, 0.2]]
    session = manager.start_source(source, court_corners=corners)

    assert session.source == source
    assert session.options["court_corners"] == corners
    assert started == [source]


def test_live_manager_accepts_an_external_camera_stream(monkeypatch) -> None:
    started = []

    class FakeSession:
        def __init__(self, source, **options) -> None:
            self.source = source
            self.options = options
            self.id = "external"

        def start(self) -> None:
            started.append(self)

        def snapshot(self) -> dict[str, str]:
            return {"state": "awaiting_stream"}

    monkeypatch.setattr(
        "netcast_tennisvision.streaming.live_experiment.LiveExperimentSession", FakeSession
    )
    manager = LiveExperimentManager()

    corners = [[0.1, 0.8], [0.9, 0.8], [0.7, 0.2], [0.3, 0.2]]
    session = manager.start_external_stream(
        "netcast-camera01", fps_hint=30.0, source_name="court.mp4", court_corners=corners
    )

    assert session.source is None
    assert session.options["external_stream_name"] == "netcast-camera01"
    assert session.options["source_name"] == "court.mp4"
    assert session.options["court_corners"] == corners
    assert started == [session]


def test_live_manager_finishes_only_the_named_camera_session(monkeypatch) -> None:
    class FakeSession:
        def __init__(self, source, **_options) -> None:
            self.id = "external"
            self.finished = False

        def start(self) -> None:
            pass

        def snapshot(self) -> dict[str, str]:
            return {"state": "running"}

        def finish_source(self) -> None:
            self.finished = True

    monkeypatch.setattr(
        "netcast_tennisvision.streaming.live_experiment.LiveExperimentSession", FakeSession
    )
    manager = LiveExperimentManager()
    session = manager.start_external_stream("netcast-camera01")

    assert manager.finish("missing") is False
    assert manager.finish(session.id) is True
    assert session.finished is True


def test_live_comparison_requires_time_and_position_agreement() -> None:
    reference = [
        {"frame": 100, "x": 4.0, "y": 6.0},
        {"frame": 200, "x": 7.0, "y": 18.0},
    ]
    online = [
        {"frame": 103, "x": 4.2, "y": 6.1},
        {"frame": 201, "x": 1.0, "y": 2.0},
        {"frame": 450, "x": 7.0, "y": 18.0},
    ]

    result = compare_landing_events(online, reference, fps=30.0)

    assert result["matched_events"] == 1
    assert result["recall"] == 0.5
    assert result["precision"] == 1 / 3


def test_explicit_live_court_corners_are_scaled_to_the_camera_frame() -> None:
    corners = [[0.1, 0.8], [0.9, 0.8], [0.7, 0.2], [0.3, 0.2]]

    scaled = _camera_corners(1920, 1080, corners)

    assert scaled.tolist() == [
        [192.0, 864.0],
        [1728.0, 864.0],
        [1344.0, 216.0],
        [576.0, 216.0],
    ]


def test_webrtc_origin_accepts_only_http_server_origins(monkeypatch) -> None:
    monkeypatch.setenv("TENNISVISION_ZLM_WEBRTC_ORIGIN", "https://media.example.com")
    assert _webrtc_origin("127.0.0.1") == "https://media.example.com"
