import io
import json
import math
import os

from netcast_tennisvision.api import server
from netcast_tennisvision.api.server import (
    parse_live_court_corners,
    supports_native_fps,
    video_fingerprint,
)


def test_ui_responses_revalidate_but_video_preserves_range_support():
    for path in ("/web/index.html", "/web/app.js?v=old", "/web/styles.css", "/"):
        handler = object.__new__(server.Handler)
        handler.path = path
        handler.request_version = "HTTP/1.1"
        handler.wfile = io.BytesIO()
        handler._headers_buffer = []
        handler.end_headers()
        assert b"Cache-Control: no-cache, must-revalidate" in handler.wfile.getvalue()
    handler.path = "/data/clip.mp4"
    handler.wfile = io.BytesIO()
    handler.end_headers()
    assert b"Accept-Ranges: bytes" in handler.wfile.getvalue()
    assert b"Cache-Control" not in handler.wfile.getvalue()


def test_accepts_any_positive_native_frame_rate():
    for fps in (1.0, 23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0, 120.0, 240.0):
        assert supports_native_fps(fps)


def test_rejects_only_invalid_frame_rates():
    for fps in (0.0, -1.0, math.nan, math.inf, -math.inf):
        assert not supports_native_fps(fps)


def test_live_court_confirmation_accepts_a_normalized_trapezoid():
    corners = [[0.2, 0.8], [0.8, 0.8], [0.65, 0.2], [0.35, 0.2]]

    assert parse_live_court_corners(corners) == corners


def test_live_court_confirmation_rejects_missing_or_malformed_points():
    for corners in (
        None,
        [],
        [[0.2, 0.8], [0.8, 0.8], [0.65, 0.2]],
        [[0.2, 0.8, 1], [0.8, 0.8], [0.65, 0.2], [0.35, 0.2]],
        [[float("nan"), 0.8], [0.8, 0.8], [0.65, 0.2], [0.35, 0.2]],
    ):
        try:
            parse_live_court_corners(corners)
        except ValueError:
            continue
        raise AssertionError(f"invalid corners were accepted: {corners!r}")


def test_status_keeps_job_identity_across_pipeline_progress_updates(tmp_path, monkeypatch):
    status_path = tmp_path / "job_status.json"
    current_job_path = tmp_path / "current_job.json"
    status_path.write_text(json.dumps({"state": "running", "progress": 54, "stage": "追踪网球"}), encoding="utf-8")
    current_job_path.write_text(json.dumps({
        "job_id": "job-1",
        "filename": "match.mp4",
        "video_fingerprint": "sha256-sample:abc",
        "fps": 29.97,
    }), encoding="utf-8")
    monkeypatch.setattr(server, "STATUS", status_path)
    monkeypatch.setattr(server, "CURRENT_JOB", current_job_path)
    monkeypatch.setattr(server, "cloud_manager", None)

    payload = server.status_payload()

    assert payload["state"] == "running"
    assert payload["progress"] == 54
    assert payload["job_id"] == "job-1"
    assert payload["filename"] == "match.mp4"
    assert payload["execution_target"] == "local"


def test_cloud_bearer_secret_is_required(monkeypatch):
    class Request:
        headers = {"Authorization": "Bearer wrong"}
        response = None

        def send_json(self, payload, status):
            self.response = (payload, status)

    monkeypatch.setattr(server, "CLOUD_SHARED_SECRET", "relay-secret")
    request = Request()

    assert not server.Handler.cloud_request_authorized(request)
    assert request.response[1] == 401
    request.headers["Authorization"] = "Bearer relay-secret"
    assert server.Handler.cloud_request_authorized(request)


def test_running_job_is_resumable_only_for_the_same_video(tmp_path, monkeypatch):
    status_path = tmp_path / "job_status.json"
    current_job_path = tmp_path / "current_job.json"
    status_path.write_text(json.dumps({"state": "running", "progress": 28}), encoding="utf-8")
    current_job_path.write_text(json.dumps({
        "job_id": "job-2",
        "video_fingerprint": "sha256-sample:same",
    }), encoding="utf-8")
    monkeypatch.setattr(server, "STATUS", status_path)
    monkeypatch.setattr(server, "CURRENT_JOB", current_job_path)

    assert server.resumable_job("sha256-sample:same")["job_id"] == "job-2"
    assert server.resumable_job("sha256-sample:different") is None
    assert server.resumable_job(None) is None


def test_report_ready_job_remains_resumable_while_video_renders(tmp_path, monkeypatch):
    status_path = tmp_path / "job_status.json"
    current_job_path = tmp_path / "current_job.json"
    status_path.write_text(json.dumps({"state": "report_ready", "progress": 96}), encoding="utf-8")
    current_job_path.write_text(json.dumps({
        "job_id": "job-report",
        "video_fingerprint": "sha256-sample:report",
    }), encoding="utf-8")
    monkeypatch.setattr(server, "STATUS", status_path)
    monkeypatch.setattr(server, "CURRENT_JOB", current_job_path)

    assert server.resumable_job("sha256-sample:report")["job_id"] == "job-report"


def test_video_fingerprint_uses_content_and_size(tmp_path):
    first = tmp_path / "first.mp4"
    renamed = tmp_path / "renamed.mp4"
    changed = tmp_path / "changed.mp4"
    first.write_bytes(b"tennis-video" * 100)
    renamed.write_bytes(first.read_bytes())
    changed.write_bytes(first.read_bytes() + b"!")

    assert video_fingerprint(first) == video_fingerprint(renamed)
    assert video_fingerprint(first) != video_fingerprint(changed)


def test_analysis_version_matches_the_cloud_release():
    assert server.ANALYSIS_VERSION == "production-v31"


def test_completed_analysis_is_archived_and_can_be_deleted(tmp_path, monkeypatch):
    data = tmp_path / "data"
    outputs = data / "outputs"
    history = data / "history"
    outputs.mkdir(parents=True)
    (data / "clip.mp4").write_bytes(b"video")
    (outputs / "scene3d.json").write_text(
        json.dumps({"fps": 30, "n_frames": 300, "bounces": [{}, {}]}),
        encoding="utf-8",
    )
    (outputs / "rally3d.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(server, "DATA", data)
    monkeypatch.setattr(server, "HISTORY", history)

    server.archive_completed_analysis({
        "state": "complete",
        "job_id": "history-job-1",
        "filename": "match.mp4",
        "started_at": 100,
        "fps": 30,
        "file_size": 5,
        "video_fingerprint": server.video_fingerprint(data / "clip.mp4"),
        "event_overlay_ready": True,
    })

    records = server.analysis_history()
    assert len(records) == 1
    assert records[0]["duration"] == 10
    assert records[0]["bounce_count"] == 2
    assert records[0]["assets"]["original"].endswith("/source.mp4")
    assert server.delete_history_record("history-job-1")
    assert server.analysis_history() == []
    server.archive_completed_analysis({
        "state": "complete",
        "job_id": "history-job-1",
        "filename": "match.mp4",
        "video_fingerprint": server.video_fingerprint(data / "clip.mp4"),
    })
    assert server.analysis_history() == []


def test_history_delete_removes_matching_resume_state_outputs_and_caches(
    tmp_path, monkeypatch,
):
    data = tmp_path / "data"
    history = data / "history"
    cache = data / "cache"
    outputs = data / "outputs"
    record_dir = history / "history-job-2"
    record_dir.mkdir(parents=True)
    cache.mkdir()
    outputs.mkdir()
    source = record_dir / "source.mp4"
    source.write_bytes(b"deleted-video")
    os.utime(source, (1234567890, 1234567890))
    fingerprint = server.video_fingerprint(source)
    (record_dir / "record.json").write_text(json.dumps({
        "job_id": "history-job-2", "video_fingerprint": fingerprint,
    }), encoding="utf-8")
    current_source = data / "clip.mp4"
    current_source.write_bytes(source.read_bytes())
    (outputs / "scene3d.json").write_text("{}", encoding="utf-8")
    current_job = data / "current_job.json"
    status = data / "job_status.json"
    current_job.write_text(json.dumps({
        "job_id": "history-job-2", "video_fingerprint": fingerprint,
    }), encoding="utf-8")
    status.write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    identities = data / "video_identities.json"
    identities.write_text(json.dumps({server.file_sha256(source): 1234567890}), encoding="utf-8")
    matching = cache / f"passA_{source.stat().st_size}_1234567890_test.pkl"
    matching.write_bytes(b"cache")
    unrelated = cache / "passA_99_1_other.pkl"
    unrelated.write_bytes(b"keep")

    monkeypatch.setattr(server, "DATA", data)
    monkeypatch.setattr(server, "HISTORY", history)
    monkeypatch.setattr(server, "CACHE", cache)
    monkeypatch.setattr(server, "OUTPUTS", outputs)
    monkeypatch.setattr(server, "CURRENT_JOB", current_job)
    monkeypatch.setattr(server, "STATUS", status)
    monkeypatch.setattr(server, "VIDEO_IDENTITIES", identities)
    monkeypatch.setattr(server, "LOG", data / "pipeline.log")
    monkeypatch.setattr(server, "CALIBRATION_REQUEST", data / "request.json")
    monkeypatch.setattr(server, "CALIBRATION_RESPONSE", data / "response.json")

    assert server.delete_history_record("history-job-2")
    assert not record_dir.exists()
    assert not current_source.exists()
    assert not outputs.exists()
    assert not current_job.exists()
    assert not matching.exists()
    assert unrelated.exists()
    assert server.read_json(status)["state"] == "idle"
    assert json.loads(identities.read_text(encoding="utf-8")) == {}


def test_history_delete_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "HISTORY", tmp_path / "history")

    assert not server.delete_history_record("../outside")


def test_clear_all_analysis_records_preserves_non_runtime_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    history = data / "history"
    cache = data / "cache"
    uploads = data / "uploads"
    outputs = data / "outputs"
    for folder in (history, cache, uploads, outputs):
        folder.mkdir(parents=True)
        (folder / "artifact.bin").write_bytes(b"runtime")
    (data / "clip.mp4").write_bytes(b"video")
    (data / "current_job.json").write_text("{}", encoding="utf-8")
    (data / "video_identities.json").write_text("{}", encoding="utf-8")
    (data / "camera_profiles.json").write_text("{\"keep\": true}", encoding="utf-8")

    monkeypatch.setattr(server, "DATA", data)
    monkeypatch.setattr(server, "HISTORY", history)
    monkeypatch.setattr(server, "CACHE", cache)
    monkeypatch.setattr(server, "UPLOADS", uploads)
    monkeypatch.setattr(server, "OUTPUTS", outputs)
    monkeypatch.setattr(server, "CURRENT_JOB", data / "current_job.json")
    monkeypatch.setattr(server, "STATUS", data / "job_status.json")
    monkeypatch.setattr(server, "VIDEO_IDENTITIES", data / "video_identities.json")
    monkeypatch.setattr(server, "LOG", data / "pipeline.log")
    monkeypatch.setattr(server, "CALIBRATION_REQUEST", data / "request.json")
    monkeypatch.setattr(server, "CALIBRATION_RESPONSE", data / "response.json")

    server.clear_all_analysis_records()

    assert list(history.iterdir()) == []
    assert list(cache.iterdir()) == []
    assert list(uploads.iterdir()) == []
    assert not outputs.exists()
    assert not (data / "clip.mp4").exists()
    assert not (data / "current_job.json").exists()
    assert not (data / "video_identities.json").exists()
    assert server.read_json(data / "job_status.json")["state"] == "idle"
    assert json.loads((data / "camera_profiles.json").read_text(encoding="utf-8")) == {"keep": True}


def test_force_stop_clears_stale_job_and_can_repeat(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(server, "CURRENT_JOB", tmp_path / "current.json")
    monkeypatch.setattr(server, "cloud_manager", None)
    monkeypatch.setattr(server, "job_process", None)
    server.write_json_atomic(server.STATUS, {"state": "running", "progress": 36})
    server.stop_analysis()
    server.stop_analysis()
    assert server.status_payload()["state"] == "cancelled"
    assert server.resumable_job(None) is None


def test_stop_kills_local_process_tree_before_clearing_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    calls = []
    monkeypatch.setattr(server, "STATUS", tmp_path / "status.json")
    monkeypatch.setattr(server, "cloud_manager", None)
    process = SimpleNamespace(pid=123, poll=lambda: None, wait=lambda **_: calls.append("wait"))
    monkeypatch.setattr(server, "job_process", process)
    if os.name == "nt":
        monkeypatch.setattr(server.subprocess, "run", lambda command, **_: calls.append(command))
    else:
        monkeypatch.setattr(server.os, "killpg", lambda *args: calls.append(args))
    server.stop_analysis()
    assert calls[-1] == "wait"
    if os.name == "nt":
        assert calls[0] == ["taskkill", "/PID", "123", "/T", "/F"]
    assert server.job_process is None
    assert server.read_json(server.STATUS)["state"] == "cancelled"

def test_local_gpu_upload_bypasses_configured_cloud(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    for name in ('STATUS', 'CURRENT_JOB', 'LOG', 'VIDEO_IDENTITIES'):
        monkeypatch.setattr(server, name, tmp_path / f'{name}.json')
    monkeypatch.setattr(server, 'DATA', tmp_path)
    monkeypatch.setattr(server, 'CLOUD_API_URL', 'https://cloud.invalid')
    monkeypatch.setattr(server, 'job_process', None)
    manager = SimpleNamespace(active=False, stopping=False, runtime_path=tmp_path / 'absent', start=Mock(), stop=Mock())
    monkeypatch.setattr(server, 'cloud_manager', manager)
    monkeypatch.setattr(server, 'require_local_gpu', lambda: 'test GPU')
    monkeypatch.setattr(server, 'probe_native_fps', lambda _: 30.0)
    monkeypatch.setattr(server, 'ffmpeg_directory', lambda: None)
    popen = Mock(return_value=SimpleNamespace(pid=123, poll=lambda: 0))
    monkeypatch.setattr(server.subprocess, 'Popen', popen)
    handler = object.__new__(server.Handler)
    handler.path = '/api/analyze'
    handler.headers = {'Content-Length': '5', 'X-Filename': 'clip.mp4', 'X-Execution-Target': 'local-gpu'}
    handler.rfile = io.BytesIO(b'video')
    handler.send_json = Mock()
    handler.cloud_request_authorized = lambda: True
    handler.proxy_cloud_request = Mock()
    handler.do_POST()
    assert handler.send_json.call_args.args[1] == 202
    assert handler.send_json.call_args.args[0]['execution_target'] == 'local-gpu'
    assert popen.call_args.kwargs['env']['TENNISVISION_REQUIRE_LOCAL_CUDA'] == '1'
    manager.start.assert_not_called()
    handler.proxy_cloud_request.assert_not_called()
    handler.path = '/api/court-calibration'
    handler.save_court_calibration = Mock()
    handler.do_POST()
    handler.save_court_calibration.assert_called_once()
    server.stop_analysis()
    manager.stop.assert_not_called()
    assert server.read_json(server.STATUS)['state'] == 'cancelled'


def test_unavailable_local_gpu_rejects_without_upload_or_cloud(tmp_path, monkeypatch):
    from unittest.mock import Mock

    monkeypatch.setattr(server, 'STATUS', tmp_path / 'status.json')
    monkeypatch.setattr(server, 'CURRENT_JOB', tmp_path / 'current.json')
    monkeypatch.setattr(server, 'cloud_manager', None)
    monkeypatch.setattr(server, 'job_process', None)
    monkeypatch.setattr(server, 'CLOUD_API_URL', '')
    def unavailable():
        raise ValueError('GPU unavailable')
    monkeypatch.setattr(server, 'require_local_gpu', unavailable)
    handler = object.__new__(server.Handler)
    handler.path = '/api/analyze'
    handler.headers = {'X-Execution-Target': 'local-gpu'}
    handler.cloud_request_authorized = lambda: True
    handler.send_json = Mock()
    handler.do_POST()
    assert handler.send_json.call_args.args[1] == 422
    assert handler.send_json.call_args.args[0]['code'] == 'local_gpu_unavailable'
    assert not server.CURRENT_JOB.exists()

