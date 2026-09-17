import json
import math

from netcast_tennisvision.api import server
from netcast_tennisvision.api.server import (
    parse_live_court_corners,
    supports_native_fps,
    video_fingerprint,
)


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
        [[-0.1, 0.8], [0.8, 0.8], [0.65, 0.2], [0.35, 0.2]],
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
