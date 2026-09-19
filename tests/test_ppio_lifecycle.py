import os
import threading
from pathlib import Path

import pytest

from netcast_tennisvision.cloud.ppio_lifecycle import (
    CloudLifecycleError,
    PPIOJobManager,
    PPIOLiveJobManager,
)


def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PPIOJobManager:
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    return PPIOJobManager(tmp_path, tmp_path / "data" / "job_status.json")


def test_manager_requires_both_provider_and_relay_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("PPIO_API_KEY", raising=False)
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")

    assert not PPIOJobManager(tmp_path, tmp_path / "status.json").configured


def test_instance_id_can_be_read_from_nested_provider_response(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)

    assert lifecycle._find_string({"data": {"instanceId": "gpu-123"}}, ("instanceId", "id")) == "gpu-123"


def test_mirrored_version_comes_from_deployed_image(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)
    lifecycle.image = "registry/example:production-v29"
    monkeypatch.setattr(lifecycle, "_remote_request", lambda *_: (
        200, {"state": "complete", "algorithm_version": "production-v27"}))
    monkeypatch.setattr(lifecycle, "_download_report", lambda *_: None)
    monkeypatch.setattr(lifecycle, "_download_outputs", lambda *_, **__: None)
    monkeypatch.setattr(lifecycle, "_download_path", lambda *_, **__: None)
    lifecycle._mirror_until_complete("https://worker.example")
    assert lifecycle._read_json(lifecycle.status_path)["algorithm_version"] == "production-v29"


def test_worker_releases_instance_after_analysis_failure(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    released = []
    monkeypatch.setattr(lifecycle, "_create_instance", lambda: "gpu-456")
    monkeypatch.setattr(lifecycle, "_wait_for_endpoint", lambda _instance: (_ for _ in ()).throw(RuntimeError("startup failed")))
    monkeypatch.setattr(lifecycle, "_release_instance", lambda instance: released.append(instance) or True)

    lifecycle._run(clip, {})

    assert released == ["gpu-456"]
    assert lifecycle._read_json(lifecycle.status_path)["state"] == "error"
    assert not lifecycle.runtime_path.exists()


def test_failed_release_keeps_recovery_journal(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    monkeypatch.setattr(lifecycle, "_create_instance", lambda: "gpu-789")
    monkeypatch.setattr(lifecycle, "_wait_for_endpoint", lambda _instance: (_ for _ in ()).throw(RuntimeError("startup failed")))
    monkeypatch.setattr(lifecycle, "_release_instance", lambda _instance: False)

    lifecycle._run(clip, {})

    assert lifecycle._read_json(lifecycle.runtime_path)["instance_id"] == "gpu-789"


def test_new_instance_command_clears_runtime_cache_but_not_models(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)
    captured = {}

    def provider_request(method, _path, payload=None):
        if method == "GET":
            return {"data": [{"id": lifecycle.product_id, "minRootFS": 10, "maxRootFS": 63}]}
        captured.update(payload)
        return {"instanceId": "gpu-1"}

    monkeypatch.setattr(lifecycle, "_provider_request", provider_request)

    assert lifecycle._create_instance() == "gpu-1"
    assert captured["rootfsSize"] == 60
    assert "find data/cache data/outputs" in captured["command"]
    assert "data/camera_profiles.json" in captured["command"]
    assert "NETCAST_X264_CRF=22" in captured["command"]
    assert "TENNISVISION_OUTPUT_MODE=event-overlay" in captured["command"]
    assert "models" not in captured["command"]


def test_rootfs_size_is_clamped_to_live_product_limit(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)
    monkeypatch.setenv("TENNISVISION_PPIO_ROOTFS_GB", "80")
    monkeypatch.setattr(
        lifecycle,
        "_provider_request",
        lambda _method, _path: {
            "data": [{"id": lifecycle.product_id, "minRootFS": 20, "maxRootFS": 48}]
        },
    )

    assert lifecycle._rootfs_size_for_product() == 48


def test_create_retries_new_provider_rootfs_limit_without_allocating_twice(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)
    requested_sizes = []

    def provider_request(method, _path, payload=None):
        if method == "GET":
            raise CloudLifecycleError("temporary product-list failure")
        requested_sizes.append(payload["rootfsSize"])
        if len(requested_sizes) == 1:
            raise CloudLifecycleError("PPIO 请求失败：rootfs size must not be more than 42 GB")
        return {"instanceId": "gpu-retried"}

    monkeypatch.setattr(lifecycle, "_provider_request", provider_request)

    assert lifecycle._create_instance() == "gpu-retried"
    assert requested_sizes == [60, 42]


def test_event_overlay_completion_does_not_require_annotated_video(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)
    downloads = []
    monkeypatch.setattr(
        lifecycle,
        "_download_path",
        lambda _remote, path, _destination, *, required: downloads.append((path, required)),
    )

    lifecycle._download_outputs("https://worker.example", annotated_required=False)

    assert ("/data/outputs/annotated_clip.mp4", False) in downloads


def test_endpoint_wait_tolerates_transient_provider_disconnect(tmp_path, monkeypatch):
    lifecycle = manager(tmp_path, monkeypatch)
    responses = iter(
        [
            CloudLifecycleError("temporary TLS EOF"),
            {
                "status": "running",
                "portMappings": [{"port": 8000, "endpoint": "https://worker.example"}],
            },
        ]
    )

    def provider_request(_method, _path):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(lifecycle, "_provider_request", provider_request)
    monkeypatch.setattr("netcast_tennisvision.cloud.ppio_lifecycle.time.sleep", lambda _delay: None)

    assert lifecycle._wait_for_endpoint("gpu-1") == "https://worker.example"


def test_cloud_status_write_retries_a_transient_windows_reader_lock(tmp_path, monkeypatch):
    destination = tmp_path / "status.json"
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "locked")
        real_replace(source, target)

    monkeypatch.setattr("netcast_tennisvision.cloud.ppio_lifecycle.os.replace", flaky_replace)
    monkeypatch.setattr("netcast_tennisvision.cloud.ppio_lifecycle.time.sleep", lambda _delay: None)

    PPIOJobManager._write_json(destination, {"state": "running"})

    assert attempts == 3
    assert PPIOJobManager._read_json(destination) == {"state": "running"}


def test_live_manager_returns_only_new_events_for_local_session(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    lifecycle = PPIOLiveJobManager(tmp_path)
    lifecycle._write_json(
        lifecycle.status_path,
        {
            "session_id": "local-live",
            "state": "running",
            "events": [{"id": 0}, {"id": 1}],
            "event_cursor": 2,
        },
    )

    assert lifecycle.snapshot("missing") is None
    assert lifecycle.snapshot("local-live", after_event=1)["events"] == [{"id": 1}]


def test_live_worker_receives_media_relay_from_runtime_config(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    monkeypatch.delenv("TENNISVISION_ZLM_HOST", raising=False)
    config = tmp_path / "data" / "live_lab_config.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        '{"zlm_host":"relay.example",'
        '"zlm_webrtc_origin":"https://media.example"}',
        encoding="utf-8",
    )

    lifecycle = PPIOLiveJobManager(tmp_path)

    assert {item["key"]: item["value"] for item in lifecycle._instance_envs()} == {
        "TENNISVISION_CLOUD_SHARED_SECRET": "relay-secret",
        "TENNISVISION_ZLM_HOST": "relay.example",
        "TENNISVISION_ZLM_WEBRTC_ORIGIN": "https://media.example",
    }


def test_live_upload_progress_keeps_browser_session_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    lifecycle = PPIOLiveJobManager(tmp_path)
    lifecycle._local_session_id = "live-session"
    lifecycle._write_json(
        lifecycle.status_path,
        {"session_id": "live-session", "events": [], "event_cursor": 0},
    )

    lifecycle._write_status("queued", 7, "正在上传（86%）")

    payload = lifecycle.snapshot("live-session")
    assert payload is not None
    assert payload["state"] == "preparing"
    assert payload["progress"] == 7
    assert payload["session_id"] == "live-session"


def test_new_live_upload_can_replace_an_active_session(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    monkeypatch.setenv("TENNISVISION_RESULT_RELAY_URL", "https://relay.example/results")
    monkeypatch.setenv("TENNISVISION_RESULT_RELAY_TOKEN", "result-secret")
    lifecycle = PPIOLiveJobManager(tmp_path)
    old_session = lifecycle._local_session_id = "old-session"
    lifecycle._stop_event.clear()
    old_thread = threading.Thread(target=lifecycle._stop_event.wait)
    lifecycle._thread = old_thread
    old_thread.start()
    clip = tmp_path / "next.mp4"
    clip.write_bytes(b"video")
    monkeypatch.setattr(lifecycle, "_run_live", lambda *_args: None)

    snapshot = lifecycle.start_live(
        clip,
        {"X-Filename": "next.mp4"},
        replace_active=True,
    )

    lifecycle._thread.join(timeout=2)
    assert not old_thread.is_alive()
    assert snapshot["session_id"] != old_session
    assert snapshot["filename"] == "next.mp4"
