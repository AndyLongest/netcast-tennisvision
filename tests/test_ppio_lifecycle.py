from pathlib import Path

import pytest

from netcast_tennisvision.cloud.ppio_lifecycle import CloudLifecycleError, PPIOJobManager


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
    monkeypatch.setattr(
        lifecycle,
        "_provider_request",
        lambda _method, _path, payload: captured.update(payload)
        or {"instanceId": "gpu-1"},
    )

    assert lifecycle._create_instance() == "gpu-1"
    assert captured["rootfsSize"] == 80
    assert "find data/cache data/outputs" in captured["command"]
    assert "data/camera_profiles.json" in captured["command"]
    assert "models" not in captured["command"]


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
