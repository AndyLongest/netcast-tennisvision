import json
from pathlib import Path

from netcast_tennisvision.cloud.ppio_lifecycle import PPIOLiveJobManager


def test_external_camera_uses_relay_without_demo_producer(tmp_path, monkeypatch):
    from netcast_tennisvision.cloud import ppio_lifecycle as module
    monkeypatch.setenv("PPIO_API_KEY", "test")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "test")
    live = PPIOLiveJobManager(tmp_path)
    monkeypatch.setattr(live, "_result_relay_config", lambda: ("https://relay.example", "test"))
    monkeypatch.setattr(live, "_create_instance", lambda: "gpu1")
    monkeypatch.setattr(live, "_wait_for_endpoint", lambda _: "https://worker.example")
    monkeypatch.setattr(live, "_wait_for_service", lambda _: None)
    monkeypatch.setattr(live, "_upload_camera_profiles", lambda _: None)
    monkeypatch.setattr(live, "_start_camera_simulator", lambda *a: (_ for _ in ()).throw(AssertionError("demo must not start")))
    released = []
    monkeypatch.setattr(live, "_release_instance", lambda i: released.append(i) or True)
    def request(url, method, path, **kwargs):
        if path == "/api/live-lab/start":
            body = json.loads(kwargs["body"])
            assert body["stream_name"] == "camera-test"
            assert body["source"] == "external_rtmp"
            return 202, {"session_id": "remote"}
        return 200, {"session_id": "remote", "state": "running"}
    monkeypatch.setattr(live, "_remote_request", request)
    monkeypatch.setattr(module, "fetch_result_snapshot", lambda *a: {
        "state": "complete", "result_relay": "ecs", "relay_updated_at": 1, "events": [], "speed": {"count": 1}})
    monkeypatch.setattr(module, "delete_result_snapshot", lambda *a: None)
    live._run_live(Path("nonexistent-demo.mp4"), {
        "X-External-Stream": "camera-test", "X-Filename": "camera",
        "X-Court-Corners": json.dumps([[0,1],[1,1],[1,0],[0,0]])}, "local")
    assert live.snapshot("local")["state"] == "complete"
    assert released == ["gpu1"]
