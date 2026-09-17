from __future__ import annotations

import os
import threading
from http.server import ThreadingHTTPServer

from netcast_tennisvision.streaming.result_relay import (
    ResultRelayPublisher,
    ResultRelayStore,
    _handler,
    delete_result_snapshot,
    fetch_result_snapshot,
)


def test_gpu_publishes_and_client_reads_only_through_result_relay(tmp_path) -> None:
    token = "test-token-that-is-long-enough"
    store = ResultRelayStore(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(store, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        publisher = ResultRelayPublisher(base_url, token, "session_12345678")
        publisher.publish(
            {
                "session_id": "session_12345678",
                "state": "running",
                "processed_frames": 48,
                "events": [{"id": 0, "frame": 42}],
            }
        )
        publisher.close(timeout=3)

        payload = fetch_result_snapshot(base_url, token, "session_12345678")
        assert payload is not None
        assert payload["result_relay"] == "ecs"
        assert payload["processed_frames"] == 48
        assert payload["events"] == [{"id": 0, "frame": 42}]
        assert delete_result_snapshot(base_url, token, "session_12345678") is True
        assert fetch_result_snapshot(base_url, token, "session_12345678") is None
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_result_relay_rejects_cross_session_snapshot(tmp_path) -> None:
    store = ResultRelayStore(tmp_path)
    try:
        store.put("session_12345678", {"session_id": "different_12345678"})
    except ValueError as exc:
        assert "mismatch" in str(exc)
    else:
        raise AssertionError("cross-session snapshot was accepted")


def test_result_relay_prunes_only_expired_snapshots(tmp_path) -> None:
    store = ResultRelayStore(tmp_path)
    store.put("session_old_1234", {"session_id": "session_old_1234"})
    store.put("session_new_1234", {"session_id": "session_new_1234"})
    os.utime(tmp_path / "session_old_1234.json", (800, 800))
    os.utime(tmp_path / "session_new_1234.json", (950, 950))

    assert store.prune_expired(100, now=1000) == 1
    assert store.get("session_old_1234") is None
    assert store.get("session_new_1234") is not None
