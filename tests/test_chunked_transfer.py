import hashlib
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

from netcast_tennisvision.api import server
from netcast_tennisvision.cloud.ppio_lifecycle import PPIOJobManager


def start_upload_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(server, "CLOUD_SHARED_SECRET", "")
    monkeypatch.setattr(server, "CLOUD_API_URL", "")
    monkeypatch.setattr(server, "cloud_manager", None)
    monkeypatch.setattr(server, "cloud_live_manager", None)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def request_json(connection, method, path, payload, headers=None):
    body = json.dumps(payload).encode()
    request_headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    request_headers.update(headers or {})
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    return response.status, json.loads(response.read())


def test_chunk_upload_accepts_only_correct_size_and_checksum(tmp_path, monkeypatch):
    httpd, thread = start_upload_server(tmp_path, monkeypatch)
    connection = http.client.HTTPConnection(*httpd.server_address, timeout=5)
    try:
        status, initialized = request_json(
            connection,
            "POST",
            "/api/upload/init",
            {"filename": "match.mp4", "total_size": 5},
        )
        assert status == 201
        upload_id = initialized["upload_id"]
        chunk = b"video"
        connection.request(
            "PUT",
            f"/api/upload/chunk/{upload_id}/0",
            body=chunk,
            headers={
                "Content-Length": str(len(chunk)),
                "X-Chunk-SHA256": hashlib.sha256(chunk).hexdigest(),
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        assert (tmp_path / "uploads" / upload_id / "000000.part").read_bytes() == chunk

        connection.request(
            "PUT",
            f"/api/upload/chunk/{upload_id}/0",
            body=chunk,
            headers={"Content-Length": str(len(chunk)), "X-Chunk-SHA256": "0" * 64},
        )
        response = connection.getresponse()
        assert response.status == 422
        response.read()

        monkeypatch.setattr(
            server.Handler,
            "start_assembled_upload",
            lambda _self, path, _metadata: (202, {"accepted": path.read_bytes() == chunk}),
        )
        status, completed = request_json(
            connection,
            "POST",
            "/api/upload/complete",
            {"upload_id": upload_id},
        )
        assert status == 202
        assert completed["accepted"]
        assert not (tmp_path / "uploads" / upload_id).exists()
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_lifecycle_sends_multiple_independent_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    lifecycle = PPIOJobManager(tmp_path, tmp_path / "status.json")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"a" * 19)
    calls = []

    initialized_payload = {}

    def remote_request(_url, _method, path, **kwargs):
        if path == "/api/upload/init":
            initialized_payload.update(json.loads(kwargs["body"]))
            return 201, {"upload_id": "a" * 32, "chunk_size": 8}
        assert path == "/api/upload/complete"
        return 202, {"accepted": True}

    monkeypatch.setattr(lifecycle, "_remote_request", remote_request)
    monkeypatch.setattr(
        lifecycle,
        "_upload_part",
        lambda _url, upload_id, index, body: calls.append((upload_id, index, body)),
    )

    completed = lifecycle._upload_video(
        "https://cloud.example",
        clip,
        {"X-Filename": "clip.mp4"},
        purpose="live-lab",
    )

    calls.sort(key=lambda item: item[1])
    assert [len(body) for _, _, body in calls] == [8, 8, 3]
    assert [index for _, index, _ in calls] == [0, 1, 2]
    assert initialized_payload["purpose"] == "live-lab"
    assert completed == {"accepted": True}


def test_live_lab_chunk_completion_stores_source_without_starting_offline_job(
    tmp_path, monkeypatch
):
    httpd, thread = start_upload_server(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "DATA", tmp_path / "data")
    connection = http.client.HTTPConnection(*httpd.server_address, timeout=5)
    try:
        status, initialized = request_json(
            connection,
            "POST",
            "/api/upload/init",
            {"filename": "practice.mp4", "total_size": 5, "purpose": "live-lab"},
        )
        assert status == 201
        upload_id = initialized["upload_id"]
        chunk = b"video"
        connection.request(
            "PUT",
            f"/api/upload/chunk/{upload_id}/0",
            body=chunk,
            headers={
                "Content-Length": str(len(chunk)),
                "X-Chunk-SHA256": hashlib.sha256(chunk).hexdigest(),
            },
        )
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        monkeypatch.setattr(
            server.Handler,
            "start_assembled_upload",
            lambda *_args: (_ for _ in ()).throw(AssertionError("offline path called")),
        )

        status, completed = request_json(
            connection, "POST", "/api/upload/complete", {"upload_id": upload_id}
        )

        assert status == 201
        assert completed["source"] == "uploaded"
        assert (server.DATA / "live_lab_source.mp4").read_bytes() == chunk
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_ranged_video_download_reassembles_all_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    lifecycle = PPIOJobManager(tmp_path, tmp_path / "status.json")
    source = b"annotated-video"

    def fetch_range(_url, _path, start, end):
        returned_end = min(end, len(source) - 1)
        body = source[start : returned_end + 1]
        return 206, f"bytes {start}-{returned_end}/{len(source)}", body

    monkeypatch.setattr(lifecycle, "_fetch_range", fetch_range)
    destination = tmp_path / "annotated.mp4"

    lifecycle._download_video_ranged(
        "https://cloud.example", "/data/outputs/annotated.mp4", destination, required=True
    )

    assert destination.read_bytes() == source


def test_parallel_ranged_download_preserves_file_order(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    monkeypatch.setenv("TENNISVISION_TRANSFER_WORKERS", "3")
    lifecycle = PPIOJobManager(tmp_path, tmp_path / "status.json")
    source = b"parts-can-complete-out-of-order"
    monkeypatch.setattr(
        "netcast_tennisvision.cloud.ppio_lifecycle.TRANSFER_CHUNK_SIZE", 5
    )

    def fetch_range(_url, _path, start, end):
        returned_end = min(end, len(source) - 1)
        body = source[start : returned_end + 1]
        return 206, f"bytes {start}-{returned_end}/{len(source)}", body

    monkeypatch.setattr(lifecycle, "_fetch_range", fetch_range)
    destination = tmp_path / "parallel.mp4"

    lifecycle._download_video_ranged(
        "https://cloud.example", "/data/outputs/parallel.mp4", destination, required=True
    )

    assert destination.read_bytes() == source


def test_camera_profiles_are_uploaded_only_when_present(tmp_path, monkeypatch):
    monkeypatch.setenv("PPIO_API_KEY", "provider-secret")
    monkeypatch.setenv("TENNISVISION_CLOUD_TOKEN", "relay-secret")
    lifecycle = PPIOJobManager(tmp_path, tmp_path / "status.json")
    profile = tmp_path / "data" / "camera_profiles.json"
    profile.parent.mkdir(parents=True)
    profile.write_text('{"version":1,"profiles":[]}', encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        lifecycle,
        "_remote_request",
        lambda url, method, path, **kwargs: calls.append((url, method, path, kwargs))
        or (200, {"accepted": True}),
    )

    lifecycle._upload_camera_profiles("https://cloud.example")

    assert calls[0][2] == "/api/camera-profiles"
    assert calls[0][3]["body"] == profile.read_bytes()


def test_camera_profile_endpoint_accepts_bounded_profile_store(tmp_path, monkeypatch):
    httpd, thread = start_upload_server(tmp_path, monkeypatch)
    monkeypatch.setattr(server, "CAMERA_PROFILES", tmp_path / "camera_profiles.json")
    connection = http.client.HTTPConnection(*httpd.server_address, timeout=5)
    try:
        status, response = request_json(
            connection,
            "POST",
            "/api/camera-profiles",
            {"version": 1, "profiles": []},
        )
        assert status == 200
        assert response == {"accepted": True, "profiles": 0}
        assert json.loads(server.CAMERA_PROFILES.read_text(encoding="utf-8")) == {
            "version": 1,
            "profiles": [],
        }
    finally:
        connection.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)
