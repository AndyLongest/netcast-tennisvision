"""Authenticated ECS relay for live inference snapshots.

ZLMediaKit owns media bytes.  This companion service owns only small JSON snapshots so
the GPU worker and the venue client never exchange inference results directly.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{8,96}$")
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024


class ResultRelayStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, session_id: str) -> Path:
        if not SESSION_RE.fullmatch(session_id):
            raise ValueError("invalid session id")
        return self.root / f"{session_id}.json"

    def put(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if str(payload.get("session_id", "")) != session_id:
            raise ValueError("session id mismatch")
        stored = dict(payload)
        stored["result_relay"] = "ecs"
        stored["relay_updated_at"] = time.time()
        encoded = json.dumps(stored, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot too large")
        destination = self._path(session_id)
        temporary = destination.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_bytes(encoded)
            os.replace(temporary, destination)
        return stored

    def get(self, session_id: str) -> dict[str, Any] | None:
        path = self._path(session_id)
        with self._lock:
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        with self._lock:
            existed = path.is_file()
            path.unlink(missing_ok=True)
        return existed


def _handler(store: ResultRelayStore, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "NetcastResultRelay/1"

        def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            if token and hmac.compare_digest(supplied, expected):
                return True
            self._json({"error": "unauthorized"}, HTTPStatus.UNAUTHORIZED)
            return False

        def _session_id(self) -> str:
            prefix = "/v1/sessions/"
            path = self.path.split("?", 1)[0]
            marker = path.find(prefix)
            return path[marker + len(prefix):] if marker >= 0 else ""

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            if self.path.split("?", 1)[0].endswith("/healthz"):
                self._json({"ok": True, "service": "netcast-result-relay"})
                return
            if not self._authorized():
                return
            try:
                payload = store.get(self._session_id())
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if payload is None:
                self._json({"error": "session not found"}, HTTPStatus.NOT_FOUND)
            else:
                self._json(payload)

        def do_PUT(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._authorized():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_SNAPSHOT_BYTES:
                    raise ValueError("invalid snapshot size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("snapshot must be an object")
                stored = store.put(self._session_id(), payload)
                self._json({"accepted": True, "relay_updated_at": stored["relay_updated_at"]})
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler API
            if not self._authorized():
                return
            try:
                deleted = store.delete(self._session_id())
            except ValueError as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self._json({"deleted": deleted})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


class ResultRelayPublisher:
    """Coalesce frequent inference updates and publish the newest snapshot off-thread."""

    def __init__(self, base_url: str, token: str, session_id: str) -> None:
        self.url = f"{base_url.rstrip('/')}/v1/sessions/{session_id}"
        self.token = token
        self._condition = threading.Condition()
        self._latest: bytes | None = None
        self._revision = 0
        self._sent_revision = 0
        self._closing = False
        self._thread = threading.Thread(target=self._run, name="result-relay-publisher", daemon=True)
        self._thread.start()

    @classmethod
    def from_environment(cls, session_id: str) -> ResultRelayPublisher | None:
        base_url = os.environ.get("TENNISVISION_RESULT_RELAY_URL", "").strip()
        token = os.environ.get("TENNISVISION_RESULT_RELAY_TOKEN", "").strip()
        if not base_url or not token or not session_id:
            return None
        return cls(base_url, token, session_id)

    def publish(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > MAX_SNAPSHOT_BYTES:
            raise RuntimeError("result relay snapshot exceeded 2 MiB")
        with self._condition:
            self._latest = encoded
            self._revision += 1
            self._condition.notify_all()

    def close(self, timeout: float = 5.0) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._revision > self._sent_revision or self._closing,
                    timeout=0.5,
                )
                if self._closing and self._revision <= self._sent_revision:
                    return
                body, revision = self._latest, self._revision
            if body is None:
                continue
            request = Request(
                self.url,
                data=body,
                method="PUT",
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/json",
                    "Content-Length": str(len(body)),
                },
            )
            try:
                with urlopen(request, timeout=4) as response:
                    if response.status >= 300:
                        raise RuntimeError(f"result relay returned {response.status}")
                with self._condition:
                    self._sent_revision = max(self._sent_revision, revision)
            except (HTTPError, URLError, OSError, TimeoutError):
                time.sleep(0.25)


def fetch_result_snapshot(base_url: str, token: str, session_id: str) -> dict[str, Any] | None:
    request = Request(
        f"{base_url.rstrip('/')}/v1/sessions/{session_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == HTTPStatus.NOT_FOUND:
            return None
        raise
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Netcast ECS result relay")
    parser.add_argument("--host", default=os.environ.get("NETCAST_RESULT_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("NETCAST_RESULT_PORT", "18081")))
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("NETCAST_RESULT_DATA", "/data")))
    args = parser.parse_args()
    token = os.environ.get("NETCAST_RESULT_TOKEN", "").strip()
    if len(token) < 24:
        raise SystemExit("NETCAST_RESULT_TOKEN must contain at least 24 characters")
    server = ThreadingHTTPServer((args.host, args.port), _handler(ResultRelayStore(args.data), token))
    server.serve_forever()


if __name__ == "__main__":
    main()
