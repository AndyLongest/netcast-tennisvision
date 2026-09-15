"""Minimal HTTP adapter for PPIO sync Serverless when its async gateway is unavailable."""
from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from netcast_tennisvision.cloud.worker import handle_job


class BenchmarkHandler(BaseHTTPRequestHandler):
    server_version = "NetcastTennisVision/1.0"

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/health"}:
            self._json({"ok": True, "service": "netcast-tennisvision-benchmark"})
            return
        self._json({"ok": False, "error": "接口不存在"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/benchmark":
            self._json({"ok": False, "error": "接口不存在"}, HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= 64 * 1024:
            self._json({"ok": False, "error": "请求体无效"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象")
            result = handle_job({"input": payload})
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        print(f"http: {format % args}", flush=True)


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    ThreadingHTTPServer(("0.0.0.0", port), BenchmarkHandler).serve_forever()


if __name__ == "__main__":
    main()
