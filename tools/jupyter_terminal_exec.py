"""Run one shell command through an authenticated Jupyter terminal endpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import requests
import websocket
from websocket import WebSocketTimeoutException


def _endpoint_urls(address: str) -> tuple[str, str, str]:
    parsed = urlsplit(address)
    origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    query = f"?{parsed.query}" if parsed.query else ""
    websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
    return origin, websocket_scheme, query


def run_command(address: str, command: str, timeout_seconds: float) -> int:
    origin, websocket_scheme, query = _endpoint_urls(address)
    response = requests.post(f"{origin}/api/terminals{query}", json={}, timeout=30)
    response.raise_for_status()
    terminal_name = response.json()["name"]
    marker = f"__NETCAST_DONE_{secrets.token_hex(8)}__"
    socket_url = (
        f"{websocket_scheme}://{urlsplit(address).netloc}"
        f"/terminals/websocket/{terminal_name}{query}"
    )
    connection = websocket.create_connection(socket_url, timeout=30, origin=origin)
    started = time.monotonic()
    exit_code: int | None = None
    try:
        connection.send(json.dumps(["stdin", f"{command}; printf '\\n{marker}%s\\n' $?\r"]))
        while time.monotonic() - started < timeout_seconds:
            connection.settimeout(min(30, max(1, timeout_seconds - (time.monotonic() - started))))
            try:
                message = json.loads(connection.recv())
            except WebSocketTimeoutException:
                # Long-running commands may legitimately stay silent while the
                # pipeline writes progress to its own log file.
                continue
            if not isinstance(message, list) or len(message) < 2 or message[0] != "stdout":
                continue
            output = str(message[1])
            sys.stdout.write(output)
            sys.stdout.flush()
            matches = re.findall(rf"{re.escape(marker)}(\d+)", output)
            if matches:
                exit_code = int(matches[-1])
                break
    finally:
        connection.close()
        requests.delete(f"{origin}/api/terminals/{terminal_name}{query}", timeout=30)
    if exit_code is None:
        raise TimeoutError(f"Jupyter terminal command exceeded {timeout_seconds:.0f} seconds")
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    address = os.environ.get("NETCAST_JUPYTER_URL")
    if not address:
        raise SystemExit("NETCAST_JUPYTER_URL is required")
    raise SystemExit(run_command(address, args.command, args.timeout))


if __name__ == "__main__":
    main()
