"""Local Netcast TennisVision UI server and single-job analysis API."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from netcast_tennisvision.cloud.ppio_lifecycle import (
    CloudLifecycleError,
    PPIOJobManager,
    PPIOLiveJobManager,
)
from netcast_tennisvision.paths import REPOSITORY_ROOT
from netcast_tennisvision.vision.display_correction import parse_display_correction

ROOT = REPOSITORY_ROOT
DATA = ROOT / "data"
STATUS = DATA / "job_status.json"
LOG = DATA / "pipeline.log"
VIDEO_IDENTITIES = DATA / "video_identities.json"
CAMERA_PROFILES = DATA / "camera_profiles.json"
CURRENT_JOB = DATA / "current_job.json"
CALIBRATION_REQUEST = DATA / "court_calibration_request.json"
CALIBRATION_RESPONSE = DATA / "court_calibration_response.json"
UPLOADS = DATA / "uploads"
UPLOAD_CHUNK_SIZE = 8 * 1024**2
MAX_VIDEO_SIZE = 4 * 1024**3
job_lock = threading.Lock()
job_process: subprocess.Popen[bytes] | None = None
ACTIVE_STATES = {"queued", "running", "needs_court_calibration", "report_ready"}
CLOUD_API_URL = os.environ.get("TENNISVISION_CLOUD_URL", "").strip().rstrip("/")
CLOUD_API_TOKEN = os.environ.get("TENNISVISION_CLOUD_TOKEN", "").strip()
CLOUD_TIMEOUT_SECONDS = float(os.environ.get("TENNISVISION_CLOUD_TIMEOUT", "3600"))
CLOUD_SHARED_SECRET = os.environ.get("TENNISVISION_CLOUD_SHARED_SECRET", "").strip()
CLOUD_PROVIDER = os.environ.get("TENNISVISION_CLOUD_PROVIDER", "").strip().lower()
cloud_manager = PPIOJobManager(ROOT, STATUS) if CLOUD_PROVIDER == "ppio" else None
cloud_live_manager = PPIOLiveJobManager(ROOT) if CLOUD_PROVIDER == "ppio" else None
_live_experiment_manager = None


def live_experiment_manager():
    """Import the GPU experiment lazily so the ordinary upload UI stays lightweight."""
    global _live_experiment_manager
    if _live_experiment_manager is None:
        from netcast_tennisvision.streaming.live_experiment import live_experiments

        _live_experiment_manager = live_experiments
    return _live_experiment_manager


def read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def status_payload() -> dict[str, object]:
    status = read_json(STATUS) or {"state": "idle", "progress": 0, "stage": "等待视频"}
    # Pipeline progress updates replace job_status.json. Keep durable job identity in a
    # separate file and merge it into every response so a refreshed browser can reattach.
    for key, value in read_json(CURRENT_JOB).items():
        status.setdefault(key, value)
    default_target = (
        "cloud-on-demand"
        if cloud_manager
        else os.environ.get("TENNISVISION_EXECUTION_TARGET", "local")
    )
    status.setdefault("execution_target", default_target)
    return status


def video_fingerprint(path: Path, sample_size: int = 256 * 1024) -> str:
    """Fast content identity shared with the browser: size + first/last sample."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        first = stream.read(sample_size)
        stream.seek(max(0, size - sample_size))
        last = stream.read(sample_size)
    digest = hashlib.sha256(size.to_bytes(8, "big") + first + last).hexdigest()
    return f"sha256-sample:{digest}"


def resumable_job(request_fingerprint: str | None) -> dict[str, object] | None:
    status = status_payload()
    if status.get("state") not in ACTIVE_STATES:
        return None
    current_fingerprint = status.get("video_fingerprint")
    if request_fingerprint and current_fingerprint == request_fingerprint:
        return status
    return None


def ffmpeg_directory() -> Path | None:
    executable = shutil.which("ffmpeg")
    if executable:
        directory = Path(executable).resolve().parent
        if (directory / "ffprobe.exe").exists() or (directory / "ffprobe").exists():
            return directory
    packages = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    candidates = sorted(packages.glob("Gyan.FFmpeg.Shared_*/ffmpeg-*/bin/ffmpeg.exe"), reverse=True)
    return candidates[0].parent if candidates else None


def same_file(first: Path, second: Path) -> bool:
    if not second.exists() or first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        return (
            hashlib.file_digest(left, "sha256").digest()
            == hashlib.file_digest(right, "sha256").digest()
        )


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_video_mtime(video: Path) -> int:
    """Give identical video bytes one stable cache identity across later uploads."""
    try:
        identities = json.loads(VIDEO_IDENTITIES.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        identities = {}
    digest = file_sha256(video)
    if digest in identities:
        return int(identities[digest])

    # Preserve the already-validated bundled sample's original cache identity.
    canonical = int(video.stat().st_mtime)
    for candidate in (ROOT / "assets" / "demo" / "demo.mp4",):
        if candidate.exists() and candidate.stat().st_size == video.stat().st_size:
            if file_sha256(candidate) == digest:
                canonical = int(candidate.stat().st_mtime)
                break
    identities[digest] = canonical
    VIDEO_IDENTITIES.write_text(json.dumps(identities, indent=2), encoding="utf-8")
    return canonical


def probe_native_fps(video: Path) -> float:
    """Read the captured frame rate without transcoding or inventing frames."""
    directory = ffmpeg_directory()
    executable = (
        (directory / ("ffprobe.exe" if os.name == "nt" else "ffprobe")) if directory else None
    )
    if executable is None or not executable.exists():
        raise RuntimeError("本机未找到 ffprobe，无法检查视频帧率")
    completed = subprocess.run(
        [
            str(executable),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    rate = payload["streams"][0]["avg_frame_rate"]
    numerator, denominator = (float(value) for value in rate.split("/", 1))
    if denominator == 0:
        raise ValueError("视频没有有效帧率")
    return numerator / denominator


def supports_native_fps(fps: float) -> bool:
    """Accept every valid native frame rate without resampling."""
    return math.isfinite(fps) and fps > 0.0


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if cloud_manager:
            self.send_header("X-Netcast-Execution-Target", "cloud-on-demand")
        self.send_header(
            "Access-Control-Allow-Origin",
            "null" if self.headers.get("Origin") == "null" else "http://127.0.0.1:4173",
        )
        self.end_headers()
        self.wfile.write(body)

    def send_jpeg(self, body: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        """Tell browsers that generated match videos support seeking."""
        if urlparse(self.path).path.lower().endswith(".mp4"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def cloud_request_authorized(self) -> bool:
        """Protect the public GPU port with a relay-only bearer secret."""
        if not CLOUD_SHARED_SECRET:
            return True
        expected = f"Bearer {CLOUD_SHARED_SECRET}"
        provided = self.headers.get("Authorization", "")
        if hmac.compare_digest(provided, expected):
            return True
        self.send_json({"error": "云端访问未授权"}, HTTPStatus.UNAUTHORIZED)
        return False

    def proxy_cloud_request(self, method: str) -> None:
        """Stream API and generated artifacts through the trusted local relay."""
        parsed = urlparse(CLOUD_API_URL)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            self.send_json(
                {"error": "云端分析地址配置无效", "code": "cloud_configuration_error"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        connection_class = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        request = urlparse(self.path)
        target = f"{parsed.path.rstrip('/')}{request.path}"
        if request.query:
            target += f"?{request.query}"
        connection = connection_class(parsed.hostname, port, timeout=CLOUD_TIMEOUT_SECONDS)
        response_started = False
        try:
            connection.putrequest(method, target)
            for name in (
                "Content-Type",
                "Content-Length",
                "Range",
                "X-Filename",
                "X-Video-Fingerprint",
                "X-Display-Correction",
                "X-Display-Corners",
            ):
                value = self.headers.get(name)
                if value:
                    connection.putheader(name, value)
            if CLOUD_API_TOKEN:
                connection.putheader("Authorization", f"Bearer {CLOUD_API_TOKEN}")
            connection.endheaders()
            if method == "POST":
                remaining = int(self.headers.get("Content-Length", "0"))
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("浏览器上传提前中断")
                    connection.send(chunk)
                    remaining -= len(chunk)
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for name in (
                "Content-Type",
                "Content-Length",
                "Content-Range",
                "Accept-Ranges",
                "Last-Modified",
                "Cache-Control",
            ):
                value = response.getheader(name)
                if value:
                    self.send_header(name, value)
            self.send_header(
                "Access-Control-Allow-Origin",
                "null" if self.headers.get("Origin") == "null" else "http://127.0.0.1:4173",
            )
            self.send_header("X-Netcast-Execution-Target", "cloud")
            self.end_headers()
            response_started = True
            while chunk := response.read(1024 * 1024):
                self.wfile.write(chunk)
        except (OSError, TimeoutError, http.client.HTTPException, ValueError) as exc:
            if not response_started and not self.wfile.closed:
                try:
                    self.send_json(
                        {"error": f"无法连接云端分析服务：{exc}", "code": "cloud_unavailable"},
                        HTTPStatus.BAD_GATEWAY,
                    )
                except (BrokenPipeError, ConnectionResetError):
                    pass
        finally:
            connection.close()

    def send_video_range(self, path: Path, range_header: str) -> None:
        """Serve one HTTP byte range so large analysis videos start immediately."""
        size = path.stat().st_size
        try:
            unit, requested = range_header.strip().split("=", 1)
            start_text, end_text = requested.split("-", 1)
            if unit.lower() != "bytes" or "," in requested:
                raise ValueError
            if start_text:
                start = int(start_text)
                end = min(int(end_text), size - 1) if end_text else size - 1
            else:
                suffix = int(end_text)
                start, end = max(0, size - suffix), size - 1
            if start < 0 or start >= size or end < start:
                raise ValueError
        except (TypeError, ValueError):
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        try:
            with path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def do_OPTIONS(self) -> None:
        request_path = urlparse(self.path).path
        if request_path not in {
            "/api/status",
            "/api/analyze",
            "/api/court-calibration",
            "/api/upload/init",
            "/api/upload/complete",
        } and not request_path.startswith("/api/upload/chunk/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header(
            "Access-Control-Allow-Origin",
            "null" if self.headers.get("Origin") == "null" else "http://127.0.0.1:4173",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Filename, X-Video-Fingerprint, X-Display-Correction, X-Display-Corners",
        )
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_PUT(self) -> None:
        """Accept one bounded, checksummed part through the public HTTP mapping."""
        request_path = urlparse(self.path).path
        if not self.cloud_request_authorized():
            return
        prefix = "/api/upload/chunk/"
        if not request_path.startswith(prefix):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        pieces = request_path[len(prefix) :].split("/")
        if len(pieces) != 2 or not self.valid_upload_id(pieces[0]):
            self.send_json({"error": "上传分片地址无效"}, HTTPStatus.BAD_REQUEST)
            return
        upload_id = pieces[0]
        try:
            index = int(pieces[1])
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            index, length = -1, 0
        session = UPLOADS / upload_id
        metadata = read_json(session / "metadata.json")
        total_size = int(metadata.get("total_size", 0))
        chunk_size = int(metadata.get("chunk_size", UPLOAD_CHUNK_SIZE))
        chunk_count = math.ceil(total_size / chunk_size) if total_size else 0
        expected = min(chunk_size, total_size - index * chunk_size) if 0 <= index < chunk_count else 0
        if not metadata or expected <= 0 or length != expected:
            self.send_json({"error": "上传分片大小或编号无效"}, HTTPStatus.BAD_REQUEST)
            return
        body = self.rfile.read(length)
        if len(body) != length:
            self.send_json({"error": "上传分片提前中断"}, HTTPStatus.BAD_REQUEST)
            return
        expected_hash = self.headers.get("X-Chunk-SHA256", "").lower()
        actual_hash = hashlib.sha256(body).hexdigest()
        if not expected_hash or not hmac.compare_digest(actual_hash, expected_hash):
            self.send_json({"error": "上传分片校验失败"}, HTTPStatus.UNPROCESSABLE_ENTITY)
            return
        destination = session / f"{index:06d}.part"
        temporary = destination.with_suffix(".part.uploading")
        temporary.write_bytes(body)
        os.replace(temporary, destination)
        self.send_json({"accepted": True, "index": index, "sha256": actual_hash})

    def do_GET(self) -> None:
        request = urlparse(self.path)
        request_path = request.path
        if not self.cloud_request_authorized():
            return
        if request_path in {"/api/live-lab/status", "/api/live-lab/frame"}:
            query = parse_qs(request.query)
            session_id = query.get("session_id", [""])[0]
            cloud_payload = (
                cloud_live_manager.snapshot(session_id, after_event=0)
                if cloud_live_manager is not None
                else None
            )
            session = None if cloud_payload is not None else live_experiment_manager().get(session_id)
            if cloud_payload is not None:
                if request_path.endswith("/frame"):
                    self.send_error(HTTPStatus.NO_CONTENT)
                    return
                try:
                    after_event = max(0, int(query.get("after_event", ["0"])[0]))
                except ValueError:
                    after_event = 0
                events = cloud_payload.get("events", [])
                cloud_payload["events"] = events[after_event:] if isinstance(events, list) else []
                self.send_json(cloud_payload)
                return
            if session is None:
                self.send_json({"error": "实时实验会话不存在或已过期"}, HTTPStatus.NOT_FOUND)
                return
            if request_path.endswith("/frame"):
                jpeg = session.jpeg()
                if jpeg is None:
                    self.send_error(HTTPStatus.NO_CONTENT)
                else:
                    self.send_jpeg(jpeg)
                return
            try:
                after_event = max(0, int(query.get("after_event", ["0"])[0]))
            except ValueError:
                after_event = 0
            self.send_json(session.snapshot(after_event))
            return
        if CLOUD_API_URL and (
            request_path.startswith("/api/") or request_path.startswith("/data/")
        ):
            self.proxy_cloud_request("GET")
            return
        if request_path == "/api/status":
            self.send_json(status_payload())
            return
        range_header = self.headers.get("Range")
        if range_header and urlparse(self.path).path.lower().endswith(".mp4"):
            path = Path(self.translate_path(urlparse(self.path).path))
            if path.is_file():
                self.send_video_range(path, range_header)
                return
        super().do_GET()

    def do_POST(self) -> None:
        global job_process
        request_path = urlparse(self.path).path
        if not self.cloud_request_authorized():
            return
        if request_path == "/api/live-lab/start":
            try:
                if cloud_live_manager is not None:
                    if cloud_manager is not None and cloud_manager.active:
                        raise CloudLifecycleError("普通视频分析正在运行，请完成后再启动实时实验")
                    snapshot = cloud_live_manager.start_live(
                        ROOT / "assets" / "demo" / "demo.mp4",
                        {"X-Filename": "demo.mp4"},
                    )
                    self.send_json(snapshot, HTTPStatus.ACCEPTED)
                else:
                    source = ROOT / "assets" / "demo" / "demo.mp4"
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                    except ValueError:
                        length = 0
                    if length:
                        payload = self.read_bounded_json(maximum=4096)
                        if payload.get("source") == "external_rtmp":
                            session = live_experiment_manager().start_external_stream(
                                str(payload.get("stream_name", "")),
                                fps_hint=float(payload.get("fps", 30.0)),
                                source_name=Path(str(payload.get("filename", "camera"))).name,
                            )
                            self.send_json(session.snapshot(), HTTPStatus.ACCEPTED)
                            return
                        if payload.get("source") == "uploaded":
                            candidates = sorted(DATA.glob("live_lab_source.*"))
                            if not candidates:
                                raise RuntimeError("云端没有收到实时实验素材")
                            source = candidates[-1]
                    session = live_experiment_manager().start_source(source)
                    self.send_json(session.snapshot(), HTTPStatus.ACCEPTED)
            except (CloudLifecycleError, OSError, RuntimeError, ValueError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if request_path == "/api/live-lab/upload":
            self.start_live_lab_upload()
            return
        if request_path == "/api/live-lab/webrtc-offer":
            session_id = parse_qs(urlparse(self.path).query).get("session_id", [""])[0]
            session = live_experiment_manager().get(session_id)
            if session is None:
                self.send_json({"error": "实时实验会话不存在或已过期"}, HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 256 * 1024:
                    raise ValueError("WebRTC SDP 内容无效")
                offer = self.rfile.read(length).decode("utf-8")
                self.send_json(session.exchange_webrtc_offer(offer))
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_GATEWAY)
            return
        if request_path == "/api/live-lab/stop":
            try:
                payload = self.read_bounded_json(maximum=4096)
                session_id = str(payload.get("session_id", ""))
                stopped = (
                    cloud_live_manager.stop_live(session_id)
                    if cloud_live_manager is not None
                    and cloud_live_manager.snapshot(session_id) is not None
                    else live_experiment_manager().stop(session_id)
                )
                self.send_json({"stopped": stopped}, HTTPStatus.OK if stopped else HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if CLOUD_API_URL and request_path in {"/api/analyze", "/api/court-calibration"}:
            self.proxy_cloud_request("POST")
            return
        if request_path == "/api/upload/init":
            self.initialize_chunked_upload()
            return
        if request_path == "/api/upload/complete":
            self.complete_chunked_upload()
            return
        if request_path == "/api/court-calibration":
            if cloud_manager:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 64 * 1024:
                        raise CloudLifecycleError("球场校准数据无效")
                    response = cloud_manager.submit_calibration(self.rfile.read(length))
                    self.send_json(response)
                except (ValueError, CloudLifecycleError) as exc:
                    self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.save_court_calibration()
            return
        if request_path == "/api/camera-profiles":
            self.save_camera_profiles()
            return
        if request_path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with job_lock:
            requested_fingerprint = self.headers.get("X-Video-Fingerprint")
            current_status = status_payload()
            process_active = job_process is not None and job_process.poll() is None
            status_active = current_status.get("state") in ACTIVE_STATES
            if process_active or status_active:
                resumed = resumable_job(requested_fingerprint)
                if resumed is not None:
                    self.send_json(
                        {
                            "accepted": True,
                            "resumed": True,
                            "job_id": resumed.get("job_id"),
                            "filename": resumed.get("filename"),
                            "fps": resumed.get("fps"),
                            "workload_factor": resumed.get("workload_factor", 1),
                            "display_correction": resumed.get("display_correction"),
                        },
                        HTTPStatus.ACCEPTED,
                    )
                    return
                self.send_json(
                    {
                        "error": "另一段视频正在分析，请等待当前分析完成",
                        "code": "analysis_in_progress",
                        "filename": current_status.get("filename"),
                        "progress": current_status.get("progress", 0),
                    },
                    HTTPStatus.CONFLICT,
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_VIDEO_SIZE:
                self.send_json({"error": "视频为空或超过 4GB"}, HTTPStatus.BAD_REQUEST)
                return
            filename = Path(unquote(self.headers.get("X-Filename", "clip.mp4"))).name
            suffix = Path(filename).suffix.lower()
            if suffix not in {".mp4", ".mov", ".webm", ".mkv"}:
                self.send_json({"error": "不支持的视频格式"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                display_correction = parse_display_correction(
                    self.headers.get("X-Display-Correction"),
                    self.headers.get("X-Display-Corners"),
                )
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            DATA.mkdir(parents=True, exist_ok=True)
            temporary = DATA / "clip.uploading"
            remaining = length
            with temporary.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("上传提前中断")
                    output.write(chunk)
                    remaining -= len(chunk)
            try:
                fps = probe_native_fps(temporary)
            except (
                OSError,
                ValueError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
                subprocess.SubprocessError,
                RuntimeError,
            ) as exc:
                temporary.unlink(missing_ok=True)
                self.send_json({"error": f"无法读取视频规格：{exc}"}, HTTPStatus.BAD_REQUEST)
                return
            # The pipeline derives every temporal window from the measured FPS. Accept
            # any valid native rate and never silently drop or invent frames.
            if not supports_native_fps(fps):
                temporary.unlink(missing_ok=True)
                self.send_json(
                    {"error": f"视频没有有效帧率（读到{fps!r}fps），无法建立时间轴。"},
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                )
                return
            clip = DATA / "clip.mp4"
            if same_file(temporary, clip):
                temporary.unlink()
            else:
                os.replace(temporary, clip)
            # The notebook's frozen cache key includes mtime. Restore a stable timestamp
            # for identical bytes so revisiting a video cannot silently retrain BallNet.
            cache_mtime = stable_video_mtime(clip)
            os.utime(clip, (cache_mtime, cache_mtime))
            fingerprint = video_fingerprint(clip)
            job_id = uuid.uuid4().hex
            workload_factor = round(fps / 30.0, 2)
            job_metadata: dict[str, object] = {
                "job_id": job_id,
                "filename": filename,
                "file_size": length,
                "video_fingerprint": fingerprint,
                "fps": round(fps, 3),
                "workload_factor": workload_factor,
                "display_correction": display_correction,
                "started_at": int(time.time()),
            }
            # Publish a self-contained queued snapshot first. Until CURRENT_JOB is replaced,
            # readers still see the new identity because status fields win during merging.
            write_json_atomic(
                STATUS,
                {
                    "state": "queued",
                    "progress": 1,
                    "stage": "视频已接收，准备分析",
                    **job_metadata,
                },
            )
            write_json_atomic(CURRENT_JOB, job_metadata)
            if cloud_manager:
                try:
                    cloud_manager.start(
                        clip,
                        {
                            "X-Filename": filename,
                            "X-Video-Fingerprint": fingerprint,
                            "X-Display-Correction": self.headers.get("X-Display-Correction", ""),
                            "X-Display-Corners": self.headers.get("X-Display-Corners", ""),
                        },
                    )
                except CloudLifecycleError as exc:
                    write_json_atomic(
                        STATUS,
                        {
                            "state": "error",
                            "progress": 0,
                            "stage": "云端分析未启动",
                            "error": str(exc),
                            "execution_target": "cloud-on-demand",
                        },
                    )
                    self.send_json(
                        {"error": str(exc), "code": "cloud_configuration_error"},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
                self.send_json(
                    {
                        "accepted": True,
                        "resumed": False,
                        "job_id": job_id,
                        "filename": filename,
                        "fps": round(fps, 3),
                        "high_fps": fps >= 48.0,
                        "workload_factor": workload_factor,
                        "video_fingerprint": fingerprint,
                        "display_correction": display_correction,
                        "execution_target": "cloud-on-demand",
                    },
                    HTTPStatus.ACCEPTED,
                )
                return
            environment = os.environ.copy()
            ffmpeg_dir = ffmpeg_directory()
            if ffmpeg_dir:
                environment["PATH"] = str(ffmpeg_dir) + os.pathsep + environment.get("PATH", "")
            log_handle = LOG.open("wb")
            job_process = subprocess.Popen(
                [sys.executable, "-m", "netcast_tennisvision.pipeline.runner"],
                cwd=ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            job_metadata["pid"] = job_process.pid
            write_json_atomic(CURRENT_JOB, job_metadata)
            self.send_json(
                {
                    "accepted": True,
                    "resumed": False,
                    "job_id": job_id,
                    "filename": filename,
                    "fps": round(fps, 3),
                    "high_fps": fps >= 48.0,
                    "workload_factor": workload_factor,
                    "video_fingerprint": fingerprint,
                    "display_correction": display_correction,
                },
                HTTPStatus.ACCEPTED,
            )

    @staticmethod
    def valid_upload_id(value: str) -> bool:
        return len(value) == 32 and all(character in "0123456789abcdef" for character in value)

    def read_bounded_json(self, maximum: int = 64 * 1024) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= maximum:
            raise ValueError("请求数据大小无效")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求数据格式无效") from exc
        if not isinstance(payload, dict):
            raise ValueError("请求数据格式无效")
        return payload

    def initialize_chunked_upload(self) -> None:
        """Create a resumable upload session with fixed-size independent parts."""
        try:
            payload = self.read_bounded_json()
            total_size = int(payload.get("total_size", 0))
            filename = Path(str(payload.get("filename", "clip.mp4"))).name
            if not 0 < total_size <= MAX_VIDEO_SIZE:
                raise ValueError("视频为空或超过 4GB")
            if Path(filename).suffix.lower() not in {".mp4", ".mov", ".webm", ".mkv"}:
                raise ValueError("不支持的视频格式")
        except (TypeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        upload_id = uuid.uuid4().hex
        session = UPLOADS / upload_id
        session.mkdir(parents=True, exist_ok=False)
        metadata: dict[str, object] = {
            "upload_id": upload_id,
            "filename": filename,
            "total_size": total_size,
            "chunk_size": UPLOAD_CHUNK_SIZE,
            "video_fingerprint": str(payload.get("video_fingerprint", "")),
            "display_correction": str(payload.get("display_correction", "")),
            "display_corners": str(payload.get("display_corners", "")),
            "purpose": str(payload.get("purpose", "analysis")),
            "created_at": int(time.time()),
        }
        write_json_atomic(session / "metadata.json", metadata)
        self.send_json(
            {
                "upload_id": upload_id,
                "chunk_size": UPLOAD_CHUNK_SIZE,
                "chunk_count": math.ceil(total_size / UPLOAD_CHUNK_SIZE),
            },
            HTTPStatus.CREATED,
        )

    def complete_chunked_upload(self) -> None:
        """Join verified parts locally, then enter the unchanged analysis endpoint."""
        try:
            payload = self.read_bounded_json()
            upload_id = str(payload.get("upload_id", ""))
            if not self.valid_upload_id(upload_id):
                raise ValueError("上传会话无效")
            session = UPLOADS / upload_id
            metadata = read_json(session / "metadata.json")
            total_size = int(metadata.get("total_size", 0))
            chunk_size = int(metadata.get("chunk_size", 0))
            chunk_count = math.ceil(total_size / chunk_size) if chunk_size else 0
            parts = [session / f"{index:06d}.part" for index in range(chunk_count)]
            if not metadata or not parts or any(not part.is_file() for part in parts):
                raise ValueError("视频分片尚未全部上传")
            if sum(part.stat().st_size for part in parts) != total_size:
                raise ValueError("视频分片总大小不一致")
            assembled = session / "assembled.video"
            with assembled.open("wb") as output:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            if metadata.get("purpose") == "live-lab":
                suffix = Path(str(metadata.get("filename", "clip.mp4"))).suffix.lower()
                DATA.mkdir(parents=True, exist_ok=True)
                for old_source in DATA.glob("live_lab_source.*"):
                    old_source.unlink(missing_ok=True)
                destination = DATA / f"live_lab_source{suffix}"
                os.replace(assembled, destination)
                response_status, response_payload = HTTPStatus.CREATED, {
                    "accepted": True,
                    "source": "uploaded",
                    "filename": str(metadata.get("filename", destination.name)),
                }
            else:
                response_status, response_payload = self.start_assembled_upload(assembled, metadata)
            if response_status < 300:
                shutil.rmtree(session)
            self.send_json(response_payload, HTTPStatus(response_status))
        except (OSError, TypeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def start_live_lab_upload(self) -> None:
        """Store one test clip locally, then run it through a temporary L40S live worker."""
        if cloud_live_manager is None:
            self.send_json(
                {"error": "上传素材的 L40S 实时实验尚未配置"},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        if cloud_live_manager.active or (cloud_manager is not None and cloud_manager.active):
            self.send_json({"error": "已有云端任务正在运行"}, HTTPStatus.CONFLICT)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        filename = Path(unquote(self.headers.get("X-Filename", "clip.mp4"))).name
        suffix = Path(filename).suffix.lower()
        if not 0 < length <= MAX_VIDEO_SIZE:
            self.send_json({"error": "视频为空或超过 4GB"}, HTTPStatus.BAD_REQUEST)
            return
        if suffix not in {".mp4", ".mov", ".webm", ".mkv"}:
            self.send_json({"error": "不支持的视频格式"}, HTTPStatus.BAD_REQUEST)
            return
        DATA.mkdir(parents=True, exist_ok=True)
        temporary = DATA / "live_lab_source.uploading"
        try:
            remaining = length
            with temporary.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError("上传提前中断")
                    output.write(chunk)
                    remaining -= len(chunk)
            fps = probe_native_fps(temporary)
            if not supports_native_fps(fps):
                raise ValueError("视频没有有效帧率")
            for old_source in DATA.glob("live_lab_input.*"):
                old_source.unlink(missing_ok=True)
            clip = DATA / f"live_lab_input{suffix}"
            os.replace(temporary, clip)
            snapshot = cloud_live_manager.start_live(
                clip,
                {
                    "X-Filename": filename,
                    "X-Video-Fingerprint": video_fingerprint(clip),
                    "X-Fps": f"{fps:.6f}",
                },
            )
            snapshot["fps"] = round(fps, 3)
            self.send_json(snapshot, HTTPStatus.ACCEPTED)
        except (CloudLifecycleError, ConnectionError, OSError, RuntimeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def start_assembled_upload(
        self, assembled: Path, metadata: dict[str, object]
    ) -> tuple[int, dict[str, object]]:
        """Loop a reassembled file into the stable direct-upload API inside the container."""
        port = int(os.environ.get("TENNISVISION_PORT", "4173"))
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=CLOUD_TIMEOUT_SECONDS)
        try:
            connection.putrequest("POST", "/api/analyze")
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("Content-Length", str(assembled.stat().st_size))
            connection.putheader("X-Filename", str(metadata["filename"]))
            for source, target in (
                ("video_fingerprint", "X-Video-Fingerprint"),
                ("display_correction", "X-Display-Correction"),
                ("display_corners", "X-Display-Corners"),
            ):
                value = str(metadata.get(source, ""))
                if value:
                    connection.putheader(target, value)
            if CLOUD_SHARED_SECRET:
                connection.putheader("Authorization", f"Bearer {CLOUD_SHARED_SECRET}")
            connection.endheaders()
            with assembled.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    connection.send(chunk)
            response = connection.getresponse()
            raw = response.read()
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {"error": "云端无法启动已上传的视频"}
            return response.status, payload if isinstance(payload, dict) else {}
        finally:
            connection.close()

    def save_court_calibration(self) -> None:
        """Accept four guided clicks for the currently paused analysis job."""
        status = status_payload()
        calibration = status.get("calibration")
        if status.get("state") != "needs_court_calibration" or not isinstance(calibration, dict):
            self.send_json({"error": "当前分析不需要人工确认球场"}, HTTPStatus.CONFLICT)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= 64 * 1024:
            self.send_json({"error": "球场校准数据无效"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            request_id = str(payload.get("request_id", ""))
            corners = payload.get("corners")
            if request_id != calibration.get("request_id"):
                raise ValueError("校准请求已经过期")
            if not isinstance(corners, list) or len(corners) != 4:
                raise ValueError("请依次标记四个球场角点")
            width = float(calibration["width"])
            height = float(calibration["height"])
            normalized: list[list[float]] = []
            for point in corners:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("角点坐标格式不正确")
                x, y = float(point[0]), float(point[1])
                if (
                    not math.isfinite(x)
                    or not math.isfinite(y)
                    or not (0 <= x < width and 0 <= y < height)
                ):
                    raise ValueError("角点超出了画面范围")
                normalized.append([x, y])
            import numpy as np

            from netcast_tennisvision.vision.court_calibration import validate_manual_calibration

            world_quad = np.array([[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]], np.float32)
            validate_manual_calibration(normalized, (int(height), int(width)), world_quad)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        temporary = CALIBRATION_RESPONSE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "request_id": request_id,
                    "corners": normalized,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, CALIBRATION_RESPONSE)
        self.send_json({"accepted": True})

    def save_camera_profiles(self) -> None:
        """Restore bounded, local fixed-camera evidence into an isolated cloud worker."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= 3 * 1024**2:
            self.send_json({"error": "固定机位资料大小无效"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            profiles = payload.get("profiles") if isinstance(payload, dict) else None
            if payload.get("version") != 1 or not isinstance(profiles, list):
                raise ValueError("固定机位资料版本无效")
            if len(profiles) > 12 or not all(isinstance(item, dict) for item in profiles):
                raise ValueError("固定机位资料数量无效")
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        write_json_atomic(CAMERA_PROFILES, payload)
        self.send_json({"accepted": True, "profiles": len(profiles)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Netcast TennisVision local analysis server")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("TENNISVISION_PORT", "4173"))
    )
    args = parser.parse_args()
    port = args.port
    host = os.environ.get("TENNISVISION_HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    if cloud_manager:
        cloud_manager.recover_orphan_async()
    if cloud_live_manager:
        cloud_live_manager.recover_orphan_async()
    print(f"Netcast TennisVision: http://{host}:{port}/web/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
