"""Local Netcast TennisVision UI server and single-job analysis API."""
from __future__ import annotations

import argparse
import hashlib
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
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
STATUS = DATA / "job_status.json"
LOG = DATA / "pipeline.log"
VIDEO_IDENTITIES = DATA / "video_identities.json"
CURRENT_JOB = DATA / "current_job.json"
CALIBRATION_REQUEST = DATA / "court_calibration_request.json"
CALIBRATION_RESPONSE = DATA / "court_calibration_response.json"
job_lock = threading.Lock()
job_process: subprocess.Popen[bytes] | None = None
ACTIVE_STATES = {"queued", "running", "needs_court_calibration"}


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
        return hashlib.file_digest(left, "sha256").digest() == hashlib.file_digest(right, "sha256").digest()


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
    executable = (directory / ("ffprobe.exe" if os.name == "nt" else "ffprobe")) if directory else None
    if executable is None or not executable.exists():
        raise RuntimeError("本机未找到 ffprobe，无法检查视频帧率")
    completed = subprocess.run(
        [str(executable), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate", "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
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
        self.send_header("Access-Control-Allow-Origin", "null" if self.headers.get("Origin") == "null" else "http://127.0.0.1:4173")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        """Tell browsers that generated match videos support seeking."""
        if urlparse(self.path).path.lower().endswith(".mp4"):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

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
        if urlparse(self.path).path not in {
            "/api/status", "/api/analyze", "/api/court-calibration"
        }:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "null" if self.headers.get("Origin") == "null" else "http://127.0.0.1:4173")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Filename, X-Video-Fingerprint")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/api/status":
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
        if request_path == "/api/court-calibration":
            self.save_court_calibration()
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
                    self.send_json({
                        "accepted": True,
                        "resumed": True,
                        "job_id": resumed.get("job_id"),
                        "filename": resumed.get("filename"),
                        "fps": resumed.get("fps"),
                        "workload_factor": resumed.get("workload_factor", 1),
                    }, HTTPStatus.ACCEPTED)
                    return
                self.send_json({
                    "error": "另一段视频正在分析，请等待当前分析完成",
                    "code": "analysis_in_progress",
                    "filename": current_status.get("filename"),
                    "progress": current_status.get("progress", 0),
                }, HTTPStatus.CONFLICT)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > 4 * 1024**3:
                self.send_json({"error": "视频为空或超过 4GB"}, HTTPStatus.BAD_REQUEST)
                return
            filename = Path(unquote(self.headers.get("X-Filename", "clip.mp4"))).name
            suffix = Path(filename).suffix.lower()
            if suffix not in {".mp4", ".mov", ".webm", ".mkv"}:
                self.send_json({"error": "不支持的视频格式"}, HTTPStatus.BAD_REQUEST)
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
            except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError,
                    subprocess.SubprocessError, RuntimeError) as exc:
                temporary.unlink(missing_ok=True)
                self.send_json({"error": f"无法读取视频规格：{exc}"}, HTTPStatus.BAD_REQUEST)
                return
            # The pipeline derives every temporal window from the measured FPS. Accept
            # any valid native rate and never silently drop or invent frames.
            if not supports_native_fps(fps):
                temporary.unlink(missing_ok=True)
                self.send_json({
                    "error": f"视频没有有效帧率（读到{fps!r}fps），无法建立时间轴。"
                }, HTTPStatus.UNPROCESSABLE_ENTITY)
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
                "started_at": int(time.time()),
            }
            # Publish a self-contained queued snapshot first. Until CURRENT_JOB is replaced,
            # readers still see the new identity because status fields win during merging.
            write_json_atomic(STATUS, {
                "state": "queued", "progress": 1, "stage": "视频已接收，准备分析",
                **job_metadata,
            })
            write_json_atomic(CURRENT_JOB, job_metadata)
            environment = os.environ.copy()
            ffmpeg_dir = ffmpeg_directory()
            if ffmpeg_dir:
                environment["PATH"] = str(ffmpeg_dir) + os.pathsep + environment.get("PATH", "")
            log_handle = LOG.open("wb")
            job_process = subprocess.Popen(
                [sys.executable, str(ROOT / "pipeline_runner.py")], cwd=ROOT,
                env=environment, stdout=log_handle, stderr=subprocess.STDOUT
            )
            job_metadata["pid"] = job_process.pid
            write_json_atomic(CURRENT_JOB, job_metadata)
            self.send_json({
                "accepted": True,
                "resumed": False,
                "job_id": job_id,
                "filename": filename,
                "fps": round(fps, 3),
                "high_fps": fps >= 48.0,
                "workload_factor": workload_factor,
                "video_fingerprint": fingerprint,
            }, HTTPStatus.ACCEPTED)

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
                if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x < width and 0 <= y < height):
                    raise ValueError("角点超出了画面范围")
                normalized.append([x, y])
            import numpy as np

            from court_calibration import validate_manual_calibration
            world_quad = np.array(
                [[0, 0], [10.97, 0], [10.97, 23.77], [0, 23.77]], np.float32)
            validate_manual_calibration(
                normalized, (int(height), int(width)), world_quad)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        temporary = CALIBRATION_RESPONSE.with_suffix(".tmp")
        temporary.write_text(json.dumps({
            "request_id": request_id, "corners": normalized,
        }, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, CALIBRATION_RESPONSE)
        self.send_json({"accepted": True})


def main() -> None:
    parser = argparse.ArgumentParser(description="Netcast TennisVision local analysis server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TENNISVISION_PORT", "4173")))
    args = parser.parse_args()
    port = args.port
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Netcast TennisVision: http://127.0.0.1:{port}/web/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
