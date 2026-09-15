"""Create one PPIO GPU instance per analysis job and release it afterwards.

The browser never receives provider credentials.  This controller runs in the local
relay, mirrors remote progress to the local status file, downloads finished artifacts,
and deletes the instance in a ``finally`` block.  A small runtime journal also lets the
next relay process clean up an instance left behind by an unexpected local shutdown.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

PPIO_API = "https://api.ppio.com/gpu-instance/openapi/v1"
REMOTE_OUTPUTS = {
    "scene3d.json": True,
    "rally3d.html": True,
    "annotated_clip.mp4": True,
    "corrected_clip.mp4": False,
}
TRANSFER_CHUNK_SIZE = 8 * 1024**2
TRANSFER_ATTEMPTS = 5


class CloudLifecycleError(RuntimeError):
    """A user-facing cloud provisioning or transfer failure."""


class PPIOJobManager:
    """Own the single temporary GPU used by the local single-job API."""

    def __init__(self, root: Path, status_path: Path) -> None:
        self.root = root
        self.status_path = status_path
        self.runtime_path = root / "data" / "cloud_runtime.json"
        self.output_dir = root / "data" / "outputs"
        self.api_key = os.environ.get("PPIO_API_KEY", "").strip()
        self.shared_secret = os.environ.get("TENNISVISION_CLOUD_TOKEN", "").strip()
        self.image = os.environ.get(
            "TENNISVISION_PPIO_IMAGE",
            "image.ppinfra.com/prod-ahskpcitxxwcgdnfqfpu/netcast-tennisvision:production-v4",
        ).strip()
        self.product_id = os.environ.get("TENNISVISION_PPIO_PRODUCT_ID", "L40S.22c125g")
        self.cluster_id = os.environ.get("TENNISVISION_PPIO_CLUSTER_ID", "cn-south-1")
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._instance_id: str | None = None
        self._remote_url: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.shared_secret and self.image)

    @property
    def active(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def recover_orphan_async(self) -> None:
        """Release a journaled instance after an unclean relay shutdown."""
        runtime = self._read_json(self.runtime_path)
        instance_id = str(runtime.get("instance_id", ""))
        if not instance_id or not self.configured:
            return
        with self._lock:
            self._thread = threading.Thread(
                target=self._release_orphan,
                args=(instance_id,),
                name="ppio-orphan-cleanup",
                daemon=True,
            )
            self._thread.start()

    def start(self, clip: Path, upload_headers: dict[str, str]) -> None:
        if not self.configured:
            raise CloudLifecycleError("云端按需算力尚未配置完整")
        with self._lock:
            if self.active:
                raise CloudLifecycleError("已有云端分析任务正在运行")
            self._thread = threading.Thread(
                target=self._run,
                args=(clip, upload_headers),
                name="ppio-analysis-job",
                daemon=True,
            )
            self._thread.start()

    def submit_calibration(self, payload: bytes) -> dict[str, Any]:
        with self._lock:
            remote_url = self._remote_url
        if not remote_url:
            raise CloudLifecycleError("云端任务尚未准备好接收球场确认")
        status, response = self._remote_request(
            remote_url,
            "POST",
            "/api/court-calibration",
            body=payload,
            headers={"Content-Type": "application/json", "Content-Length": str(len(payload))},
        )
        if status >= 300:
            raise CloudLifecycleError(str(response.get("error", "球场确认提交失败")))
        return response

    def _run(self, clip: Path, upload_headers: dict[str, str]) -> None:
        instance_id: str | None = None
        try:
            self._write_status("queued", 2, "正在启动临时云端算力")
            instance_id = self._create_instance()
            with self._lock:
                self._instance_id = instance_id
            self._write_json(
                self.runtime_path, {"instance_id": instance_id, "created_at": int(time.time())}
            )
            remote_url = self._wait_for_endpoint(instance_id)
            with self._lock:
                self._remote_url = remote_url
            self._wait_for_service(remote_url)
            self._write_status("queued", 4, "云端算力已就绪，正在上传比赛视频")
            self._upload_video(remote_url, clip, upload_headers)
            self._mirror_until_complete(remote_url)
        except Exception as exc:  # worker boundary: always expose the failure and release GPU
            self._write_json(
                self.status_path,
                {
                    "state": "error",
                    "progress": 0,
                    "stage": "云端分析未完成",
                    "error": str(exc),
                    "execution_target": "cloud-on-demand",
                },
            )
        finally:
            if instance_id:
                released = self._release_instance(instance_id)
            else:
                released = True
            with self._lock:
                self._instance_id = None
                self._remote_url = None
            if released:
                self.runtime_path.unlink(missing_ok=True)

    def _create_instance(self) -> str:
        payload = {
            "name": f"netcast-job-{int(time.time())}",
            "productId": self.product_id,
            "clusterId": self.cluster_id,
            "gpuNum": 1,
            "rootfsSize": 100,
            "imageUrl": self.image,
            "imageAuth": "",
            "imageAuthId": "",
            "ports": "8000/http",
            "envs": [{"key": "TENNISVISION_CLOUD_SHARED_SECRET", "value": self.shared_secret}],
            "tools": [],
            "command": (
                "bash -lc 'cd /app && "
                "mkdir -p data/cache data/outputs && "
                "find data/cache data/outputs -mindepth 1 -maxdepth 1 "
                "-exec rm -rf -- {} + && "
                "rm -f data/current_job.json data/job_status.json "
                "data/camera_profiles.json data/clip.mp4 && "
                "exec timeout --signal=TERM 7200 env "
                "TENNISVISION_HOST=0.0.0.0 "
                "TENNISVISION_PORT=8000 TENNISVISION_EXECUTION_TARGET=cloud "
                "PYTHONPATH=/app/src MPLBACKEND=Agg "
                "python -m netcast_tennisvision.api.server --port 8000'"
            ),
            "entrypoint": "",
            "networkStorages": [],
            "kind": "gpu",
            "billingMode": "onDemand",
            "minCudaVersion": "12.8",
        }
        response = self._provider_request("POST", "/gpu/instance/create", payload)
        instance_id = self._find_string(response, ("instanceId", "id"))
        if not instance_id:
            raise CloudLifecycleError(f"云端没有返回实例编号：{response}")
        return instance_id

    def _wait_for_endpoint(self, instance_id: str) -> str:
        deadline = time.monotonic() + 10 * 60
        last_connection_error: CloudLifecycleError | None = None
        while time.monotonic() < deadline:
            try:
                detail = self._provider_request("GET", f"/gpu/instance?instanceId={instance_id}")
                last_connection_error = None
            except CloudLifecycleError as exc:
                # PPIO's control plane can briefly reset TLS connections while an
                # image is being scheduled or pulled.  The instance already exists,
                # so a failed status read must not fail the user's whole analysis.
                last_connection_error = exc
                time.sleep(2)
                continue
            state = str(detail.get("status", ""))
            if state in {"error", "failed"}:
                message = detail.get("statusError", {}).get("message", "实例启动失败")
                raise CloudLifecycleError(str(message))
            for mapping in detail.get("portMappings", []):
                if int(mapping.get("port", 0)) == 8000 and mapping.get("endpoint"):
                    if state == "running":
                        return str(mapping["endpoint"]).rstrip("/")
            time.sleep(2)
        if last_connection_error is not None:
            raise CloudLifecycleError(f"云端 GPU 启动超时：{last_connection_error}")
        raise CloudLifecycleError("云端 GPU 启动超时")

    def _wait_for_service(self, remote_url: str) -> None:
        deadline = time.monotonic() + 5 * 60
        while time.monotonic() < deadline:
            try:
                status, _ = self._remote_request(remote_url, "GET", "/api/status")
                if status == 200:
                    return
            except (OSError, TimeoutError, http.client.HTTPException):
                pass
            time.sleep(2)
        raise CloudLifecycleError("云端分析服务启动超时")

    def _upload_video(self, remote_url: str, clip: Path, upload_headers: dict[str, str]) -> None:
        metadata = {
            "filename": upload_headers.get("X-Filename", clip.name),
            "total_size": clip.stat().st_size,
            "video_fingerprint": upload_headers.get("X-Video-Fingerprint", ""),
            "display_correction": upload_headers.get("X-Display-Correction", ""),
            "display_corners": upload_headers.get("X-Display-Corners", ""),
        }
        encoded = json.dumps(metadata).encode("utf-8")
        status, initialized = self._remote_request(
            remote_url,
            "POST",
            "/api/upload/init",
            body=encoded,
            headers={"Content-Type": "application/json", "Content-Length": str(len(encoded))},
        )
        if status >= 300:
            raise CloudLifecycleError(str(initialized.get("error", "云端无法建立上传会话")))
        upload_id = str(initialized.get("upload_id", ""))
        chunk_size = int(initialized.get("chunk_size", TRANSFER_CHUNK_SIZE))
        if not upload_id or not 0 < chunk_size <= 32 * 1024**2:
            raise CloudLifecycleError("云端返回了无效的上传会话")
        total_size = clip.stat().st_size
        uploaded = 0
        with clip.open("rb") as source:
            index = 0
            while chunk := source.read(chunk_size):
                self._upload_part(remote_url, upload_id, index, chunk)
                uploaded += len(chunk)
                progress = 4 + round(4 * uploaded / total_size)
                self._write_status(
                    "queued",
                    progress,
                    f"正在可靠上传比赛视频（{uploaded * 100 // total_size}%）",
                )
                index += 1
        completed_body = json.dumps({"upload_id": upload_id}).encode("utf-8")
        status, completed = self._remote_request(
            remote_url,
            "POST",
            "/api/upload/complete",
            body=completed_body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(completed_body)),
            },
        )
        if status >= 300:
            raise CloudLifecycleError(str(completed.get("error", "云端无法合并上传的视频")))

    def _upload_part(self, remote_url: str, upload_id: str, index: int, body: bytes) -> None:
        parsed = urlparse(remote_url)
        target = f"{parsed.path.rstrip('/')}/api/upload/chunk/{upload_id}/{index}"
        last_error: Exception | None = None
        for attempt in range(TRANSFER_ATTEMPTS):
            connection = self._connection(parsed, timeout=90)
            try:
                connection.request(
                    "PUT",
                    target,
                    body=body,
                    headers={
                        "Authorization": f"Bearer {self.shared_secret}",
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(len(body)),
                        "X-Chunk-SHA256": hashlib.sha256(body).hexdigest(),
                    },
                )
                response = connection.getresponse()
                payload = self._decode_json(response.read())
                if response.status < 300:
                    return
                if response.status < 500:
                    raise CloudLifecycleError(
                        str(payload.get("error", f"第 {index + 1} 个视频分片被拒绝"))
                    )
                last_error = CloudLifecycleError(
                    str(payload.get("error", f"第 {index + 1} 个视频分片上传失败"))
                )
            except CloudLifecycleError:
                raise
            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                connection.close()
            time.sleep(0.6 * (2**attempt))
        raise CloudLifecycleError(f"第 {index + 1} 个视频分片重试后仍失败：{last_error}")

    def _mirror_until_complete(self, remote_url: str) -> None:
        report_downloaded = False
        while True:
            status_code, remote = self._remote_request(remote_url, "GET", "/api/status")
            if status_code != 200:
                raise CloudLifecycleError("无法读取云端分析进度")
            state = str(remote.get("state", ""))
            remote["execution_target"] = "cloud-on-demand"
            if state == "needs_court_calibration":
                self._download_path(
                    remote_url,
                    "/data/court_calibration_preview.jpg",
                    self.root / "data" / "court_calibration_preview.jpg",
                    required=True,
                )
            if state in {"report_ready", "complete"} and not report_downloaded:
                self._download_report(remote_url)
                report_downloaded = True
            if state == "complete":
                self._download_outputs(remote_url)
                self._write_json(self.status_path, remote)
                return
            self._write_json(self.status_path, remote)
            if state == "error":
                raise CloudLifecycleError(str(remote.get("error", "云端分析失败")))
            time.sleep(1.8)

    def _download_report(self, remote_url: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("scene3d.json", "rally3d.html"):
            self._download_path(
                remote_url, f"/data/outputs/{name}", self.output_dir / name, required=True
            )
        self._download_path(
            remote_url,
            "/data/outputs/corrected_clip.mp4",
            self.output_dir / "corrected_clip.mp4",
            required=False,
        )

    def _download_outputs(self, remote_url: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, required in REMOTE_OUTPUTS.items():
            self._download_path(
                remote_url, f"/data/outputs/{name}", self.output_dir / name, required=required
            )

    def _download_path(
        self, remote_url: str, remote_path: str, destination: Path, *, required: bool
    ) -> None:
        if destination.suffix.lower() == ".mp4":
            self._download_video_ranged(remote_url, remote_path, destination, required=required)
            return
        parsed = urlparse(remote_url)
        connection = self._connection(parsed, timeout=3600)
        temporary = destination.with_name(f".{destination.name}.cloud-download")
        try:
            target = f"{parsed.path.rstrip('/')}{remote_path}"
            connection.request(
                "GET", target, headers={"Authorization": f"Bearer {self.shared_secret}"}
            )
            response = connection.getresponse()
            if response.status == 404 and not required:
                response.read()
                return
            if response.status != 200:
                response.read()
                raise CloudLifecycleError(f"云端结果下载失败：{destination.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
            connection.close()

    def _download_video_ranged(
        self, remote_url: str, remote_path: str, destination: Path, *, required: bool
    ) -> None:
        temporary = destination.with_name(f".{destination.name}.cloud-download")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            start = 0
            total: int | None = None
            with temporary.open("wb") as output:
                while total is None or start < total:
                    end = start + TRANSFER_CHUNK_SIZE - 1
                    status, content_range, body = self._fetch_range(
                        remote_url, remote_path, start, end
                    )
                    if status == 404 and not required:
                        return
                    if status != HTTPStatus.PARTIAL_CONTENT:
                        raise CloudLifecycleError(f"云端结果不支持分片下载：{destination.name}")
                    try:
                        range_unit, range_value = content_range.split(" ", 1)
                        returned, total_text = range_value.split("/", 1)
                        returned_start, returned_end = (int(value) for value in returned.split("-", 1))
                        parsed_total = int(total_text)
                    except (TypeError, ValueError) as exc:
                        raise CloudLifecycleError("云端返回了无效的下载范围") from exc
                    if range_unit != "bytes" or returned_start != start:
                        raise CloudLifecycleError("云端返回的下载分片顺序不正确")
                    expected_length = returned_end - returned_start + 1
                    if len(body) != expected_length:
                        raise CloudLifecycleError("云端下载分片不完整")
                    total = parsed_total
                    output.write(body)
                    start = returned_end + 1
            if total is None or temporary.stat().st_size != total:
                raise CloudLifecycleError("云端视频下载不完整")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _fetch_range(
        self, remote_url: str, remote_path: str, start: int, end: int
    ) -> tuple[int, str, bytes]:
        parsed = urlparse(remote_url)
        target = f"{parsed.path.rstrip('/')}{remote_path}"
        last_error: Exception | None = None
        for attempt in range(TRANSFER_ATTEMPTS):
            connection = self._connection(parsed, timeout=90)
            try:
                connection.request(
                    "GET",
                    target,
                    headers={
                        "Authorization": f"Bearer {self.shared_secret}",
                        "Range": f"bytes={start}-{end}",
                    },
                )
                response = connection.getresponse()
                body = response.read()
                if response.status in {HTTPStatus.PARTIAL_CONTENT, HTTPStatus.NOT_FOUND}:
                    return response.status, response.getheader("Content-Range", ""), body
                if response.status < 500:
                    raise CloudLifecycleError(f"云端视频分片下载被拒绝（{response.status}）")
                last_error = CloudLifecycleError(f"云端视频分片下载失败（{response.status}）")
            except CloudLifecycleError:
                raise
            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                last_error = exc
            finally:
                connection.close()
            time.sleep(0.6 * (2**attempt))
        raise CloudLifecycleError(f"云端视频分片重试后仍下载失败：{last_error}")

    def _remote_request(
        self,
        remote_url: str,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        parsed = urlparse(remote_url)
        connection = self._connection(parsed, timeout=30)
        request_headers = {"Authorization": f"Bearer {self.shared_secret}"}
        request_headers.update(headers or {})
        try:
            connection.request(method, f"{parsed.path.rstrip('/')}{path}", body, request_headers)
            response = connection.getresponse()
            return response.status, self._decode_json(response.read())
        finally:
            connection.close()

    @staticmethod
    def _connection(parsed: Any, timeout: float) -> http.client.HTTPConnection:
        cls = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        return cls(parsed.hostname, parsed.port, timeout=timeout)

    def _provider_request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{PPIO_API}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return self._decode_json(response.read())
        except urllib.error.HTTPError as exc:
            detail = self._decode_json(exc.read())
            message = detail.get("message") or detail.get("error") or str(exc)
            raise CloudLifecycleError(f"PPIO 请求失败：{message}") from exc
        except urllib.error.URLError as exc:
            raise CloudLifecycleError(f"无法连接 PPIO：{exc.reason}") from exc

    def _release_orphan(self, instance_id: str) -> None:
        if self._release_instance(instance_id):
            self.runtime_path.unlink(missing_ok=True)

    def _release_instance(self, instance_id: str) -> bool:
        try:
            self._provider_request("POST", "/gpu/instance/stop", {"instanceId": instance_id})
        except CloudLifecycleError:
            pass
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                detail = self._provider_request("GET", f"/gpu/instance?instanceId={instance_id}")
                if str(detail.get("status", "")) in {"exited", "stopped"}:
                    break
            except CloudLifecycleError:
                break
            time.sleep(2)
        try:
            self._provider_request("POST", "/gpu/instance/delete", {"instanceId": instance_id})
        except CloudLifecycleError:
            # Keep the journal so the next relay startup retries cleanup.
            return False
        return True

    def _write_status(self, state: str, progress: int, stage: str) -> None:
        self._write_json(
            self.status_path,
            {
                "state": state,
                "progress": progress,
                "stage": stage,
                "execution_target": "cloud-on-demand",
            },
        )

    @staticmethod
    def _find_string(payload: Any, names: tuple[str, ...]) -> str | None:
        if isinstance(payload, dict):
            for name in names:
                value = payload.get(name)
                if isinstance(value, str) and value:
                    return value
            for value in payload.values():
                found = PPIOJobManager._find_string(value, names)
                if found:
                    return found
        return None

    @staticmethod
    def _decode_json(raw: bytes) -> dict[str, Any]:
        try:
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
