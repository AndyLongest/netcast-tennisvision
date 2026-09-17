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
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from netcast_tennisvision.streaming.result_relay import (
    delete_result_snapshot,
    fetch_result_snapshot,
)

PPIO_API = "https://api.ppio.com/gpu-instance/openapi/v1"
REMOTE_OUTPUTS = {"scene3d.json": True, "rally3d.html": True, "corrected_clip.mp4": False}
TRANSFER_CHUNK_SIZE = 8 * 1024**2
TRANSFER_ATTEMPTS = 5
DEFAULT_TRANSFER_WORKERS = 4
MAX_TRANSFER_WORKERS = 8
CAMERA_PROFILE_TRANSFER_LIMIT = 3 * 1024**2
DEFAULT_ROOTFS_SIZE_GB = 60
MINIMUM_ROOTFS_SIZE_GB = 10


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
            "image.ppinfra.com/prod-ahskpcitxxwcgdnfqfpu/netcast-tennisvision:production-v14",
        ).strip()
        self.product_id = os.environ.get("TENNISVISION_PPIO_PRODUCT_ID", "L40S.22c125g")
        self.cluster_id = os.environ.get("TENNISVISION_PPIO_CLUSTER_ID", "cn-south-1")
        self.transfer_workers = self._bounded_worker_count(
            os.environ.get("TENNISVISION_TRANSFER_WORKERS", str(DEFAULT_TRANSFER_WORKERS))
        )
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
                self.runtime_path,
                {"instance_id": instance_id, "created_at": int(time.time())},
            )
            remote_url = self._wait_for_endpoint(instance_id)
            with self._lock:
                self._remote_url = remote_url
            self._wait_for_service(remote_url)
            self._upload_camera_profiles(remote_url)
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
        rootfs_size = self._rootfs_size_for_product()
        payload = {
            "name": f"netcast-job-{int(time.time())}",
            "productId": self.product_id,
            "clusterId": self.cluster_id,
            "gpuNum": 1,
            "rootfsSize": rootfs_size,
            "imageUrl": self.image,
            "imageAuth": "",
            "imageAuthId": "",
            "ports": "8000/http",
            "envs": self._instance_envs(),
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
                "TENNISVISION_OUTPUT_MODE=event-overlay "
                "NETCAST_VIDEO_ENCODER=auto NETCAST_X264_CRF=22 "
                "PYTHONPATH=/app/src MPLBACKEND=Agg "
                "python -m netcast_tennisvision.api.server --port 8000'"
            ),
            "entrypoint": "",
            "networkStorages": [],
            "kind": "gpu",
            "billingMode": "onDemand",
            "minCudaVersion": "12.8",
        }
        try:
            response = self._provider_request("POST", "/gpu/instance/create", payload)
        except CloudLifecycleError as exc:
            # Product inventory can change between the products query and instance
            # creation. A validation rejection is safe to retry because no instance
            # was created and therefore no GPU billing has started.
            provider_limit = self._rootfs_limit_from_error(str(exc))
            if provider_limit is None or provider_limit == rootfs_size:
                raise
            payload["rootfsSize"] = provider_limit
            response = self._provider_request("POST", "/gpu/instance/create", payload)
        instance_id = self._find_string(response, ("instanceId", "id"))
        if not instance_id:
            raise CloudLifecycleError(f"云端没有返回实例编号：{response}")
        return instance_id

    def _instance_envs(self) -> list[dict[str, str]]:
        """Return environment shared by every temporary analysis worker."""
        return [{"key": "TENNISVISION_CLOUD_SHARED_SECRET", "value": self.shared_secret}]

    def _rootfs_size_for_product(self) -> int:
        """Choose a safe root filesystem size from the live product constraints."""
        configured = os.environ.get("TENNISVISION_PPIO_ROOTFS_GB", "").strip()
        try:
            desired = int(configured) if configured else DEFAULT_ROOTFS_SIZE_GB
        except ValueError:
            desired = DEFAULT_ROOTFS_SIZE_GB
        desired = max(MINIMUM_ROOTFS_SIZE_GB, desired)
        try:
            response = self._provider_request("GET", "/products")
        except CloudLifecycleError:
            return desired
        products = response.get("data", []) if isinstance(response, dict) else []
        for product in products if isinstance(products, list) else []:
            if not isinstance(product, dict) or str(product.get("id", "")) != self.product_id:
                continue
            try:
                minimum = max(MINIMUM_ROOTFS_SIZE_GB, int(product.get("minRootFS", 0)))
                maximum = int(product.get("maxRootFS", 0))
            except (TypeError, ValueError):
                return desired
            if maximum < minimum:
                return desired
            return max(minimum, min(desired, maximum))
        return desired

    @staticmethod
    def _rootfs_limit_from_error(message: str) -> int | None:
        match = re.search(r"rootfs size must not be more than\s+(\d+)\s*GB", message, re.I)
        if not match:
            return None
        limit = int(match.group(1))
        return limit if limit >= MINIMUM_ROOTFS_SIZE_GB else None

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

    def _upload_video(
        self,
        remote_url: str,
        clip: Path,
        upload_headers: dict[str, str],
        *,
        purpose: str = "analysis",
    ) -> dict[str, Any]:
        metadata = {
            "filename": upload_headers.get("X-Filename", clip.name),
            "total_size": clip.stat().st_size,
            "video_fingerprint": upload_headers.get("X-Video-Fingerprint", ""),
            "display_correction": upload_headers.get("X-Display-Correction", ""),
            "display_corners": upload_headers.get("X-Display-Corners", ""),
            "purpose": purpose,
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
        with clip.open("rb") as source, ThreadPoolExecutor(
            max_workers=self.transfer_workers,
            thread_name_prefix="netcast-upload",
        ) as executor:
            pending: dict[Future[None], int] = {}
            index = 0

            def submit_next() -> bool:
                nonlocal index
                chunk = source.read(chunk_size)
                if not chunk:
                    return False
                future = executor.submit(self._upload_part, remote_url, upload_id, index, chunk)
                pending[future] = len(chunk)
                index += 1
                return True

            for _ in range(self.transfer_workers * 2):
                if not submit_next():
                    break
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    chunk_length = pending.pop(future)
                    future.result()
                    uploaded += chunk_length
                progress = 4 + round(4 * uploaded / total_size)
                self._write_status(
                    "queued",
                    progress,
                    f"正在并行可靠上传比赛视频（{uploaded * 100 // total_size}%）",
                )
                while len(pending) < self.transfer_workers * 2 and submit_next():
                    pass
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
        return completed

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
                self._download_outputs(
                    remote_url,
                    annotated_required=bool(remote.get("annotated_video_ready", False)),
                )
                self._download_path(
                    remote_url,
                    "/data/camera_profiles.json",
                    self.root / "data" / "camera_profiles.json",
                    required=False,
                )
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

    def _download_outputs(self, remote_url: str, *, annotated_required: bool = True) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, required in REMOTE_OUTPUTS.items():
            self._download_path(
                remote_url, f"/data/outputs/{name}", self.output_dir / name, required=required
            )
        self._download_path(
            remote_url,
            "/data/outputs/annotated_clip.mp4",
            self.output_dir / "annotated_clip.mp4",
            required=annotated_required,
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
            status, content_range, body = self._fetch_range(
                remote_url, remote_path, 0, TRANSFER_CHUNK_SIZE - 1
            )
            if status == HTTPStatus.NOT_FOUND and not required:
                return
            if status != HTTPStatus.PARTIAL_CONTENT:
                raise CloudLifecycleError(f"云端结果不支持分片下载：{destination.name}")
            first_start, first_end, total = self._validated_content_range(
                content_range, body, expected_start=0
            )
            with temporary.open("w+b") as output:
                output.truncate(total)
                output.seek(first_start)
                output.write(body)
                with ThreadPoolExecutor(
                    max_workers=self.transfer_workers,
                    thread_name_prefix="netcast-download",
                ) as executor:
                    pending: dict[Future[tuple[int, str, bytes]], int] = {}
                    next_start = first_end + 1

                    def submit_next() -> bool:
                        nonlocal next_start
                        if next_start >= total:
                            return False
                        start = next_start
                        end = min(total - 1, start + TRANSFER_CHUNK_SIZE - 1)
                        pending[executor.submit(
                            self._fetch_range, remote_url, remote_path, start, end
                        )] = start
                        next_start = end + 1
                        return True

                    for _ in range(self.transfer_workers * 2):
                        if not submit_next():
                            break
                    while pending:
                        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                        for future in completed:
                            expected_start = pending.pop(future)
                            part_status, part_range, part_body = future.result()
                            if part_status != HTTPStatus.PARTIAL_CONTENT:
                                raise CloudLifecycleError(
                                    f"云端结果分片下载失败：{destination.name}"
                                )
                            returned_start, _, returned_total = self._validated_content_range(
                                part_range, part_body, expected_start=expected_start
                            )
                            if returned_total != total:
                                raise CloudLifecycleError("云端下载文件大小在传输中发生变化")
                            output.seek(returned_start)
                            output.write(part_body)
                        while len(pending) < self.transfer_workers * 2 and submit_next():
                            pass
            if temporary.stat().st_size != total:
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

    def _upload_camera_profiles(self, remote_url: str) -> None:
        """Copy local fixed-camera evidence into the isolated worker when available."""
        profile_path = self.root / "data" / "camera_profiles.json"
        try:
            body = profile_path.read_bytes()
        except OSError:
            return
        if not body or len(body) > CAMERA_PROFILE_TRANSFER_LIMIT:
            return
        try:
            self._remote_request(
                remote_url,
                "POST",
                "/api/camera-profiles",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
        except (OSError, TimeoutError, http.client.HTTPException):
            # Profile reuse is an optimization. A fresh calibration remains the safe fallback.
            return

    @staticmethod
    def _validated_content_range(
        content_range: str, body: bytes, *, expected_start: int
    ) -> tuple[int, int, int]:
        try:
            range_unit, range_value = content_range.split(" ", 1)
            returned, total_text = range_value.split("/", 1)
            returned_start, returned_end = (int(value) for value in returned.split("-", 1))
            total = int(total_text)
        except (TypeError, ValueError) as exc:
            raise CloudLifecycleError("云端返回了无效的下载范围") from exc
        if range_unit != "bytes" or returned_start != expected_start:
            raise CloudLifecycleError("云端返回的下载分片位置不正确")
        if returned_end < returned_start or total <= returned_end:
            raise CloudLifecycleError("云端返回的下载范围超出文件大小")
        if len(body) != returned_end - returned_start + 1:
            raise CloudLifecycleError("云端下载分片不完整")
        return returned_start, returned_end, total

    @staticmethod
    def _bounded_worker_count(value: str) -> int:
        try:
            workers = int(value)
        except ValueError:
            workers = DEFAULT_TRANSFER_WORKERS
        return max(1, min(MAX_TRANSFER_WORKERS, workers))

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
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            for attempt in range(30):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError:
                    if attempt == 29:
                        raise
                    time.sleep(0.01 * (1 + attempt // 5))
        finally:
            temporary.unlink(missing_ok=True)


class PPIOLiveJobManager(PPIOJobManager):
    """Run one uploaded pseudo-live source on an auto-released PPIO worker."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, root / "data" / "live_lab_cloud_status.json")
        self.runtime_path = root / "data" / "live_lab_cloud_runtime.json"
        self._stop_event = threading.Event()
        self._local_session_id = ""
        self._remote_session_id = ""
        self._producer: subprocess.Popen[bytes] | None = None

    def _instance_envs(self) -> list[dict[str, str]]:
        """Pass the external media relay to the isolated live worker.

        The live worker cannot read the local relay's ignored runtime config file.  It
        therefore receives only the non-secret ZLMediaKit host through the provider
        environment while provider credentials remain local.
        """
        envs = super()._instance_envs()
        config = self._read_json(self.root / "data" / "live_lab_config.json")
        host = os.environ.get("TENNISVISION_ZLM_HOST", "").strip()
        if not host:
            host = str(config.get("zlm_host", "")).strip()
        if host:
            envs.append({"key": "TENNISVISION_ZLM_HOST", "value": host})
        playback_origin = os.environ.get("TENNISVISION_ZLM_WEBRTC_ORIGIN", "").strip()
        if not playback_origin:
            playback_origin = str(config.get("zlm_webrtc_origin", "")).strip()
        if playback_origin:
            envs.append(
                {"key": "TENNISVISION_ZLM_WEBRTC_ORIGIN", "value": playback_origin}
            )
        relay_url, relay_token = self._result_relay_config()
        if relay_url and relay_token:
            envs.extend(
                [
                    {"key": "TENNISVISION_RESULT_RELAY_URL", "value": relay_url},
                    {"key": "TENNISVISION_RESULT_RELAY_TOKEN", "value": relay_token},
                ]
            )
        return envs

    def _result_relay_config(self) -> tuple[str, str]:
        config = self._read_json(self.root / "data" / "live_lab_config.json")
        url = os.environ.get("TENNISVISION_RESULT_RELAY_URL", "").strip()
        token = os.environ.get("TENNISVISION_RESULT_RELAY_TOKEN", "").strip()
        return (
            url or str(config.get("result_relay_url", "")).strip(),
            token or str(config.get("result_relay_token", "")).strip(),
        )

    def start_live(
        self,
        clip: Path,
        upload_headers: dict[str, str],
        *,
        replace_active: bool = False,
    ) -> dict[str, Any]:
        if not self.configured:
            raise CloudLifecycleError("L40S 按需实时实验尚未配置完整")
        relay_url, relay_token = self._result_relay_config()
        if not relay_url or not relay_token:
            raise CloudLifecycleError("ECS 推理结果中继尚未配置完整")
        previous_thread: threading.Thread | None = None
        with self._lock:
            if self.active:
                if not replace_active:
                    raise CloudLifecycleError("已有实时实验正在运行")
                self._stop_event.set()
                previous_thread = self._thread
        if previous_thread is not None:
            previous_thread.join(timeout=120)
            if previous_thread.is_alive():
                raise CloudLifecycleError("上一场实验仍在释放云端资源，请稍后重试")
        with self._lock:
            self._stop_event.clear()
            self._local_session_id = uuid.uuid4().hex
            self._remote_session_id = ""
            initial = {
                "session_id": self._local_session_id,
                "state": "preparing",
                "stage": "正在启动临时 L40S",
                "source_time": 0.0,
                "analysis_time": 0.0,
                "events": [],
                "event_cursor": 0,
                "execution_target": "cloud-live-l40s",
                "filename": upload_headers.get("X-Filename", clip.name),
            }
            self._write_json(self.status_path, initial)
            self._thread = threading.Thread(
                target=self._run_live,
                args=(clip, upload_headers, self._local_session_id),
                name="ppio-live-lab",
                daemon=True,
            )
            self._thread.start()
            return initial

    def snapshot(self, session_id: str, after_event: int = 0) -> dict[str, Any] | None:
        payload = self._read_json(self.status_path)
        if not session_id or payload.get("session_id") != session_id:
            return None
        events = payload.get("events", [])
        payload["events"] = events[after_event:] if isinstance(events, list) else []
        return payload

    def stop_live(self, session_id: str) -> bool:
        with self._lock:
            if session_id != self._local_session_id or not self.active:
                return False
            self._stop_event.set()
            return True

    def _run_live(
        self,
        clip: Path,
        upload_headers: dict[str, str],
        local_session_id: str,
    ) -> None:
        instance_id: str | None = None
        remote_url = ""
        remote_session_id = ""
        relay_url, relay_token = self._result_relay_config()
        try:
            instance_id = self._create_instance()
            with self._lock:
                self._instance_id = instance_id
            self._write_json(
                self.runtime_path,
                {
                    "instance_id": instance_id,
                    "session_id": local_session_id,
                    "created_at": int(time.time()),
                },
            )
            remote_url = self._wait_for_endpoint(instance_id)
            with self._lock:
                self._remote_url = remote_url
            self._wait_for_service(remote_url)
            self._upload_camera_profiles(remote_url)
            self._write_json(
                self.status_path,
                {
                    "session_id": local_session_id,
                    "state": "preparing",
                    "stage": "L40S 已就绪，正在加载在线模型",
                    "source_time": 0.0,
                    "analysis_time": 0.0,
                    "events": [],
                    "event_cursor": 0,
                    "execution_target": "cloud-live-l40s",
                    "filename": upload_headers.get("X-Filename", clip.name),
                },
            )
            stream_name = f"netcast-{local_session_id[:12]}"
            try:
                fps = float(upload_headers.get("X-Fps", "30"))
            except ValueError:
                fps = 30.0
            start_body = json.dumps(
                {
                    "source": "external_rtmp",
                    "stream_name": stream_name,
                    "fps": fps,
                    "filename": upload_headers.get("X-Filename", clip.name),
                    "result_session_id": local_session_id,
                }
            ).encode("utf-8")
            status, started = self._remote_request(
                remote_url,
                "POST",
                "/api/live-lab/start",
                body=start_body,
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": str(len(start_body)),
                },
            )
            if status >= 300:
                raise CloudLifecycleError(str(started.get("error", "云端实时实验无法启动")))
            remote_session_id = str(started.get("session_id", ""))
            if not remote_session_id:
                raise CloudLifecycleError("云端实时实验没有返回会话编号")
            with self._lock:
                self._remote_session_id = remote_session_id

            # The inference server is a pure ZLM subscriber.  Start the camera
            # simulator only after the remote models are ready, exactly as a venue
            # camera would publish independently of the GPU worker.
            readiness_deadline = time.monotonic() + 180.0
            while time.monotonic() < readiness_deadline and not self._stop_event.is_set():
                status, payload = self._remote_request(
                    remote_url,
                    "GET",
                    f"/api/live-lab/status?session_id={remote_session_id}&after_event=0",
                )
                if status >= 300:
                    raise CloudLifecycleError(str(payload.get("error", "无法读取云端实时状态")))
                payload["remote_session_id"] = remote_session_id
                payload["session_id"] = local_session_id
                payload["execution_target"] = "cloud-live-l40s"
                payload["filename"] = upload_headers.get("X-Filename", clip.name)
                self._write_json(self.status_path, payload)
                if str(payload.get("state", "")) == "awaiting_stream":
                    break
                if str(payload.get("state", "")) in {"error", "stopped"}:
                    raise CloudLifecycleError(str(payload.get("error", "在线模型未能就绪")))
                time.sleep(0.2)
            else:
                raise CloudLifecycleError("L40S 在线模型准备超时")

            self._producer = self._start_camera_simulator(clip, stream_name, fps)

            relay_deadline = time.monotonic() + 20.0
            last_relay_update = 0.0
            while not self._stop_event.is_set():
                try:
                    payload = fetch_result_snapshot(relay_url, relay_token, local_session_id)
                except (OSError, TimeoutError, urllib.error.URLError) as exc:
                    if time.monotonic() >= relay_deadline:
                        raise CloudLifecycleError(f"无法从 ECS 读取推理结果：{exc}") from exc
                    time.sleep(0.12)
                    continue
                if payload is None:
                    if time.monotonic() >= relay_deadline:
                        raise CloudLifecycleError("L40S 未向 ECS 上报推理结果")
                    time.sleep(0.12)
                    continue
                if payload.get("result_relay") != "ecs":
                    raise CloudLifecycleError("拒绝未经过 ECS 中继的推理结果")
                relay_update = float(payload.get("relay_updated_at", 0.0) or 0.0)
                if relay_update > last_relay_update:
                    last_relay_update = relay_update
                    relay_deadline = time.monotonic() + 20.0
                elif time.monotonic() >= relay_deadline:
                    raise CloudLifecycleError("ECS 上的 L40S 推理结果已停止更新")
                payload["remote_session_id"] = remote_session_id
                payload["session_id"] = local_session_id
                payload["execution_target"] = "cloud-live-l40s"
                payload["result_path"] = "l40s->ecs-result-relay->local"
                payload["filename"] = upload_headers.get("X-Filename", clip.name)
                self._write_json(self.status_path, payload)
                if str(payload.get("state", "")) in {"complete", "error", "stopped"}:
                    break
                if self._producer.poll() is not None and str(payload.get("state", "")) != "complete":
                    # A normal zero exit means the finite camera simulation reached
                    # EOF; give the remote puller time to drain and close naturally.
                    if self._producer.returncode not in {0, None}:
                        detail = (
                            self._producer.stderr.read() if self._producer.stderr else b""
                        ).decode("utf-8", errors="replace")
                        raise CloudLifecycleError(
                            f"摄像头模拟推流中断：{detail.strip() or self._producer.returncode}"
                        )
                time.sleep(0.12)
            if self._stop_event.is_set():
                self._stop_remote_live(remote_url, remote_session_id)
                stopped = self._read_json(self.status_path)
                stopped.update(state="stopped", stage="实验已停止")
                self._write_json(self.status_path, stopped)
        except Exception as exc:
            self._write_json(
                self.status_path,
                {
                    "session_id": local_session_id,
                    "state": "error",
                    "stage": "L40S 实时实验未完成",
                    "error": str(exc),
                    "events": [],
                    "event_cursor": 0,
                    "execution_target": "cloud-live-l40s",
                },
            )
        finally:
            if self._producer is not None and self._producer.poll() is None:
                self._producer.terminate()
                try:
                    self._producer.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._producer.kill()
            self._producer = None
            if self._stop_event.is_set() and remote_url and remote_session_id:
                self._stop_remote_live(remote_url, remote_session_id)
            released = self._release_instance(instance_id) if instance_id else True
            # ECS is a transient result relay, not a report store. The final state is
            # already durable in status_path. Delete the relay copy only after the GPU
            # can no longer publish a late update; its TTL janitor covers hard crashes.
            for attempt in range(3):
                try:
                    delete_result_snapshot(relay_url, relay_token, local_session_id)
                    break
                except (OSError, TimeoutError, urllib.error.URLError):
                    if attempt < 2:
                        time.sleep(0.25 * (attempt + 1))
            with self._lock:
                self._instance_id = None
                self._remote_url = None
                self._remote_session_id = ""
            if released:
                self.runtime_path.unlink(missing_ok=True)

    def _start_camera_simulator(
        self, clip: Path, stream_name: str, fps: float
    ) -> subprocess.Popen[bytes]:
        packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        candidates = sorted(
            packages.glob("Gyan.FFmpeg.Shared_*/ffmpeg-*/bin/ffmpeg.exe"), reverse=True
        )
        executable = str(candidates[0]) if candidates else shutil.which("ffmpeg")
        if not executable:
            raise CloudLifecycleError("本机没有 FFmpeg，无法模拟摄像头推流")
        config = self._read_json(self.root / "data" / "live_lab_config.json")
        host = os.environ.get("TENNISVISION_ZLM_HOST", "").strip() or str(
            config.get("zlm_host", "")
        ).strip()
        if not host:
            raise CloudLifecycleError("尚未配置 ZLMediaKit 地址")
        stream_url = f"rtmp://{host}:1935/live/{stream_name}"
        command = [
            executable,
            "-hide_banner", "-loglevel", "error", "-re", "-i", str(clip),
            "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
            "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", str(max(1, round(fps))),
            "-bf", "0", "-f", "flv", stream_url,
        ]
        return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _write_status(self, state: str, progress: int, stage: str) -> None:
        """Keep the browser session addressable during chunked cloud uploads."""
        payload = self._read_json(self.status_path)
        payload.update(
            state="preparing" if state == "queued" else state,
            progress=progress,
            stage=stage,
            execution_target="cloud-live-l40s",
        )
        payload.setdefault("session_id", self._local_session_id)
        payload.setdefault("source_time", 0.0)
        payload.setdefault("analysis_time", 0.0)
        payload.setdefault("events", [])
        payload.setdefault("event_cursor", 0)
        self._write_json(self.status_path, payload)

    def _stop_remote_live(self, remote_url: str, remote_session_id: str) -> None:
        body = json.dumps({"session_id": remote_session_id}).encode("utf-8")
        try:
            self._remote_request(
                remote_url,
                "POST",
                "/api/live-lab/stop",
                body=body,
                headers={"Content-Type": "application/json", "Content-Length": str(len(body))},
            )
        except (OSError, TimeoutError, http.client.HTTPException):
            pass
