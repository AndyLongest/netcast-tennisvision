"""Run the true pseudo-live landing experiment on one temporary PPIO GPU.

The benchmark deliberately avoids the offline upload/download workflow.  The remote
worker pushes the bundled demo to ZLMediaKit at source speed, pulls that RTMP stream
back, and performs online inference.  The provider instance is deleted in ``finally``
even when provisioning, streaming, or inference fails.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from netcast_tennisvision.cloud.ppio_lifecycle import CloudLifecycleError, PPIOJobManager
from netcast_tennisvision.paths import REPOSITORY_ROOT

DEFAULT_IMAGE = (
    "image.ppinfra.com/prod-ahskpcitxxwcgdnfqfpu/"
    "netcast-tennisvision:production-v10"
)
DEFAULT_PRODUCT = "L40S.22c125g"
TERMINAL_STATES = {"complete", "error", "stopped"}


def _instance_payload(manager: PPIOJobManager) -> dict[str, Any]:
    rootfs_size = manager._rootfs_size_for_product()
    media_host = os.environ.get("TENNISVISION_ZLM_HOST", "39.108.95.121").strip()
    media_origin = os.environ.get(
        "TENNISVISION_ZLM_WEBRTC_ORIGIN", "https://zlmediakit.moralspace.com"
    ).strip()
    return {
        "name": f"netcast-live-l40s-{int(time.time())}",
        "productId": manager.product_id,
        "clusterId": manager.cluster_id,
        "gpuNum": 1,
        "rootfsSize": rootfs_size,
        "imageUrl": manager.image,
        "imageAuth": "",
        "imageAuthId": "",
        "ports": "8000/http",
        "envs": [
            {"key": "TENNISVISION_CLOUD_SHARED_SECRET", "value": manager.shared_secret},
            {"key": "TENNISVISION_ZLM_HOST", "value": media_host},
            {"key": "TENNISVISION_ZLM_WEBRTC_ORIGIN", "value": media_origin},
        ],
        "tools": [],
        "command": (
            "bash -lc 'cd /app && mkdir -p data && "
            "rm -f data/live_lab_last.json && "
            "exec timeout --signal=TERM 1800 env "
            "TENNISVISION_HOST=0.0.0.0 TENNISVISION_PORT=8000 "
            "TENNISVISION_EXECUTION_TARGET=cloud-live-l40s "
            "PYTHONPATH=/app/src MPLBACKEND=Agg "
            "python -m netcast_tennisvision.api.server --port 8000'"
        ),
        "entrypoint": "",
        "networkStorages": [],
        "kind": "gpu",
        "billingMode": "onDemand",
        "minCudaVersion": "12.8",
    }


def _create_instance(manager: PPIOJobManager) -> str:
    payload = _instance_payload(manager)
    response = manager._provider_request("POST", "/gpu/instance/create", payload)
    instance_id = manager._find_string(response, ("instanceId", "id"))
    if not instance_id:
        raise CloudLifecycleError(f"云端没有返回实例编号：{response}")
    return instance_id


def _request(
    manager: PPIOJobManager,
    remote_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = None
    if body is not None:
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    status, response = manager._remote_request(
        remote_url, method, path, body=body, headers=headers
    )
    if status >= 300:
        raise CloudLifecycleError(str(response.get("error", f"远端请求失败（{status}）")))
    return response


def run_benchmark(output: Path, timeout_seconds: float) -> dict[str, Any]:
    os.environ.setdefault("TENNISVISION_PPIO_IMAGE", DEFAULT_IMAGE)
    os.environ.setdefault("TENNISVISION_PPIO_PRODUCT_ID", DEFAULT_PRODUCT)
    manager = PPIOJobManager(REPOSITORY_ROOT, REPOSITORY_ROOT / "data" / "job_status.json")
    if not manager.configured:
        raise CloudLifecycleError("缺少 PPIO_API_KEY、TENNISVISION_CLOUD_TOKEN 或镜像配置")

    started = time.time()
    instance_id: str | None = None
    remote_url = ""
    session_id = ""
    final: dict[str, Any] = {}
    released = False
    try:
        print(f"创建临时 {manager.product_id} 实例……", flush=True)
        instance_id = _create_instance(manager)
        print(f"实例已创建：{instance_id}", flush=True)
        remote_url = manager._wait_for_endpoint(instance_id)
        endpoint_ready = time.time()
        print(f"公网端点已就绪（{endpoint_ready - started:.1f}s）", flush=True)
        manager._wait_for_service(remote_url)
        service_ready = time.time()
        print(f"分析服务已就绪（{service_ready - started:.1f}s）", flush=True)

        initial = _request(manager, remote_url, "POST", "/api/live-lab/start")
        session_id = str(initial.get("session_id", ""))
        if not session_id:
            raise CloudLifecycleError("远端没有返回实时实验会话编号")
        print(f"实时实验已启动：{session_id}", flush=True)

        deadline = time.monotonic() + timeout_seconds
        last_print = 0.0
        while time.monotonic() < deadline:
            final = _request(
                manager,
                remote_url,
                "GET",
                f"/api/live-lab/status?session_id={session_id}&after_event=0",
            )
            now = time.monotonic()
            if now - last_print >= 4.0:
                print(
                    "state={state} source={source:.1f}s analysis={analysis:.1f}s "
                    "backlog={backlog:.1f}s frames={frames}".format(
                        state=final.get("state", "?"),
                        source=float(final.get("source_time", 0.0) or 0.0),
                        analysis=float(final.get("analysis_time", 0.0) or 0.0),
                        backlog=float(final.get("backlog_seconds", 0.0) or 0.0),
                        frames=int(final.get("processed_frames", 0) or 0),
                    ),
                    flush=True,
                )
                last_print = now
            if str(final.get("state", "")) in TERMINAL_STATES:
                break
            time.sleep(1.0)
        else:
            raise CloudLifecycleError(f"L40S 实验超过 {timeout_seconds:.0f} 秒仍未结束")

        finished = time.time()
        result = {
            "provider": "PPIO",
            "product_id": manager.product_id,
            "image": manager.image,
            "instance_id": instance_id,
            "cold_start_seconds": endpoint_ready - started,
            "service_ready_seconds": service_ready - started,
            "experiment_wall_seconds": finished - service_ready,
            "total_wall_seconds": finished - started,
            "remote": final,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    finally:
        if session_id and remote_url:
            try:
                _request(
                    manager,
                    remote_url,
                    "POST",
                    "/api/live-lab/stop",
                    {"session_id": session_id},
                )
            except Exception:
                pass
        if instance_id:
            print("正在停止并删除临时 GPU 实例……", flush=True)
            released = manager._release_instance(instance_id)
            print(f"实例释放：{'成功' if released else '失败，请立即人工检查'}", flush=True)
        if instance_id and not released:
            raise CloudLifecycleError(f"临时实例 {instance_id} 未能自动删除")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "l40s_live_benchmark.json",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    result = run_benchmark(args.output.resolve(), args.timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("remote", {}).get("state") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
