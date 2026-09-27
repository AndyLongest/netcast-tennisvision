"""Provision one temporary L40S, upload an isolated RTMP pose A/B experiment, release it."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import time
import zipfile

from netcast_tennisvision.cloud.ppio_lifecycle import PPIOJobManager
from netcast_tennisvision.paths import REPOSITORY_ROOT as ROOT
from netcast_tennisvision.streaming.live_experiment import _ffmpeg

# An authenticated, single-purpose bootstrap, running only inside our temporary container.
# Chunks are idempotent; no shell command or arbitrary path is accepted over HTTP.
BOOTSTRAP = r"""
import hashlib, hmac, io, json, os, pathlib, subprocess, sys, zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
root = pathlib.Path('/app').resolve()
parts = root / 'pose_parts'
parts.mkdir(exist_ok=True)
child = None
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def reply(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def authorized(self):
        return hmac.compare_digest(self.headers.get('Authorization',''),
            'Bearer ' + os.environ['TENNISVISION_CLOUD_SHARED_SECRET'])
    def do_GET(self):
        if not self.authorized(): return self.reply(403, {})
        result = {'ready': True, 'exit': child.poll() if child else None}
        for name in ('pose_progress', 'pose_results'):
            try: result[name] = json.loads((root / (name+'.json')).read_text())
            except (OSError, ValueError): pass
        if child and child.poll() is not None:
            result['log_tail'] = (root / 'pose_run.log').read_text(errors='replace')[-4000:]
        self.reply(200, result)
    def do_POST(self):
        try: self.post()
        except Exception as error: self.reply(500, {'error':str(error)})
    def post(self):
        global child
        if not self.authorized(): return self.reply(403, {})
        length = int(self.headers.get('Content-Length',0))
        if not 0 < length <= 2*1024*1024: return self.reply(400, {})
        body = self.rfile.read(length)
        if self.path.startswith('/part/'):
            index = self.path.rsplit('/',1)[-1]
            if not index.isdigit() or int(index)>200: return self.reply(400,{})
            (parts / index).write_bytes(body)
            return self.reply(200, {'sha256':hashlib.sha256(body).hexdigest()})
        if self.path != '/run': return self.reply(404,{})
        if child: return self.reply(200, {'started':True})
        config = json.loads(body)
        archive = b''.join((parts / str(i)).read_bytes() for i in range(config['parts']))
        if hashlib.sha256(archive).hexdigest()!=config['sha256']: return self.reply(400,{'error':'archive checksum'})
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for item in z.infolist():
                target = (root / item.filename).resolve()
                if not target.is_relative_to(root) or item.file_size>150*1024*1024:
                    return self.reply(400,{'error':'archive path or size: '+item.filename})
            z.extractall(root)
        env = dict(os.environ, PYTHONPATH='/app/src', MPLBACKEND='Agg')
        child = subprocess.Popen([sys.executable,'tools/benchmark_pose_live_remote.py'],
            cwd=root, env=env, stdout=open(root/'pose_run.log','w'), stderr=subprocess.STDOUT)
        self.reply(200, {'started':True})
HTTPServer(('0.0.0.0',8000), Handler).serve_forever()
"""


def user_environment():
    if os.name != "nt":
        return
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        for name in (
            "PPIO_API_KEY",
            "TENNISVISION_CLOUD_TOKEN",
            "TENNISVISION_PPIO_IMAGE",
            "TENNISVISION_PPIO_CLUSTER_ID",
        ):
            if not os.environ.get(name):
                try:
                    os.environ[name] = winreg.QueryValueEx(key, name)[0]
                except FileNotFoundError:
                    pass


def main():
    user_environment()
    out = ROOT / "outputs/pose_cloud_live"
    out.mkdir(parents=True, exist_ok=True)
    manager = PPIOJobManager(ROOT, out / "unused_status.json")
    manager.product_id = "L40S.22c125g"
    if not manager.configured:
        raise RuntimeError("Cloud credentials missing")
    journal = out / "instance.json"
    if journal.exists():
        old = json.loads(journal.read_text())
        if not old.get("released"):
            raise RuntimeError("An unreleased benchmark instance exists; inspect journal first")
    source = ROOT / "data/history/2023a642c4a4431bb9dfb45494236965/source.mp4"
    clip = out / "source_45_80.mp4"
    if not clip.exists():
        subprocess.run(
            [
                _ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "45",
                "-i",
                str(source),
                "-t",
                "35",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-y",
                str(clip),
            ],
            check=True,
        )
    archive = out / "payload.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in (ROOT / "src").rglob("*.py"):
            z.write(path, path.relative_to(ROOT).as_posix())
        z.write(ROOT / "tools/benchmark_pose_live_remote.py", "tools/benchmark_pose_live_remote.py")
        z.write(ROOT / "models/yolo11n-pose.pt", "pose_model.pt")
        z.write(clip, "pose_source.mp4")
        calibration = json.loads((source.parent / "scene3d.json").read_text())
        z.writestr(
            "pose_calibration.json",
            json.dumps({"court_image_corners": calibration["court_image_corners"]}),
        )
    encoded = base64.b64encode(BOOTSTRAP.encode()).decode()
    command = (
        "cd /app && exec timeout --signal=TERM 1800 python -u -c "
        "\"import base64,os;exec(base64.b64decode(os.environ['POSE_BOOTSTRAP']))\""
    )
    payload = {
        "name": f"netcast-pose-ab-{int(time.time())}",
        "productId": manager.product_id,
        "clusterId": manager.cluster_id,
        "gpuNum": 1,
        "rootfsSize": manager._rootfs_size_for_product(),
        "imageUrl": manager.image,
        "imageAuth": "",
        "imageAuthId": "",
        "ports": "8000/http",
        "envs": [
            {"key": "TENNISVISION_CLOUD_SHARED_SECRET", "value": manager.shared_secret},
            {"key": "POSE_BOOTSTRAP", "value": encoded},
            {
                "key": "TENNISVISION_ZLM_HOST",
                "value": os.environ.get("TENNISVISION_ZLM_HOST", "39.108.95.121"),
            },
            {"key": "TENNISVISION_ZLM_WEBRTC_ORIGIN", "value": "https://zlmediakit.moralspace.com"},
        ],
        "tools": [],
        "command": "bash -lc " + __import__("shlex").quote(command),
        "entrypoint": "",
        "networkStorages": [],
        "kind": "gpu",
        "billingMode": "onDemand",
        "minCudaVersion": "12.8",
    }
    instance = None
    try:
        print("Provisioning", manager.product_id, manager.image.rsplit(":", 1)[-1], flush=True)
        response = manager._provider_request("POST", "/gpu/instance/create", payload)
        instance = manager._find_string(response, ("instanceId", "id"))
        if not instance:
            raise RuntimeError("Provider returned no instance ID")
        journal.write_text(json.dumps({"instance_id": instance, "released": False}))
        print("Instance", instance, flush=True)
        endpoint = manager._wait_for_endpoint(instance)

        def request(method, path, body=None):
            detail = ""
            for _attempt in range(5):
                try:
                    status, data = manager._remote_request(endpoint, method, path, body=body)
                    if status == 200:
                        return data
                    detail = str(status) + ": " + str(data.get("error", ""))
                except (OSError, TimeoutError) as error:
                    detail = str(error)
                time.sleep(2)
            raise RuntimeError(f"Benchmark endpoint failed: {method} {path}: {detail}")

        request("GET", "/")
        data = archive.read_bytes()
        chunk_size = 1024 * 1024
        count = (len(data) + chunk_size - 1) // chunk_size
        for index in range(count):
            chunk = data[index * chunk_size : (index + 1) * chunk_size]
            answer = request("POST", f"/part/{index}", chunk)
            if answer.get("sha256") != hashlib.sha256(chunk).hexdigest():
                raise RuntimeError("Upload checksum mismatch")
        print("Uploaded", len(data), "bytes; starting ABBA", flush=True)
        request(
            "POST",
            "/run",
            json.dumps({"parts": count, "sha256": hashlib.sha256(data).hexdigest()}).encode(),
        )
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            result = request("GET", "/")
            (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps(result.get("pose_progress", {})), flush=True)
            if result.get("exit") is not None:
                if result["exit"] != 0:
                    raise RuntimeError("Remote benchmark failed; see result.json log_tail")
                print("ABBA complete", flush=True)
                break
            time.sleep(10)
        else:
            raise TimeoutError("Cloud benchmark timed out")
    finally:
        if instance:
            released = manager._release_instance(instance)
            journal.write_text(json.dumps({"instance_id": instance, "released": released}))
            print("GPU released:", released, flush=True)
            if not released:
                raise RuntimeError("Cloud cleanup failed; inspect instance journal")


if __name__ == "__main__":
    main()
