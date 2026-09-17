"""End-to-end pseudo-live experiment for the internal comparison page.

The experiment deliberately uses the real media path and real inference path:

``camera simulator -> RTMP -> ZLMediaKit -> GPU pull -> frozen models -> fixed-lag event``

Court calibration and the median background are prepared before the stream starts.  That
matches a fixed venue camera: session setup is not repeated during a rally.  No report or
ball-candidate cache is read by the online worker.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch

from netcast_tennisvision.events.landing_event_detector import detect_landing_impulses
from netcast_tennisvision.events.line_call import classify_line_call
from netcast_tennisvision.paths import REPOSITORY_ROOT
from netcast_tennisvision.tracking.geometry import inside_player_body
from netcast_tennisvision.tracking.world_tracker import track_ball_persistent
from netcast_tennisvision.vision.racketvision import (
    DEFAULT_THRESHOLD,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    SEQUENCE_LENGTH,
    _decode_candidates,
    _median_background,
    load_model,
)

ROOT = REPOSITORY_ROOT
DEMO_VIDEO = ROOT / "assets" / "demo" / "demo.mp4"
BALL_WEIGHT = ROOT / "models" / "racketvision_balltrack_state_v1.pt"
PERSON_WEIGHT = ROOT / "models" / "yolo11n-seg.pt"
RUNTIME_CONFIG = ROOT / "data" / "live_lab_config.json"
LAST_RESULT = ROOT / "data" / "live_lab_last.json"
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
DEFAULT_LIVE_BATCH_SIZE = 16
DEFAULT_PERSON_STRIDE = 4
WORLD_CORNERS = np.asarray(
    [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
    dtype=np.float32,
)


def compare_landing_events(
    online: list[dict[str, Any]], reference: list[dict[str, Any]], *, fps: float
) -> dict[str, Any]:
    """Greedily match causal events to the offline reviewed baseline for evaluation only."""
    available = set(range(len(online)))
    matches: list[dict[str, Any]] = []
    for expected in reference:
        expected_frame = float(expected.get("touchdown_frame_f", expected.get("frame", 0)))
        candidates: list[tuple[float, int, float, float]] = []
        for index in available:
            actual = online[index]
            frame_delta = abs(float(actual.get("touchdown_frame_f", actual["frame"])) - expected_frame)
            distance = float(np.hypot(
                float(actual["x"]) - float(expected["x"]),
                float(actual["y"]) - float(expected["y"]),
            ))
            if frame_delta <= max(12.0, 0.45 * fps) and distance <= 2.0:
                candidates.append((frame_delta / max(fps, 1e-9) + distance, index,
                                   frame_delta, distance))
        if not candidates:
            continue
        _score, index, frame_delta, distance = min(candidates)
        available.remove(index)
        matches.append({
            "reference_frame": expected_frame,
            "online_frame": float(online[index]["frame"]),
            "time_error_ms": 1000.0 * frame_delta / fps,
            "position_error_m": distance,
        })
    matched = len(matches)
    return {
        "online_events": len(online),
        "reference_events": len(reference),
        "matched_events": matched,
        "recall": matched / len(reference) if reference else None,
        "precision": matched / len(online) if online else None,
        "matches": matches,
    }


def _ffmpeg() -> str:
    # Prefer the validated full Gyan build.  Some vendor tools place a reduced binary
    # earlier on PATH which can decode video but lacks libx264/preset support.
    packages = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    candidates = sorted(packages.glob("Gyan.FFmpeg.Shared_*/ffmpeg-*/bin/ffmpeg.exe"), reverse=True)
    if candidates:
        return str(candidates[0])
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    candidate = Path("E:/Program File/ffmpeg.exe")
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError("本机未找到 ffmpeg，无法发起伪直播")


def _runtime_config() -> dict[str, Any]:
    try:
        payload = json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _media_host() -> str:
    host = os.environ.get("TENNISVISION_ZLM_HOST", "").strip()
    if not host:
        host = str(_runtime_config().get("zlm_host", "")).strip()
    if not host or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-" for character in host):
        raise RuntimeError("尚未配置 ZLMediaKit 地址（TENNISVISION_ZLM_HOST）")
    return host


def _webrtc_origin(host: str) -> str:
    origin = os.environ.get("TENNISVISION_ZLM_WEBRTC_ORIGIN", "").strip()
    if not origin:
        origin = str(_runtime_config().get("zlm_webrtc_origin", "")).strip()
    if not origin:
        origin = f"http://{host}"
    parsed = urlparse(origin)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.path not in {"", "/"}:
        raise RuntimeError("ZLMediaKit WebRTC 信令地址无效")
    return origin.rstrip("/")


def _camera_corners(width: int, height: int) -> np.ndarray:
    """Load the latest verified fixed-camera profile without decoding its JPEG."""
    profile_path = ROOT / "data" / "camera_profiles.json"
    try:
        profiles = json.loads(profile_path.read_text(encoding="utf-8")).get("profiles", [])
        profile = max(profiles, key=lambda item: int(item.get("last_used_at", 0)))
        normalized = np.asarray(profile["corners"], dtype=np.float32)
        if normalized.shape == (4, 2) and np.isfinite(normalized).all():
            return normalized * np.asarray([width, height], dtype=np.float32)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    # The bundled sample is a checked-in fixed-camera regression asset.  These are the
    # reviewed demo calibration corners, not per-frame detections or landing labels.
    return np.asarray(
        [[291.1, 529.3], [1069.3, 532.4], [804.4, 117.3], [543.6, 118.2]],
        dtype=np.float32,
    ) * np.asarray([width / 1280.0, height / 720.0], dtype=np.float32)


def _person_boxes(result: Any) -> np.ndarray:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None:
        return np.empty((0, 4), dtype=np.float32)
    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
    confidence = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    keep = (classes == 0) & (confidence >= 0.25)
    return np.asarray(xyxy[keep], dtype=np.float32).reshape(-1, 4)


def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


class LiveExperimentSession:
    """Own one RTMP consumer and online inference worker.

    Local-only experiments may still create their own producer.  Production-shaped
    cloud experiments receive an external stream name and never receive the source file.
    """

    def __init__(
        self,
        source: Path | None = DEMO_VIDEO,
        *,
        external_stream_name: str = "",
        fps_hint: float = 30.0,
        source_name: str = "camera",
    ) -> None:
        self.id = uuid.uuid4().hex
        self.source = source
        self.external_stream_name = external_stream_name
        self.fps_hint = fps_hint if fps_hint > 0 else 30.0
        self.source_name = source.name if source is not None else source_name
        self.created_at = time.time()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"live-lab-{self.id[:8]}", daemon=True)
        self._producer: subprocess.Popen[bytes] | None = None
        self._remote_webrtc_url = ""
        self._latest_jpeg: bytes | None = None
        self._events: list[dict[str, Any]] = []
        self._status: dict[str, Any] = {
            "session_id": self.id,
            "state": "preparing",
            "stage": "准备固定机位与冻结模型",
            "source_time": 0.0,
            "analysis_time": 0.0,
            "processed_frames": 0,
            "detector_frames": 0,
            "events": [],
            "execution": "真实 RTMP / ZLMediaKit / 在线推理",
            "source_name": self.source_name,
        }

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._producer is not None and self._producer.poll() is None:
            self._producer.terminate()

    def snapshot(self, after_event: int = 0) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._status)
            payload["events"] = [dict(item) for item in self._events[after_event:]]
            payload["event_cursor"] = len(self._events)
            return payload

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def exchange_webrtc_offer(self, offer_sdp: str) -> dict[str, Any]:
        """Proxy only the small SDP handshake; WebRTC media remains browser-to-ZLM."""
        if not self._remote_webrtc_url:
            raise RuntimeError("WebRTC 信令尚未准备完成")
        if not offer_sdp.strip() or len(offer_sdp.encode("utf-8")) > 256 * 1024:
            raise ValueError("WebRTC SDP 内容无效")
        request = Request(
            self._remote_webrtc_url,
            data=offer_sdp.encode("utf-8"),
            headers={"Content-Type": "text/plain;charset=utf-8"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("ZLMediaKit 返回了无效的信令响应")
        answer_sdp = str(payload.get("sdp", ""))
        self._update(
            webrtc_offer_count=int(self.snapshot().get("webrtc_offer_count", 0)) + 1,
            webrtc_answer_code=payload.get("code"),
            webrtc_answer_has_video="m=video" in answer_sdp,
        )
        return payload

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._status.update(values)

    def _fail(self, error: BaseException) -> None:
        self._update(state="error", stage="实验中断", error=str(error))

    def _emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(event)
            self._status["latest_event_delay_ms"] = event["end_to_end_delay_ms"]

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            external_stream = bool(self.external_stream_name)
            if not BALL_WEIGHT.is_file() or not PERSON_WEIGHT.is_file():
                raise RuntimeError("冻结模型文件不完整")
            if not external_stream:
                if self.source is None or not self.source.is_file():
                    raise RuntimeError("Demo 视频不存在")
                metadata = cv2.VideoCapture(str(self.source))
                width = int(metadata.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(metadata.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(metadata.get(cv2.CAP_PROP_FPS) or 30.0)
                total_frames = int(metadata.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata.release()
                if min(width, height, total_frames) <= 0:
                    raise RuntimeError("Demo 视频规格无效")
                background = _median_background(self.source, total_frames)
                background_channels = np.moveaxis(
                    background.astype(np.float32) / 255.0, -1, 0
                )
            else:
                width = height = total_frames = 0
                fps = self.fps_hint
                background_channels = None

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ball_model = load_model(BALL_WEIGHT, device)
            from ultralytics import YOLO

            person_model = YOLO(str(PERSON_WEIGHT))
            if external_stream:
                # Compile CUDA kernels and initialize both frozen models before telling
                # the camera simulator to publish. Otherwise the first inference call
                # can stall for several seconds and discard the opening rally even
                # though steady-state throughput is faster than real time.
                warm_batch = _bounded_int_env(
                    "TENNISVISION_LIVE_BATCH_SIZE", DEFAULT_LIVE_BATCH_SIZE, 1, 64
                )
                warm_inputs = torch.zeros(
                    (
                        warm_batch,
                        3 * (SEQUENCE_LENGTH + 1),
                        MODEL_HEIGHT,
                        MODEL_WIDTH,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            ball_model(warm_inputs)
                    else:
                        ball_model(warm_inputs)
                person_model.predict(
                    np.zeros((720, 1280, 3), dtype=np.uint8),
                    device=str(device),
                    verbose=False,
                    imgsz=640,
                    conf=0.25,
                    classes=[0],
                )
                del warm_inputs
            host = _media_host()
            webrtc_origin = _webrtc_origin(host)
            stream_name = self.external_stream_name or f"netcast-{self.id[:12]}"
            stream_url = f"rtmp://{host}:1935/live/{stream_name}"
            self._remote_webrtc_url = (
                f"{webrtc_origin}/index/api/webrtc"
                f"?app=live&stream={stream_name}&type=play"
            )
            self._update(state="awaiting_stream" if external_stream else "connecting",
                         stage="模型已就绪，等待摄像头推流" if external_stream else "正在建立 RTMP / ZLMediaKit 链路", fps=fps,
                         total_frames=total_frames, media_host=host, device=str(device),
                         stream_id=stream_name,
                         fmp4_playback_url=(
                             f"{webrtc_origin}/live/{stream_name}.live.mp4"
                         ),
                         webrtc_signaling_url=(
                             f"/api/live-lab/webrtc-offer?session_id={self.id}"
                         ))
            producer_started = time.time()
            if not external_stream:
                command = [
                    _ffmpeg(), "-hide_banner", "-loglevel", "error", "-re", "-i", str(self.source),
                    "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "ultrafast",
                    "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", str(max(1, round(fps))),
                    "-bf", "0", "-f", "flv", stream_url,
                ]
                self._producer = subprocess.Popen(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
                )

            deadline = time.monotonic() + (120.0 if external_stream else 12.0)
            while time.monotonic() < deadline and not self._stop.is_set():
                capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                if capture.isOpened():
                    ok, first_frame = capture.read()
                    if ok:
                        break
                capture.release()
                capture = None
                if self._producer is not None and self._producer.poll() is not None:
                    detail = (self._producer.stderr.read() if self._producer.stderr else b"").decode(
                        "utf-8", errors="replace"
                    )
                    raise RuntimeError(f"RTMP 推流失败：{detail.strip() or '媒体服务拒绝连接'}")
                time.sleep(0.25)
            else:
                if self._stop.is_set():
                    self._update(state="stopped", stage="实验已停止")
                    return
                raise RuntimeError("等待摄像头推流超时，未能从 ZLMediaKit 拉回画面")

            prefetched_frames = [first_frame]
            if external_stream:
                # A fixed venue camera is normally online before play begins.  Keep the
                # first short causal window both for background initialization and later
                # inference, so the stream is never scanned ahead and no source frame is
                # silently discarded.
                for _ in range(15):
                    ok, warmup_frame = capture.read()
                    if not ok:
                        break
                    prefetched_frames.append(warmup_frame)
                height, width = first_frame.shape[:2]
                reported_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                if reported_fps > 0:
                    fps = reported_fps
                resized_background = [
                    cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
                    for frame in prefetched_frames
                ]
                background = np.median(np.stack(resized_background), axis=0).astype(np.uint8)
                background_channels = np.moveaxis(
                    background.astype(np.float32) / 255.0, -1, 0
                )
                producer_started = time.time() - len(prefetched_frames) / max(fps, 1e-9)

            corners = _camera_corners(width, height)
            image_to_world = cv2.getPerspectiveTransform(corners, WORLD_CORNERS)
            world_to_image = np.linalg.inv(image_to_world)
            net_point = cv2.perspectiveTransform(
                np.asarray([[[COURT_WIDTH_M / 2.0, COURT_LENGTH_M / 2.0]]], np.float32),
                world_to_image.astype(np.float32),
            )[0, 0]
            spatial = min(width / 1280.0, height / 720.0)

            self._update(state="running", stage="真实链路在线分析中", started_at=producer_started)
            frames_meta: list[dict[str, Any]] = []
            frame_window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
            pending_inputs: list[np.ndarray] = []
            pending_frames: list[np.ndarray] = []
            pending_source_indices: list[int] = []
            live_batch_size = _bounded_int_env(
                "TENNISVISION_LIVE_BATCH_SIZE", DEFAULT_LIVE_BATCH_SIZE, 1, 64
            )
            person_stride = _bounded_int_env(
                "TENNISVISION_LIVE_PERSON_STRIDE", DEFAULT_PERSON_STRIDE, 1, 12
            )
            last_person_boxes = np.empty((0, 4), dtype=np.float32)
            emitted_frames: set[int] = set()
            last_event_frame = -10_000
            rally_id = 0
            last_preview = 0.0
            peak_backlog = 0.0

            def process_batch() -> None:
                nonlocal last_event_frame, rally_id, last_person_boxes
                if not pending_inputs:
                    return
                inputs = torch.from_numpy(np.stack(pending_inputs)).to(device, non_blocking=True)
                with torch.inference_mode():
                    if device.type == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            heatmaps = ball_model(inputs)
                    else:
                        heatmaps = ball_model(inputs)
                decoded = [
                    _decode_candidates(
                        heatmap, DEFAULT_THRESHOLD, width / MODEL_WIDTH, height / MODEL_HEIGHT,
                        max_candidates=1,
                    )
                    for heatmap in heatmaps.float().cpu().numpy()
                ]
                first_index = pending_source_indices[0]
                sampled_offsets = [
                    offset
                    for offset in range(len(pending_frames))
                    if pending_source_indices[offset] % person_stride == 0
                ]
                if not sampled_offsets and last_person_boxes.size == 0:
                    sampled_offsets = [0]
                sampled_results = person_model.predict(
                    [pending_frames[offset] for offset in sampled_offsets],
                    device=str(device), verbose=False, imgsz=640, conf=0.25, classes=[0],
                ) if sampled_offsets else []
                boxes_by_offset = {
                    offset: _person_boxes(result)
                    for offset, result in zip(sampled_offsets, sampled_results, strict=True)
                }
                active_boxes = last_person_boxes
                for offset, candidates in enumerate(decoded):
                    if offset in boxes_by_offset:
                        active_boxes = boxes_by_offset[offset]
                    source_index = pending_source_indices[offset]
                    while len(frames_meta) < source_index:
                        frames_meta.append({
                            "candidates": [],
                            "person_boxes": active_boxes.copy(),
                            "is_court": True,
                            "corners": corners,
                            "M": world_to_image,
                            "M_inv": image_to_world,
                            "net_y_px": float(net_point[1]),
                            "source_frame": len(frames_meta),
                            "dropped_before_inference": True,
                        })
                    frames_meta.append({
                        "candidates": candidates,
                        # A fixed-camera player cannot teleport between adjacent frames.
                        # Hold the latest native-frame detection causally; do not
                        # interpolate future evidence or alter any ball input frame.
                        "person_boxes": active_boxes.copy(),
                        "is_court": True,
                        "corners": corners,
                        "M": world_to_image,
                        "M_inv": image_to_world,
                        "net_y_px": float(net_point[1]),
                        "source_frame": source_index,
                    })
                last_person_boxes = active_boxes
                pending_inputs.clear()
                pending_frames.clear()
                pending_source_indices.clear()

                # Re-evaluate only a bounded eight-second fixed-lag window.  Re-running
                # the whole match after every four frames is quadratic and is not how a
                # streaming service would retain state.
                window_size = max(120, round(8.0 * fps))
                window_start = max(0, len(frames_meta) - window_size)
                tracking_window = [dict(item) for item in frames_meta[window_start:]]
                track_ball_persistent(
                    tracking_window, fps=fps, spatial=spatial, speed_scale=spatial,
                    frame_size=(width, height), play_mode="singles",
                )
                fixed_lag = max(10, round(0.35 * fps))
                latest_decidable = len(frames_meta) - 1 - fixed_lag
                if latest_decidable < 0:
                    return
                impulses = detect_landing_impulses(
                    tracking_window, radius=8, min_score=0.52,
                    min_gap_frames=max(10, round(0.38 * fps)),
                )
                for impulse in impulses:
                    local_frame = int(impulse["frame"])
                    timeline_frame = window_start + local_frame
                    if timeline_frame > latest_decidable:
                        continue
                    meta = tracking_window[local_frame]
                    event_frame = int(meta.get("source_frame", timeline_frame))
                    if event_frame in emitted_frames:
                        continue
                    point = meta.get("ball_px")
                    if point is None or impulse.get("impulse_y_px_frame", 0.0) >= -0.15 * spatial:
                        continue
                    if inside_player_body(meta, np.asarray(point, dtype=float), spatial):
                        continue
                    world = cv2.perspectiveTransform(
                        np.asarray([[point]], dtype=np.float32), image_to_world,
                    )[0, 0]
                    x, y = map(float, world)
                    # Keep a bounded apron so a genuine out ball remains observable.
                    if not (-1.8 <= x <= COURT_WIDTH_M + 1.8 and -2.5 <= y <= COURT_LENGTH_M + 2.5):
                        continue
                    if event_frame - last_event_frame < max(12, round(0.55 * fps)):
                        continue
                    if event_frame - last_event_frame > round(4.0 * fps):
                        rally_id += 1
                    last_event_frame = event_frame
                    emitted_frames.add(event_frame)
                    line = classify_line_call(x, y, uncertainty_m=0.12)
                    # With a fixed baseline camera, a far-half landing was struck by the
                    # near-side player and vice versa. Stable A/B identity can be layered
                    # on this side assignment after the live identity branch is validated.
                    player_id = "A" if y >= COURT_LENGTH_M / 2.0 else "B"
                    now = time.time()
                    event_time = event_frame / fps
                    self._emit({
                        "id": len(self._events), "frame": event_frame,
                        "touchdown_frame_f": float(event_frame),
                        "decision_frame": int(frames_meta[-1].get("source_frame", len(frames_meta) - 1)),
                        "t": event_time, "x": x, "y": y,
                        "zone": "Out" if line.call == "out" else "在线候选",
                        "line_call": line.call, "player_id": player_id, "rally_id": rally_id,
                        "impulse_score": float(impulse["score"]),
                        "emitted_at": now,
                        "end_to_end_delay_ms": max(0.0, (now - producer_started - event_time) * 1000.0),
                    })
                inferred_frames = int(self.snapshot().get("detector_frames", 0)) + len(decoded)
                self._update(detector_frames=inferred_frames, tracker_frames=len(frames_meta),
                             batch_first_frame=first_index)

            # OpenCV already returns an independent ndarray for every decoded frame.
            # Queue it directly: the previous implementation JPEG-encoded every frame
            # and immediately decoded it again, wasting CPU while changing no model input.
            queue_capacity = max(4, round(0.75 * fps))
            frame_queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(
                maxsize=queue_capacity
            )
            reader_done = threading.Event()
            ingested_frames = 0
            dropped_frames = 0

            def ingest() -> None:
                nonlocal ingested_frames, last_preview, dropped_frames

                def publish(item: tuple[int, np.ndarray]) -> None:
                    nonlocal dropped_frames
                    try:
                        frame_queue.put_nowait(item)
                    except queue.Full:
                        try:
                            frame_queue.get_nowait()
                            dropped_frames += 1
                        except queue.Empty:
                            pass
                        frame_queue.put_nowait(item)

                incoming_index = 0
                try:
                    for incoming in prefetched_frames:
                        if self._stop.is_set():
                            break
                        publish((incoming_index, incoming))
                        ingested_frames = incoming_index + 1
                        now = time.time()
                        if now - last_preview >= 0.08:
                            encoded_ok, encoded = cv2.imencode(
                                ".jpg", incoming, [cv2.IMWRITE_JPEG_QUALITY, 88]
                            )
                            if encoded_ok:
                                with self._lock:
                                    self._latest_jpeg = encoded.tobytes()
                                last_preview = now
                        self._update(
                            source_time=ingested_frames / fps,
                            ingested_frames=ingested_frames,
                            queued_frames=max(0, ingested_frames - len(frames_meta)),
                            dropped_frames=dropped_frames,
                        )
                        incoming_index += 1
                    while not self._stop.is_set():
                        ok, incoming = capture.read()
                        if not ok:
                            break
                        publish((incoming_index, incoming))
                        ingested_frames = incoming_index + 1
                        now = time.time()
                        if now - last_preview >= 0.08:
                            encoded_ok, encoded = cv2.imencode(
                                ".jpg", incoming, [cv2.IMWRITE_JPEG_QUALITY, 88]
                            )
                            if encoded_ok:
                                with self._lock:
                                    self._latest_jpeg = encoded.tobytes()
                                last_preview = now
                        self._update(
                            source_time=ingested_frames / fps,
                            ingested_frames=ingested_frames,
                            queued_frames=max(0, ingested_frames - len(frames_meta)),
                            dropped_frames=dropped_frames,
                        )
                        incoming_index += 1
                finally:
                    reader_done.set()
                    frame_queue.put(None)

            reader = threading.Thread(target=ingest, name=f"live-ingest-{self.id[:8]}", daemon=True)
            reader.start()
            while not self._stop.is_set():
                item = frame_queue.get()
                if item is None:
                    break
                source_index, current_frame = item
                resized = cv2.resize(current_frame, (MODEL_WIDTH, MODEL_HEIGHT))
                frame_window.append(np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0))
                sequence = list(frame_window)
                while len(sequence) < SEQUENCE_LENGTH:
                    sequence.insert(0, sequence[0])
                pending_inputs.append(np.concatenate([background_channels, *sequence], axis=0))
                pending_frames.append(current_frame)
                pending_source_indices.append(source_index)
                if len(pending_inputs) >= live_batch_size:
                    process_batch()

                update_now = time.time()
                processed = len(frames_meta)
                analysis_time = processed / fps
                backlog_seconds = max(0.0, (ingested_frames - processed) / fps)
                peak_backlog = max(peak_backlog, backlog_seconds)
                self._update(
                    source_time=ingested_frames / fps, analysis_time=analysis_time,
                    processed_frames=processed,
                    backlog_seconds=backlog_seconds, peak_backlog_seconds=peak_backlog,
                    ingest_delay_ms=max(
                        0.0, (update_now - producer_started - analysis_time) * 1000.0
                    ),
                    live_batch_size=live_batch_size,
                    person_stride=person_stride,
                    queue_capacity=queue_capacity,
                    dropped_frames=dropped_frames,
                )

            process_batch()
            reader_done.wait(timeout=2)
            if self._stop.is_set():
                self._update(state="stopped", stage="实验已停止")
            else:
                elapsed = time.time() - producer_started
                comparison = None
                if self.source is not None and self.source.resolve() == DEMO_VIDEO.resolve():
                    reference_payload = json.loads(
                        (ROOT / "assets" / "demo" / "scene3d.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    comparison = compare_landing_events(
                        self._events, list(reference_payload.get("bounces", [])), fps=fps
                    )
                final = {
                    "state": "complete", "stage": "真实端到端实验完成",
                    "source_time": ingested_frames / fps,
                    "analysis_time": len(frames_meta) / fps,
                    "ingested_frames": ingested_frames,
                    "processed_frames": len(frames_meta),
                    "elapsed_seconds": elapsed,
                    "realtime_factor": elapsed / max(ingested_frames / fps, 1e-9),
                    "peak_backlog_seconds": peak_backlog,
                    "comparison": comparison,
                }
                self._update(**final)
                LAST_RESULT.parent.mkdir(parents=True, exist_ok=True)
                LAST_RESULT.write_text(
                    json.dumps({**self.snapshot(), "events": self._events}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except BaseException as error:
            if self._stop.is_set():
                self._update(state="stopped", stage="实验已停止", error=None)
            else:
                self._fail(error)
        finally:
            if capture is not None:
                capture.release()
            if self._producer is not None and self._producer.poll() is None:
                self._producer.terminate()
                try:
                    self._producer.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._producer.kill()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LiveExperimentManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: LiveExperimentSession | None = None

    def start_demo(self) -> LiveExperimentSession:
        return self.start_source(DEMO_VIDEO)

    def start_source(self, source: Path) -> LiveExperimentSession:
        with self._lock:
            if self._session is not None and self._session.snapshot().get("state") in {
                "preparing", "connecting", "running",
            }:
                return self._session
            if not source.is_file():
                raise RuntimeError("实时实验素材不存在")
            self._session = LiveExperimentSession(source)
            self._session.start()
            return self._session

    def start_external_stream(
        self,
        stream_name: str,
        *,
        fps_hint: float = 30.0,
        source_name: str = "camera",
    ) -> LiveExperimentSession:
        """Analyze a camera-shaped RTMP stream without receiving its source file."""
        if not stream_name or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in stream_name
        ):
            raise RuntimeError("实时流名称无效")
        with self._lock:
            if self._session is not None and self._session.snapshot().get("state") in {
                "preparing", "awaiting_stream", "connecting", "running",
            }:
                return self._session
            self._session = LiveExperimentSession(
                None,
                external_stream_name=stream_name,
                fps_hint=fps_hint,
                source_name=source_name,
            )
            self._session.start()
            return self._session

    def get(self, session_id: str) -> LiveExperimentSession | None:
        with self._lock:
            if self._session is not None and self._session.id == session_id:
                return self._session
            return None

    def stop(self, session_id: str) -> bool:
        session = self.get(session_id)
        if session is None:
            return False
        session.stop()
        return True


live_experiments = LiveExperimentManager()
