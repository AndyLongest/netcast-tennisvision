"""Opt-in, process-local model sharing for concurrent live streams.

Each stream still owns its temporal window, tracker and events.  Only frozen model
inference is shared.  The single-stream call remains synchronous and unbatched with
other streams; concurrent callers may be combined for at most a few milliseconds.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _BallRequest:
    inputs: torch.Tensor
    result: Future[torch.Tensor]


@dataclass
class _PersonRequest:
    frames: list[Any]
    options: dict[str, Any]
    result: Future[Any]


class LiveInferencePool:
    def __init__(self, *, max_batch: int | None = None,
                 gather_ms: float | None = None) -> None:
        self._state_lock = threading.Lock()
        self._ball_lock = threading.Lock()
        self._person_lock = threading.Lock()
        self._requests: queue.Queue[_BallRequest] = queue.Queue()
        self._person_requests: queue.Queue[_PersonRequest] = queue.Queue()
        self._max_batch = max_batch or 16
        self._person_max_batch = 16
        self._gather_seconds = (3.0 if gather_ms is None else gather_ms) / 1000.0
        self._person_gather_seconds = (10.0 if gather_ms is None else gather_ms) / 1000.0
        self._fixed_configuration = max_batch is not None or gather_ms is not None
        self._active = 0
        self._ball_model: Any = None
        self._person_model: Any = None
        self._device: torch.device | None = None
        self._worker: threading.Thread | None = None
        self._person_worker: threading.Thread | None = None
        self._ball_calls = 0
        self._person_calls = 0
        self._ball_frames = 0
        self._person_frames = 0

    @property
    def active_count(self) -> int:
        with self._state_lock:
            return self._active

    def acquire(
        self,
        prepare: Callable[[], tuple[Any, Any]],
        device: torch.device,
    ) -> None:
        with self._state_lock:
            if self._active == 0:
                if not self._fixed_configuration:
                    self._max_batch = max(1, min(64, int(os.environ.get(
                        "TENNISVISION_LIVE_SHARED_MAX_BATCH", "16"
                    ))))
                    self._person_max_batch = max(1, min(32, int(os.environ.get(
                        "TENNISVISION_LIVE_SHARED_PERSON_BATCH", "16"
                    ))))
                    self._gather_seconds = max(0.0, min(0.03, float(os.environ.get(
                        "TENNISVISION_LIVE_SHARED_GATHER_MS", "3"
                    )) / 1000.0))
                    self._person_gather_seconds = max(0.0, min(0.03, float(os.environ.get(
                        "TENNISVISION_LIVE_SHARED_PERSON_GATHER_MS", "10"
                    )) / 1000.0))
                self._ball_model, self._person_model = prepare()
                self._device = device
            elif self._device != device:
                raise RuntimeError("共享直播推理不能混用不同设备")
            self._active += 1
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._serve_ball, name="live-shared-ball", daemon=True,
                )
                self._worker.start()
            if self._person_worker is None or not self._person_worker.is_alive():
                self._person_worker = threading.Thread(
                    target=self._serve_people, name="live-shared-people", daemon=True,
                )
                self._person_worker.start()

    def release(self) -> None:
        with self._state_lock:
            if self._active <= 0:
                raise RuntimeError("共享直播推理释放次数超过获取次数")
            self._active -= 1
            if self._active == 0:
                self._ball_model = None
                self._person_model = None
                self._device = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _forward_ball(self, inputs: torch.Tensor) -> torch.Tensor:
        with self._ball_lock, torch.inference_mode():
            self._ball_calls += 1
            self._ball_frames += len(inputs)
            if self._device is not None and self._device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    return self._ball_model(inputs)
            return self._ball_model(inputs)

    def infer_ball(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.active_count <= 1:
            return self._forward_ball(inputs)
        if len(inputs) > self._max_batch:
            return torch.cat(
                [self.infer_ball(part) for part in inputs.split(self._max_batch, dim=0)],
                dim=0,
            )
        request = _BallRequest(inputs, Future())
        self._requests.put(request)
        return request.result.result(timeout=30)

    def _serve_ball(self) -> None:
        carry: _BallRequest | None = None
        while True:
            first = carry or self._requests.get()
            carry = None
            requests = [first]
            size = len(first.inputs)
            deadline = time.monotonic() + self._gather_seconds
            while size < self._max_batch:
                try:
                    following = self._requests.get(timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if size + len(following.inputs) > self._max_batch:
                    carry = following
                    break
                requests.append(following)
                size += len(following.inputs)
            try:
                merged = torch.cat([item.inputs for item in requests], dim=0)
                outputs = self._forward_ball(merged)
                for item, output in zip(
                    requests, outputs.split([len(item.inputs) for item in requests], dim=0),
                    strict=True,
                ):
                    item.result.set_result(output)
            except BaseException as error:
                for item in requests:
                    item.result.set_exception(error)

    def predict_people(self, frames: list[Any], **options: Any) -> Any:
        if self.active_count <= 1 or len(frames) > self._person_max_batch:
            return self._forward_people(frames, options)
        request = _PersonRequest(frames, options, Future())
        self._person_requests.put(request)
        return request.result.result(timeout=30)

    def _forward_people(self, frames: list[Any], options: dict[str, Any]) -> Any:
        # Ultralytics keeps mutable predictor state, so only one worker calls it.
        with self._person_lock:
            self._person_calls += 1
            self._person_frames += len(frames)
            return self._person_model.predict(frames, **options)

    def _serve_people(self) -> None:
        carry: _PersonRequest | None = None
        while True:
            first = carry or self._person_requests.get()
            carry = None
            requests = [first]
            size = len(first.frames)
            deadline = time.monotonic() + self._person_gather_seconds
            while size < self._person_max_batch:
                try:
                    following = self._person_requests.get(
                        timeout=max(0.0, deadline - time.monotonic())
                    )
                except queue.Empty:
                    break
                if (size + len(following.frames) > self._person_max_batch
                        or following.options != first.options):
                    carry = following
                    break
                requests.append(following)
                size += len(following.frames)
            try:
                frames = [frame for item in requests for frame in item.frames]
                outputs = self._forward_people(frames, first.options)
                offset = 0
                for item in requests:
                    item.result.set_result(outputs[offset:offset + len(item.frames)])
                    offset += len(item.frames)
            except BaseException as error:
                for item in requests:
                    item.result.set_exception(error)

    def statistics(self) -> dict[str, int]:
        with self._ball_lock, self._person_lock:
            return {
                "ball_calls": self._ball_calls,
                "ball_frames": self._ball_frames,
                "person_calls": self._person_calls,
                "person_frames": self._person_frames,
            }


shared_live_inference = LiveInferencePool()
