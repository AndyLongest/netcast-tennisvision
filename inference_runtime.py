"""Fast, result-preserving video inference helpers.

The detection models are frozen.  This module only groups adjacent frames into a small
GPU batch so CUDA does less launch and preprocessing work; it does not change confidence
thresholds, image sizes, model precision, NMS, tracking, or event classification.
"""
from __future__ import annotations

import os
import queue
import threading
from collections.abc import Iterator
from typing import Any

import torch


def recommended_batch_size(device: str) -> int:
    """Choose a conservative batch for the available device, with an env override."""
    override = os.environ.get("TENNISVISION_INFERENCE_BATCH")
    if override:
        return max(1, min(8, int(override)))
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return 1
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if memory_gib >= 3.5:
        return 4
    if memory_gib >= 2.0:
        return 2
    return 1


def _predict_pair(
    frames: list[Any], ball_model: Any, person_model: Any, *, device: str,
    ball_kwargs: dict[str, Any], person_kwargs: dict[str, Any],
) -> list[tuple[Any, Any, Any]]:
    """Infer one batch, recursively reducing it if CUDA memory is insufficient."""
    try:
        ball_results = ball_model.predict(frames, device=device, verbose=False, **ball_kwargs)
        person_results = person_model.predict(frames, device=device, verbose=False, **person_kwargs)
        return list(zip(frames, ball_results, person_results, strict=False))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(frames) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(frames) // 2
        return (_predict_pair(frames[:middle], ball_model, person_model, device=device,
                              ball_kwargs=ball_kwargs, person_kwargs=person_kwargs) +
                _predict_pair(frames[middle:], ball_model, person_model, device=device,
                              ball_kwargs=ball_kwargs, person_kwargs=person_kwargs))


def _iter_synchronous(
    capture: Any, ball_model: Any, person_model: Any, *, device: str,
    batch_size: int, ball_kwargs: dict[str, Any], person_kwargs: dict[str, Any],
) -> Iterator[tuple[Any, Any, Any]]:
    while True:
        frames = []
        for _ in range(batch_size):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            return
        yield from _predict_pair(frames, ball_model, person_model, device=device,
                                 ball_kwargs=ball_kwargs, person_kwargs=person_kwargs)


def iter_batched_detections(
    capture: Any, ball_model: Any, person_model: Any, *, device: str,
    batch_size: int, ball_kwargs: dict[str, Any], person_kwargs: dict[str, Any],
) -> Iterator[tuple[Any, Any, Any]]:
    """Decode/infer ahead while the caller performs CPU court fitting on the prior batch."""
    items: queue.Queue[Any] = queue.Queue(maxsize=max(2, batch_size * 2))
    finished = object()

    def produce() -> None:
        try:
            for item in _iter_synchronous(
                capture, ball_model, person_model, device=device, batch_size=batch_size,
                ball_kwargs=ball_kwargs, person_kwargs=person_kwargs,
            ):
                items.put(item)
        except BaseException as exc:  # re-raised in the notebook's main thread
            items.put(exc)
        finally:
            items.put(finished)

    worker = threading.Thread(target=produce, name="tennisvision-gpu-prefetch", daemon=True)
    worker.start()
    while True:
        item = items.get()
        if item is finished:
            worker.join()
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def _predict_people(
    frames: list[Any], person_model: Any, *, device: str,
    person_kwargs: dict[str, Any],
) -> list[tuple[Any, Any]]:
    """Run only the person model after RacketVision has produced ball candidates."""
    try:
        results = person_model.predict(frames, device=device, verbose=False, **person_kwargs)
        return list(zip(frames, results, strict=False))
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(frames) == 1:
            raise
        torch.cuda.empty_cache()
        middle = len(frames) // 2
        return (
            _predict_people(frames[:middle], person_model, device=device,
                            person_kwargs=person_kwargs)
            + _predict_people(frames[middle:], person_model, device=device,
                              person_kwargs=person_kwargs)
        )


def iter_batched_person_detections(
    capture: Any, person_model: Any, *, device: str, batch_size: int,
    person_kwargs: dict[str, Any],
) -> Iterator[tuple[Any, Any]]:
    """Decode sequentially and batch the unchanged player-segmentation pass."""
    while True:
        frames = []
        for _ in range(batch_size):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            return
        yield from _predict_people(
            frames, person_model, device=device, person_kwargs=person_kwargs
        )


def iter_sparse_person_detections(
    capture: Any, person_model: Any, *, device: str, batch_size: int,
    person_kwargs: dict[str, Any], stride: int = 2,
) -> Iterator[tuple[Any, Any, Any | None, float, bool]]:
    """Infer slow-moving players on keyframes while yielding every native video frame.

    The ball path is not involved.  Each skipped frame reuses the immediately preceding
    keyframe segmentation for masking and player geometry.  Reading and output order stay
    native-rate, and stride=1 is exactly the ordinary batched person path.
    """
    stride = max(1, int(stride))
    if stride == 1:
        for frame, result in iter_batched_person_detections(
            capture, person_model, device=device, batch_size=batch_size,
            person_kwargs=person_kwargs,
        ):
            yield frame, result, result, 0.0, True
        return

    frame_index = 0
    previous_result: Any | None = None
    previous_index: int | None = None
    # One extra frame exposes the next keyframe, allowing true before/after interpolation
    # instead of carrying a stale box across each batch boundary.
    chunk_size = max(stride + 1, batch_size * stride + 1)
    while True:
        frames = []
        for _ in range(chunk_size):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if not frames:
            return

        key_offsets = [offset for offset in range(len(frames))
                       if (frame_index + offset) % stride == 0]
        if previous_result is None and 0 not in key_offsets:
            key_offsets.insert(0, 0)
        keyframes = [frames[offset] for offset in key_offsets]
        inferred = _predict_people(
            keyframes, person_model, device=device, person_kwargs=person_kwargs)
        by_offset = {offset: result for offset, (_frame, result)
                     in zip(key_offsets, inferred, strict=False)}
        for offset, frame in enumerate(frames):
            is_keyframe = offset in by_offset
            if is_keyframe:
                previous_result = by_offset[offset]
                previous_index = frame_index + offset
            if previous_result is None:  # defensive; the first frame is always a keyframe
                raise RuntimeError("person inference did not produce an initial keyframe")
            later = [key for key in key_offsets if key > offset]
            next_result = (previous_result if is_keyframe else
                           (by_offset[later[0]] if later else None))
            next_index = (previous_index if is_keyframe else
                          (frame_index + later[0] if later else None))
            alpha = (0.0 if is_keyframe or next_index is None or previous_index is None
                     else (frame_index + offset - previous_index) / (next_index - previous_index))
            yield frame, previous_result, next_result, float(alpha), is_keyframe
        frame_index += len(frames)
