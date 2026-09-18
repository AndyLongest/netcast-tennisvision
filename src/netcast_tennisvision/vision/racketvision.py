"""Production inference for the public RacketVision BallTrack checkpoint.

The architecture and preprocessing follow OrcustD/RacketVision's MIT-licensed
BallTrack module. This adapter returns candidates in the format consumed by the
existing Netcast TennisVision world tracker; it never trains on an uploaded video.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .adaptive_ball import plan_event_preserving_frames, scout_frame_indices

MODEL_WIDTH = 512
MODEL_HEIGHT = 288
SEQUENCE_LENGTH = 4
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 4
DEFAULT_PREFETCH = True
DEFAULT_BACKGROUND_WORKERS = 4
ProgressCallback = Callable[[int, int], None]


class Conv2DBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, padding="same", bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(inputs)))


class Double2DConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv_1 = Conv2DBlock(input_channels, output_channels)
        self.conv_2 = Conv2DBlock(output_channels, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv_2(self.conv_1(inputs))


class Triple2DConv(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv_1 = Conv2DBlock(input_channels, output_channels)
        self.conv_2 = Conv2DBlock(output_channels, output_channels)
        self.conv_3 = Conv2DBlock(output_channels, output_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv_3(self.conv_2(self.conv_1(inputs)))


class RacketVisionBallTrack(nn.Module):
    """Four-frame TrackNetV3 with an explicit median-background input."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.down_block_1 = Double2DConv(15, channels)
        self.down_block_2 = Double2DConv(channels, channels * 2)
        self.down_block_3 = Triple2DConv(channels * 2, channels * 4)
        self.bottleneck = Triple2DConv(channels * 4, channels * 8)
        self.up_block_1 = Triple2DConv(channels * 12, channels * 4)
        self.up_block_2 = Double2DConv(channels * 6, channels * 2)
        self.up_block_3 = Double2DConv(channels * 3, channels)
        self.predictor = nn.Conv2d(channels, SEQUENCE_LENGTH, 1)

    def forward_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        level_1 = self.down_block_1(inputs)
        level_2 = self.down_block_2(nn.functional.max_pool2d(level_1, 2))
        level_3 = self.down_block_3(nn.functional.max_pool2d(level_2, 2))
        encoded = self.bottleneck(nn.functional.max_pool2d(level_3, 2))
        decoded = self.up_block_1(
            torch.cat([nn.functional.interpolate(encoded, scale_factor=2), level_3], dim=1)
        )
        decoded = self.up_block_2(
            torch.cat([nn.functional.interpolate(decoded, scale_factor=2), level_2], dim=1)
        )
        decoded = self.up_block_3(
            torch.cat([nn.functional.interpolate(decoded, scale_factor=2), level_1], dim=1)
        )
        return torch.sigmoid(self.predictor(decoded))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(inputs)[:, -1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(checkpoint: Path, device: torch.device) -> RacketVisionBallTrack:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RacketVisionBallTrack()
    model.load_state_dict(state)
    return model.to(device).eval()


def _metadata(video: Path) -> tuple[int, int, int]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if width <= 0 or height <= 0 or total <= 0:
        raise RuntimeError("视频元数据无效，无法运行 RacketVision")
    return width, height, total


def _sample_background_frames(
    video: Path, frame_indices: list[int],
) -> list[tuple[int, np.ndarray]]:
    """Read one deterministic subset with its own decoder instance."""
    capture = cv2.VideoCapture(str(video))
    frames: list[tuple[int, np.ndarray]] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok:
            frames.append((int(frame_index), cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))))
    capture.release()
    return frames


def _median_background(video: Path, total_frames: int, samples: int = 180) -> np.ndarray:
    frame_indices = list(map(
        int, np.linspace(0, total_frames - 1, min(samples, total_frames), dtype=int)
    ))
    workers = max(1, min(
        8, int(os.environ.get("TENNISVISION_BACKGROUND_WORKERS", DEFAULT_BACKGROUND_WORKERS))
    ))
    if workers == 1:
        indexed_frames = _sample_background_frames(video, frame_indices)
    else:
        chunks = [frame_indices[offset::workers] for offset in range(workers)]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="background-sample") as pool:
            indexed_frames = [
                item
                for result in pool.map(
                    lambda chunk: _sample_background_frames(video, chunk), chunks
                )
                for item in result
            ]
    frames = [frame for _index, frame in sorted(indexed_frames, key=lambda item: item[0])]
    if not frames:
        raise RuntimeError("无法为 RacketVision 建立视频背景")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _decode_candidates(
    heatmap: np.ndarray,
    threshold: float,
    scale_x: float,
    scale_y: float,
    *,
    max_candidates: int = 1,
    alternative_threshold: float | None = None,
) -> list[tuple]:
    """Decode distinct heatmap components while preserving the public primary choice.

    Match footage normally contains one relevant ball, so production historically kept
    only the largest connected component.  A training court can contain many stationary
    loose balls: one of those can win the largest-component decision even while the fed
    ball is also present in the heatmap.  Training-mode association therefore needs the
    bounded list of alternatives.  The largest component remains item zero, making
    ``max_candidates=1`` bit-for-bit compatible with the original decoder.
    """
    def contours_at(level: float) -> list[np.ndarray]:
        mask = (heatmap > level).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    def contour_score(contour: np.ndarray) -> tuple[float, float]:
        x, y, width, height = cv2.boundingRect(contour)
        return float(np.max(heatmap[y:y + height, x:x + width])), cv2.contourArea(contour)

    def decode_contour(contour: np.ndarray) -> tuple[float, float, float, float, float]:
        x, y, width, height = cv2.boundingRect(contour)
        confidence = float(np.mean(heatmap[y:y + height, x:x + width]))
        return (
            (x + width / 2.0) * scale_x,
            (y + height / 2.0) * scale_y,
            confidence,
            max(1.0, width * scale_x),
            max(1.0, height * scale_y),
        )

    contours = contours_at(threshold)
    primary = max(contours, key=cv2.contourArea) if contours else None
    decoded: list[tuple] = []
    if alternative_threshold is None:
        if primary is None:
            return []
        remaining = [contour for contour in contours if contour is not primary]
        remaining.sort(key=contour_score, reverse=True)
        return [
            decode_contour(contour)
            for contour in [primary, *remaining][:max(1, int(max_candidates))]
        ]

    # Mark the public-threshold component explicitly. Match-mode routing can then discard
    # every low-threshold alternative and preserve the canonical input, including a
    # genuinely empty frame when no component crossed the public 0.5 threshold.
    if primary is not None:
        decoded.append((*decode_contour(primary), 1.0))
    alternatives = contours_at(min(float(alternative_threshold), threshold))
    alternatives.sort(key=contour_score, reverse=True)
    for contour in alternatives:
        candidate = decode_contour(contour)
        if primary is not None:
            px, py, _, pw, ph = decoded[0][:5]
            if abs(candidate[0] - px) <= max(pw, candidate[3]) and abs(candidate[1] - py) <= max(ph, candidate[4]):
                continue
        if any(np.hypot(candidate[0] - prior[0], candidate[1] - prior[1]) <= 4.0 * max(scale_x, scale_y)
               for prior in decoded):
            continue
        decoded.append((*candidate, 0.0))
        if len(decoded) >= max(1, int(max_candidates)):
            break
    return decoded


def _decode(
    heatmap: np.ndarray, threshold: float, scale_x: float, scale_y: float
) -> tuple[float, float, float, float, float] | None:
    """Compatibility wrapper for callers that require the historic single candidate."""
    decoded = _decode_candidates(
        heatmap, threshold, scale_x, scale_y, max_candidates=1,
    )
    return decoded[0] if decoded else None


def _cache_key(
    video: Path,
    checkpoint: Path,
    threshold: float,
    batch_size: int = 1,
    max_candidates: int = 1,
    alternative_threshold: float | None = None,
) -> str:
    stat = video.stat()
    identity = (
        f"{stat.st_size}:{stat.st_mtime_ns}:{sha256(checkpoint)}:{threshold}:"
        f"{MODEL_WIDTH}x{MODEL_HEIGHT}:rv3-production-v1"
        f"{'' if batch_size == 1 else f':batch{batch_size}'}"
        f"{'' if max_candidates == 1 else f':top{max_candidates}'}"
        f"{'' if alternative_threshold is None else f':alt{alternative_threshold:.3f}'}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _prepared_input_batches(
    video: Path,
    background_channels: np.ndarray,
    batch_size: int,
) -> Iterator[np.ndarray]:
    """Decode and prepare native-rate temporal inputs in their original order."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")
    window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    pending: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
            # A frame belongs to four consecutive temporal windows. Convert it once here
            # instead of repeating the same uint8->CHW float work on every reuse.
            window.append(np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0))
            sequence = list(window)
            while len(sequence) < SEQUENCE_LENGTH:
                sequence.insert(0, sequence[0])
            pending.append(np.concatenate(
                [background_channels, *sequence[-SEQUENCE_LENGTH:]], axis=0,
            ))
            if len(pending) >= batch_size:
                yield np.stack(pending, axis=0)
                pending.clear()
        if pending:
            yield np.stack(pending, axis=0)
    finally:
        capture.release()


def _prepared_selected_input_batches(
    video: Path, background_channels: np.ndarray,
    selected_frames: set[int] | frozenset[int], batch_size: int,
) -> Iterator[tuple[list[int], np.ndarray]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")
    wanted, window = set(map(int, selected_frames)), deque(maxlen=SEQUENCE_LENGTH)
    indices: list[int] = []
    inputs: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
            window.append(np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0))
            if frame_index in wanted:
                sequence = list(window)
                while len(sequence) < SEQUENCE_LENGTH:
                    sequence.insert(0, sequence[0])
                indices.append(frame_index)
                inputs.append(np.concatenate([background_channels, *sequence], axis=0))
                if len(inputs) >= batch_size:
                    yield indices, np.stack(inputs)
                    indices, inputs = [], []
            frame_index += 1
        if inputs:
            yield indices, np.stack(inputs)
    finally:
        capture.release()


def _prepared_grouped_input_batches(
    video: Path, background_channels: np.ndarray, batch_size: int,
) -> Iterator[tuple[list[list[int]], np.ndarray]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{video}")
    batch_indices: list[list[int]] = []
    batch_inputs: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    indices: list[int] = []
    frame_index = 0

    def finish_group() -> tuple[list[list[int]], np.ndarray] | None:
        nonlocal batch_indices, batch_inputs, frames, indices
        if not frames:
            return None
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1])
        batch_indices.append(indices.copy())
        batch_inputs.append(np.concatenate([background_channels, *frames], axis=0))
        frames, indices = [], []
        if len(batch_inputs) < batch_size:
            return None
        result = batch_indices, np.stack(batch_inputs)
        batch_indices, batch_inputs = [], []
        return result

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
            frames.append(np.moveaxis(resized.astype(np.float32) / 255.0, -1, 0))
            indices.append(frame_index)
            frame_index += 1
            if len(frames) == SEQUENCE_LENGTH:
                if (ready := finish_group()) is not None:
                    yield ready
        if (ready := finish_group()) is not None:
            yield ready
        if batch_inputs:
            yield batch_indices, np.stack(batch_inputs)
    finally:
        capture.release()


def _input_batches(
    video: Path,
    background_channels: np.ndarray,
    batch_size: int,
    *,
    prefetch: bool,
) -> Iterator[np.ndarray]:
    """Optionally overlap CPU decoding/preparation with GPU inference."""
    source = _prepared_input_batches(video, background_channels, batch_size)
    if not prefetch:
        yield from source
        return

    items: queue.Queue[np.ndarray | BaseException | None] = queue.Queue(maxsize=2)

    def produce() -> None:
        try:
            for prepared in source:
                items.put(prepared)
        except BaseException as error:
            items.put(error)
        finally:
            items.put(None)

    worker = threading.Thread(target=produce, name="racketvision-prefetch", daemon=True)
    worker.start()
    while True:
        item = items.get()
        if item is None:
            break
        if isinstance(item, BaseException):
            raise item
        yield item
    worker.join()


def detect_video_candidates(
    video: Path,
    checkpoint: Path,
    cache_dir: Path,
    *,
    device: str = "cuda",
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int | None = None,
    max_candidates: int = 1,
    alternative_threshold: float | None = None,
    progress: ProgressCallback | None = None,
    prefetch: bool | None = None,
) -> list[list[tuple[float, float, float, float, float]]]:
    """Return one candidate row per source frame.

    ``batch_size=1`` preserves the original sequential execution path. Larger
    batches only group identical inputs into one GPU call; preprocessing,
    weights, thresholding and per-frame decoding stay unchanged. The environment
    switch makes rollback possible without changing code.
    """
    if batch_size is None:
        batch_size = int(os.environ.get("TENNISVISION_RACKETVISION_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    batch_size = max(1, int(batch_size))
    if prefetch is None:
        default_prefetch = "1" if DEFAULT_PREFETCH else "0"
        prefetch = os.environ.get("TENNISVISION_RACKETVISION_PREFETCH", default_prefetch) != "0"
    max_candidates = max(1, int(max_candidates))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"racketvision_{_cache_key(video, checkpoint, threshold, batch_size, max_candidates, alternative_threshold)}.pkl"
    )
    if cache_path.exists():
        with cache_path.open("rb") as stream:
            return pickle.load(stream)

    source_width, source_height, total_frames = _metadata(video)
    background = _median_background(video, total_frames)
    background_channels = np.moveaxis(background.astype(np.float32) / 255.0, -1, 0)
    scale_x = source_width / MODEL_WIDTH
    scale_y = source_height / MODEL_HEIGHT

    runtime_device = torch.device(
        device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    model = load_model(checkpoint, runtime_device)
    candidates: list[list[tuple[float, float, float, float, float]]] = []
    started = time.perf_counter()

    def infer_batch(prepared: np.ndarray) -> None:
        completed_before = len(candidates)
        inputs = torch.from_numpy(prepared).to(
            runtime_device, non_blocking=True
        )
        with torch.inference_mode():
            if runtime_device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    heatmaps = model(inputs)
            else:
                heatmaps = model(inputs)
        for heatmap in heatmaps.float().cpu().numpy():
            candidates.append(_decode_candidates(
                heatmap, threshold, scale_x, scale_y,
                max_candidates=max_candidates,
                alternative_threshold=alternative_threshold,
            ))
        crossed_progress_step = len(candidates) // 100 > completed_before // 100
        if progress is not None and (crossed_progress_step or len(candidates) >= total_frames):
            progress(len(candidates), total_frames)

    for prepared in _input_batches(
        video, background_channels, batch_size, prefetch=bool(prefetch),
    ):
        infer_batch(prepared)
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()
    if progress is not None:
        progress(len(candidates), total_frames)
    if not candidates:
        raise RuntimeError("RacketVision没有解码出任何视频帧")

    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(candidates, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    elapsed = time.perf_counter() - started
    detected = sum(bool(row) for row in candidates)
    print(
        f"RacketVision: {detected}/{len(candidates)} frames have a public-model candidate "
        f"({len(candidates) / max(elapsed, 1e-9):.1f} fps, batch={batch_size}, "
        f"prefetch={int(bool(prefetch))})",
        flush=True,
    )
    return candidates


def detect_video_candidates_adaptive(
    video: Path,
    checkpoint: Path,
    cache_dir: Path,
    frame_metadata: list[dict],
    *,
    device: str = "cuda",
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int | None = None,
    max_candidates: int = 1,
    alternative_threshold: float | None = None,
    scout_stride: int = 2,
    context_signature: str = "",
    progress: ProgressCallback | None = None,
) -> list[list[tuple[float, ...]]]:
    """Run a sparse scout and recover native-rate windows around possible events.

    Output contains one row for every source frame. Empty rows outside inferred
    timestamps are intentional missing observations; the tracker may coast across them.
    Every actual model invocation still consumes the original four consecutive frames.
    """
    if batch_size is None:
        batch_size = int(os.environ.get("TENNISVISION_RACKETVISION_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    batch_size = max(1, int(batch_size))
    scout_stride = max(1, int(scout_stride))
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_width, source_height, total_frames = _metadata(video)
    if len(frame_metadata) != total_frames:
        raise ValueError(
            f"自适应球检测需要逐帧人物上下文：视频{total_frames}帧，上下文{len(frame_metadata)}帧"
        )
    stat = video.stat()
    identity = (
        f"{stat.st_size}:{stat.st_mtime_ns}:{sha256(checkpoint)}:{threshold}:"
        f"{MODEL_WIDTH}x{MODEL_HEIGHT}:rv3-event-adaptive-v2:s{scout_stride}:"
        f"b{batch_size}:top{max_candidates}:alt{alternative_threshold}:{context_signature}"
    )
    cache_path = cache_dir / f"racketvision_adaptive_{hashlib.sha256(identity.encode()).hexdigest()[:24]}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as stream:
            return pickle.load(stream)

    background = _median_background(video, total_frames)
    background_channels = np.moveaxis(background.astype(np.float32) / 255.0, -1, 0)
    scale_x = source_width / MODEL_WIDTH
    scale_y = source_height / MODEL_HEIGHT
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    capture.release()
    runtime_device = torch.device(
        device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    model = load_model(checkpoint, runtime_device)
    candidates: list[list[tuple[float, ...]]] = [[] for _ in range(total_frames)]
    started = time.perf_counter()

    def infer(selected: set[int] | frozenset[int]) -> None:
        for frame_indices, prepared in _prepared_selected_input_batches(
                video, background_channels, selected, batch_size):
            inputs = torch.from_numpy(prepared).to(runtime_device, non_blocking=True)
            with torch.inference_mode():
                if runtime_device.type == "cuda":
                    with torch.autocast("cuda", dtype=torch.float16):
                        heatmaps = model(inputs)
                else:
                    heatmaps = model(inputs)
            for frame_index, heatmap in zip(
                    frame_indices, heatmaps.float().cpu().numpy(), strict=True):
                candidates[frame_index] = _decode_candidates(
                    heatmap, threshold, scale_x, scale_y,
                    max_candidates=max_candidates,
                    alternative_threshold=alternative_threshold,
                )

    scout = scout_frame_indices(total_frames, scout_stride)
    infer(scout)
    if progress is not None:
        progress(max(1, int(total_frames * 0.45)), total_frames)
    plan = plan_event_preserving_frames(
        candidates, frame_metadata, fps=fps, stride=scout_stride,
    )
    recovery = plan.dense_frames - scout
    infer(recovery)
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()
    if progress is not None:
        progress(total_frames, total_frames)

    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(candidates, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    elapsed = time.perf_counter() - started
    inferred = len(plan.inference_frames)
    detected = sum(bool(row) for row in candidates)
    print(
        f"RacketVision adaptive: inferred {inferred}/{total_frames} frames "
        f"({100.0 * inferred / max(total_frames, 1):.1f}%), {detected} candidates, "
        f"{elapsed:.1f}s; guards={plan.reasons}",
        flush=True,
    )
    return candidates


def detect_video_candidates_grouped(
    video: Path,
    checkpoint: Path,
    cache_dir: Path,
    *,
    device: str = "cuda",
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int | None = None,
    max_candidates: int = 1,
    alternative_threshold: float | None = None,
    progress: ProgressCallback | None = None,
) -> list[list[tuple[float, ...]]]:
    """Decode all four trained heatmaps from each non-overlapping model window.

    This preserves one candidate row per native source frame while reducing temporal
    window forward passes by approximately four.  It is an opt-in A/B path until event
    regression proves that non-causal within-window context preserves production output.
    """
    if batch_size is None:
        batch_size = int(os.environ.get("TENNISVISION_RACKETVISION_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    batch_size = max(1, int(batch_size))
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_width, source_height, total_frames = _metadata(video)
    stat = video.stat()
    identity = (
        f"{stat.st_size}:{stat.st_mtime_ns}:{sha256(checkpoint)}:{threshold}:"
        f"{MODEL_WIDTH}x{MODEL_HEIGHT}:rv3-grouped4-v1:b{batch_size}:"
        f"top{max_candidates}:alt{alternative_threshold}"
    )
    cache_path = cache_dir / f"racketvision_grouped_{hashlib.sha256(identity.encode()).hexdigest()[:24]}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as stream:
            return pickle.load(stream)

    background = _median_background(video, total_frames)
    background_channels = np.moveaxis(background.astype(np.float32) / 255.0, -1, 0)
    scale_x = source_width / MODEL_WIDTH
    scale_y = source_height / MODEL_HEIGHT
    runtime_device = torch.device(
        device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    model = load_model(checkpoint, runtime_device)
    candidates: list[list[tuple[float, ...]]] = [[] for _ in range(total_frames)]
    completed = 0
    started = time.perf_counter()
    for grouped_indices, prepared in _prepared_grouped_input_batches(
            video, background_channels, batch_size):
        inputs = torch.from_numpy(prepared).to(runtime_device, non_blocking=True)
        with torch.inference_mode():
            if runtime_device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    grouped_heatmaps = model.forward_sequence(inputs)
            else:
                grouped_heatmaps = model.forward_sequence(inputs)
        for source_indices, heatmaps in zip(
                grouped_indices, grouped_heatmaps.float().cpu().numpy(), strict=True):
            for frame_index, heatmap in zip(source_indices, heatmaps, strict=False):
                candidates[frame_index] = _decode_candidates(
                    heatmap, threshold, scale_x, scale_y,
                    max_candidates=max_candidates,
                    alternative_threshold=alternative_threshold,
                )
                completed += 1
        if progress is not None:
            progress(completed, total_frames)
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()
    if progress is not None:
        progress(total_frames, total_frames)

    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(candidates, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    elapsed = time.perf_counter() - started
    detected = sum(bool(row) for row in candidates)
    windows = (total_frames + SEQUENCE_LENGTH - 1) // SEQUENCE_LENGTH
    print(
        f"RacketVision grouped4: {detected}/{total_frames} candidate frames from "
        f"{windows} temporal windows in {elapsed:.1f}s "
        f"({total_frames / max(elapsed, 1e-9):.1f} source fps)",
        flush=True,
    )
    return candidates


def detect_video_candidates_selected(
    video: Path,
    checkpoint: Path,
    cache_dir: Path,
    selected_frames: set[int] | frozenset[int],
    *,
    device: str = "cuda",
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int | None = None,
    max_candidates: int = 1,
    alternative_threshold: float | None = None,
    cache_label: str = "selected",
    progress: ProgressCallback | None = None,
) -> list[list[tuple[float, ...]]]:
    """Infer exact causal BallTrack outputs only at requested native timestamps."""
    if batch_size is None:
        batch_size = int(os.environ.get("TENNISVISION_RACKETVISION_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    batch_size = max(1, int(batch_size))
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_width, source_height, total_frames = _metadata(video)
    selected = frozenset(int(frame) for frame in selected_frames if 0 <= int(frame) < total_frames)
    selection_hash = hashlib.sha256(
        np.asarray(sorted(selected), dtype=np.int32).tobytes()
    ).hexdigest()[:16]
    stat = video.stat()
    identity = (
        f"{stat.st_size}:{stat.st_mtime_ns}:{sha256(checkpoint)}:{threshold}:"
        f"{MODEL_WIDTH}x{MODEL_HEIGHT}:rv3-selected-v1:b{batch_size}:"
        f"top{max_candidates}:alt{alternative_threshold}:{cache_label}:{selection_hash}"
    )
    cache_path = cache_dir / f"racketvision_selected_{hashlib.sha256(identity.encode()).hexdigest()[:24]}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as stream:
            return pickle.load(stream)

    background = _median_background(video, total_frames)
    background_channels = np.moveaxis(background.astype(np.float32) / 255.0, -1, 0)
    scale_x = source_width / MODEL_WIDTH
    scale_y = source_height / MODEL_HEIGHT
    runtime_device = torch.device(
        device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    model = load_model(checkpoint, runtime_device)
    candidates: list[list[tuple[float, ...]]] = [[] for _ in range(total_frames)]
    completed = 0
    started = time.perf_counter()
    for frame_indices, prepared in _prepared_selected_input_batches(
            video, background_channels, selected, batch_size):
        inputs = torch.from_numpy(prepared).to(runtime_device, non_blocking=True)
        with torch.inference_mode():
            if runtime_device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    heatmaps = model(inputs)
            else:
                heatmaps = model(inputs)
        for frame_index, heatmap in zip(
                frame_indices, heatmaps.float().cpu().numpy(), strict=True):
            candidates[frame_index] = _decode_candidates(
                heatmap, threshold, scale_x, scale_y,
                max_candidates=max_candidates,
                alternative_threshold=alternative_threshold,
            )
            completed += 1
        if progress is not None:
            progress(completed, max(len(selected), 1))
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()
    if progress is not None:
        progress(max(len(selected), 1), max(len(selected), 1))

    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        pickle.dump(candidates, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    elapsed = time.perf_counter() - started
    print(
        f"RacketVision {cache_label}: inferred {len(selected)}/{total_frames} frames in "
        f"{elapsed:.1f}s ({len(selected) / max(elapsed, 1e-9):.1f} inferred fps)",
        flush=True,
    )
    return candidates
