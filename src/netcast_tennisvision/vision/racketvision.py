"""Production inference for the public RacketVision BallTrack checkpoint.

The architecture and preprocessing follow OrcustD/RacketVision's MIT-licensed
BallTrack module. This adapter returns candidates in the format consumed by the
existing Netcast TennisVision world tracker; it never trains on an uploaded video.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

MODEL_WIDTH = 512
MODEL_HEIGHT = 288
SEQUENCE_LENGTH = 4
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 4
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
        return torch.sigmoid(self.predictor(decoded)[:, -1])


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


def _median_background(video: Path, total_frames: int, samples: int = 180) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    frame_indices = np.linspace(0, total_frames - 1, min(samples, total_frames), dtype=int)
    frames = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok:
            frames.append(cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT)))
    capture.release()
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
    """Decode distinct heatmap components without changing the legacy first choice.

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

    # Mark the legacy component explicitly.  Match-mode routing can then discard every
    # low-threshold alternative and exactly reproduce the historical input, including a
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
    capture = cv2.VideoCapture(str(video))
    window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    candidates: list[list[tuple[float, float, float, float, float]]] = []
    pending_inputs: list[np.ndarray] = []
    started = time.perf_counter()

    def flush_batch() -> None:
        if not pending_inputs:
            return
        completed_before = len(candidates)
        inputs = torch.from_numpy(np.stack(pending_inputs, axis=0)).to(
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
        pending_inputs.clear()
        crossed_progress_step = len(candidates) // 100 > completed_before // 100
        if progress is not None and (crossed_progress_step or len(candidates) >= total_frames):
            progress(len(candidates), total_frames)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        window.append(resized)
        sequence = list(window)
        while len(sequence) < SEQUENCE_LENGTH:
            sequence.insert(0, sequence[0])
        channels = [background_channels]
        channels.extend(
            np.moveaxis(item.astype(np.float32) / 255.0, -1, 0)
            for item in sequence[-SEQUENCE_LENGTH:]
        )
        pending_inputs.append(np.concatenate(channels, axis=0))
        if len(pending_inputs) >= batch_size:
            flush_batch()

    capture.release()
    flush_batch()
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
        f"({len(candidates) / max(elapsed, 1e-9):.1f} fps, batch={batch_size})",
        flush=True,
    )
    return candidates
