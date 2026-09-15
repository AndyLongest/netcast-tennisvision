"""Check whether parallel random access preserves the production median background."""
from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from netcast_tennisvision.vision.racketvision import MODEL_HEIGHT, MODEL_WIDTH


def sample(video: Path, indices: list[int]) -> list[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video))
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            frames.append((index, cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))))
    capture.release()
    return frames


def median(items: list[tuple[int, np.ndarray]]) -> tuple[np.ndarray, str]:
    ordered = [frame for _index, frame in sorted(items, key=lambda item: item[0])]
    result = np.median(np.stack(ordered), axis=0).astype(np.uint8)
    return result, hashlib.sha256(result.tobytes()).hexdigest()


def main() -> None:
    video = Path("assets/demo/demo.mp4")
    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    indices = list(map(int, np.linspace(0, total - 1, min(180, total), dtype=int)))
    started = time.perf_counter()
    serial_items = sample(video, indices)
    serial_seconds = time.perf_counter() - started
    _, serial_hash = median(serial_items)
    chunks = [indices[offset::4] for offset in range(4)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel_items = [item for result in pool.map(lambda chunk: sample(video, chunk), chunks)
                          for item in result]
    parallel_seconds = time.perf_counter() - started
    _, parallel_hash = median(parallel_items)
    print({
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "serial_frames": len(serial_items),
        "parallel_frames": len(parallel_items),
        "serial_hash": serial_hash,
        "parallel_hash": parallel_hash,
        "exact": serial_hash == parallel_hash,
    })


if __name__ == "__main__":
    main()
