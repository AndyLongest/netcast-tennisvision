from pathlib import Path

import numpy as np

import netcast_tennisvision.vision.racketvision as racketvision
from netcast_tennisvision.vision.racketvision import (
    DEFAULT_BACKGROUND_WORKERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_PREFETCH,
    _cache_key,
)


def test_production_uses_validated_four_frame_gpu_batch() -> None:
    assert DEFAULT_BATCH_SIZE == 4
    assert DEFAULT_PREFETCH is True
    assert DEFAULT_BACKGROUND_WORKERS == 4


def test_sequential_and_batched_modes_use_separate_caches(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    checkpoint = tmp_path / "weights.pt"
    video.write_bytes(b"video")
    checkpoint.write_bytes(b"weights")

    sequential = _cache_key(video, checkpoint, 0.5, batch_size=1)
    batched = _cache_key(video, checkpoint, 0.5, batch_size=4)

    assert sequential != batched


def test_prefetch_preserves_every_prepared_batch_in_order(monkeypatch) -> None:
    expected = [np.full((2, 3), index, dtype=np.float32) for index in range(5)]

    def prepared(*_args):
        yield from (item.copy() for item in expected)

    monkeypatch.setattr(racketvision, "_prepared_input_batches", prepared)
    actual = list(racketvision._input_batches(
        Path("unused.mp4"), np.empty((3, 2, 2)), 4, prefetch=True,
    ))

    assert len(actual) == len(expected)
    assert all(
        np.array_equal(left, right)
        for left, right in zip(actual, expected, strict=True)
    )


def test_parallel_background_sampling_preserves_order_and_pixels(monkeypatch) -> None:
    def sample(_video, indices):
        return [
            (index, np.full((3, 4, 3), index % 251, dtype=np.uint8))
            for index in indices
        ]

    monkeypatch.setattr(racketvision, "_sample_background_frames", sample)
    monkeypatch.setenv("TENNISVISION_BACKGROUND_WORKERS", "1")
    serial = racketvision._median_background(Path("unused.mp4"), 400, samples=37)
    monkeypatch.setenv("TENNISVISION_BACKGROUND_WORKERS", "4")
    parallel = racketvision._median_background(Path("unused.mp4"), 400, samples=37)

    assert np.array_equal(parallel, serial)
