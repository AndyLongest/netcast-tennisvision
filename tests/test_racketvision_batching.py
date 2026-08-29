from pathlib import Path

from netcast_tennisvision.vision.racketvision import DEFAULT_BATCH_SIZE, _cache_key


def test_production_uses_validated_four_frame_gpu_batch() -> None:
    assert DEFAULT_BATCH_SIZE == 4


def test_sequential_and_batched_modes_use_separate_caches(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    checkpoint = tmp_path / "weights.pt"
    video.write_bytes(b"video")
    checkpoint.write_bytes(b"weights")

    sequential = _cache_key(video, checkpoint, 0.5, batch_size=1)
    batched = _cache_key(video, checkpoint, 0.5, batch_size=4)

    assert sequential != batched
