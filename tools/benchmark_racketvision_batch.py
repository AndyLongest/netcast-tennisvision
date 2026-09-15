"""Measure one native-rate RacketVision batch size against a frozen candidate cache."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import torch

from netcast_tennisvision.vision.racketvision import detect_video_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args()
    if hasattr(torch.backends, "cudnn") and args.cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    with args.baseline.open("rb") as stream:
        baseline = pickle.load(stream)
    started = time.perf_counter()
    rows = detect_video_candidates(
        args.video,
        Path("models/racketvision_balltrack_state_v1.pt"),
        args.cache,
        device="cuda" if torch.cuda.is_available() else "cpu",
        threshold=0.5,
        batch_size=args.batch_size,
        max_candidates=8,
        alternative_threshold=0.30,
    )
    elapsed = time.perf_counter() - started
    print(
        f"batch={args.batch_size} seconds={elapsed:.3f} fps={len(rows) / elapsed:.3f} "
        f"exact={rows == baseline}",
        flush=True,
    )


if __name__ == "__main__":
    main()
