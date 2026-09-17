"""Benchmark frozen BallTrack inference backends without changing production.

This intentionally measures model execution only. Video decoding, background
construction and candidate decoding are identical across backends and are excluded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from netcast_tennisvision.vision.racketvision import MODEL_HEIGHT, MODEL_WIDTH, load_model


def timed(callable_, rounds: int) -> float:
    for _ in range(5):
        callable_()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(rounds):
        callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / rounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("models/racketvision_balltrack_state_v1.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--provider", choices=("cuda", "tensorrt"), default="cuda")
    parser.add_argument("--onnx", type=Path, default=Path("data/cache/balltrack_benchmark.onnx"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda")
    model = load_model(args.checkpoint, device)
    generator = torch.Generator().manual_seed(20260916)
    sample = torch.rand(
        args.batch_size, 15, MODEL_HEIGHT, MODEL_WIDTH, generator=generator
    ).to(device)

    def eager() -> torch.Tensor:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            return model(sample)

    eager_output = eager().float().cpu().numpy()
    eager_seconds = timed(eager, args.rounds)
    result: dict[str, object] = {
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "eager_ms": round(eager_seconds * 1000, 3),
    }

    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    model = model.half()
    onnx_sample = sample.half()
    torch.onnx.export(
        model,
        onnx_sample,
        args.onnx,
        input_names=["frames"],
        output_names=["heatmap"],
        dynamic_axes={"frames": {0: "batch"}, "heatmap": {0: "batch"}},
        opset_version=17,
    )
    import onnxruntime as ort

    providers = ort.get_available_providers()
    requested_provider = (
        "TensorrtExecutionProvider" if args.provider == "tensorrt" else "CUDAExecutionProvider"
    )
    session = ort.InferenceSession(str(args.onnx), providers=[requested_provider])
    numpy_sample = onnx_sample.cpu().numpy()

    def ort_cuda() -> np.ndarray:
        return session.run(None, {"frames": numpy_sample})[0]

    ort_output = ort_cuda()
    started = time.perf_counter()
    for _ in range(args.rounds):
        ort_cuda()
    ort_seconds = (time.perf_counter() - started) / args.rounds
    result.update({
        "providers": providers,
        "provider_used": session.get_providers(),
        "onnxruntime_ms": round(ort_seconds * 1000, 3),
        "speedup": round(eager_seconds / ort_seconds, 3),
        "max_abs_difference": float(np.max(np.abs(eager_output - ort_output))),
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
