"""Render an auditable preview of the production player-identity module."""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from netcast_tennisvision.paths import REPOSITORY_ROOT
from netcast_tennisvision.vision.player_identity import (
    PLAYER_COLORS_RGB,
    WEIGHT_NAME,
    classify_observations,
    collect_observations,
    enrol_identities,
    load_osnet,
    select_side_boxes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, default=REPOSITORY_ROOT / "data/clip.mp4")
    parser.add_argument("--cache", type=Path, help="Pass-A pickle containing person_boxes")
    parser.add_argument("--weights", type=Path, default=REPOSITORY_ROOT / "models" / WEIGHT_NAME)
    parser.add_argument(
        "--output", type=Path,
        default=REPOSITORY_ROOT / "outputs/player_identity_osnet_preview.mp4",
    )
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--enrol-seconds", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def find_cache(video: Path, explicit: Path | None) -> Path:
    if explicit:
        return explicit
    matches = sorted(
        (REPOSITORY_ROOT / "data/cache").glob(f"passA_{video.stat().st_size}_*.pkl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"没有找到与 {video.name} 匹配的 Pass-A 人物缓存")
    return matches[0]


def select_portraits(
    observations: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, np.ndarray]:
    portraits = {}
    for side, identity in (("near", "A"), ("far", "B")):
        choices = [
            observation for by_side in observations.values()
            if (observation := by_side.get(side)) is not None
        ]
        if not choices:
            raise RuntimeError(f"没有可展示的{side}球员画面")
        best = max(
            choices,
            key=lambda item: float(
                (item["box"][2] - item["box"][0]) * (item["box"][3] - item["box"][1])
            ),
        )
        portraits[identity] = best["crop"]
    return portraits


@lru_cache(maxsize=8)
def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def render_text(
    frame: np.ndarray,
    lines: list[tuple[tuple[int, int], str, tuple[int, int, int], int]],
) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for position, label, color, size in lines:
        draw.text(position, label, fill=color, font=font(size), stroke_width=1, stroke_fill=(22, 14, 31))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_video(
    video: Path,
    output: Path,
    frames_meta: list[dict[str, Any]],
    decisions: dict[int, dict[str, Any]],
    portraits: dict[str, np.ndarray],
    fps: float,
    source_size: tuple[int, int],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output_width = 1280
    output_height = round(output_width * source_size[1] / source_size[0])
    packages = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    full_ffmpeg = sorted(packages.glob("Gyan.FFmpeg.Shared_*/ffmpeg-*/bin/ffmpeg.exe"))
    ffmpeg = str(full_ffmpeg[-1]) if full_ffmpeg else "ffmpeg"
    encoder = subprocess.Popen([
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{output_width}x{output_height}", "-r", f"{fps}", "-i", "-",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        str(output),
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    capture = cv2.VideoCapture(str(video))
    latest = {"mapping": {"near": "A", "far": "B"}, "scores": {}}
    scale_x, scale_y = output_width / source_size[0], output_height / source_size[1]
    portrait_tiles = {
        identity: cv2.resize(crop, (58, 116), interpolation=cv2.INTER_AREA)
        for identity, crop in portraits.items()
    }
    for frame_index, meta in enumerate(frames_meta):
        ok, source = capture.read()
        if not ok:
            break
        latest = decisions.get(frame_index, latest)
        frame = cv2.resize(source, (output_width, output_height), interpolation=cv2.INTER_AREA)
        overlay = frame.copy()
        cv2.rectangle(overlay, (18, 18), (430, 112), (20, 13, 30), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        lines = [
            ((34, 29), "球员身份识别 · OSNet-AIN", (245, 242, 248), 21),
            ((34, 64), "外观重识别 + 球场位置 + 三次确认防抖", (190, 181, 199), 15),
            ((34, 87), f"时间 {frame_index / fps:06.2f}s", (196, 241, 44), 15),
        ]
        for side, box in select_side_boxes(meta).items():
            identity = latest["mapping"].get(side, "?")
            rgb = PLAYER_COLORS_RGB.get(identity, (190, 190, 190))
            x1, y1, x2, y2 = box * np.array([scale_x, scale_y, scale_x, scale_y])
            p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.rectangle(frame, p1, p2, rgb[::-1], 3, cv2.LINE_AA)
            scores = latest.get("scores", {}).get(side, {})
            advantage = abs(scores.get("A", 0.0) - scores.get("B", 0.0))
            side_name = "近端" if side == "near" else "远端"
            lines.append((
                (p1[0], max(8, p1[1] - 28)),
                f"球员 {identity} · {side_name} · 优势 {advantage:.2f}", rgb, 18,
            ))
        tile_x = output_width - 170
        for row, identity in enumerate(("A", "B")):
            y = 18 + row * 132
            frame[y:y + 116, tile_x:tile_x + 58] = portrait_tiles[identity]
            cv2.rectangle(frame, (tile_x, y), (tile_x + 58, y + 116), PLAYER_COLORS_RGB[identity][::-1], 2)
            lines.append(((tile_x + 68, y + 38), f"球员 {identity}", PLAYER_COLORS_RGB[identity], 18))
        frame = render_text(frame, lines)
        assert encoder.stdin is not None
        encoder.stdin.write(frame.tobytes())
    capture.release()
    assert encoder.stdin is not None
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg 无法生成身份识别预览视频")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    with find_cache(args.video, args.cache).open("rb") as stream:
        all_frames_meta = pickle.load(stream)
    frames_meta = all_frames_meta[:args.max_frames] if args.max_frames else all_frames_meta
    capture = cv2.VideoCapture(str(args.video))
    source_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_osnet(args.weights, device)
    observations, fps = collect_observations(
        args.video, frames_meta, model, device, stride=max(1, args.sample_stride),
    )
    prototypes = enrol_identities(observations, fps, args.enrol_seconds)
    decisions, metrics = classify_observations(
        observations, prototypes, int(fps * args.enrol_seconds), fps=fps,
    )
    render_video(
        args.video, args.output, frames_meta, decisions,
        select_portraits(observations), fps, source_size,
    )
    metrics.update({
        "video": str(args.video), "fps": round(fps, 4),
        "sample_stride": args.sample_stride, "osnet_samples": len(observations),
        "elapsed_seconds": round(time.perf_counter() - started, 2), "output": str(args.output),
    })
    args.output.with_suffix(".json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
