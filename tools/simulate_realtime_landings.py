"""Render a timeline-honest live-landing demonstration from a frozen report.

This tool validates event delivery and presentation, not online neural-network throughput.
It deliberately delays every frozen touchdown until all future frames required by the
current centred bounce descriptor have arrived.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from netcast_tennisvision.streaming.causal_events import schedule_landing_events

ROOT = Path(__file__).resolve().parents[1]
COURT_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77
NET_Y_M = COURT_LENGTH_M / 2.0
SERVICE_LINE_M = 5.485
SINGLES_MARGIN_M = 1.37


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _ffmpeg() -> str | None:
    packages = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    full_builds = sorted(packages.glob("Gyan.FFmpeg.Shared_*/ffmpeg-*/bin/ffmpeg.exe"))
    if full_builds:
        return str(full_builds[-1])
    return shutil.which("ffmpeg")


def _court_point(rect: tuple[int, int, int, int], x: float, y: float) -> tuple[int, int]:
    left, top, right, bottom = rect
    return (
        round(left + np.clip(x / COURT_WIDTH_M, 0.0, 1.0) * (right - left)),
        round(bottom - np.clip(y / COURT_LENGTH_M, 0.0, 1.0) * (bottom - top)),
    )


def _zone_bounds(event: dict) -> tuple[float, float, float, float] | None:
    if event.get("zone") == "Out":
        return None
    x, y = float(event["x"]), float(event["y"])
    if y < SERVICE_LINE_M:
        return SINGLES_MARGIN_M, 0.0, COURT_WIDTH_M - SINGLES_MARGIN_M, SERVICE_LINE_M
    if y < NET_Y_M:
        x0, x1 = ((SINGLES_MARGIN_M, COURT_WIDTH_M / 2.0)
                  if x < COURT_WIDTH_M / 2.0
                  else (COURT_WIDTH_M / 2.0, COURT_WIDTH_M - SINGLES_MARGIN_M))
        return x0, SERVICE_LINE_M, x1, NET_Y_M
    if y < COURT_LENGTH_M - SERVICE_LINE_M:
        x0, x1 = ((SINGLES_MARGIN_M, COURT_WIDTH_M / 2.0)
                  if x < COURT_WIDTH_M / 2.0
                  else (COURT_WIDTH_M / 2.0, COURT_WIDTH_M - SINGLES_MARGIN_M))
        return x0, NET_Y_M, x1, COURT_LENGTH_M - SERVICE_LINE_M
    return (SINGLES_MARGIN_M, COURT_LENGTH_M - SERVICE_LINE_M,
            COURT_WIDTH_M - SINGLES_MARGIN_M, COURT_LENGTH_M)


def _draw_cross(draw: ImageDraw.ImageDraw, point: tuple[int, int], radius: int = 8) -> None:
    x, y = point
    draw.line((x - radius, y - radius, x + radius, y + radius), fill=(255, 67, 83, 255), width=4)
    draw.line((x - radius, y + radius, x + radius, y - radius), fill=(255, 67, 83, 255), width=4)


def _render_overlay(
    frame: np.ndarray,
    *,
    frame_index: int,
    fps: float,
    emitted: list[dict],
    latest: dict | None,
    latest_age: int,
) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    scale = max(0.72, min(width / 1280.0, height / 720.0))
    title_font = _font(round(23 * scale), bold=True)
    body_font = _font(round(16 * scale))
    small_font = _font(round(13 * scale))

    pad = round(22 * scale)
    status_w, status_h = round(360 * scale), round(92 * scale)
    draw.rounded_rectangle((pad, pad, pad + status_w, pad + status_h), radius=round(16 * scale),
                           fill=(15, 8, 31, 220), outline=(179, 239, 36, 150), width=1)
    draw.ellipse((pad + 17 * scale, pad + 18 * scale, pad + 31 * scale, pad + 32 * scale),
                 fill=(185, 242, 35, 255))
    draw.text((pad + 42 * scale, pad + 12 * scale), "实时落点分析 · 在线",
              font=title_font, fill=(255, 255, 255, 255))
    elapsed = frame_index / fps
    draw.text((pad + 18 * scale, pad + 52 * scale),
              f"比赛时间  {elapsed:05.1f}s    已确认 {len(emitted)} 个落点",
              font=body_font, fill=(214, 207, 228, 255))

    panel_w, panel_h = round(220 * scale), round(390 * scale)
    panel_right, panel_bottom = width - pad, height - pad
    panel_left, panel_top = panel_right - panel_w, panel_bottom - panel_h
    draw.rounded_rectangle((panel_left, panel_top, panel_right, panel_bottom),
                           radius=round(18 * scale), fill=(18, 8, 36, 225),
                           outline=(255, 255, 255, 42), width=1)
    draw.text((panel_left + 16 * scale, panel_top + 13 * scale), "本回合落点",
              font=body_font, fill=(255, 255, 255, 255))
    court = (round(panel_left + 34 * scale), round(panel_top + 55 * scale),
             round(panel_right - 34 * scale), round(panel_bottom - 28 * scale))

    if latest is not None and latest_age < round(1.15 * fps):
        zone = _zone_bounds(latest)
        if zone is not None:
            x0, y0 = _court_point(court, zone[0], zone[1])
            x1, y1 = _court_point(court, zone[2], zone[3])
            draw.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)),
                           fill=(232, 255, 62, 75), outline=(232, 255, 62, 220), width=2)

    line = (241, 237, 246, 230)
    left, top, right, bottom = court
    draw.rectangle(court, outline=line, width=2)
    for y in (SERVICE_LINE_M, NET_Y_M, COURT_LENGTH_M - SERVICE_LINE_M):
        p0 = _court_point(court, 0.0 if y == NET_Y_M else SINGLES_MARGIN_M, y)
        p1 = _court_point(court, COURT_WIDTH_M if y == NET_Y_M else COURT_WIDTH_M - SINGLES_MARGIN_M, y)
        draw.line((*p0, *p1), fill=line, width=2 if y != NET_Y_M else 3)
    for x in (SINGLES_MARGIN_M, COURT_WIDTH_M - SINGLES_MARGIN_M):
        p0, p1 = _court_point(court, x, 0.0), _court_point(court, x, COURT_LENGTH_M)
        draw.line((*p0, *p1), fill=line, width=1)
    p0 = _court_point(court, COURT_WIDTH_M / 2.0, SERVICE_LINE_M)
    p1 = _court_point(court, COURT_WIDTH_M / 2.0, COURT_LENGTH_M - SERVICE_LINE_M)
    draw.line((*p0, *p1), fill=line, width=1)

    for event in emitted:
        point = _court_point(court, float(event["x"]), float(event["y"]))
        if event.get("zone") == "Out":
            _draw_cross(draw, point, radius=round(7 * scale))
        else:
            color = (181, 239, 38, 255) if event.get("player_id") == "A" else (157, 91, 255, 255)
            radius = round(6 * scale)
            draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
                         fill=color, outline=(255, 255, 255, 245), width=2)

    if latest is not None and latest_age < round(1.15 * fps):
        card_w, card_h = round(410 * scale), round(80 * scale)
        cx, cy = width // 2 - card_w // 2, height - pad - card_h
        draw.rounded_rectangle((cx, cy, cx + card_w, cy + card_h), radius=round(15 * scale),
                               fill=(16, 8, 31, 232), outline=(185, 242, 35, 180), width=2)
        result = "界外" if latest.get("zone") == "Out" else "界内"
        delay = round(float(latest["causal_delay_ms"]))
        draw.text((cx + 18 * scale, cy + 10 * scale), f"落点已确认 · {result}",
                  font=title_font, fill=(185, 242, 35, 255) if result == "界内" else (255, 86, 101, 255))
        draw.text((cx + 18 * scale, cy + 45 * scale),
                  f"触地后 {delay} ms 发出 · 球员 {latest.get('player_id', '—')}",
                  font=small_font, fill=(222, 215, 232, 255))

    return cv2.cvtColor(np.asarray(Image.alpha_composite(image, overlay).convert("RGB")), cv2.COLOR_RGB2BGR)


def render(source: Path, scene_path: Path, output: Path, metrics_path: Path,
           future_support_frames: int) -> dict:
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or scene["fps"])
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    scheduled = schedule_landing_events(
        list(scene["bounces"]), fps=fps, future_support_frames=future_support_frames,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="netcast-live-") as temp_dir:
        silent = Path(temp_dir) / "silent.mp4"
        writer = cv2.VideoWriter(
            str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("无法创建实时模拟视频")

        pending = iter(scheduled)
        next_event = next(pending, None)
        current_rally = None
        emitted: list[dict] = []
        latest = None
        latest_frame = -10_000
        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                while next_event is not None and next_event["available_frame"] <= frame_index:
                    if current_rally != next_event.get("rally_id"):
                        current_rally = next_event.get("rally_id")
                        emitted = []
                    emitted.append(next_event)
                    latest, latest_frame = next_event, frame_index
                    next_event = next(pending, None)
                writer.write(_render_overlay(
                    frame, frame_index=frame_index, fps=fps, emitted=emitted,
                    latest=latest, latest_age=frame_index - latest_frame,
                ))
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        ffmpeg = _ffmpeg()
        if ffmpeg:
            subprocess.run([
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(silent), "-i", str(source),
                "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264",
                "-preset", "fast", "-crf", "20", "-c:a", "aac", "-shortest", str(output),
            ], check=True)
        else:
            shutil.copy2(silent, output)

    delays = sorted(float(event["causal_delay_ms"]) for event in scheduled)
    def percentile(p: float) -> float:
        return delays[min(len(delays) - 1, math.ceil(p * len(delays)) - 1)]

    metrics = {
        "source": str(source.relative_to(ROOT)),
        "fps": fps,
        "frames": total_frames,
        "duration_seconds": total_frames / fps,
        "events": len(scheduled),
        "future_support_frames": future_support_frames,
        "algorithmic_delay_ms": {
            "min": min(delays), "median": percentile(0.5),
            "p95": percentile(0.95), "max": max(delays),
        },
        "scope": "事件时序与前端显示模拟；不代表在线神经网络吞吐",
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "assets/demo/demo.mp4")
    parser.add_argument("--scene", type=Path, default=ROOT / "assets/demo/scene3d.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/realtime/realtime_landing_demo.mp4")
    parser.add_argument("--metrics", type=Path,
                        default=ROOT / "outputs/realtime/realtime_landing_metrics.json")
    parser.add_argument("--future-support-frames", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.scene, args.output, args.metrics,
                            args.future_support_frames), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
