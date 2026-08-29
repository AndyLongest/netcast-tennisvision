"""Create per-frame ASS overlays for Netcast TennisVision tracking-state review videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STATE_STYLE = {
    "observed": ("Observed / real detection", "&H004AA316"),
    "occluded_predicted": ("Occlusion prediction", "&H000B9EF5"),
    "unacquired": ("Unknown / no reliable position", "&H0080726B"),
}


def ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def build_ass(data: dict, width: int, height: int) -> str:
    fps = float(data["fps"])
    frames = data["frames"]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Status,Microsoft YaHei,25,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,3,10,0,7,24,24,22,1
Style: Footer,Microsoft YaHei,18,&H00FFFFFF,&H00FFFFFF,&HCC000000,&H90000000,0,0,0,0,100,100,0,0,3,6,0,1,24,24,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    for frame_number, frame in enumerate(frames):
        start = ass_time(frame_number / fps)
        end = ass_time((frame_number + 1) / fps)
        state = frame.get("s", "unacquired")
        label, panel_colour = STATE_STYLE.get(state, STATE_STYLE["unacquired"])
        confidence = round(float(frame.get("v", 0.0)) * 100)
        timestamp = frame_number / fps
        speed = frame.get("sp")
        radius = frame.get("r")
        axes = frame.get("ax")
        motion_mode = frame.get("mo") or "flight"
        search_line = ""
        if speed is not None:
            radius_label = f"{float(radius):.0f}px" if radius is not None else "no match"
            search_line = (
                rf"\NEst. speed {float(speed):.1f}px/frame  |  Search radius {radius_label}"
            )
            if axes is not None:
                search_line += (
                    rf"  |  Ellipse {float(axes[0]):.0f}x{float(axes[1]):.0f}px"
                    rf"\NMotion mode: {motion_mode}"
                )
        panel = (
            rf"{{\3c{panel_colour}}}BALL STATE: {label}\N"
            rf"Frame {frame_number}  |  {timestamp:05.2f}s  |  Confidence {confidence}%"
            rf"{search_line}"
        )
        events.append(f"Dialogue: 1,{start},{end},Status,,0,0,0,,{panel}")

    footer = (
        "Dialogue: 0,0:00:00.00,9:59:59.99,Footer,,0,0,0,,"
        "GREEN = real observation   YELLOW = predicted through occlusion   GRAY = unknown"
    )
    return header + "\n".join(events) + "\n" + footer + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_json", type=Path)
    parser.add_argument("output_ass", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    data = json.loads(args.scene_json.read_text(encoding="utf-8"))
    args.output_ass.write_text(build_ass(data, args.width, args.height), encoding="utf-8")


if __name__ == "__main__":
    main()
