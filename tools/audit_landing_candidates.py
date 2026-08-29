"""Run the cached analysis through contact detection and print a bounded landing audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "tennis_detection.ipynb"
sys.path.insert(0, str(ROOT / "src"))


def main(seconds: float = 10.0, start_seconds: float = 0.0) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__"}
    os.chdir(ROOT)
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code" and index <= 25 and index not in (23, 24):
            source = "".join(cell.get("source", []))
            exec(compile(source, f"audit-cell-{index}", "exec"), namespace)

    fps = float(namespace["FPS"])
    frames = namespace["frames_meta"]
    score_landing = namespace["score_landing_impulse"]
    nframes = namespace["nframes"]
    racket_gap = namespace["racket_gap"]
    player_gap = namespace["player_gap"]
    horizontal_reversal = namespace["horizontal_reversal"]
    sequence_probability = namespace["sequence_bounce_p"]
    rows = []
    first_frame = max(0, int(start_seconds * fps))
    last_frame = min(len(frames), int(seconds * fps))
    for frame in range(first_frame, last_frame):
        result = score_landing(frame, frames, radius=nframes(7))
        if result is None:
            continue
        meta = frames[frame]
        rows.append(
            {
                "frame": frame,
                "time_s": round(frame / fps, 3),
                "score": round(result.score, 3),
                "bic_gain": round(result.bic_gain, 3),
                "error_reduction": round(result.error_reduction, 3),
                "impulse_xy": (
                    round(result.impulse_x_px_frame, 3),
                    round(result.impulse_y_px_frame, 3),
                ),
                "support": (result.support_before, result.support_after),
                "seen": bool(meta.get("ball_seen")),
                "confidence": round(float(meta.get("ball_confidence", 0.0)), 3),
                "world_ground": _rounded(meta.get("world_ground")),
                "racket_gap": (
                    round(float(racket_gap(frame, *meta["ball_px"])), 3)
                    if meta.get("ball_px") is not None
                    else None
                ),
                "player_gap": (
                    round(float(player_gap(frame, *meta["ball_px"])[0]), 3)
                    if meta.get("ball_px") is not None
                    else None
                ),
                "horizontal_reversal": bool(horizontal_reversal(frame)),
                "sequence_p": (
                    round(float(sequence_probability[frame]), 3)
                    if np.isfinite(sequence_probability[frame])
                    else None
                ),
            }
        )

    events = [
        {
            key: _json_value(event.get(key))
            for key in (
                "frame",
                "kind",
                "source",
                "player_gap",
                "racket_gap",
                "sequence_p",
                "impulse_score",
                "evidence",
                "h_reverses",
                "down_up",
                "contact_side",
            )
        }
        for event in namespace["events"]
        if first_frame <= event["frame"] < last_frame
    ]
    bounces = [
        {
            key: _json_value(bounce.get(key))
            for key in (
                "frame",
                "time_s",
                "zone",
                "world",
                "source",
                "position_source",
                "touchdown_quality",
                "landing_support",
                "landing_uncertainty_px",
                "player_gap",
                "sequence_p",
                "impulse_score",
                "evidence",
            )
        }
        for bounce in namespace["bounces"]
        if first_frame <= bounce["frame"] < last_frame
    ]
    audio_times = namespace.get("AUDIO_IMPACTS", [])
    print(
        "AUDIO="
        + json.dumps(
            [
                {"frame": int(round(float(time_s) * fps)), "time_s": round(float(time_s), 3)}
                for time_s in audio_times
                if first_frame <= float(time_s) * fps < last_frame
            ]
        )
    )
    print("AUDIT_TOP=" + json.dumps(sorted(rows, key=lambda row: row["score"], reverse=True)[:50]))
    print(
        "NEAR_CHANGES="
        + json.dumps(
            [event for event in events if str(event.get("evidence") or "").startswith("near-")]
        )
    )
    print("EVENTS=" + json.dumps(events))
    print("BOUNCES=" + json.dumps(bounces))
    raw_rows = []
    for frame in range(first_frame, last_frame):
        if frames[frame].get("ball_px") is not None:
            continue
        for candidate in frames[frame].get("candidates", ()):
            x, y, confidence = candidate[:3]
            raw_rows.append({
                "frame": frame,
                "time_s": round(frame / fps, 3),
                "xy": [round(float(x), 2), round(float(y), 2)],
                "confidence": round(float(confidence), 3),
                "racket_gap": round(float(racket_gap(frame, x, y)), 3),
                "player_gap": round(float(player_gap(frame, x, y)[0]), 3),
            })
    print("UNTRACKED_CANDIDATES=" + json.dumps(raw_rows))


def _rounded(value):
    return tuple(round(float(item), 3) for item in value) if value is not None else None


def _json_value(value):
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.floating | float):
        return round(float(value), 4) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=10.0)
    arguments = parser.parse_args()
    main(arguments.end, arguments.start)
