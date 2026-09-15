"""FFmpeg settings for report videos.

Rendering is deliberately separate from inference.  Changing these settings cannot
change court, player, ball, trajectory, or landing results.  The environment
variables make the faster default easy to benchmark or roll back.
"""
from __future__ import annotations

import os


def raw_h264_output_args() -> list[str]:
    """Return quality-oriented x264 arguments with a faster production preset."""
    preset = os.environ.get("NETCAST_X264_PRESET", "veryfast").strip() or "veryfast"
    crf = os.environ.get("NETCAST_X264_CRF", "20").strip() or "20"
    return [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", crf,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
