"""FFmpeg settings for report videos.

Rendering is deliberately separate from inference.  Changing these settings cannot
change court, player, ball, trajectory, or landing results.  The environment
variables make the faster default easy to benchmark or roll back.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache


def _x264_output_args() -> list[str]:
    preset = os.environ.get("NETCAST_X264_PRESET", "veryfast").strip() or "veryfast"
    crf = os.environ.get("NETCAST_X264_CRF", "20").strip() or "20"
    return [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", crf,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]


def _nvenc_output_args() -> list[str]:
    preset = os.environ.get("NETCAST_NVENC_PRESET", "p4").strip() or "p4"
    quality = os.environ.get("NETCAST_NVENC_CQ", "20").strip() or "20"
    return [
        "-c:v", "h264_nvenc",
        "-preset", preset,
        "-tune", "hq",
        "-rc", "vbr",
        "-cq", quality,
        "-b:v", "0",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]


@lru_cache(maxsize=1)
def _nvenc_usable() -> bool:
    """Prove that FFmpeg can open the NVIDIA encoder, not merely list it."""
    executable = shutil.which("ffmpeg")
    if not executable:
        return False
    command = [
        executable,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", "16x16",
        "-r", "1",
        "-i", "pipe:0",
        "-frames:v", "1",
        *_nvenc_output_args()[:-4],
        "-f", "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            input=bytes(16 * 16 * 3),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def raw_h264_output_args() -> list[str]:
    """Use verified NVENC when available, with the frozen x264 path as fallback."""
    mode = os.environ.get("NETCAST_VIDEO_ENCODER", "auto").strip().lower() or "auto"
    if mode not in {"auto", "nvenc", "x264"}:
        raise ValueError(f"unsupported NETCAST_VIDEO_ENCODER: {mode!r}")
    if mode in {"auto", "nvenc"} and _nvenc_usable():
        return _nvenc_output_args()
    return _x264_output_args()
