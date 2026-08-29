"""Fail-fast installation audit for a new developer or coding agent."""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_IMPORTS = (
    "cv2",
    "netcast_tennisvision",
    "numpy",
    "scipy",
    "sklearn",
    "torch",
    "ultralytics",
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    state = "ok" if condition else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{state}] {label}{suffix}")
    return condition


def main() -> int:
    results: list[bool] = []
    results.append(check("Python 3.10–3.12", (3, 10) <= sys.version_info[:2] <= (3, 12), sys.version.split()[0]))
    for module in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module)
            results.append(check(f"import {module}", True))
        except Exception as exc:  # installation audit should report every missing component
            results.append(check(f"import {module}", False, str(exc)))

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    results.append(check("ffmpeg", bool(ffmpeg), ffmpeg or "not found on PATH"))
    results.append(check("ffprobe", bool(ffprobe), ffprobe or "not found on PATH"))

    asset_check = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "install_assets.py"), "--verify-only"],
        cwd=ROOT,
        check=False,
    )
    results.append(check("frozen assets", asset_check.returncode == 0))

    manifest_path = ROOT / "tests" / "fixtures" / "production_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        demo = manifest["demo"]
        results.append(check("production manifest", demo["bounces"] == 28 and demo["hits"] == 32))
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        results.append(check("production manifest", False, str(exc)))

    results.append(check("web entrypoint", (ROOT / "web" / "index.html").is_file()))
    print("Installation is ready." if all(results) else "Installation is incomplete.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
