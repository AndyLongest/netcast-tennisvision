"""Regenerate the browser's embedded demo scene from the frozen JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = ROOT / "assets" / "demo" / "scene3d.json"
DEFAULT_OUTPUT = ROOT / "web" / "demo-scene.js"


def build(source: Path, output: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload.get("frames"), list):
        raise ValueError("demo scene is missing frames")
    if not isinstance(payload.get("bounces"), list) or not isinstance(payload.get("hits"), list):
        raise ValueError("demo scene is missing event arrays")
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    output.write_text(f"globalThis.TENNIS_DEMO_SCENE={body};\n", encoding="utf-8")
    print(
        f"wrote {output.relative_to(ROOT)}: {len(payload['frames'])} frames, "
        f"{len(payload['bounces'])} bounces, {len(payload['hits'])} hits"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
