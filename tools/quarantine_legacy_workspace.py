"""Reversibly isolate files outside the canonical Tennis_Vision repository.

Nothing is deleted. The parent workspace's other children are atomically moved to a
sibling quarantine directory after a SHA-256 inventory is written. ``--restore`` moves
every entry back using that inventory. Dry-run is the default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent.resolve()
PROTECTED_NAMES = {ROOT.name, ".git"}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inventory(source: Path) -> dict[str, object]:
    files = (
        [source]
        if source.is_file()
        else sorted(path for path in source.rglob("*") if path.is_file())
    )
    return {
        "name": source.name,
        "kind": "directory" if source.is_dir() else "file",
        "bytes": sum(path.stat().st_size for path in files),
        "files": [
            {
                "path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }


def validate_quarantine(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.parent != WORKSPACE.parent or resolved == WORKSPACE:
        raise ValueError("quarantine must be a new sibling of the parent workspace")
    return resolved


def apply(quarantine: Path) -> int:
    if quarantine.exists():
        raise FileExistsError(f"quarantine already exists: {quarantine}")
    sources = sorted(path for path in WORKSPACE.iterdir() if path.name not in PROTECTED_NAMES)
    if not sources:
        print("No legacy siblings found.")
        return 0
    print("Will quarantine:")
    for source in sources:
        print(f"  {source}")
    entries = [inventory(source) for source in sources]
    quarantine.mkdir(parents=False)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "workspace": str(WORKSPACE),
        "canonical_repository": str(ROOT),
        "entries": entries,
    }
    manifest_path = quarantine / "quarantine-manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    moved: list[tuple[Path, Path]] = []
    try:
        for source in sources:
            destination = quarantine / source.name
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
            print(f"[isolated] {source.name}")
    except Exception:
        for source, destination in reversed(moved):
            if destination.exists() and not source.exists():
                shutil.move(str(destination), str(source))
        raise
    print(f"Manifest: {manifest_path}")
    print("Nothing was deleted. Keep the quarantine until post-cleanup E2E passes.")
    return 0


def restore(quarantine: Path) -> int:
    manifest_path = quarantine / "quarantine-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if Path(str(manifest["workspace"])).resolve() != WORKSPACE:
        raise ValueError("manifest belongs to a different workspace")
    for entry in manifest["entries"]:
        source = quarantine / str(entry["name"])
        destination = WORKSPACE / str(entry["name"])
        if destination.exists():
            raise FileExistsError(f"restore target already exists: {destination}")
        shutil.move(str(source), str(destination))
        print(f"[restored] {destination.name}")
    print("Restore complete. The manifest remains in the quarantine directory.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.apply and args.restore:
        parser.error("choose either --apply or --restore")
    default_name = f"{WORKSPACE.name}-legacy-quarantine-{datetime.now():%Y%m%d-%H%M%S}"
    quarantine = validate_quarantine(args.quarantine or WORKSPACE.parent / default_name)
    if args.restore:
        return restore(quarantine)
    if not args.apply:
        print(f"Dry run only. Planned quarantine: {quarantine}")
        for path in sorted(
            item for item in WORKSPACE.iterdir() if item.name not in PROTECTED_NAMES
        ):
            print(f"  {path}")
        print("Re-run with --apply after reviewing this list.")
        return 0
    return apply(quarantine)


if __name__ == "__main__":
    sys.exit(main())
