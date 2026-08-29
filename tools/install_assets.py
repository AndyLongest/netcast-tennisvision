"""Install checksummed runtime/demo assets without ever training on user video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets" / "manifest.json"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_manifest(path: Path = MANIFEST) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("asset manifest has no assets list")
    return assets


def safe_destination(relative: str) -> Path:
    destination = (ROOT / relative).resolve()
    if destination != ROOT and ROOT not in destination.parents:
        raise ValueError(f"asset path escapes repository: {relative}")
    return destination


def valid(asset: dict[str, Any], path: Path) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(asset["size"])
        and sha256(path) == asset["sha256"]
    )


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Netcast-TennisVision-asset-installer/1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _verify_source(asset: dict[str, Any], path: Path) -> None:
    expected_size = int(asset.get("download_size", asset["size"]))
    expected_hash = str(asset.get("download_sha256", asset["sha256"]))
    if path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise RuntimeError(f"downloaded source checksum mismatch: {asset['id']}")


def _download_final(asset: dict[str, Any], destination: Path, staging: Path) -> None:
    _download(str(asset["download_url"]), staging)
    if not valid(asset, staging):
        raise RuntimeError(f"downloaded checkpoint checksum mismatch: {asset['id']}")
    os.replace(staging, destination)


def _extract_racketvision(asset: dict[str, Any], destination: Path, directory: Path) -> None:
    """Download the pinned MMEngine checkpoint and save its unchanged state_dict.

    The upstream checkpoint contains pickle metadata. It is loaded only after both its
    byte length and SHA-256 match the immutable manifest entry.
    """
    import torch

    source = directory / "balltrack_best.pth"
    _download(str(asset["download_url"]), source)
    _verify_source(asset, source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not state_dict:
        raise RuntimeError("RacketVision checkpoint contains no state_dict")
    staging = directory / destination.name
    torch.save(state_dict, staging)
    if not valid(asset, staging):
        raise RuntimeError("extracted RacketVision state_dict is not the frozen production file")
    os.replace(staging, destination)


def _rebuild_bounce_classifier(
    asset: dict[str, Any], destination: Path, directory: Path
) -> None:
    from netcast_tennisvision.events.bounce_sequence import train_open_classifier

    reference = directory / "bigDF.csv"
    _download(str(asset["download_url"]), reference)
    _verify_source(asset, reference)
    model = train_open_classifier(reference)
    staging = directory / destination.name
    with staging.open("wb") as stream:
        # Protocol 5 is part of the frozen artifact contract and reproduces the exact hash.
        pickle.dump(model, stream, protocol=5)
    if not valid(asset, staging):
        raise RuntimeError("rebuilt bounce classifier is not the frozen production file")
    os.replace(staging, destination)


def install_one(asset: dict[str, Any], source_dir: Path | None, verify_only: bool) -> bool:
    destination = safe_destination(str(asset["path"]))
    if valid(asset, destination):
        print(f"[ok] {asset['id']} -> {destination.relative_to(ROOT)}")
        return True
    if verify_only:
        print(f"[missing or invalid] {asset['id']} -> {destination.relative_to(ROOT)}")
        return False

    candidate = source_dir / Path(str(asset["path"])).name if source_dir else None
    if candidate and candidate.is_file():
        if not valid(asset, candidate):
            print(f"[rejected] checksum mismatch: {candidate}")
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, destination)
        print(f"[installed] {asset['id']} from {candidate}")
        return True

    delivery = str(asset.get("delivery", "download"))
    if delivery == "git":
        print(f"[missing] {asset['id']}: restore this bundled file from Git")
        return False
    url = asset.get("download_url")
    if not url:
        print(f"[not hosted] {asset['id']}: no verified download source is configured")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"[download] {asset['id']}")
        with tempfile.TemporaryDirectory(prefix=".asset-", dir=destination.parent) as temporary:
            directory = Path(temporary)
            if delivery == "download":
                _download_final(asset, destination, directory / destination.name)
            elif delivery == "download_and_extract_state_dict":
                _extract_racketvision(asset, destination, directory)
            elif delivery == "rebuild_from_pinned_reference":
                _rebuild_bounce_classifier(asset, destination, directory)
            else:
                raise ValueError(f"unsupported delivery method: {delivery}")
        print(f"[installed] {asset['id']} -> {destination.relative_to(ROOT)}")
        return True
    except Exception as error:  # keep setup output actionable for non-developers
        print(f"[failed] {asset['id']}: {error}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["runtime", "demo", "all"], default="all")
    parser.add_argument("--source-dir", type=Path, help="offline handoff directory")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    selected = [
        asset for asset in load_manifest()
        if args.group == "all" or asset.get("group") == args.group
    ]
    results = [install_one(asset, args.source_dir, args.verify_only) for asset in selected]
    if all(results):
        print(f"All {len(results)} selected assets are ready.")
        return 0
    print("Some assets are unavailable. See docs/PUBLICATION_CHECKLIST.md.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
