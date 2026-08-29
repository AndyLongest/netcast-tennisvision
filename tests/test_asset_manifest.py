import hashlib
import json
from pathlib import Path

import pytest

from tools import install_assets


def test_manifest_paths_stay_inside_repository():
    for asset in install_assets.load_manifest():
        destination = install_assets.safe_destination(asset["path"])
        assert destination == install_assets.ROOT / asset["path"]


def test_every_asset_has_an_explicit_supported_delivery_method():
    supported = {
        "git",
        "download",
        "download_and_extract_state_dict",
        "rebuild_from_pinned_reference",
    }
    for asset in install_assets.load_manifest():
        assert asset["delivery"] in supported
        if asset["delivery"] == "git":
            assert install_assets.valid(asset, install_assets.ROOT / asset["path"])
        else:
            assert asset["download_url"].startswith("https://")


def test_safe_destination_rejects_escape():
    with pytest.raises(ValueError):
        install_assets.safe_destination("../outside.bin")


def test_asset_validation_checks_size_and_hash(tmp_path):
    payload = b"frozen-asset"
    path = tmp_path / "asset.bin"
    path.write_bytes(payload)
    asset = {
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    assert install_assets.valid(asset, path)
    path.write_bytes(payload + b"changed")
    assert not install_assets.valid(asset, path)


def test_production_manifest_matches_embedded_demo():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "tests/fixtures/production_manifest.json").read_text(encoding="utf-8")
    )
    scene = json.loads((root / "assets/demo/scene3d.json").read_text(encoding="utf-8"))
    assert len(scene["frames"]) == manifest["demo"]["frames"]
    assert len(scene["bounces"]) == manifest["demo"]["bounces"] == 28
    assert len(scene["hits"]) == manifest["demo"]["hits"] == 32
