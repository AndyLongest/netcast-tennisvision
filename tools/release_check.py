"""Repository handoff/publication gate with no network or destructive actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ESSENTIAL = (
    "AGENTS.md",
    "README.md",
    "docs/HANDOFF.md",
    "docs/CURRENT_ARCHITECTURE.md",
    "docs/DEVELOPMENT.md",
    "docs/NEXT_STEPS.md",
    "docs/API_AND_SCHEMAS.md",
    "docs/PUBLICATION_CHECKLIST.md",
    "assets/manifest.json",
    "assets/MODELS.md",
    "assets/demo/demo.mp4",
    "assets/demo/annotated_clip.mp4",
    "tests/fixtures/production_manifest.json",
    "web/index.html",
    "server.py",
    "pipeline_runner.py",
    "download_models.ps1",
)
SECRET_PATTERN = re.compile(
    r"(?:github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|"
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
TEXT_SUFFIXES = {".py", ".js", ".css", ".html", ".json", ".md", ".ps1", ".toml", ".yml", ".yaml"}
EXCLUDED_PARTS = {".git", ".venv", "data", "models", "outputs", "__pycache__"}


class Audit:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        print(f"[{'ok' if condition else 'FAIL'}] {message}")
        if not condition:
            self.failures.append(message)

    def warn(self, condition: bool, message: str) -> None:
        if not condition:
            print(f"[warn] {message}")
            self.warnings.append(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_SUFFIXES
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and not ("docs" in path.parts and path.suffix.lower() == ".log")
    ]


def check_demo(audit: Audit) -> None:
    production = json.loads(
        (ROOT / "tests" / "fixtures" / "production_manifest.json").read_text(encoding="utf-8")
    )
    scene_path = ROOT / production["demo"]["outputs"]["scene"]["path"]
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    audit.require(len(scene["frames"]) == production["demo"]["frames"], "demo frame count matches manifest")
    audit.require(len(scene["bounces"]) == production["demo"]["bounces"] == 28, "demo has frozen 28 bounces")
    audit.require(len(scene["hits"]) == production["demo"]["hits"] == 32, "demo has frozen 32 hits")
    audit.require(sha256(scene_path) == production["demo"]["outputs"]["scene"]["sha256"], "demo scene checksum matches manifest")
    compact = json.dumps(scene, ensure_ascii=False, separators=(",", ":"))
    expected_js = f"globalThis.TENNIS_DEMO_SCENE={compact};\n"
    audit.require((ROOT / "web" / "demo-scene.js").read_text(encoding="utf-8") == expected_js, "embedded browser demo is current")


def check_assets(audit: Audit, require_binaries: bool, require_urls: bool) -> None:
    manifest = json.loads((ROOT / "assets" / "manifest.json").read_text(encoding="utf-8"))
    for asset in manifest["assets"]:
        path = (ROOT / asset["path"]).resolve()
        audit.require(ROOT in path.parents, f"asset path is repository-local: {asset['id']}")
        delivery = asset.get("delivery", "download")
        if require_binaries or delivery == "git":
            valid = path.is_file() and path.stat().st_size == asset["size"] and sha256(path) == asset["sha256"]
            audit.require(valid, f"asset is present and frozen: {asset['id']}")
        if require_urls and asset.get("required") and delivery != "git":
            audit.require(bool(asset.get("download_url")), f"asset has publication URL: {asset['id']}")
    if require_urls:
        audit.require(manifest.get("publication_status") == "ready", "asset manifest is approved for publication")


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "handoff", "public"], default="handoff")
    args = parser.parse_args()
    audit = Audit()

    audit.require((ROOT / ".git").exists(), "command runs from the intended Tennis_Vision repository")
    for relative in ESSENTIAL:
        audit.require((ROOT / relative).is_file(), f"essential file exists: {relative}")

    files = source_files()
    secret_hits = [str(path.relative_to(ROOT)) for path in files if SECRET_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore"))]
    audit.require(not secret_hits, f"no obvious embedded credentials ({', '.join(secret_hits) or 'clean'})")
    absolute_hits = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"(?i)\b[A-Z]:\\(?:Users|Built|Alpha|anaconda)\\", text):
            absolute_hits.append(str(path.relative_to(ROOT)))
    audit.require(not absolute_hits, f"no workstation-specific absolute paths ({', '.join(absolute_hits) or 'clean'})")

    oversized = [
        str(path.relative_to(ROOT)) for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size >= 95 * 1024 * 1024
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    ]
    audit.require(not oversized, f"no publishable file approaches GitHub's 100MB limit ({', '.join(oversized) or 'clean'})")
    check_demo(audit)
    check_assets(audit, require_binaries=args.mode == "handoff", require_urls=args.mode == "public")

    if (ROOT.parent / ".git").exists():
        audit.warn(False, "parent workspace also contains .git; publish only the inner Tennis_Vision root")

    if args.mode == "public":
        audit.require((ROOT / "LICENSE").is_file(), "approved project LICENSE exists")
        audit.require(not git_output("status", "--porcelain"), "Git worktree is clean")
        remote = git_output("remote", "get-url", "origin")
        audit.require("vahehambardzumyan/Tennis_Vision" not in remote, "origin is owner-controlled, not upstream")

    print(f"\n{len(audit.failures)} failure(s), {len(audit.warnings)} warning(s)")
    if audit.failures:
        return 1
    print("Repository gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
