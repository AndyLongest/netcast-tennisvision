from pathlib import Path

from netcast_tennisvision.paths import REPOSITORY_ROOT


def test_repository_root_is_resolved_from_the_installed_package() -> None:
    assert (REPOSITORY_ROOT / "pyproject.toml").is_file()
    assert (REPOSITORY_ROOT / "src" / "netcast_tennisvision").is_dir()


def test_repository_root_contains_no_python_modules() -> None:
    root_modules = sorted(path.name for path in REPOSITORY_ROOT.glob("*.py"))
    assert root_modules == []


def test_production_package_has_explicit_ownership_layers() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "netcast_tennisvision"
    assert {path.name for path in package.iterdir() if path.is_dir()} >= {
        "api",
        "events",
        "pipeline",
        "tracking",
        "vision",
    }
