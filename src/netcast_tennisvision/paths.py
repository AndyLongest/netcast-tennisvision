"""Repository-local paths shared by production entry points."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]


def repository_path(*parts: str) -> Path:
    """Return a path inside the movable repository boundary."""
    return REPOSITORY_ROOT.joinpath(*parts)
