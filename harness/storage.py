"""Filesystem placement policy for final-evaluation data and artifacts."""
from pathlib import Path

from .common import ROOT


def default_private_root():
    return Path.home() / "SocialSciBench-private"


def private_package_file(entry):
    """Only development inputs may be kept with the development checkout."""
    path = Path(entry["path"])
    return path.parts[0] == "data" and path.parts[1:2] != ("development",)


def external_directory(path, *, workspace=None, other_forbidden=()):
    """Reject repo-contained paths, including aliases/symlinks; this is not a sandbox."""
    workspace = Path(workspace or ROOT).absolute()
    boundaries = [workspace, *map(Path, other_forbidden)]
    # Frozen evaluators may live below a checkout's runs/ or freezes/ directory.
    boundaries.extend(p for p in workspace.parents if (p / ".git").exists())
    supplied = Path(path).absolute()
    resolved = supplied.resolve()
    for boundary in boundaries:
        if supplied.is_relative_to(boundary.absolute()) or resolved.is_relative_to(boundary.resolve()):
            raise ValueError(f"Private data and final-evaluation artifacts must be outside the repository "
                             f"and development workspace: {path}")
    return resolved
