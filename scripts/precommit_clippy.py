"""Clippy for the pre-commit hook, on what a commit changes.

Lints the workspace packages that hold a staged Rust file or manifest, and
every workspace package that depends on one of them (a change to a library
can break its dependents), with all targets and features, warnings denied.
A staged file outside every package (the root manifest, the lock file), or
no staged file at all (`prek run --all-files`), lints the whole workspace.

Usage: python scripts/precommit_clippy.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

LINT = ["--all-targets", "--all-features", "--", "-D", "warnings"]


def staged_files() -> list[str]:
    """The paths of the staged files, relative to the repository root."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line for line in out.splitlines() if line]


def is_rust(path: str) -> bool:
    return path.endswith((".rs", "Cargo.toml", "Cargo.lock"))


def owner(path: Path, packages: dict[str, Path]) -> str | None:
    """The package whose directory holds `path` most closely, or None."""
    best = None
    for name, directory in packages.items():
        if path.is_relative_to(directory) and (
            best is None or len(directory.parts) > len(packages[best].parts)
        ):
            best = name
    return best


def selection(metadata: dict, staged: list[str]) -> list[str] | None:
    """The packages to lint, or None for the whole workspace."""
    root = Path(metadata["workspace_root"])
    packages = {
        p["name"]: Path(p["manifest_path"]).parent for p in metadata["packages"]
    }
    rust = [root / path for path in staged if is_rust(path)]
    if not rust:
        return None
    changed = set()
    for path in rust:
        name = owner(path, packages)
        if name is None:
            return None
        changed.add(name)
    depends_on = {
        p["name"]: {d["name"] for d in p["dependencies"] if d["name"] in packages}
        for p in metadata["packages"]
    }
    selected = set(changed)
    grown = True
    while grown:
        grown = False
        for name, dependencies in depends_on.items():
            if name not in selected and dependencies & selected:
                selected.add(name)
                grown = True
    return sorted(selected)


def main() -> int:
    metadata = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    packages = selection(metadata, staged_files())
    if packages is None:
        command = ["cargo", "clippy", "--workspace", *LINT]
    else:
        command = [
            "cargo",
            "clippy",
            *(f"--package={name}" for name in packages),
            *LINT,
        ]
    print(" ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
