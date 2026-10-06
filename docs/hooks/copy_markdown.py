"""MkDocs hooks for markdown file management.

Pre-build:
- Copies root CHANGELOG.md to docs/changelog.md (single source of truth)

Post-build:
- Copies llms.txt to site root
- Copies all markdown files to site/md/ for LLM consumption

Inspired by:
    https://github.com/koaning/wigglystuff/blob/main/scripts/copy_docs_md.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig

# Files to sync from project root to docs directory
ROOT_TO_DOCS_FILES = {
    "CHANGELOG.md": "changelog.md",
}


def on_pre_build(config: MkDocsConfig, **kwargs) -> None:
    """Copy files from project root to docs directory before build."""
    docs_dir = Path(config["docs_dir"])
    project_root = docs_dir.parent

    for source_name, dest_name in ROOT_TO_DOCS_FILES.items():
        source_file = project_root / source_name
        dest_file = docs_dir / dest_name

        if source_file.exists():
            shutil.copy2(source_file, dest_file)
            print(f"Synced {source_name} -> docs/{dest_name}")


def should_exclude(path: Path, exclude_patterns: list[str]) -> bool:
    """Check if a path should be excluded based on patterns."""
    path_str = str(path)
    for pattern in exclude_patterns:
        if pattern in path_str:
            return True
    return False


def on_post_build(config: MkDocsConfig, **kwargs) -> None:
    """Hook that runs after mkdocs build completes."""
    docs_dir = Path(config["docs_dir"])
    site_dir = Path(config["site_dir"])
    output_dir = site_dir / "md"
    exclude_patterns = ["includes/", "overrides/"]

    # Copy llms.txt to site root
    llms_file = docs_dir / "llms.txt"
    if llms_file.exists():
        shutil.copy2(llms_file, site_dir / "llms.txt")
        print(f"Copied llms.txt to {site_dir / 'llms.txt'}")

    # Clean existing output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Copy all markdown files
    copied_count = 0
    for md_file in docs_dir.rglob("*.md"):
        relative_path = md_file.relative_to(docs_dir)

        if should_exclude(relative_path, exclude_patterns):
            continue

        dest_path = output_dir / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, dest_path)
        copied_count += 1

    print(f"Copied {copied_count} markdown files to {output_dir}")
