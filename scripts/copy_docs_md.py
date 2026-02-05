#!/usr/bin/env python3
"""Copy markdown documentation files to the built mkdocs site for LLM consumption.

This script copies the source markdown files from the docs/ directory to the
built site/ directory, making them accessible at /md/ paths. This allows LLMs
to fetch clean markdown instead of parsing HTML.

Usage:
    python scripts/copy_docs_md.py

The script reads mkdocs.yml to determine the docs and site directories.

Inspired by:
    https://github.com/koaning/wigglystuff/blob/main/scripts/copy_docs_md.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path


def parse_mkdocs_config(config_path: Path) -> dict[str, str]:
    """Parse mkdocs.yml to extract docs_dir and site_dir."""
    config = {
        "docs_dir": "docs",
        "site_dir": "site",
    }

    if not config_path.exists():
        return config

    content = config_path.read_text(encoding="utf-8")

    for key in config:
        match = re.search(rf"^{key}:\s*(.+)$", content, re.MULTILINE)
        if match:
            value = match.group(1).strip().strip("'\"")
            config[key] = value

    return config


def should_exclude(path: Path, exclude_patterns: list[str]) -> bool:
    """Check if a path should be excluded based on patterns."""
    path_str = str(path)
    for pattern in exclude_patterns:
        if pattern in path_str:
            return True
    return False


def copy_markdown_files(
    docs_dir: Path,
    site_dir: Path,
    output_subdir: str = "md",
    exclude_patterns: list[str] | None = None,
) -> int:
    """Copy markdown files from docs to site directory.

    Args:
        docs_dir: Source documentation directory
        site_dir: Built site directory
        output_subdir: Subdirectory in site for markdown files
        exclude_patterns: Patterns to exclude from copying

    Returns:
        Number of files copied
    """
    if exclude_patterns is None:
        exclude_patterns = ["includes/", "overrides/"]

    output_dir = site_dir / output_subdir
    copied_count = 0

    # Clean existing output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Find and copy all markdown files
    for md_file in docs_dir.rglob("*.md"):
        relative_path = md_file.relative_to(docs_dir)

        if should_exclude(relative_path, exclude_patterns):
            continue

        dest_path = output_dir / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(md_file, dest_path)
        copied_count += 1

    return copied_count


def copy_llms_txt(docs_dir: Path, site_dir: Path) -> bool:
    """Copy llms.txt to site root if it exists.

    Args:
        docs_dir: Source documentation directory
        site_dir: Built site directory

    Returns:
        True if file was copied, False otherwise
    """
    llms_file = docs_dir / "llms.txt"
    if llms_file.exists():
        shutil.copy2(llms_file, site_dir / "llms.txt")
        return True
    return False


def main() -> None:
    """Main entry point."""
    # Find project root (where mkdocs.yml is)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    config_path = project_root / "mkdocs.yml"
    config = parse_mkdocs_config(config_path)

    docs_dir = project_root / config["docs_dir"]
    site_dir = project_root / config["site_dir"]

    if not docs_dir.exists():
        print(f"Error: docs directory not found: {docs_dir}")
        return

    if not site_dir.exists():
        print(f"Warning: site directory not found: {site_dir}")
        print("Run 'mkdocs build' first, then run this script.")
        return

    # Copy llms.txt to site root
    if copy_llms_txt(docs_dir, site_dir):
        print(f"Copied llms.txt to {site_dir / 'llms.txt'}")

    # Copy markdown files
    copied = copy_markdown_files(docs_dir, site_dir)
    print(f"Copied {copied} markdown files to {site_dir / 'md'}")


if __name__ == "__main__":
    main()
