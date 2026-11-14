#!/usr/bin/env python3
"""Post-build extras for the Zensical docs site.

Run after `zensical build`:

- copies every page's Markdown source to `site/md/<path>.md` (for LLMs),
- places `llms.txt` at the site root,
- writes a forwarding page (stub) at every old page path listed in `docs/redirects.json`,
- copies the new page's Markdown to `site/md/<old path>.md` so old Markdown links keep working.

Stubs are written after the build, so the site never indexes them (no sitemap or search entry).

Usage:
    python scripts/docs_extras.py [--docs docs] [--site site]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

STUB = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Moved</title>\n'
    '<meta name="robots" content="noindex"><link rel="canonical" href="{new}">\n'
    '<meta http-equiv="refresh" content="0; url={new}">\n'
    '<script>location.replace("{new}" + location.hash)</script></head>\n'
    '<body><p>This page moved to <a href="{new}">{new}</a>.</p></body></html>\n'
)
STUB_MARKER = "<title>Moved</title>"
SKIP_DIRS = ("includes",)


class StubCollisionError(Exception):
    """A stub would overwrite a real page."""


def page_url(md_path: str) -> str:
    """URL path (no leading slash) of a page: `a/b.md` -> `a/b/`, `a/index.md` -> `a/`."""
    if md_path.endswith("/index.md"):
        return md_path[: -len("index.md")]
    if md_path == "index.md":
        return ""
    return md_path[: -len(".md")] + "/"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def copy_markdown(docs: Path, site: Path) -> int:
    out = site / "md"
    if out.exists():
        shutil.rmtree(out)
    count = 0
    for src in sorted(docs.rglob("*.md")):
        rel = src.relative_to(docs)
        if rel.parts[0] in SKIP_DIRS:
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        count += 1
    return count


def write_stubs(table: dict[str, str], site: Path, base: str = "/") -> int:
    count = 0
    for old, new in sorted(table.items()):
        if old == new:
            continue
        target = site / (page_url(old) + "index.html")
        if target.exists() and STUB_MARKER not in target.read_text(encoding="utf-8"):
            raise StubCollisionError(f"{old}: {target} is a built page, not a stub")
        _write(target, STUB.format(new=base + page_url(new)))
        count += 1
    return count


def copy_old_markdown(table: dict[str, str], docs: Path, site: Path) -> int:
    count = 0
    for old, new in sorted(table.items()):
        if old == new:
            continue
        src = docs / new
        if not src.is_file():
            continue
        dest = site / "md" / old
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--docs", default="docs", type=Path)
    ap.add_argument("--site", default="site", type=Path)
    args = ap.parse_args(argv)
    docs, site = args.docs, args.site
    if not docs.is_dir() or not site.is_dir():
        print(
            f"error: need built {site} and sources {docs} (run zensical build first)",
            file=sys.stderr,
        )
        return 1
    table = json.loads((docs / "redirects.json").read_text(encoding="utf-8"))
    try:
        stubs = write_stubs(table, site)
    except StubCollisionError as err:
        print(f"error: stub collision: {err}", file=sys.stderr)
        return 2
    if (docs / "llms.txt").is_file():
        shutil.copy2(docs / "llms.txt", site / "llms.txt")
        print(f"Placed llms.txt at {site / 'llms.txt'}")
    print(f"Copied {copy_markdown(docs, site)} Markdown files to {site / 'md'}")
    print(f"Copied {copy_old_markdown(table, docs, site)} Markdown files to old paths")
    print(f"Wrote {stubs} stubs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
