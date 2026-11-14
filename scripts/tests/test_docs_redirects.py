"""The redirect table from old grafeo.dev page paths to the new page paths.

`fixtures/grafeo_dev_pages.txt` is the frozen list of every page under docs/ before the
restructure (never regenerate it). `docs/redirects.json` maps each of them to the page that
replaces it.
"""

import json
import os
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Anchors that were already missing on the target page before the restructure: broken links
# in shipped doc comments and READMEs, reported to the maintainer. They are listed so that the
# test still guards against new breakage; remove an entry when the link or heading is fixed.
# (None found: every anchor linked from the repo existed on its target page before the move.)
KNOWN_MISSING_ANCHORS = set()

# Links to pages that never existed (no page at that path before the restructure).
# packages/grafeo-cli-python/pyproject.toml points its Documentation URL at /user-guide/cli,
# the CLI page is getting-started/cli.
KNOWN_MISSING_PAGES = {"/user-guide/cli"}


def fixture_lines(name):
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def table():
    return json.loads((DOCS / "redirects.json").read_text(encoding="utf-8"))


def test_fixture_is_frozen_at_204_pages():
    old = fixture_lines("grafeo_dev_pages.txt")
    assert len(old) == 204
    assert old == sorted(old)


def test_every_old_page_has_a_redirect():
    old = fixture_lines("grafeo_dev_pages.txt")
    assert sorted(table()) == old


def test_every_target_exists():
    for src, dst in table().items():
        assert (DOCS / dst).is_file(), f"{src} -> {dst}"


def test_no_old_path_is_a_new_page():
    # stubs are written at old paths; an old path that is also a new page would be overwritten
    for src, dst in table().items():
        if src != dst:
            assert not (DOCS / src).exists(), src


# --- links from the rest of the repo -------------------------------------------------------

URL = re.compile(r"https://(?:www\.)?grafeo\.dev(/[^\s)\"'<>`\]\\,;*|{}]*)?")
SKIP_DIRS = {
    "target",
    "node_modules",
    ".git",
    "pkg",
    "dist",
    ".cache",
    "site",
    "docs",
    ".venv",
    "__pycache__",
}
SCAN_ROOTS = ["crates", "examples", "packages"]
TEXT_SUFFIXES = {
    ".rs",
    ".md",
    ".py",
    ".ts",
    ".js",
    ".mjs",
    ".toml",
    ".json",
    ".go",
    ".cs",
    ".dart",
    ".c",
    ".h",
    ".txt",
    ".yml",
    ".yaml",
    ".html",
    ".cfg",
    ".csproj",
    ".rst",
    ".pyi",
}


def repo_files():
    yield from ROOT.glob("*.md")
    for name in SCAN_ROOTS:
        for dirpath, dirnames, filenames in os.walk(ROOT / name):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                p = Path(dirpath) / filename
                if p.suffix.lower() in TEXT_SUFFIXES:
                    yield p


def repo_links():
    """Yield (file, url_path_with_fragment) for every grafeo.dev URL."""
    for path in repo_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in URL.finditer(text):
            rest = (match.group(1) or "/").rstrip(".:!?")
            yield path.relative_to(ROOT).as_posix(), rest


def slugify(text):
    # Python-Markdown's toc slugify
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def heading_ids(page):
    ids, seen = set(), {}
    in_fence = False
    for line in page.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for explicit in re.findall(r"\{[^}]*#([\w-]+)[^}]*\}", line):
            ids.add(explicit)
        for explicit in re.findall(r'<a\s+[^>]*(?:id|name)="([^"]+)"', line):
            ids.add(explicit)
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        title = re.sub(r"\s*\{[^}]*\}\s*$", "", m.group(2))
        title = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", title)
        title = re.sub(r"<[^>]+>", "", title)
        title = (
            re.sub(r"[`*_]{1,3}", lambda _m: "", title)
            if "`" in title or "*" in title
            else title
        )
        base = slugify(title) or "section"
        n = seen.get(base, 0)
        seen[base] = n + 1
        ids.add(base if n == 0 else f"{base}_{n}")
    return ids


def resolve(url_path, pages):
    """Return (kind, target) for a URL path: ('page', new page), ('static', path) or (None, None)."""
    path = url_path.split("#", 1)[0].split("?", 1)[0].lstrip("/")
    if path == "":
        return "page", pages.get("index.md", "index.md")
    if (DOCS / path).is_file() and not path.endswith(".md"):
        return "static", path
    stem = path.rstrip("/")
    for cand in (f"{stem}/index.md", f"{stem}.md"):
        if cand in pages:
            return "page", pages[cand]
    return None, None


def test_repo_links_resolve():
    pages = table()
    problems = []
    seen = set()
    for source, url_path in repo_links():
        key = (source, url_path)
        if key in seen:
            continue
        seen.add(key)
        kind, target = resolve(url_path, pages)
        if kind is None:
            if url_path in KNOWN_MISSING_PAGES:
                continue
            problems.append(f"{source}: {url_path}: no such page")
            continue
        fragment = url_path.split("#", 1)[1] if "#" in url_path else ""
        if kind == "page" and fragment:
            if (
                fragment in heading_ids(DOCS / target)
                or url_path in KNOWN_MISSING_ANCHORS
            ):
                continue
            problems.append(
                f"{source}: {url_path}: no heading '{fragment}' on {target}"
            )
    assert not problems, "\n".join(problems)
