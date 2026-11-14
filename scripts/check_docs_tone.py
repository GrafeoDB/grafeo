#!/usr/bin/env python3
"""Check the docs' prose against the tone of voice.

    python scripts/check_docs_tone.py <docs dir> [--section <subdir or page>]

Prints `page:line: rule: text` per finding and exits 1 when there are any.

Rules:
    em-dash   the em dash character
    person    you, your, yours, we, our, us, and the pronoun I
    pip       `pip install` anywhere, code included, unless `uv pip install`
    buzzword  utilize, leverage, robust and the rest of the banned words

Prose only, except for `pip`: fenced code (at any indentation), inline code,
URLs, link targets, HTML comments (docs-check markers included) and YAML front
matter are ignored. CRLF reads like LF. Does not import grafeo.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_docs_examples import _FENCE, _fence_end  # noqa: E402

BUZZWORDS = (
    r"utiliz(?:e|es|ed|ing)",
    r"leverag(?:e|es|ed|ing)",
    r"facilitat(?:e|es|ed|ing)",
    r"performant",
    r"robust",
    r"comprehensive(?:ly)?",
    r"seamless(?:ly)?",
    r"blazing(?:ly)?",
    r"cutting[- ]edge",
    r"next[- ]generation",
    r"revolutionary",
    r"powerful",
    r"enterprise[- ]grade",
    r"holistic(?:ally)?",
    r"paradigms?",
)

# A word not glued to other word characters or hyphens ("us-east-1"). Apostrophes are
# allowed, so "we're" and "you'll" match.
_B = r"(?<![\w-])"
_E = r"(?![\w-])"
_PERSON = re.compile(_B + r"(?:you|your|yours|we|our|us)" + _E, re.IGNORECASE)
# "I" alone. Not "I/O", not a numeral after a label ("Part I", "Phase I").
_I = re.compile(_B + r"I" + _E)
_I_LABEL = re.compile(
    r"\b(?:part|phase|stage|level|tier|type|step|chapter|section|appendix|class|group|volume)\s+$",
    re.IGNORECASE,
)
_BUZZ = re.compile(_B + "(?:" + "|".join(BUZZWORDS) + ")" + _E, re.IGNORECASE)
_EM_DASH = re.compile("\u2014|&mdash;|&#0*8212;|&#x0*2014;", re.IGNORECASE)
_PIP = re.compile(r"(?<!uv )\bpip\s+install\b")

_INLINE_CODE = re.compile(r"(?P<t>`+)(?!`).+?(?<!`)(?P=t)(?!`)")
_URL = re.compile(
    r"(?:https?|ftp)://[^\s)>\]]+|<[a-z][a-z0-9+.-]*:[^\s>]+>|\bwww\.[^\s)>\]]+",
    re.IGNORECASE,
)
_LINK_TARGET = re.compile(r"\]\([^)]*\)")
_REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+\S+")
_HTML_ATTR = re.compile(
    r"""\b(?:href|src|srcset|data-[\w-]+)\s*=\s*(?:"[^"]*"|'[^']*')"""
)


def _blank(m: re.Match) -> str:
    return " " * len(m.group(0))


def _prose(line: str) -> str:
    """The line with inline code, URLs and link targets blanked out."""
    line = _INLINE_CODE.sub(_blank, line)
    line = _REF_DEF.sub(_blank, line)
    line = _LINK_TARGET.sub(_blank, line)
    line = _HTML_ATTR.sub(_blank, line)
    return _URL.sub(_blank, line)


def _strip_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """Blank out HTML comment text; return the line and whether a comment is still open."""
    out: list[str] = []
    i = 0
    while i < len(line):
        if in_comment:
            end = line.find("-->", i)
            if end < 0:
                out.append(" " * (len(line) - i))
                return "".join(out), True
            out.append(" " * (end + 3 - i))
            i, in_comment = end + 3, False
        else:
            start = line.find("<!--", i)
            if start < 0:
                out.append(line[i:])
                break
            out.append(line[i:start])
            i, in_comment = start, True
    return "".join(out), in_comment


def _has_pronoun_i(prose: str) -> bool:
    for m in _I.finditer(prose):
        if _I_LABEL.search(prose[: m.start()]):
            continue
        if (
            prose[m.end() : m.end() + 1] == "/"
            or prose[max(m.start() - 1, 0) : m.start()] == "/"
        ):
            continue
        return True
    return False


def findings(text: str) -> list[tuple[int, str, str]]:
    """Return (line, rule, text) for every finding, in page order. CRLF reads like LF."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[tuple[int, str, str]] = []

    def add(n: int, rule: str, raw: str) -> None:
        out.append((n, rule, raw.strip()))

    i = 0
    if lines and lines[0].strip() == "---":  # YAML front matter
        for j in range(1, len(lines)):
            if lines[j].strip() in ("---", "..."):
                i = j + 1
                break
    in_comment = False
    while i < len(lines):
        raw = lines[i]
        if not in_comment and _FENCE.match(raw):
            end = _fence_end(lines, i)
            for k in range(i, end + 1):
                if _PIP.search(lines[k]):
                    add(k + 1, "pip", lines[k])
            i = end + 1
            continue
        visible, in_comment = _strip_comments(raw, in_comment)
        if _PIP.search(visible):
            add(i + 1, "pip", raw)
        prose = _prose(visible)
        if _EM_DASH.search(prose):
            add(i + 1, "em-dash", raw)
        # "US" in capitals is the country, not the pronoun.
        if any(m.group(0) != "US" for m in _PERSON.finditer(prose)) or _has_pronoun_i(
            prose
        ):
            add(i + 1, "person", raw)
        if _BUZZ.search(prose):
            add(i + 1, "buzzword", raw)
        i += 1
    return out


def _pages(docs: Path, section: str | None) -> list[Path]:
    root = docs / section if section else docs
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise SystemExit(f"error: {root} is neither a directory nor a page")
    return sorted(root.rglob("*.md"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check the docs' prose against the tone of voice."
    )
    ap.add_argument("docs", type=Path)
    ap.add_argument(
        "--section", help="a subdirectory or a single page, relative to the docs dir"
    )
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8"
        )  # pages hold arrows and the like; Windows defaults to cp1252

    counts: dict[str, dict[str, int]] = {}
    total = 0
    for path in _pages(args.docs, args.section):
        rel = path.relative_to(args.docs).as_posix()
        section = rel.split("/", 1)[0] if "/" in rel else "(root)"
        for line, rule, text in findings(path.read_bytes().decode("utf-8")):
            print(f"{rel}:{line}: {rule}: {text}")
            counts.setdefault(section, {}).setdefault(rule, 0)
            counts[section][rule] += 1
            total += 1
    print(f"\n{total} finding(s)")
    for section, rules in sorted(counts.items()):
        print(
            f"  {section}: " + ", ".join(f"{r} {n}" for r, n in sorted(rules.items()))
        )
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
