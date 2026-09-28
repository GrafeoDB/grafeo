#!/usr/bin/env python3
"""Repository policy checks for Grafeo.

Usage:
    python scripts/check_policy.py tree [--metadata FILE|none]
    python scripts/check_policy.py diff [--base REF | --staged] [--paths P ...] [--allow RULE ...]
    python scripts/check_policy.py commit-msg FILE

`tree` checks invariants that hold for the whole repository today (T rules). `diff` checks only
lines added since REF (default HEAD, compared through the merge base) or in the index, so existing
code is never flagged (D rules, W rules warn). `commit-msg` rejects AI co-author trailers (P3).
Add `--json` for machine-readable output. Exit code 1 when an error is found.

    T1  crate boundaries (cargo metadata, normal and build dependencies)
    T2  workflow toolchain pins match rust-toolchain.toml or the MSRV
    T3  the private planning directory is never tracked or referenced
    T4  no em or en dashes in public docs, README, CHANGELOG, CONTRIBUTING, .github
    D1  an added #[allow(...)] states a reason = "..."
    D2  no em or en dashes in added Markdown lines or code comments
    D3  no added references to private planning notes; no internal phase wording in docs
    D4  no added #[ignore] on crash-injection tests (they must run in CI)
    D5  no new GraphStore, GraphStoreMut or GraphStoreSearch wrapper in library code
    D6  no `let _ =` in WAL, replay and recovery code (errors must propagate)
    W1  warning: new feature cfg in grafeo-core or grafeo-engine (features add modules)
    P3  no AI co-author or "Generated with" lines in commit messages

Standard library only (Python 3.11+). Tests: scripts/tests/test_check_policy.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

DASH = re.compile("[–—]")
PRIVATE_DIR = re.compile(r"(?<![\w])\.claude(?![\w.-])")
PRIVATE_ALLOWED = {
    ".gitignore",
    "scripts/check_policy.py",
    "scripts/tests/test_check_policy.py",
}
INTERNAL_PHRASES = re.compile(
    r"\bPhase [0-9]|consolidated plan|internal roadmap", re.IGNORECASE
)
ALLOW_ATTRIBUTE = re.compile(r"#!?\[\s*(?:cfg_attr\s*\(.*?,\s*)?allow\s*\(")
IGNORE_ATTRIBUTE = re.compile(r"#\[\s*ignore\b")
STORE_IMPL = re.compile(
    r"^impl\b[^{]*\b(GraphStore|GraphStoreMut|GraphStoreSearch)\b[^{]*\bfor\b"
)
LET_UNDERSCORE = re.compile(r"\blet\s+_\s*(:[^=]*)?=")
REPLAY_PATH = re.compile(
    r"(^|/)(wal|recovery|replay)(/|[_.])|/(recovery|replay)[^/]*\.rs$"
)
FEATURE_CFG = re.compile(r"#!?\[\s*cfg(_attr)?\s*\(.*\bfeature\s*=")
LIBRARY_SOURCE = re.compile(r"^crates/(bindings/)?[^/]+/src/")
AI_TRAILER = re.compile(
    r"^\s*co-authored-by:.*(claude|anthropic|openai|chatgpt|codex|copilot|gemini|mistral|devin|"
    r"aider|cursoragent|cursor agent|jules\[bot\]|google-labs-jules)",
    re.IGNORECASE,
)
GENERATED = re.compile(
    r"generated (with|by) .*(claude|chatgpt|copilot|codex|gemini|cursor|aider|devin)|\U0001f916",
    re.I,
)
PUBLIC_TEXT = re.compile(
    r"^(docs/|\.github/.*\.(md|ya?ml)$|(README|CHANGELOG|CONTRIBUTING)\.md$)"
)
COMMENT_MARKERS = {
    ".rs": "//",
    ".py": "#",
    ".toml": "#",
    ".yml": "#",
    ".yaml": "#",
    ".sh": "#",
    ".ps1": "#",
}


# Allowed internal dependencies and forbidden I/O crates per crate (normal and build edges).
BOUNDARIES: dict[str, dict[str, set[str]]] = {
    "grafeo-common": {"internal": set()},
    "grafeo-core": {
        "internal": {"grafeo-common"},
        "transitive": {"tokio", "memmap2", "fs2"},
    },
    "grafeo-storage": {"internal": {"grafeo-common"}},
    "grafeo-adapters": {
        "internal": {"grafeo-common", "grafeo-core"},
        "direct": {"memmap2", "crc32fast", "fs2"},
        "transitive": {"tokio", "memmap2", "fs2"},
    },
}

HINTS = {
    "T2": "bump the pin together with rust-toolchain.toml (or rust-version for the MSRV job)",
    "T3": "restate what matters inline or link a public issue",
    "T4": "rewrite with a comma, colon or parentheses",
    "D1": 'add reason = "..." naming the bound or invariant that makes it safe',
    "D2": "em or en dash; rewrite with a comma, colon or parentheses",
    "D3": "public text must stand on its own; restate the content or link a public issue",
    "D4": "crash-injection tests must run in CI; make it fast enough instead of ignoring it",
    "D5": "cross-cutting concerns derive from the transaction change set, not from store wrappers",
    "D6": "replay and recovery must propagate errors; handle or return the result",
    "W1": "features should add modules, not fork core types",
    "P3": "land the change without AI co-author or generated-by lines",
}


@dataclass
class Finding:
    rule: str
    path: str
    line: int | None
    message: str
    level: str = "error"

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"{self.rule} {where} {self.level}: {self.message}"


def finding(rule: str, path: str, line: int | None, detail: str = "") -> Finding:
    message = f"{detail}; {HINTS[rule]}" if detail else HINTS[rule]
    return Finding(
        rule, path, line, message, "warning" if rule.startswith("W") else "error"
    )


# --------------------------------------------------------------------- helpers


def git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def comment_part(text: str, marker: str) -> str:
    """The part of a line after its comment marker, ignoring markers inside string literals."""
    quotes = '"' if marker == "//" else "\"'"
    quote = None
    i = 0
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in quotes:
            quote = c
        elif text.startswith(marker, i):
            return text[i + len(marker) :]
        i += 1
    return ""


def attribute_text(lines: list[str], start: int) -> str:
    """The attribute starting on line `start` (1-based), up to its closing bracket."""
    collected: list[str] = []
    depth = 0
    for line in lines[start - 1 : start + 29]:
        collected.append(line)
        in_string = False
        for i, c in enumerate(line):
            if c == '"' and (i == 0 or line[i - 1] != "\\"):
                in_string = not in_string
            elif not in_string and c == "[":
                depth += 1
            elif not in_string and c == "]":
                depth -= 1
        if depth <= 0:
            break
    return "\n".join(collected)


# ------------------------------------------------------------------ tree rules


def boundary_findings(metadata: dict) -> list[Finding]:
    """T1: allowed internal dependencies and forbidden I/O crates, from `cargo metadata` JSON."""
    packages = {p["id"]: p for p in metadata["packages"]}
    ids_by_name = {
        p["name"]: p["id"] for p in metadata["packages"] if p.get("source") is None
    }
    edges: dict[str, list[str]] = {}
    for node in metadata["resolve"]["nodes"]:
        edges[node["id"]] = [
            dep["pkg"]
            for dep in node["deps"]
            if any(kind.get("kind") in (None, "build") for kind in dep["dep_kinds"])
        ]

    def name(package_id: str) -> str:
        return packages[package_id]["name"]

    findings: list[Finding] = []
    for crate, rules in BOUNDARIES.items():
        crate_id = ids_by_name.get(crate)
        if crate_id is None:
            findings.append(
                Finding(
                    "T1",
                    "Cargo.toml",
                    None,
                    f"{crate} is missing from cargo metadata; update the boundary rules",
                )
            )
            continue
        direct = sorted({name(d) for d in edges.get(crate_id, [])})
        allowed = rules["internal"]
        for dep in direct:
            internal = dep.startswith("grafeo") and dep in ids_by_name
            if internal and dep not in allowed:
                others = ", ".join(sorted(allowed)) or "no other grafeo crate"
                findings.append(
                    Finding(
                        "T1",
                        "Cargo.toml",
                        None,
                        f"{crate} depends on {dep}; it may only depend on {others}",
                    )
                )
        for dep in direct:
            if dep in rules.get("direct", set()):
                findings.append(
                    Finding(
                        "T1",
                        "Cargo.toml",
                        None,
                        f"{crate} depends directly on {dep} (storage I/O belongs in grafeo-storage)",
                    )
                )
        forbidden = rules.get("transitive", set())
        if forbidden:
            parent: dict[str, str | None] = {crate_id: None}
            queue = deque([crate_id])
            reached: dict[str, str] = {}
            while queue:
                current = queue.popleft()
                for dep in edges.get(current, []):
                    if dep in parent:
                        continue
                    parent[dep] = current
                    if name(dep) in forbidden and name(dep) not in reached:
                        reached[name(dep)] = dep
                    queue.append(dep)
            for target in sorted(reached):
                chain, node = [], reached[target]
                while node is not None:
                    chain.append(name(node))
                    node = parent[node]
                findings.append(
                    Finding(
                        "T1",
                        "Cargo.toml",
                        None,
                        f"{crate} reaches {target} through {' -> '.join(reversed(chain))}",
                    )
                )
    return findings


def cargo_metadata(root: Path) -> dict:
    result = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--all-features", "--locked"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise SystemExit(f"cargo metadata failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def toolchain_findings(root: Path) -> list[Finding]:
    """T2: every pinned toolchain in the workflows is the channel or the MSRV."""
    toolchain_file, manifest = root / "rust-toolchain.toml", root / "Cargo.toml"
    if not toolchain_file.exists() or not manifest.exists():
        return []
    channel = tomllib.loads(toolchain_file.read_text(encoding="utf-8"))["toolchain"][
        "channel"
    ]
    msrv = (
        tomllib.loads(manifest.read_text(encoding="utf-8"))
        .get("workspace", {})
        .get("package", {})
        .get("rust-version")
    )
    allowed = {channel, msrv} - {None}
    version = r"(\d+\.\d+(?:\.\d+)?)"
    patterns = [
        (re.compile(rf"dtolnay/rust-toolchain@{version}\b"), allowed),
        (re.compile(rf"\bcargo \+{version}\b"), allowed),
        (re.compile(rf"rust-toolchain:\s*[\"']?{version}"), {channel}),
    ]
    findings = []
    for workflow in sorted((root / ".github" / "workflows").glob("*.y*ml")):
        rel = workflow.relative_to(root).as_posix()
        for number, line in enumerate(
            workflow.read_text(encoding="utf-8").splitlines(), 1
        ):
            for pattern, permitted in patterns:
                for match in pattern.finditer(line):
                    if match.group(1) not in permitted:
                        expected = " or ".join(sorted(permitted))
                        findings.append(
                            finding(
                                "T2",
                                rel,
                                number,
                                f"toolchain {match.group(1)} is not {expected}",
                            )
                        )
    return findings


def text_findings(root: Path, files: list[str]) -> list[Finding]:
    """T3 and T4 over tracked files."""
    findings = []
    for rel in files:
        if rel.startswith(".claude/"):
            findings.append(finding("T3", rel, None, "private directory is tracked"))
            continue
        private = rel not in PRIVATE_ALLOWED
        public = bool(PUBLIC_TEXT.match(rel))
        if not private and not public:
            continue
        text = read_text(root / rel)
        if text is None:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if private and PRIVATE_DIR.search(line):
                findings.append(
                    finding("T3", rel, number, "reference to private planning notes")
                )
            if public and DASH.search(line):
                findings.append(finding("T4", rel, number))
    return findings


def tree_findings(root: Path, metadata_arg: str | None) -> list[Finding]:
    files = git(root, "ls-files").splitlines()
    findings = toolchain_findings(root) + text_findings(root, files)
    if metadata_arg != "none" and (root / "Cargo.toml").exists():
        if metadata_arg:
            metadata = json.loads(Path(metadata_arg).read_text(encoding="utf-8"))
        else:
            metadata = cargo_metadata(root)
        findings += boundary_findings(metadata)
    return findings


# ------------------------------------------------------------------ diff rules


def added_lines(
    root: Path, base: str | None, staged: bool, paths: list[str]
) -> dict[str, list[tuple[int, str]]]:
    """Added lines per path: staged changes, or the working tree against the merge base with `base`."""
    args = ["diff", "--no-color", "--no-ext-diff", "-U0", "-M", "--diff-filter=AMR"]
    if staged:
        args.append("--cached")
    else:
        ref = base or "HEAD"
        merge_base = git(root, "merge-base", ref, "HEAD", check=False).strip()
        args.append(merge_base or ref)
    output = git(root, *args, "--", *paths)
    added: dict[str, list[tuple[int, str]]] = defaultdict(list)
    path, number = None, 0
    for line in output.splitlines():
        if line.startswith("+++ "):
            target = line[4:].strip('"')
            path = (
                None
                if target == "/dev/null"
                else target[2:]
                if target.startswith("b/")
                else target
            )
        elif line.startswith("@@"):
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)", line)
            number = int(match.group(1)) if match else 0
        elif line.startswith("+") and path is not None:
            added[path].append((number, line[1:]))
            number += 1
    if not staged:
        for rel in git(
            root, "ls-files", "--others", "--exclude-standard", "--", *paths
        ).splitlines():
            text = read_text(root / rel)
            if text is not None:
                added[rel] = list(enumerate(text.splitlines(), 1))
    return dict(added)


def is_crash_test(path: str, content: str) -> bool:
    return Path(path).name.startswith("crash") or "testing::crash" in content


def diff_findings(
    added: dict[str, list[tuple[int, str]]], read: Callable[[str], str | None]
) -> list[Finding]:
    findings: list[Finding] = []
    for path, lines in sorted(added.items()):
        suffix = Path(path).suffix
        content = read(path) or ""
        post_image = content.splitlines()
        library = bool(LIBRARY_SOURCE.match(path)) and "/tests/" not in path
        for number, text in lines:
            if suffix == ".rs" and ALLOW_ATTRIBUTE.search(text):
                if not re.search(
                    r"\breason\s*=",
                    attribute_text(post_image, number) if post_image else text,
                ):
                    findings.append(finding("D1", path, number))
            if suffix == ".md":
                dashed = DASH.search(text)
            elif suffix in COMMENT_MARKERS:
                dashed = DASH.search(comment_part(text, COMMENT_MARKERS[suffix]))
            else:
                dashed = None
            if dashed:
                findings.append(finding("D2", path, number))
            if path not in PRIVATE_ALLOWED and PRIVATE_DIR.search(text):
                findings.append(
                    finding("D3", path, number, "reference to private planning notes")
                )
            elif path.startswith("docs/") and INTERNAL_PHRASES.search(text):
                findings.append(
                    finding("D3", path, number, "internal planning wording")
                )
            if suffix != ".rs":
                continue
            if IGNORE_ATTRIBUTE.search(text) and is_crash_test(path, content):
                findings.append(finding("D4", path, number))
            if library and STORE_IMPL.match(text):
                findings.append(finding("D5", path, number))
            if library and REPLAY_PATH.search(path) and LET_UNDERSCORE.search(text):
                findings.append(finding("D6", path, number))
            if FEATURE_CFG.search(text) and path.startswith(
                ("crates/grafeo-core/src/", "crates/grafeo-engine/src/")
            ):
                findings.append(finding("W1", path, number))
    return findings


# ------------------------------------------------------------------ commit-msg


def commit_message_findings(name: str, text: str) -> list[Finding]:
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("#"):
            continue
        if AI_TRAILER.search(line) or GENERATED.search(line):
            findings.append(finding("P3", name, number, line.strip()))
    return findings


# ---------------------------------------------------------------------- output


def report(findings: list[Finding], as_json: bool) -> int:
    findings = sorted(findings, key=lambda f: (f.path, f.line or 0, f.rule))
    errors = [f for f in findings if f.level == "error"]
    if as_json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        for item in findings:
            print(item)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for item in findings:
            location = f"file={item.path}" + (f",line={item.line}" if item.line else "")
            print(
                f"::{item.level} {location}::{item.rule} {item.message}",
                file=sys.stderr,
            )
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary and findings:
            with open(summary, "a", encoding="utf-8") as handle:
                handle.write(
                    "| Rule | Location | Level | Message |\n| --- | --- | --- | --- |\n"
                )
                for item in findings:
                    where = f"{item.path}:{item.line}" if item.line else item.path
                    handle.write(
                        f"| {item.rule} | `{where}` | {item.level} | {item.message} |\n"
                    )
    print(
        f"policy: {len(errors)} error(s), {len(findings) - len(errors)} warning(s)",
        file=sys.stderr,
    )
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true", help="print findings as JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    tree = commands.add_parser("tree", help="whole-repository rules (T1 to T4)")
    tree.add_argument(
        "--metadata", help="cargo metadata JSON file, or 'none' to skip T1"
    )
    diff = commands.add_parser("diff", help="rules on added lines (D1 to D6, W1)")
    source = diff.add_mutually_exclusive_group()
    source.add_argument(
        "--base", help="compare the working tree with the merge base of REF and HEAD"
    )
    source.add_argument(
        "--staged", action="store_true", help="check the staged changes"
    )
    diff.add_argument(
        "--paths", nargs="*", default=[], help="limit the check to these paths"
    )
    diff.add_argument(
        "--allow", action="append", default=[], help="report RULE as a warning"
    )
    message = commands.add_parser("commit-msg", help="commit message rules (P3)")
    message.add_argument("file")
    for sub in (tree, diff, message):
        sub.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.command == "commit-msg":
        path = Path(args.file)
        return report(
            commit_message_findings(path.name, path.read_text(encoding="utf-8")),
            args.json,
        )

    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    if args.command == "tree":
        return report(tree_findings(root, args.metadata), args.json)

    def read(rel: str) -> str | None:
        if args.staged:
            return git(root, "show", f":{rel}", check=False) or None
        return read_text(root / rel)

    findings = diff_findings(
        added_lines(root, args.base, args.staged, args.paths), read
    )
    for item in findings:
        if item.rule in args.allow:
            item.level = "warning"
    return report(findings, args.json)


if __name__ == "__main__":
    sys.exit(main())
