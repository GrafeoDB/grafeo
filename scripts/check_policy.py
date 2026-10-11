#!/usr/bin/env python3
"""Repository policy checks for Grafeo.

Usage:
    python scripts/check_policy.py tree [--metadata FILE|none]
    python scripts/check_policy.py diff [--base REF | --staged] [--paths P ...] [--allow RULE ...]
    python scripts/check_policy.py commit-msg FILE
    python scripts/check_policy.py pr --number N --head REF [--base REF] [--context FILE]

`tree` checks invariants that hold for the whole repository today (T rules). `diff` checks only
lines added since REF (default HEAD, compared through the merge base) or in the index, so existing
code is never flagged (D rules, W rules warn). `commit-msg` rejects AI co-author trailers (P3).
`pr` checks a pull request's eligibility (P rules) and the D rules on its diff; it reads the pull
request through the GitHub API (or --context) and its commits as git data, never running them.
Add `--json` for machine-readable output. Exit code 1 when an error is found.

    T1  crate boundaries (cargo metadata, normal and build dependencies)
    T2  workflow toolchain pins match rust-toolchain.toml or the MSRV
    T3  the private planning directory is never tracked or referenced
    T4  no em or en dashes in public docs, README, CHANGELOG, CONTRIBUTING, .github
    D1  an added #[allow(...)] states a reason = "..."
    D2  no em or en dashes in added Markdown lines or code comments
    D3  no added references to private planning notes; no internal phase wording in docs
    D4  no added #[ignore] on crash-injection tests (they must run in CI)
    D5  no new GraphStore, GraphStoreMut or GraphStoreSearch wrapper in library code (the
        stores themselves, STORES, implement the traits)
    D6  no `let _ =` in WAL, replay and recovery code (errors must propagate)
    W1  warning: new feature cfg in grafeo-core or grafeo-engine (features add modules)
    P1  pull requests target release/*; main only receives release branches
    P2  contributor pull requests link a planned issue (milestone, help wanted, good first issue)
    P3  commit messages: no AI co-author or "Generated with" lines; pull requests with such
        lines need the ownership box ticked (then a warning)
    P4  new dependencies need the `approved: deps` label
    P5  CI, scripts and root configuration need the `approved: infra` label
    P6  new crates and feature flags need the `approved: arch` label
    P7  bug fixes change a test, unless labelled `no-test-needed`
    P8  warning: changes to crates/ without a CHANGELOG.md entry
    P9  warning: more than 1,500 added lines outside tests

Contributor pull requests fail P2 and P4 to P6; maintainers see warnings instead (P2: nothing);
bots only get P3 and the D rules.

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
from typing import Any

DASH = re.compile("[\u2013\u2014]")
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
# An impl of a store trait itself (`impl<S> GraphStoreMut for X`), not an impl
# that only mentions one (`impl From<Arc<dyn GraphStoreMut>> for X`).
STORE_IMPL = re.compile(
    r"^impl\b(?:\s*<[^{]*?>)?\s+(?:[\w:]+::)?(GraphStore|GraphStoreMut|GraphStoreSearch)\s+for\s+([\w:]+)"
)
# The stores themselves, which implement the store traits without being
# wrappers; any other type that does is one. A new store is added here only
# by a user decision (D5's message names them).
STORES = {"LpgStore", "RowGroupStore"}
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
# "Made with [Cursor](...)", "Generated with Claude Code": the tool name follows the verb directly,
# so ordinary sentences such as "rows made with an open cursor" do not match.
GENERATED = re.compile(
    r"\b(generated|made|created|written|built) (with|by|using) \[?"
    r"(claude|chatgpt|copilot|codex|gemini|cursor|aider|devin|windsurf|jules)\b|\U0001f916",
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
    "D5": "cross-cutting concerns derive from the transaction change set, not from store wrappers (only the stores LpgStore and RowGroupStore implement the store traits)",
    "D6": "replay and recovery must propagate errors; handle or return the result",
    "W1": "features should add modules, not fork core types",
    "P1": "open the pull request against the current release/<milestone> branch",
    "P2": 'link a planned issue with "Fixes #N" (it needs a milestone, or the "help wanted" '
    'or "good first issue" label); for anything else, open an issue or discussion first',
    "P3": "AI co-author lines are not kept in the history; remove the line and name the AI "
    "tools in the pull request description instead (AI tools used: ...)",
    "P3PR": "tick the box in the PR description confirming you have read every line of this "
    "change, understand it and can explain it in review",
    "P4": 'new dependencies need a maintainer to agree and add the "approved: deps" label',
    "P5": 'CI, scripts and root configuration need a maintainer to agree and add the "approved: '
    'infra" label',
    "P6": 'new crates and feature flags need a maintainer to agree and add the "approved: arch" '
    "label",
    "P7": 'add a regression test for the fix (or a maintainer adds "no-test-needed")',
    "P8": "add a user-facing entry to CHANGELOG.md under the unreleased version",
    "P9": "consider splitting the pull request so it can be reviewed properly",
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


def finding(
    rule: str,
    path: str,
    line: int | None,
    detail: str = "",
    level: str | None = None,
    hint: str | None = None,
) -> Finding:
    advice = HINTS[hint or rule]
    message = f"{detail}; {advice}" if detail else advice
    if level is None:
        level = "warning" if rule.startswith("W") or rule in ("P8", "P9") else "error"
    return Finding(rule, path, line, message, level)


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
    root: Path,
    base: str | None,
    staged: bool,
    paths: list[str],
    head: str | None = None,
) -> dict[str, list[tuple[int, str]]]:
    """Added lines per path: staged changes, the working tree against the merge base with `base`,
    or (with `head`) the commits of `head` since its merge base with `base`."""
    args = ["diff", "--no-color", "--no-ext-diff", "-U0", "-M", "--diff-filter=AMR"]
    if staged:
        args.append("--cached")
    else:
        ref = base or "HEAD"
        merge_base = git(root, "merge-base", ref, head or "HEAD", check=False).strip()
        args.append(merge_base or ref)
        if head:
            args.append(head)
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
    if not staged and not head:
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
        feature_cfgs: list[int] = []
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
            store_impl = STORE_IMPL.match(text)
            if (
                library
                and store_impl
                and store_impl.group(2).split("::")[-1] not in STORES
            ):
                findings.append(finding("D5", path, number))
            if library and REPLAY_PATH.search(path) and LET_UNDERSCORE.search(text):
                findings.append(finding("D6", path, number))
            if FEATURE_CFG.search(text) and path.startswith(
                ("crates/grafeo-core/src/", "crates/grafeo-engine/src/")
            ):
                feature_cfgs.append(number)
        if (
            feature_cfgs
        ):  # one warning per file: it is a design nudge, not a line-level defect
            count = len(feature_cfgs)
            detail = f"{count} new feature cfg{'s' if count > 1 else ''}"
            findings.append(finding("W1", path, feature_cfgs[0], detail))
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


# --------------------------------------------------------------- pull requests

MAINTAINERS = {"OWNER", "MEMBER", "COLLABORATOR"}
PLANNED_LABELS = {"help wanted", "good first issue"}
LINKED_ISSUE = re.compile(
    r"\b(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?|refs?)\b:?\s*"
    r"(?:https://github\.com/GrafeoDB/grafeo/issues/|#)(\d+)",
    re.IGNORECASE,
)
OWNERSHIP_BOX = re.compile(
    r"^\s*[-*]\s*\[[xX]\]\s*I have read every line of this change", re.MULTILINE
)
# The template's "AI tools used:" field; "none", "no" or "n/a" (or nothing) declares no AI help.
AI_DECLARED = re.compile(
    r"^\s*AI tools used:[ \t]*(?!(?:none|no|n/?a)\b[ \t.]*$)\S.*$",
    re.IGNORECASE | re.MULTILINE,
)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
FIX_TITLE = re.compile(r"^fix(\([^)]*\))?!?:", re.IGNORECASE)
TEST_PATH = re.compile(
    r"(^|/)tests?/|\.gtest$|(^|/)test_[^/]*\.py$|_test\.[^/.]+$|\.(test|spec)\.[^/]+$"
)
TEST_ATTRIBUTE = re.compile(r"#\[\s*(tokio::)?test\b")
PROTECTED = re.compile(
    r"^(\.github/|scripts/|\.cargo/|Cargo\.toml$|rust-toolchain\.toml$|deny\.toml$|"
    r"codecov\.yml$|_typos\.toml$|bench-thresholds\.toml$|\.pre-commit-config\.yaml$)"
)
MANIFEST = re.compile(
    r"(^|/)(Cargo\.toml|package\.json|pyproject\.toml|go\.mod|pubspec\.yaml|[^/]+\.csproj)$"
)
DEPENDENCY_TABLES = ("dependencies", "dev-dependencies", "build-dependencies")
LARGE_PULL_REQUEST = 1500


def cargo_dependencies(document: dict) -> set[str]:
    keys: set[str] = set()
    for table in DEPENDENCY_TABLES:
        keys |= {f"{table}.{name}" for name in document.get(table, {})}
    workspace = document.get("workspace", {}).get("dependencies", {})
    keys |= {f"workspace.dependencies.{name}" for name in workspace}
    for target, tables in document.get("target", {}).items():
        for table in DEPENDENCY_TABLES:
            keys |= {
                f"target.{target}.{table}.{name}" for name in tables.get(table, {})
            }
    return keys


def requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return match.group(1).lower() if match else requirement


def yaml_block_keys(text: str, block: str) -> set[str]:
    """Keys directly under a top-level `block:` in a simple YAML file (pubspec.yaml)."""
    keys, inside = set(), False
    for line in text.splitlines():
        if re.match(rf"^{block}:\s*$", line):
            inside = True
        elif inside and re.match(r"^\S", line):
            inside = False
        elif inside and (match := re.match(r"^  ([A-Za-z0-9_]+):", line)):
            keys.add(match.group(1))
    return keys


def manifest_dependencies(path: str, text: str | None) -> set[str]:
    """Dependency names declared by a manifest, prefixed with their section."""
    if not text:
        return set()
    name = Path(path).name
    try:
        if name == "Cargo.toml":
            return cargo_dependencies(tomllib.loads(text))
        if name == "package.json":
            data = json.loads(text)
            sections = (
                "dependencies",
                "devDependencies",
                "peerDependencies",
                "optionalDependencies",
            )
            return {f"{s}.{k}" for s in sections for k in data.get(s, {})}
        if name == "pyproject.toml":
            data = tomllib.loads(text)
            project = data.get("project", {})
            found = {
                f"dependencies.{requirement_name(r)}"
                for r in project.get("dependencies", [])
            }
            for group, items in project.get("optional-dependencies", {}).items():
                found |= {f"{group}.{requirement_name(r)}" for r in items}
            for group, items in data.get("dependency-groups", {}).items():
                found |= {
                    f"{group}.{requirement_name(r)}"
                    for r in items
                    if isinstance(r, str)
                }
            return found
        if name == "go.mod":
            block = re.findall(
                r"^require\s*\(\s*$(.*?)^\)", text, re.MULTILINE | re.DOTALL
            )
            entries = [entry for b in block for entry in b.splitlines()]
            entries += re.findall(r"^require\s+(\S+\s+\S+)\s*$", text, re.MULTILINE)
            return {
                f"require.{entry.split()[0]}"
                for entry in entries
                if entry.strip() and not entry.strip().startswith("//")
            }
        if name.endswith(".csproj"):
            return {
                f"PackageReference.{p}"
                for p in re.findall(r'<PackageReference\s+Include="([^"]+)"', text)
            }
        if name == "pubspec.yaml":
            return {
                f"{b}.{k}"
                for b in ("dependencies", "dev_dependencies")
                for k in yaml_block_keys(text, b)
            }
    except (tomllib.TOMLDecodeError, ValueError):
        return set()
    return set()


def cargo_features(text: str | None) -> set[str]:
    try:
        return set(tomllib.loads(text or "").get("features", {}))
    except tomllib.TOMLDecodeError:
        return set()


def gh_api(path: str) -> Any:
    result = subprocess.run(
        ["gh", "api", path], capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise SystemExit(f"gh api {path} failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def load_context(repo: str, number: int) -> dict:
    """The pull request, its commit messages and its linked issues, from the GitHub API."""
    pull = gh_api(f"repos/{repo}/pulls/{number}")
    messages: list[str] = []
    for page in range(1, 4):  # the API returns at most 250 commits
        batch = gh_api(f"repos/{repo}/pulls/{number}/commits?per_page=100&page={page}")
        messages += [c["commit"]["message"] for c in batch]
        if len(batch) < 100:
            break
    body = HTML_COMMENT.sub("", pull.get("body") or "")
    issues = {
        n: gh_api(f"repos/{repo}/issues/{n}")
        for n in sorted(set(LINKED_ISSUE.findall(body)))
    }
    return {"pull": pull, "commits": messages, "issues": issues}


def pull_findings(context: dict, root: Path, base: str, head: str) -> list[Finding]:
    """P1 to P9 and the D rules for one pull request, graded by who opened it."""
    pull = context["pull"]
    user = pull["user"]
    if user.get("type") == "Bot" or user["login"].endswith("[bot]"):
        author = "bot"
    elif pull.get("author_association") in MAINTAINERS:
        author = "maintainer"
    else:
        author = "external"
    external = author == "external"
    labels = {label["name"] for label in pull.get("labels", [])}
    title = pull.get("title") or ""
    body = HTML_COMMENT.sub("", pull.get("body") or "")
    base_ref, head_ref = pull["base"]["ref"], pull["head"]["ref"]
    head_repo = (pull["head"].get("repo") or {}).get("full_name")
    release_merge = (
        base_ref == "main"
        and head_ref.startswith("release/")
        and head_repo == pull["base"]["repo"]["full_name"]
    )
    issues = [context["issues"][n] for n in sorted(context.get("issues", {}))]

    merge_base = git(root, "merge-base", base, head).strip()
    changes = []  # (status, path, previous path)
    for line in git(root, "diff", "--name-status", "-M", merge_base, head).splitlines():
        parts = line.split("\t")
        changes.append((parts[0][0], parts[-1], parts[1]))
    paths = {path for _, path, _ in changes} | {old for _, _, old in changes}

    def show(ref: str, path: str) -> str | None:
        return git(root, "show", f"{ref}:{path}", check=False) or None

    added = added_lines(root, base, False, [], head=head)
    findings: list[Finding] = []
    where = "PR"

    def grade(rule: str, path: str, detail: str, levels: dict, hint: str | None = None):
        level = levels.get(author)
        if level:
            findings.append(finding(rule, path, None, detail, level, hint))

    # P1: target branch
    if author != "bot" and not base_ref.startswith("release/") and not release_merge:
        grade(
            "P1",
            where,
            f"targets {base_ref}",
            {"external": "error", "maintainer": "warning"},
        )

    # P2: a planned, linked issue
    planned = [
        i
        for i in issues
        if i.get("milestone")
        or PLANNED_LABELS & {label["name"] for label in i.get("labels", [])}
    ]
    if external and not release_merge and not planned:
        detail = (
            "no linked issue" if not issues else "the linked issue is not planned yet"
        )
        grade("P2", where, detail, {"external": "error"})

    # P3: AI co-author or generated-by lines need the ownership box
    ai_lines = [
        line.strip()
        for message in [*context.get("commits", []), body]
        for line in message.splitlines()
        if not line.startswith("#")
        and (AI_TRAILER.search(line) or GENERATED.search(line))
    ] + [match.group(0).strip() for match in AI_DECLARED.finditer(body)]
    if ai_lines and author != "bot":
        ticked = bool(OWNERSHIP_BOX.search(body))
        detail = f"AI assistance noted ({ai_lines[0][:80]})"
        if ticked:
            detail += "; ownership box ticked, maintainers land this as a squash"
        grade(
            "P3",
            where,
            detail,
            {"external": "warning" if ticked else "error", "maintainer": "warning"},
            hint="P3PR",
        )

    # P4 and P6: manifests
    for status, path, old in changes:
        if not MANIFEST.search(path) or status == "D":
            continue
        new_text = show(head, path)
        old_text = None if status == "A" else show(merge_base, old)
        new_deps = sorted(
            manifest_dependencies(path, new_text)
            - manifest_dependencies(path, old_text)
        )
        if new_deps:
            approved = "approved: deps" in labels
            grade(
                "P4",
                path,
                f"adds {', '.join(new_deps)}" + (" (approved)" if approved else ""),
                {
                    "external": "warning" if approved else "error",
                    "maintainer": "warning",
                },
            )
        if Path(path).name == "Cargo.toml":
            approved = "approved: arch" in labels
            levels = {
                "external": "warning" if approved else "error",
                "maintainer": "warning",
            }
            if status == "A":
                grade(
                    "P6",
                    path,
                    "new crate" + (" (approved)" if approved else ""),
                    levels,
                )
            new_features = sorted(cargo_features(new_text) - cargo_features(old_text))
            if new_features and status != "A":
                grade(
                    "P6", path, f"new feature flags {', '.join(new_features)}", levels
                )

    # P5: protected paths
    protected = sorted(p for p in paths if PROTECTED.match(p))
    if protected:
        approved = "approved: infra" in labels
        grade(
            "P5",
            where,
            f"changes {', '.join(protected[:5])}"
            + (" and more" if len(protected) > 5 else ""),
            {"external": "warning" if approved else "error"},
        )

    # P7: bug fixes come with a test
    is_fix = FIX_TITLE.match(title) or any(
        (i.get("type") or {}).get("name") == "Bug" for i in issues
    )
    has_test = any(TEST_PATH.search(p) for _, p, _ in changes) or any(
        TEST_ATTRIBUTE.search(text) for lines in added.values() for _, text in lines
    )
    if is_fix and not has_test and not release_merge and "no-test-needed" not in labels:
        grade(
            "P7",
            where,
            "a fix without a test change",
            {"external": "error", "maintainer": "error"},
        )

    # P8 and P9: changelog and size
    if not release_merge and author != "bot":
        crates = [
            p for p in paths if p.startswith("crates/") and not TEST_PATH.search(p)
        ]
        if base_ref.startswith("release/") and crates and "CHANGELOG.md" not in paths:
            grade(
                "P8",
                where,
                "no CHANGELOG.md entry",
                {"external": "warning", "maintainer": "warning"},
            )
        size = sum(len(lines) for p, lines in added.items() if not TEST_PATH.search(p))
        if size > LARGE_PULL_REQUEST:
            grade(
                "P9",
                where,
                f"{size} added lines outside tests",
                {"external": "warning", "maintainer": "warning"},
            )

    # D rules on the diff, read from the pull request's commits (never the working tree)
    for item in diff_findings(added, lambda rel: show(head, rel)):
        if item.rule == "D5" and "approved: arch" in labels:
            item.level = "warning"
        findings.append(item)

    summary = f"PR #{pull['number']} by {user['login']} ({author}); labels: {', '.join(sorted(labels)) or 'none'}"
    print(summary, file=sys.stderr)
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
    pull = commands.add_parser(
        "pr", help="pull request eligibility (P1 to P9) and the D rules on its diff"
    )
    pull.add_argument("--number", type=int, required=True)
    pull.add_argument(
        "--head",
        required=True,
        help="git ref of the pull request head (fetched as data)",
    )
    pull.add_argument(
        "--base", help="git ref of the base branch (default origin/<base>)"
    )
    pull.add_argument(
        "--repo", default=os.environ.get("GITHUB_REPOSITORY", "GrafeoDB/grafeo")
    )
    pull.add_argument(
        "--context",
        help="JSON with pull, commits and issues, instead of the GitHub API",
    )
    for sub in (tree, diff, message, pull):
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
    if args.command == "pr":
        if args.context:
            context = json.loads(Path(args.context).read_text(encoding="utf-8"))
        else:
            context = load_context(args.repo, args.number)
        base = args.base or f"origin/{context['pull']['base']['ref']}"
        return report(pull_findings(context, root, base, args.head), args.json)

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
