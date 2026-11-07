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
    T5  every published workspace crate declares the workspace's rust-version (the MSRV)
    T6  docs/stability.md's Experimental table and the `# Stability: Experimental` sections
        of the facade's crates agree both ways; the facade and its modules have a
        `# Stability` section
    T7  in a crate with `autotests = false`, every test file is a `mod` of tests/main.rs or a
        `[[test]]` entry with a comment saying why, not both
    T8  the Node.js package's platform folders, its `napi` targets and the release's build
        matrix name the same platforms, and its manifest lists no platform packages
    T9  every expected failure (a gtest `xfail:`, a strict pytest xfail mark, a Rust
        `xfail(...)`) names its spec and an issue: `#N`, or `fix in X.Y.Z` not below the
        release CHANGELOG.md's first `## [X.Y.Z]` heading names
    T10 per build: the facade's profiles, the CLI and the bindings reach no tokio, and the WAL
        no crossbeam, but through the crates allowed to depend on them (cargo tree)
    T11 per binding build (its default and each profile): no engine feature is compiled whose
        same-name feature of the binding is off (cargo tree)
    D1  an added #[allow(...)] states a reason = "..."
    D2  no em or en dashes in added Markdown lines or code comments
    D3  no added references to private planning notes; no internal phase wording in docs
    D4  no added #[ignore] on crash-injection tests (they must run in CI)
    D5  no new GraphStore, GraphStoreMut, GraphStoreSearch or GraphBackend wrapper in library
        code (the stores themselves, STORES, implement the traits; test modules are not library
        code)
    D6  no `let _ =` in WAL, replay and recovery code (errors must propagate)
    D7  no added text addressed to the project's private workflow: its session roles, its
        inbox, its private design and decision notes
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
import ast
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
# Text addressed to the private workflow (D7): its session roles by name, its inbox, and its
# design and decision notes. The roles match as names only ("the Architect", "Review's", "by
# Review"), so ordinary words stay free: "a PR review", `review_threads`, "Review the plan",
# "the architecture", "an engineer".
WORKFLOW_TEXT = re.compile(
    r"\b(?:Architect|Orchestrator|git Manager)(?:'s)?\b"
    r"|\bthe Engineer(?:'s)?\b|\(Engineer\b|\bEngineer's\b"
    r"|\bReview(?:'s| Expert| session)\b|\b(?:by|from|for|with) Review\b|\(Review\b"
    r"|\bthe review's\b"
    r"|\binbox\b"
    r"|[\w.-]+-design\.md\b|\bdecisions\.md\b|\bquestions-pending\.md\b"
)
ALLOW_ATTRIBUTE = re.compile(r"#!?\[\s*(?:cfg_attr\s*\(.*?,\s*)?allow\s*\(")
IGNORE_ATTRIBUTE = re.compile(r"#\[\s*ignore\b")
# An impl of a store trait itself (`impl<S> GraphStoreMut for X`), not an impl
# that only mentions one (`impl From<Arc<dyn GraphStoreMut>> for X`).
STORE_IMPL = re.compile(
    r"^impl\b(?:\s*<[^{]*?>)?\s+(?:[\w:]+::)?(GraphStore|GraphStoreMut|GraphStoreSearch|GraphBackend)\s+for\s+([\w:]+)"
)
# The stores themselves, which implement the store traits without being
# wrappers; any other type that does is one. A new store is added here only
# by a user decision (D5's message names them).
STORES = {"RowGroupStore"}
LET_UNDERSCORE = re.compile(r"\blet\s+_\s*(:[^=]*)?=")
REPLAY_PATH = re.compile(
    r"(^|/)(wal|recovery|replay)(/|[_.])|/(recovery|replay)[^/]*\.rs$"
)
FEATURE_CFG = re.compile(r"#!?\[\s*cfg(_attr)?\s*\(.*\bfeature\s*=")
LIBRARY_SOURCE = re.compile(r"^crates/(bindings/)?[^/]+/src/")
# A file that is only a test module (`tests.rs`, `*_tests.rs`), compiled under `#[cfg(test)]`.
TEST_MODULE_FILE = re.compile(r"(^|/)(\w+_)?tests\.rs$")
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

# T10: per build, a crate it must not reach, and the crates allowed to depend on that one directly
# (a binding's own async API). `None`: the package's default features.
RUNTIME_RULES: list[tuple[str, str | None, str, frozenset[str]]] = [
    ("grafeo", "embedded", "tokio", frozenset()),
    ("grafeo", "edge", "tokio", frozenset()),
    ("grafeo", "lpg", "tokio", frozenset()),
    ("grafeo", "rdf", "tokio", frozenset()),
    ("grafeo", "analytics", "tokio", frozenset()),
    ("grafeo", "ai", "tokio", frozenset()),
    ("grafeo", "browser", "tokio", frozenset()),
    ("grafeo-cli", None, "tokio", frozenset()),
    ("grafeo-c", None, "tokio", frozenset()),
    (
        "grafeo-python",
        None,
        "tokio",
        frozenset({"grafeo-python", "pyo3-async-runtimes"}),
    ),
    ("grafeo-node", None, "tokio", frozenset({"grafeo-node", "napi"})),
    ("grafeo-storage", "wal", "tokio", frozenset()),
    ("grafeo-storage", "wal", "crossbeam", frozenset()),
]

# A build that must reach tokio: without it a renamed crate or another cargo output would let
# every rule above pass by reading nothing.
RUNTIME_CONTROL = ("grafeo", "async-storage", "tokio")

# T11: every binding crate that depends on the engine, with its default build (`None`) and each
# profile its manifest defines.
_BINDING_PROFILES = ("lpg", "rdf", "ai", "analytics", "edge", "embedded", "full")
PROFILE_BUILDS: list[tuple[str, str | None]] = [
    (package, features)
    for package, profiles in (
        ("grafeo-c", _BINDING_PROFILES),
        ("grafeo-node", _BINDING_PROFILES),
        ("grafeo-python", _BINDING_PROFILES),
        ("grafeo-wasm", ("lpg", "rdf", "ai", "analytics", "edge", "full")),
    )
    for features in (None, *profiles)
]

# A build that must compile an engine feature and report it: without it another cargo output
# would let every build above pass by reading nothing.
PROFILE_CONTROL = ("grafeo-python", "full", "vector-index")

HINTS = {
    "T2": "bump the pin together with rust-toolchain.toml (or rust-version for the MSRV job)",
    "T3": "restate what matters inline or link a public issue",
    "T4": "rewrite with a comma, colon or parentheses",
    "T5": "add `rust-version.workspace = true` to the crate's [package]",
    "T6": "an experimental item has a `# Stability: Experimental` section and a row in docs/stability.md; a module the facade exports has a `# Stability` section",
    "T7": "declare a test file with `mod name;` in tests/main.rs; a separate `[[test]]` entry has a comment above it saying why it cannot be there",
    "T8": "a platform needs its target in `napi.targets`, in the build matrix of release.yml, and a folder under npm/; the release adds the platform packages to the manifest (scripts/node-platform-deps.sh)",
    "T9": "an expected failure names its spec and an issue (`#N`, or `fix in X.Y.Z` for "
    "the release being made or a later one); a pytest mark is `strict=True` with `raises=` "
    'and `reason="spec: <ref>; issue: <issue>"`',
    "T11": "a binding's profile lists the features it compiles: turn the dependency's default "
    "features off and forward each feature the profile needs",
    "T10": "tokio belongs to `async-storage` and `server` (and a binding's own async API); "
    "make the dependency optional behind those features, or name the binding's edge in "
    "RUNTIME_RULES",
    "T12": "gate a binding's code on the capability it needs (`rdf-model`, `sparql`, "
    "`vector-index`, ...), not on a profile or group: every profile that holds the "
    "capability then compiles it",
    "T13": "copy crates/bindings/c/grafeo.h to crates/bindings/go/grafeo.h; declare each "
    "export in the header; regenerate crates/bindings/go/grafeo_c.def and libgrafeo_c.dll.a "
    "from the built grafeo_c.dll (gendef, then dlltool)",
    "D1": 'add reason = "..." naming the bound or invariant that makes it safe',
    "D2": "em or en dash; rewrite with a comma, colon or parentheses",
    "D3": "public text must stand on its own; restate the content or link a public issue",
    "D4": "crash-injection tests must run in CI; make it fast enough instead of ignoring it",
    "D5": "cross-cutting concerns derive from the transaction change set, not from store wrappers (only the store RowGroupStore implements the store traits)",
    "D6": "replay and recovery must propagate errors; handle or return the result",
    "D7": "public text must not address the private workflow; state the rule or the date "
    "itself, or link a public issue",
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


def inverse_chains(text: str, root_package: str) -> dict[str, list[str]]:
    """From `cargo tree -i CRATE --prefix depth --format {p}` of `root_package`: each crate that
    depends on CRATE directly, with one chain from `root_package` down to CRATE through it."""
    chains: dict[str, list[str]] = {}
    stack: list[str] = []
    for line in text.splitlines():
        match = re.match(r"(\d+)(\S+)", line)
        if not match:
            continue
        depth, name = int(match.group(1)), match.group(2)
        del stack[depth:]
        stack.append(name)
        if depth == 0:
            continue
        chain = list(reversed(stack))
        known = chains.get(stack[1])
        if known is None or (known[0] != root_package and chain[0] == root_package):
            chains[stack[1]] = chain
    return chains


def cargo_tree_inverse(
    root: Path, package: str, features: str | None, crate: str
) -> str | None:
    """`cargo tree -i crate` of `package` with `features` (`None`: its defaults), normal and build
    edges; `None` when the build does not reach `crate`."""
    args = ["cargo", "tree", "--locked", "-p", package, "-e", "normal,build"]
    if features is not None:
        args += ["--no-default-features", "--features", features]
    args += ["-i", crate, "--prefix", "depth", "--format", "{p}"]
    result = subprocess.run(
        args, cwd=root, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        if "did not match any packages" in result.stderr:
            return None
        raise SystemExit(f"{' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout if result.stdout.strip() else None


def runtime_findings(tree) -> list[Finding]:
    """T10: no build of RUNTIME_RULES reaches its crate but through an allowed direct dependent.
    `tree(package, features, crate)` is the output of `cargo_tree_inverse`."""
    findings: list[Finding] = []
    package, features, crate = RUNTIME_CONTROL
    if not tree(package, features, crate):
        findings.append(
            Finding(
                "T10",
                "Cargo.toml",
                None,
                f"{package} with --features {features} reaches no {crate}: the check reads "
                "nothing; update RUNTIME_CONTROL",
            )
        )
    for package, features, crate, allowed in RUNTIME_RULES:
        text = tree(package, features, crate)
        if not text:
            continue
        build = f"--features {features}" if features else "its default features"
        for dependent, chain in sorted(inverse_chains(text, package).items()):
            if dependent not in allowed:
                findings.append(
                    Finding(
                        "T10",
                        "Cargo.toml",
                        None,
                        f"{package} with {build} reaches {crate} through {' -> '.join(chain)}",
                    )
                )
    return findings


def reportable_features(manifest: str) -> set[str]:
    """The features of a binding's manifest that forward the engine feature of their own name
    (`gql = ["grafeo-engine/gql"]`): the ones a build can be judged by. Groups, profiles and
    features of another name do not count."""
    features = tomllib.loads(manifest).get("features", {})
    return {
        name for name, enables in features.items() if f"grafeo-engine/{name}" in enables
    }


def compiled_features(text: str, package: str) -> tuple[dict[str, list[str]], set[str]]:
    """From `cargo tree -e features -i grafeo-engine --prefix depth` of `package`: each engine
    feature compiled, with the features that enable it directly (`crate/feature`, in cargo's
    order), and the features of `package` that are on."""
    pattern = re.compile(r'^(\d+)(\S+) feature "([^"]+)"')
    engine: dict[str, list[str]] = {}
    own: set[str] = set()
    current: str | None = None
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            # A package line: an engine feature's enablers end at a line of its own depth.
            if line[:1].isdigit() and int(re.match(r"\d+", line).group()) <= 1:
                current = None
            continue
        depth, crate, feature = int(match.group(1)), match.group(2), match.group(3)
        if crate == package:
            own.add(feature)
        if depth == 1 and crate == "grafeo-engine":
            current = feature
            engine.setdefault(feature, [])
        elif depth == 1:
            current = None
        elif depth == 2 and current is not None:
            enabler = f"{crate}/{feature}"
            if enabler not in engine[current]:
                engine[current].append(enabler)
    return engine, own


def profile_findings(tree, manifests: dict[str, str]) -> list[Finding]:
    """T11: no build of PROFILE_BUILDS compiles an engine feature whose same-name feature of the
    binding is off. `tree(package, features)` is the build's `cargo tree` output (`None`: no tree
    to judge); `manifests` holds each binding's Cargo.toml text."""
    findings: list[Finding] = []
    package, features, feature = PROFILE_CONTROL
    text = tree(package, features)
    engine, own = compiled_features(text, package) if text else ({}, set())
    if feature not in engine or feature not in own:
        findings.append(
            Finding(
                "T11",
                "Cargo.toml",
                None,
                f"{package} with --features {features} compiles no engine {feature}: the check "
                "reads nothing; update PROFILE_CONTROL",
            )
        )
    for package, features in PROFILE_BUILDS:
        text = tree(package, features)
        manifest = manifests.get(package)
        if not text or manifest is None:
            continue
        engine, own = compiled_features(text, package)
        build = f"--features {features}" if features else "its default features"
        for name in sorted(engine.keys() & reportable_features(manifest)):
            if name in own:
                continue
            findings.append(
                Finding(
                    "T11",
                    "Cargo.toml",
                    None,
                    f"{package} with {build} compiles the engine's {name}, which its own {name} "
                    f"feature leaves off: enabled by {', '.join(engine[name])}",
                )
            )
    return findings


def cargo_tree_features(root: Path, package: str, features: str | None) -> str | None:
    """`cargo tree -e features -i grafeo-engine` of `package` with `features` (`None`: its
    defaults); `None` when the build does not reach the engine."""
    args = ["cargo", "tree", "--locked", "-p", package, "-e", "features"]
    if features is not None:
        args += ["--no-default-features", "--features", features]
    args += ["-i", "grafeo-engine", "--prefix", "depth"]
    result = subprocess.run(
        args, cwd=root, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        if "did not match any packages" in result.stderr:
            return None
        raise SystemExit(f"{' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout if result.stdout.strip() else None


def binding_manifests(root: Path) -> dict[str, str]:
    """Each binding crate's Cargo.toml text, by package name."""
    manifests: dict[str, str] = {}
    for manifest in sorted(root.glob("crates/bindings/*/Cargo.toml")):
        text = manifest.read_text(encoding="utf-8")
        name = tomllib.loads(text).get("package", {}).get("name")
        if name:
            manifests[name] = text
    return manifests


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


def rust_version_findings(root: Path, metadata: dict) -> list[Finding]:
    """T5: every published workspace crate declares the workspace's `rust-version`.

    A crate without it tells cargo nothing about the compiler it needs, so a user on an older
    one gets build errors instead of a clear message (#390: the `grafeo` crate declared none).
    Crates with `publish = false` (an empty `publish` list) are left out.
    """
    manifest = root / "Cargo.toml"
    msrv = (
        tomllib.loads(manifest.read_text(encoding="utf-8"))
        .get("workspace", {})
        .get("package", {})
        .get("rust-version")
        if manifest.exists()
        else None
    )
    if msrv is None:
        return []
    members = set(metadata.get("workspace_members", []))
    findings = []
    for package in sorted(metadata["packages"], key=lambda p: p["name"]):
        if package["id"] not in members or package.get("publish") == []:
            continue
        declared = package.get("rust_version")
        if declared == msrv:
            continue
        path = Path(package.get("manifest_path", "Cargo.toml"))
        try:
            rel = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path.as_posix()
        findings.append(
            finding(
                "T5",
                rel,
                None,
                f"{package['name']} declares rust-version {declared or 'none'}, "
                f"not the workspace's {msrv}",
            )
        )
    return findings


# ------------------------------------------------------------------ test binaries

#: A file module of `tests/main.rs`.
# A `mod name;` declaration, at the start of a line or after another item on it.
TEST_MODULE = re.compile(
    r"(?:^|[;{}])\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;", re.MULTILINE
)
TEST_TARGET_HEADER = re.compile(r"^\s*\[\[\s*test\s*\]\]")
RUST_RAW_STRING = re.compile(r'b?r(#*)"')
RUST_CHAR = re.compile(r"'(\\u\{[0-9a-fA-F]+\}|\\.|[^\\'\n])'")


def rust_code(text: str, literals: bool = False) -> str:
    """Rust source with its comments blanked, and with `literals` the contents of its string and
    char literals too: the same length and the same lines, so a position found in it is the
    source's. Block comments nest; a `//` or `/*` inside a literal starts no comment."""
    out = list(text)
    size = len(text)

    def blank(start: int, end: int) -> None:
        for at in range(start, min(end, size)):
            if out[at] != "\n":
                out[at] = " "

    def starts_a_word(at: int) -> bool:
        return at == 0 or not (text[at - 1].isalnum() or text[at - 1] == "_")

    at = 0
    while at < size:
        if text.startswith("//", at):
            end = text.find("\n", at)
            end = size if end < 0 else end
            blank(at, end)
            at = end
        elif text.startswith("/*", at):
            depth, end = 1, at + 2
            while end < size and depth:
                if text.startswith("/*", end):
                    depth, end = depth + 1, end + 2
                elif text.startswith("*/", end):
                    depth, end = depth - 1, end + 2
                else:
                    end += 1
            blank(at, end)
            at = end
        elif (raw := RUST_RAW_STRING.match(text, at)) and starts_a_word(at):
            close = '"' + raw.group(1)
            end = text.find(close, raw.end())
            end = size if end < 0 else end
            if literals:
                blank(raw.end(), end)
            at = end + len(close)
        elif text[at] == '"':
            end = at + 1
            while end < size and text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            if literals:
                blank(at + 1, end)
            at = end + 1
        elif text[at] == "'" and (char := RUST_CHAR.match(text, at)):
            if literals:
                blank(at + 1, char.end() - 1)
            at = char.end()
        else:
            at += 1
    return "".join(out)


def test_binary_findings(root: Path, files: list[str]) -> list[Finding]:
    """T7: every test file of a crate that lists its tests is built, once, for a stated reason.

    With `autotests = false` Cargo builds only what the manifest names, so a test file that
    `tests/main.rs` does not declare would never compile and its tests would never run, with
    nothing to show for it. A separate binary links the whole crate again, so each states why.
    """
    findings = []
    for rel in files:
        if not (rel.startswith("crates/") and rel.endswith("/Cargo.toml")):
            continue
        text = read_text(root / rel)
        if text is None or not re.search(r"^autotests\s*=\s*false", text, re.MULTILINE):
            continue
        tests = rel.removesuffix("Cargo.toml") + "tests/"
        test_files = {
            file[len(tests) :]
            for file in files
            if file.startswith(tests)
            and file.endswith(".rs")
            and "/" not in file[len(tests) :]
        }
        entries = tomllib.loads(text).get("test", [])
        targets = [test_target_path(entry) for entry in entries]
        # `tests/main.rs` builds its modules only when a target builds it.
        main = (
            (read_text(root / f"{tests}main.rs") or "")
            if "tests/main.rs" in targets
            else ""
        )
        modules = set(TEST_MODULE.findall(rust_code(main, literals=True)))
        for name in sorted(test_files):
            is_target = f"tests/{name}" in targets
            is_module = name.removesuffix(".rs") in modules
            if not is_target and not is_module:
                findings.append(
                    finding(
                        "T7", f"{tests}{name}", None, "this test file is never built"
                    )
                )
            elif is_target and is_module:
                findings.append(
                    finding(
                        "T7", f"{tests}{name}", None, "this test file is built twice"
                    )
                )
        lines = text.splitlines()
        headers = [n for n, line in enumerate(lines) if TEST_TARGET_HEADER.match(line)]
        for number, entry, target in zip(headers, entries, targets, strict=False):
            if target == "tests/main.rs":
                continue
            if number == 0 or not lines[number - 1].lstrip().startswith("#"):
                name = entry.get("name", target)
                findings.append(
                    finding(
                        "T7",
                        rel,
                        number + 1,
                        f'name = "{name}" has no comment saying why it is separate',
                    )
                )
    return findings


def test_target_path(entry: dict[str, Any]) -> str:
    """The file a `[[test]]` target builds, relative to its crate: Cargo's `tests/<name>.rs`
    when it names no path."""
    path = entry.get("path", f"tests/{entry.get('name', '')}.rs")
    return Path(path).as_posix().removeprefix("./")


# ------------------------------------------------------------------ Node.js package

#: The folder under `npm/` of each target a release of the Node.js package builds.
NODE_TARGET_FOLDERS = {
    "x86_64-apple-darwin": "darwin-x64",
    "aarch64-apple-darwin": "darwin-arm64",
    "x86_64-pc-windows-msvc": "win32-x64-msvc",
    "x86_64-unknown-linux-gnu": "linux-x64-gnu",
    "aarch64-unknown-linux-gnu": "linux-arm64-gnu",
    "aarch64-unknown-linux-musl": "linux-arm64-musl",
}
NODE_BUILD_JOB = re.compile(
    r"^  build-node-native:\n(.*?)(?=^  \S|\Z)", re.MULTILINE | re.DOTALL
)
NODE_BUILD_TARGET = re.compile(r"^\s*(?:- )?target:\s*(\S+)", re.MULTILINE)


def node_package_findings(root: Path, files: list[str]) -> list[Finding]:
    """T8: a release builds, packs and lists the same platforms of the Node.js package.

    The release adds the folders under `npm/` to the manifest it publishes, so a target
    built without a folder never reaches users, and a folder not built is published empty.
    The manifest in the repository lists none of them: they are not on npm until the
    release, and `npm ci` refuses a manifest whose lockfile cannot name them.
    """
    rel = "crates/bindings/node/package.json"
    manifest = read_text(root / rel)
    if manifest is None:
        return []
    package = json.loads(manifest)
    findings = []
    if package.get("optionalDependencies"):
        findings.append(finding("T8", rel, None, "lists optional dependencies"))
    release = ".github/workflows/release.yml"
    job = NODE_BUILD_JOB.search(read_text(root / release) or "")
    built = set(NODE_BUILD_TARGET.findall(job[1])) if job else set()
    targets = set(package.get("napi", {}).get("targets", []))
    for target in sorted(targets - built):
        findings.append(finding("T8", release, None, f"does not build {target}"))
    for target in sorted(built - targets):
        findings.append(finding("T8", rel, None, f"has no napi target {target}"))
    for target in sorted((targets | built) - set(NODE_TARGET_FOLDERS)):
        findings.append(finding("T8", rel, None, f"no known npm folder for {target}"))
    prefix = "crates/bindings/node/npm/"
    folders = {
        file[len(prefix) :].split("/")[0]
        for file in files
        if file.startswith(prefix) and "/" in file[len(prefix) :]
    }
    wanted = {NODE_TARGET_FOLDERS[t] for t in targets if t in NODE_TARGET_FOLDERS}
    for folder in sorted(wanted - folders):
        findings.append(
            finding("T8", f"{prefix}{folder}", None, "this folder is missing")
        )
    for folder in sorted(folders - wanted):
        findings.append(
            finding(
                "T8", f"{prefix}{folder}", None, "no napi target builds this folder"
            )
        )
    for folder in sorted(folders & wanted):
        text = read_text(root / f"{prefix}{folder}/package.json") or "{}"
        if json.loads(text).get("name") != f"@grafeo-db/js-{folder}":
            findings.append(
                finding(
                    "T8",
                    f"{prefix}{folder}/package.json",
                    None,
                    f"its name is not @grafeo-db/js-{folder}",
                )
            )
    return findings


# ------------------------------------------------------------------- T9: expected failures

#: `issue` of an expected failure: a GitHub issue, or the release that fixes it.
T9_ISSUE = re.compile(r"#[1-9][0-9]*|fix in (\d+)\.(\d+)\.(\d+)")
#: The first release heading of CHANGELOG.md, `## [X.Y.Z]` (not `## [Unreleased]`).
T9_RELEASE_HEADING = re.compile(r"^## \[(\d+)\.(\d+)\.(\d+)\]", re.MULTILINE)
#: A pytest mark's reason: `spec: <ref>; issue: <issue>`.
T9_PYTEST_REASON = re.compile(r"spec:(?P<spec>[^;]*);\s*issue:(?P<issue>.*)")
#: A Rust literal the call scanner steps over whole: a raw string, a string, a char.
RUST_LITERAL = re.compile(
    r'r(#*)"(?:.|\n)*?"\1|"(?:[^"\\]|\\.|\n)*"|\'(?:\\.|[^\\\'\n])\''
)

Release = tuple[int, int, int]


def release_floor(root: Path) -> Release | None:
    """The release being made: the first `## [X.Y.Z]` heading of CHANGELOG.md."""
    match = T9_RELEASE_HEADING.search(read_text(root / "CHANGELOG.md") or "")
    if not match:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def issue_problem(issue: str, floor: Release | None) -> str | None:
    """Why `issue` is not a valid `issue`, or None: `#N`, or `fix in X.Y.Z` not below
    `floor` (not judged without one)."""
    issue = issue.strip()
    match = T9_ISSUE.fullmatch(issue)
    if not match:
        return f"`{issue}` is neither `#N` nor `fix in X.Y.Z`"
    if match.group(1) is not None and floor is not None:
        major, minor, patch = (int(part) for part in match.groups())
        if (major, minor, patch) < floor:
            release = ".".join(map(str, floor))
            return f"`{issue}` names a release before {release}, the one being made"
    return None


def unquoted(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def gtest_xfail_findings(rel: str, text: str, floor: Release | None) -> list[Finding]:
    """T9 for a .gtest file: each `xfail:` key has a non-empty `spec:` and a valid `issue:`."""
    findings = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != "xfail:":
            continue
        indent = len(line) - len(line.lstrip())
        keys: dict[str, tuple[int, str]] = {}
        for below in range(index + 1, len(lines)):
            child = lines[below]
            if not child.strip() or child.lstrip().startswith("#"):
                continue
            if len(child) - len(child.lstrip()) <= indent:
                break
            key, _, value = child.strip().partition(":")
            keys[key.strip()] = (below + 1, unquoted(value))
        for key in ("spec", "issue"):
            if key not in keys:
                findings.append(
                    finding("T9", rel, index + 1, f"the xfail has no `{key}:`")
                )
        if "spec" in keys and not keys["spec"][1].strip():
            findings.append(
                finding("T9", rel, keys["spec"][0], "the xfail's `spec:` is empty")
            )
        if "issue" in keys:
            problem = issue_problem(keys["issue"][1], floor)
            if problem:
                findings.append(finding("T9", rel, keys["issue"][0], problem))
    return findings


def dotted_name(node: ast.expr) -> str:
    """`pytest.mark.xfail` for the expression that names it; empty for any other."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def pytest_mark_problem(call: ast.Call) -> tuple[int, str] | None:
    """The line and the problem of an xfail mark's arguments, or None (the issue aside)."""
    keywords = {keyword.arg: keyword for keyword in call.keywords if keyword.arg}
    strict = keywords.get("strict")
    is_strict = (
        strict is not None
        and isinstance(strict.value, ast.Constant)
        and strict.value.value is True
    )
    if not is_strict or "raises" not in keywords:
        return call.lineno, "the xfail mark needs `strict=True` and `raises=`"
    reason = keywords.get("reason")
    if reason is None or not (
        isinstance(reason.value, ast.Constant) and isinstance(reason.value.value, str)
    ):
        return call.lineno, "the xfail mark needs a literal `reason=`"
    return None


def pytest_xfail_findings(rel: str, text: str, floor: Release | None) -> list[Finding]:
    """T9 for a Python file: each xfail mark is strict, names the error it expects and has a
    `spec: <ref>; issue: <issue>` reason; an imperative `pytest.xfail()` is never strict."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    findings = []
    called: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called.add(id(node.func))
        name = dotted_name(node.func)
        if name == "pytest.xfail":
            detail = "`pytest.xfail()` is never strict; mark the test instead"
            findings.append(finding("T9", rel, node.lineno, detail))
            continue
        if name != "pytest.mark.xfail":
            continue
        problem = pytest_mark_problem(node)
        if problem:
            findings.append(finding("T9", rel, problem[0], problem[1]))
            continue
        reason = next(
            keyword.value for keyword in node.keywords if keyword.arg == "reason"
        )
        assert isinstance(reason, ast.Constant) and isinstance(reason.value, str)
        match = T9_PYTEST_REASON.fullmatch(reason.value.strip())
        if not match:
            detail = "the reason is not `spec: <ref>; issue: <issue>`"
        elif not match.group("spec").strip():
            detail = "the reason's spec is empty"
        else:
            detail = issue_problem(match.group("issue"), floor) or ""
        if detail:
            findings.append(finding("T9", rel, reason.lineno, detail))
    # A mark used without arguments (`@pytest.mark.xfail`) is not strict.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and id(node) not in called
            and dotted_name(node) == "pytest.mark.xfail"
        ):
            detail = "the xfail mark needs `strict=True` and `raises=`"
            findings.append(finding("T9", rel, node.lineno, detail))
    return findings


def rust_calls(text: str, name: str) -> list[tuple[int, list[tuple[int, str]]]]:
    """The calls `name(...)` in Rust source outside comments and literals, each as its line
    and its top-level arguments (line, text); `fn name(` and `x.name(` are not calls."""
    call_start = re.compile(rf"(?<![\w.]){re.escape(name)}\s*\(")
    calls = []
    i, line, n = 0, 1, len(text)
    while i < n:
        if literal := RUST_LITERAL.match(text, i):
            line += literal.group().count("\n")
            i = literal.end()
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    line += text[i] == "\n"
                    i += 1
        elif (call := call_start.match(text, i)) and not re.search(
            r"\bfn\s+$", text[max(0, i - 8) : i]
        ):
            calls.append((line, []))
            i, line = rust_arguments(text, call.end(), line, calls[-1][1])
        else:
            line += text[i] == "\n"
            i += 1
    return calls


def rust_arguments(
    text: str, i: int, line: int, args: list[tuple[int, str]]
) -> tuple[int, int]:
    """Collects into `args` the top-level arguments of the call whose `(` ends before `i`;
    the position after its `)` and the line there."""
    depth, start, start_line = 0, i, line
    while i < len(text):
        if literal := RUST_LITERAL.match(text, i):
            line += literal.group().count("\n")
            i = literal.end()
            continue
        c = text[i]
        if c in ",)" and not depth:
            arg = text[start:i]
            lead = arg[: len(arg) - len(arg.lstrip())]
            if arg.strip():
                args.append((start_line + lead.count("\n"), arg.strip()))
            if c == ")":
                return i + 1, line
            start, start_line = i + 1, line
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        line += c == "\n"
        i += 1
    return i, line


def rust_xfail_findings(rel: str, text: str, floor: Release | None) -> list[Finding]:
    """T9 for a Rust file: each `xfail(spec, issue, ..)` call names a non-empty spec and a
    valid issue, both as string literals."""
    findings = []
    for call_line, args in rust_calls(text, "xfail"):
        if len(args) < 2:
            findings.append(
                finding("T9", rel, call_line, "the xfail names no spec and issue")
            )
            continue
        for (line, arg), what in zip(args[:2], ("spec", "issue"), strict=True):
            literal = re.fullmatch(r'"((?:[^"\\]|\\.)*)"', arg)
            if not literal:
                detail = f"the xfail's {what} is not a string literal"
            elif what == "spec":
                detail = "" if literal.group(1).strip() else "the xfail's spec is empty"
            else:
                detail = issue_problem(literal.group(1), floor) or ""
            if detail:
                findings.append(finding("T9", rel, line, detail))
    return findings


def xfail_findings(root: Path, files: list[str]) -> list[Finding]:
    """T9: every expected failure (a gtest `xfail:`, a pytest xfail mark, a Rust `xfail(...)`)
    names the spec it is held against and a valid issue: `#N`, or `fix in X.Y.Z` not below
    the release being made, the first `## [X.Y.Z]` heading of CHANGELOG.md. Without that
    heading a `fix in X.Y.Z` cannot be judged, which is a finding of its own."""
    floor = release_floor(root)
    findings = []
    placeholder = False
    forms: list[
        tuple[str, Callable[[str, str, Release | None], list[Finding]], str]
    ] = [
        (".gtest", gtest_xfail_findings, "xfail:"),
        (".py", pytest_xfail_findings, "xfail"),
        (".rs", rust_xfail_findings, "xfail("),
    ]
    for rel in files:
        # The gtest harness's own fixtures, some refused on purpose (an xfail without an
        # issue): the spec suite is judged, not them. A whole path component, as the
        # runners' `probe:` key matches it.
        if "spec-probes" in rel.split("/"):
            continue
        for suffix, check, needle in forms:
            if rel.endswith(suffix):
                text = read_text(root / rel)
                if text and needle in text:
                    findings += check(rel, text, floor)
                    placeholder = placeholder or "fix in " in text
    if floor is None and placeholder:
        detail = (
            "no `## [X.Y.Z]` heading names the release a `fix in X.Y.Z` is judged by"
        )
        findings.append(finding("T9", "CHANGELOG.md", None, detail))
    return findings


# ------------------------------------------------------------------ stability markers

#: The crates whose code the `grafeo` facade exposes: where experimental items live.
STABILITY_CRATES = ("grafeo", "grafeo-engine", "grafeo-core", "grafeo-common")
#: A `# Stability` heading of an item (`///`) or a module (`//!`).
STABILITY_HEADING = re.compile(
    r"^\s*//[/!]\s*#{1,2} Stability(?:: (Experimental))?\s*$"
)
#: A Rust path in the Experimental table: `Type`, `Type::member`, `module::item`, `a::b::c`;
#: a trailing `()` (a method named as a call) is dropped.
TABLE_PATH = re.compile(r"^(?:[A-Za-z_]\w*::)+[A-Za-z_]\w*(?:\(\))?$|^[A-Z][a-z]\w*$")
ITEM = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+(?:(?:const|async|unsafe|extern \"C\")\s+)*"
    r"(?:(fn|struct|enum|trait|type|mod)\s+(\w+)|(\w+)\s*:)"
)
CONTAINER = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:impl\b(.*)|(?:struct|enum|trait)\s+(\w+))"
)


def experimental_table(root: Path) -> list[tuple[int, str]]:
    """The Rust paths of the Experimental table in docs/stability.md, with their lines."""
    text = read_text(root / "docs" / "stability.md")
    if text is None:
        return []
    paths, inside = [], False
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            inside = line.strip() == "## Experimental APIs"
            continue
        if (
            not inside
            or not line.startswith("| ")
            or line.startswith(("| API", "| ---"))
        ):
            continue
        for token in re.findall(r"`([^`]+)`", line.split("|")[1]):
            if TABLE_PATH.match(token):
                paths.append((number, token.removesuffix("()")))
    return paths


def doc_above(lines: list[str], index: int) -> list[str]:
    """The `///` lines of the item at `index`, past its attributes."""
    docs, at = [], index - 1
    while at >= 0:
        stripped = lines[at].strip()
        if stripped.startswith("///"):
            docs.append(stripped)
        elif ITEM.match(lines[at]):
            # The item before: an undocumented item has no docs of its own.
            break
        elif stripped.startswith(("#[", "#!")) or (
            # Below the docs only: a line of an attribute that spans several. Above them,
            # such a line is the item before (a struct field ends with a comma).
            not docs
            and stripped
            and not stripped.endswith((";", "{", "}"))
            and not stripped.startswith("//")
        ):
            # An attribute, or a line of one that spans several.
            pass
        else:
            break
        at -= 1
    return docs


@dataclass
class RustItem:
    """A public item of the facade's crates: where it is, what holds it, and its markers."""

    path: str
    line: int
    kind: str
    name: str
    container: str | None
    experimental: bool


def module_name(rel: str) -> str:
    stem = Path(rel).stem
    return Path(rel).parent.name if stem in ("mod", "lib") else stem


def rust_items(
    root: Path, files: list[str]
) -> tuple[list[RustItem], list[tuple[str, int, bool]]]:
    """The public items of the facade's crates, and the `//!` stability headings of their
    modules (file, line, experimental)."""
    items, modules = [], []
    for rel in files:
        parts = rel.split("/")
        if not rel.endswith(".rs") or len(parts) < 3 or parts[0] != "crates":
            continue
        if parts[1] not in STABILITY_CRATES or parts[2] != "src":
            continue
        text = read_text(root / rel)
        if text is None:
            continue
        lines = text.splitlines()
        for number, line in enumerate(lines, 1):
            if not line.lstrip().startswith("//!"):
                break
            heading = STABILITY_HEADING.match(line)
            if heading:
                modules.append((rel, number, bool(heading.group(1))))
        container = None
        for index, line in enumerate(lines):
            if line.rstrip() == "}":
                # The end of a top-level block: what follows belongs to no type.
                container = None
            holder = CONTAINER.match(line)
            if holder and not line.strip().startswith("//"):
                if holder.group(2):
                    container = holder.group(2)
                else:
                    target = holder.group(1).split(" for ")[-1]
                    words = re.findall(r"[A-Za-z_]\w*", target.split("{")[0])
                    container = next((w for w in words if w[0].isupper()), None)
            found = ITEM.match(line)
            if not found:
                continue
            kind = found.group(1) or "field"
            name = found.group(2) or found.group(3)
            docs = doc_above(lines, index)
            experimental = any(
                (heading := STABILITY_HEADING.match(doc)) and heading.group(1)
                for doc in docs
            )
            items.append(
                RustItem(
                    rel,
                    index + 1,
                    kind,
                    name,
                    container if kind in ("fn", "field") else None,
                    experimental,
                )
            )
    return items, modules


def stability_findings(root: Path, files: list[str]) -> list[Finding]:
    """T6: the Experimental table of docs/stability.md and the code agree, and every module
    the facade re-exports says what is stable.

    Both ways: each Rust path the table names has a `# Stability: Experimental` section where it
    is defined, and each such section in the facade's crates belongs to an item or module the
    table names. The `grafeo` crate's root documentation, each `grafeo_engine` module it
    re-exports and each module it defines has a `# Stability` section.
    """
    if not (root / "docs" / "stability.md").exists():
        return []
    findings = []
    table = experimental_table(root)
    items, modules = rust_items(root, files)
    named = {segment for _, path in table for segment in path.split("::")}

    def definitions(path: str) -> list[RustItem]:
        *outer, name = path.split("::")
        holder = outer[-1] if outer else None
        candidates = [item for item in items if item.name == name]
        if holder and holder[0].isupper():
            return [item for item in candidates if item.container == holder]
        if holder:
            return [
                item
                for item in candidates
                if module_name(item.path) == holder or holder in item.path.split("/")
            ]
        return [
            item
            for item in candidates
            if item.kind in ("struct", "enum", "trait", "type")
        ]

    for line, path in table:
        *outer, name = path.split("::")
        if outer and outer[-1][0].islower() and not definitions(path):
            # A module: its own `//!` heading marks it.
            marked = [m for m in modules if module_name(m[0]) == name and m[2]]
            if marked:
                continue
        found = definitions(path)
        if not found:
            findings.append(
                finding(
                    "T6",
                    "docs/stability.md",
                    line,
                    f"`{path}` is not defined in the facade's crates",
                )
            )
        elif not any(item.experimental for item in found):
            item = found[0]
            findings.append(
                finding(
                    "T6",
                    item.path,
                    item.line,
                    f"`{path}` is in the Experimental table but has no `# Stability: Experimental` section",
                )
            )
    for item in items:
        if item.experimental and item.name not in named:
            findings.append(
                finding(
                    "T6",
                    item.path,
                    item.line,
                    f"`{item.name}` is marked experimental but not in the Experimental table of docs/stability.md",
                )
            )
    marked_files = {
        item.path for item in items if item.experimental and item.name in named
    }
    for rel, line, experimental in modules:
        if experimental and module_name(rel) not in named and rel not in marked_files:
            findings.append(
                finding(
                    "T6",
                    rel,
                    line,
                    f"module `{module_name(rel)}` is marked experimental but not in the Experimental table",
                )
            )

    facade = read_text(root / "crates" / "grafeo" / "src" / "lib.rs")
    if facade is not None:
        lines = facade.splitlines()
        if not any(
            STABILITY_HEADING.match(line)
            for line in lines
            if line.lstrip().startswith("//!")
        ):
            findings.append(
                finding(
                    "T6",
                    "crates/grafeo/src/lib.rs",
                    1,
                    "the crate documentation has no `# Stability` section",
                )
            )
        sections = {m[0] for m in modules}
        for number, line in enumerate(lines, 1):
            reexport = re.match(r"^pub use grafeo_engine::([a-z_]+);", line)
            if reexport:
                module = reexport.group(1)
                candidates = [
                    f"crates/grafeo-engine/src/{module}.rs",
                    f"crates/grafeo-engine/src/{module}/mod.rs",
                ]
                if not any(c in sections for c in candidates):
                    findings.append(
                        finding(
                            "T6",
                            "crates/grafeo/src/lib.rs",
                            number,
                            f"re-exported module `{module}` has no `# Stability` section",
                        )
                    )
            own = re.match(r"^pub mod (\w+)", line)
            if own and not any(
                STABILITY_HEADING.match(doc) for doc in doc_above(lines, number - 1)
            ):
                findings.append(
                    finding(
                        "T6",
                        "crates/grafeo/src/lib.rs",
                        number,
                        f"module `{own.group(1)}` has no `# Stability` section",
                    )
                )
    return findings


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


# The binding features that are profiles or groups of capabilities, never one capability.
PROFILE_FEATURES = frozenset(
    {
        "lpg",
        "rdf",
        "ai",
        "analytics",
        "edge",
        "browser",
        "full",
        "languages",
        "embedded",
        "server",
    }
)
BINDING_SOURCES = (
    "crates/bindings/python/src/",
    "crates/bindings/node/src/",
    "crates/bindings/c/src/",
    "crates/bindings/wasm/src/",
)
CFG_START = re.compile(r"\bcfg(?:_attr)?!?\s*\(")
CFG_FEATURE = re.compile(r'feature\s*=\s*"([A-Za-z0-9_-]+)"')


def cfg_conditions(text: str) -> list[tuple[int, str]]:
    """Each `cfg(...)`, `cfg!(...)` and `cfg_attr(...)` of Rust code, comments and the contents
    of string and char literals left out: its line and its parenthesized text, which may span
    lines."""
    code = rust_code(text, literals=True)
    found = []
    for start in CFG_START.finditer(code):
        depth, end = 0, start.end() - 1
        for end in range(start.end() - 1, len(code)):
            if code[end] == "(":
                depth += 1
            elif code[end] == ")":
                depth -= 1
                if depth == 0:
                    break
        line = code.count("\n", 0, start.start()) + 1
        found.append((line, text[start.start() : end + 1]))
    return found


def binding_profile_cfg_findings(root: Path, files: list[str]) -> list[Finding]:
    """T12: a binding's code gates on capability features, never on a profile or group.

    A profile is a set of capabilities, and another profile can hold the same capability: code
    gated on `rdf` is left out of a `full` build that has `rdf-model` through `sparql`, so
    `full` lost `importRdf`. Gated on the capability, every profile that holds it compiles it.
    """
    findings = []
    for rel in files:
        if not rel.endswith(".rs") or not rel.startswith(BINDING_SOURCES):
            continue
        text = read_text(root / rel)
        if text is None:
            continue
        for line, condition in cfg_conditions(text):
            for name in CFG_FEATURE.findall(condition):
                if name in PROFILE_FEATURES:
                    findings.append(
                        finding(
                            "T12", rel, line, f"gates on the profile or group `{name}`"
                        )
                    )
    return findings


C_SOURCES = "crates/bindings/c/src/"
C_HEADER = "crates/bindings/c/grafeo.h"
GO_HEADER = "crates/bindings/go/grafeo.h"
GO_IMPORTS = "crates/bindings/go/grafeo_c.def"

# A function grafeo-c exports: `#[unsafe(no_mangle)]`, other attributes, then the `extern "C"` fn.
C_EXPORT = re.compile(
    r'#\[unsafe\(no_mangle\)\]\s*(?:#\[[^\]]*\]\s*)*pub\s+(?:unsafe\s+)?extern\s+"C"\s+fn\s+(\w+)'
)
# A `macro_rules!` whose body exports the fn it is given (`fn $name`).
C_EXPORTING_MACRO = re.compile(r"macro_rules!\s*(\w+)\s*\{")
C_DECLARATION = re.compile(r"\b(grafeo_\w+)\s*\(")
C_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def c_exports(root: Path, files: list[str]) -> set[str]:
    """The functions grafeo-c exports: each `#[unsafe(no_mangle)]` fn, and the name an exporting
    macro is invoked with (its first `grafeo_` identifier)."""
    exports: set[str] = set()
    sources = [
        text
        for rel in files
        if rel.startswith(C_SOURCES) and rel.endswith(".rs")
        if (text := read_text(root / rel)) is not None
    ]
    macros: set[str] = set()
    for text in sources:
        exports.update(C_EXPORT.findall(text))
        for match in C_EXPORTING_MACRO.finditer(text):
            # The macro's body: up to the next `macro_rules!`, or the end of the file.
            body_end = text.find("macro_rules!", match.end())
            body = text[match.end() : body_end if body_end >= 0 else len(text)]
            if "no_mangle" in body and re.search(r"\bfn\s+\$\w+", body):
                macros.add(match.group(1))
    for text in sources:
        for name in macros:
            for call in re.finditer(rf"\b{re.escape(name)}!\s*\(", text):
                arguments = text[call.end() :]
                code = "\n".join(
                    line
                    for line in arguments.splitlines()
                    if not line.lstrip().startswith("//")
                )
                found = re.search(r"\bgrafeo_\w+", code)
                if found:
                    exports.add(found.group(0))
    return exports


def c_header_findings(root: Path, files: list[str]) -> list[Finding]:
    """T13: grafeo-c's header, its Go copy and the Go import list agree with what grafeo-c
    exports. The Go header equals the C header (line endings aside); every export is declared
    and every declaration exported; `grafeo_c.def` lists exactly the exports, which the Go
    binding's Windows import library is built from."""
    present = set(files)
    findings: list[Finding] = []
    header = read_text(root / C_HEADER) if C_HEADER in present else None
    if header is not None and GO_HEADER in present:
        go_header = read_text(root / GO_HEADER)
        if go_header is not None and go_header.replace("\r\n", "\n") != header.replace(
            "\r\n", "\n"
        ):
            findings.append(finding("T13", GO_HEADER, None, f"differs from {C_HEADER}"))
    exports = c_exports(root, files)
    if header is not None:
        declared = set(C_DECLARATION.findall(C_COMMENT.sub("", header)))
        for name in sorted(exports - declared):
            findings.append(
                finding("T13", C_HEADER, None, f"`{name}` is exported but not declared")
            )
        for name in sorted(declared - exports):
            findings.append(
                finding("T13", C_HEADER, None, f"`{name}` is declared but not exported")
            )
    imports_text = read_text(root / GO_IMPORTS) if GO_IMPORTS in present else None
    if imports_text is not None:
        listed: set[str] = set()
        in_exports = False
        for raw in imports_text.splitlines():
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line == "EXPORTS":
                in_exports = True
                continue
            if in_exports:
                listed.add(line.split()[0])
        for name in sorted(exports - listed):
            findings.append(
                finding(
                    "T13",
                    GO_IMPORTS,
                    None,
                    f"`{name}` is exported but not in the import list",
                )
            )
        for name in sorted(listed - exports):
            findings.append(
                finding("T13", GO_IMPORTS, None, f"`{name}` is listed but not exported")
            )
    return findings


def tree_findings(root: Path, metadata_arg: str | None) -> list[Finding]:
    files = git(root, "ls-files").splitlines()
    findings = toolchain_findings(root) + text_findings(root, files)
    findings += stability_findings(root, files)
    findings += test_binary_findings(root, files)
    findings += node_package_findings(root, files)
    findings += binding_profile_cfg_findings(root, files)
    findings += xfail_findings(root, files)
    findings += c_header_findings(root, files)
    if metadata_arg != "none" and (root / "Cargo.toml").exists():
        if metadata_arg:
            metadata = json.loads(Path(metadata_arg).read_text(encoding="utf-8"))
        else:
            metadata = cargo_metadata(root)
        findings += boundary_findings(metadata)
        findings += rust_version_findings(root, metadata)
        findings += runtime_findings(
            lambda package, features, crate: cargo_tree_inverse(
                root, package, features, crate
            )
        )
        findings += profile_findings(
            lambda package, features: cargo_tree_features(root, package, features),
            binding_manifests(root),
        )
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
        library = (
            bool(LIBRARY_SOURCE.match(path))
            and "/tests/" not in path
            and not TEST_MODULE_FILE.search(path)
        )
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
            if path not in PRIVATE_ALLOWED and WORKFLOW_TEXT.search(text):
                findings.append(finding("D7", path, number))
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
    tree = commands.add_parser("tree", help="whole-repository rules (T1 to T10)")
    tree.add_argument(
        "--metadata", help="cargo metadata JSON file, or 'none' to skip T1, T5 and T10"
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
