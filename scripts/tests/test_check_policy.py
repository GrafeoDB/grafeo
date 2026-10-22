"""Tests for scripts/check_policy.py.

Run: uv run --with pytest python -m pytest scripts/tests
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "check_policy.py"
sys.path.insert(0, str(SCRIPT.parent))

import check_policy  # noqa: E402

EM, EN = "\u2014", "\u2013"
# Without GITHUB_* variables the script prints plain findings, even when the tests run in CI.
ENV = {key: value for key, value in os.environ.items() if not key.startswith("GITHUB_")}


def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)


def write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def policy(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=ENV,
    )


def found(result: subprocess.CompletedProcess[str]) -> list[str]:
    """`RULE path:line` for every reported finding."""
    return [
        " ".join(line.split()[:2])
        for line in result.stdout.splitlines()
        if line.strip()
    ]


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A repository whose base commit already carries debt that the diff rules must ignore."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    write(root, "README.md", "Grafeo\n")
    write(
        root,
        "crates/grafeo-core/src/lib.rs",
        "#[allow(dead_code)]\nfn old() {} // legacy \u2014 debt\n",
    )
    write(root, "crates/grafeo-storage/src/wal/recovery.rs", "pub fn replay() {}\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "base")
    return root


# ------------------------------------------------------------------ diff rules


def test_clean_change_passes(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-core/src/lib.rs",
        "#[allow(dead_code)]\nfn old() {} // legacy \u2014 debt\nfn new() {}\n",
    )
    result = policy(repo, "diff", "--base", "HEAD")
    assert (result.returncode, found(result)) == (0, [])


def test_d1_allow_without_reason(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-core/src/extra.rs",
        "#[allow(dead_code)]\nfn a() {}\n"
        '#[allow(dead_code, reason = "used by the planner tests")]\nfn b() {}\n'
        "#![allow(clippy::cast_possible_truncation)]\n"
        "#[cfg_attr(test, allow(unused))]\nfn c() {}\n"
        '#[allow(\n    clippy::cast_sign_loss,\n    reason = "ids are below 2^32, checked in new()"\n)]\nfn d() {}\n'
        "#[allow(\n    clippy::cast_sign_loss,\n)]\nfn e() {}\n",
    )
    result = policy(repo, "diff", "--base", "HEAD")
    assert result.returncode == 1
    assert found(result) == [
        "D1 crates/grafeo-core/src/extra.rs:1",
        "D1 crates/grafeo-core/src/extra.rs:5",
        "D1 crates/grafeo-core/src/extra.rs:6",
        "D1 crates/grafeo-core/src/extra.rs:13",
    ]


def test_d2_dashes_in_docs_and_comments_only(repo: Path) -> None:
    write(repo, "docs/guide.md", f"One {EM} two\nplain\nthree {EN} four\n")
    write(
        repo,
        "crates/grafeo-core/src/text.rs",
        f'let s = "a {EM} b"; // fine string\nlet t = 1; // bad {EN} comment\nlet u = "http://x"; // ok\n',
    )
    write(repo, "scripts/tool.py", f'NAME = "a {EM} b"\nx = 1  # bad {EM} note\n')
    write(repo, "tests/spec/lpg/gql/unicode.gtest", f"query: RETURN '{EM}'\n")
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == [
        "D2 crates/grafeo-core/src/text.rs:2",
        "D2 docs/guide.md:1",
        "D2 docs/guide.md:3",
        "D2 scripts/tool.py:2",
    ]


def test_d3_internal_references(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-core/src/notes.rs",
        "// see .claude/todo/x.md\n// config in ~/.claude.json is fine\n",
    )
    write(
        repo,
        "docs/roadmap.md",
        "Planned in Phase 3.\nA two-phase commit.\nPer the internal roadmap.\n",
    )
    write(repo, "crates/grafeo-core/src/phase.rs", "// Phase 3 of the planner\n")
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == [
        "D3 crates/grafeo-core/src/notes.rs:1",
        "D3 docs/roadmap.md:1",
        "D3 docs/roadmap.md:3",
    ]


def test_d4_no_new_ignore_on_crash_tests(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-engine/tests/crash_recovery.rs",
        '#[test]\n#[ignore = "slow"]\nfn a() {}\n',
    )
    write(
        repo,
        "crates/grafeo-engine/tests/checkpoint.rs",
        'use grafeo_common::testing::crash;\n#[test]\n#[ignore = "slow"]\nfn b() {}\n',
    )
    write(
        repo,
        "crates/grafeo-engine/tests/bench.rs",
        '#[test]\n#[ignore = "benchmark"]\nfn c() {}\n',
    )
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == [
        "D4 crates/grafeo-engine/tests/checkpoint.rs:3",
        "D4 crates/grafeo-engine/tests/crash_recovery.rs:2",
    ]


def test_d5_no_new_store_wrappers(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-engine/src/database/audit_store.rs",
        "impl GraphStore for AuditStore {}\nimpl<'a> GraphStoreMut for Borrowed<'a> {}\n"
        "#[cfg(test)]\nmod tests {\n    impl GraphStore for Fake {}\n}\n",
    )
    write(
        repo,
        "crates/grafeo-engine/tests/store.rs",
        "impl GraphStore for TestStore {}\n",
    )
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == [
        "D5 crates/grafeo-engine/src/database/audit_store.rs:1",
        "D5 crates/grafeo-engine/src/database/audit_store.rs:2",
    ]


def test_d5_matches_the_store_trait_not_a_mention(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-core/src/writer.rs",
        "impl From<Arc<dyn GraphStoreMut>> for GraphWriter {}\n"
        "impl<S: GraphStore> crate::graph::GraphStore for Wrapper<S> {}\n",
    )
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == ["D5 crates/grafeo-core/src/writer.rs:2"]


def test_d5_lets_the_stores_implement_the_store_traits(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-core/src/graph/rowgroup/read.rs",
        "impl GraphStore for RowGroupStore {}\n"
        "impl crate::graph::traits::GraphStoreSearch for RowGroupStore {}\n"
        "impl GraphStoreMut for super::lpg::LpgStore {}\n"
        "impl GraphStore for RowGroupStoreWrapper {}\n"
        "impl<S: GraphStore> GraphStore for Logged<S> {}\n",
    )
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == [
        "D5 crates/grafeo-core/src/graph/rowgroup/read.rs:4",
        "D5 crates/grafeo-core/src/graph/rowgroup/read.rs:5",
    ]


def test_d5_can_be_allowed_as_a_warning(repo: Path) -> None:
    write(repo, "crates/grafeo-engine/src/x.rs", "impl GraphStore for X {}\n")
    result = policy(repo, "diff", "--base", "HEAD", "--allow", "D5")
    assert result.returncode == 0
    assert found(result) == ["D5 crates/grafeo-engine/src/x.rs:1"]
    assert "warning" in result.stdout


def test_d6_no_ignored_results_in_replay_code(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-storage/src/wal/recovery.rs",
        "pub fn replay() {}\nfn f() { let _ = apply(); }\n",
    )
    write(
        repo, "crates/grafeo-engine/src/query/plan.rs", "fn g() { let _ = apply(); }\n"
    )
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == ["D6 crates/grafeo-storage/src/wal/recovery.rs:2"]


def test_w1_feature_cfg_warns_without_failing(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-core/src/store.rs",
        '#[cfg(feature = "temporal")]\nfield: u8,\n',
    )
    write(repo, "crates/grafeo-cli/src/main.rs", '#[cfg(feature = "wal")]\nfn x() {}\n')
    result = policy(repo, "diff", "--base", "HEAD")
    assert result.returncode == 0
    assert found(result) == ["W1 crates/grafeo-core/src/store.rs:1"]


def test_existing_debt_is_not_reported(repo: Path) -> None:
    write(repo, "README.md", "Grafeo\nMore text.\n")
    result = policy(repo, "diff", "--base", "HEAD")
    assert (result.returncode, found(result)) == (0, [])


def test_base_uses_the_merge_base(repo: Path) -> None:
    """Debt fixed on the base branch after the fork must not show up as added on the topic branch."""
    base = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True
    ).stdout.strip()
    write(repo, "docs/legacy.md", f"old {EM} text\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "debt")
    git(repo, "switch", "-q", "-c", "topic")
    write(repo, "docs/a.md", f"new {EM} line\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "topic")
    git(repo, "switch", "-q", base)
    write(repo, "docs/legacy.md", "old, text\n")
    git(repo, "commit", "-q", "-am", "fix the debt")
    git(repo, "switch", "-q", "topic")
    result = policy(repo, "diff", "--base", base)
    assert found(result) == ["D2 docs/a.md:1"]


def test_staged_mode_ignores_unstaged_changes(repo: Path) -> None:
    write(repo, "docs/staged.md", f"staged {EM} line\n")
    git(repo, "add", "docs/staged.md")
    write(repo, "docs/unstaged.md", f"unstaged {EM} line\n")
    result = policy(repo, "diff", "--staged")
    assert found(result) == ["D2 docs/staged.md:1"]


def test_paths_limit_the_diff_and_include_untracked_files(repo: Path) -> None:
    write(repo, "docs/new.md", f"x {EM} y\n")
    write(repo, "docs/other.md", f"x {EM} y\n")
    result = policy(repo, "diff", "--base", "HEAD", "--paths", "docs/new.md")
    assert found(result) == ["D2 docs/new.md:1"]


def test_json_output(repo: Path) -> None:
    write(repo, "docs/a.md", f"x {EM} y\n")
    result = policy(repo, "diff", "--base", "HEAD", "--json")
    data = json.loads(result.stdout)
    assert data == [
        {
            "rule": "D2",
            "path": "docs/a.md",
            "line": 1,
            "level": "error",
            "message": data[0]["message"],
        }
    ]
    assert "dash" in data[0]["message"]


# ------------------------------------------------------------------ tree rules


def test_t2_toolchain_pins(repo: Path) -> None:
    write(repo, "rust-toolchain.toml", '[toolchain]\nchannel = "1.98.1"\n')
    write(repo, "Cargo.toml", '[workspace.package]\nrust-version = "1.91.1"\n')
    write(
        repo,
        ".github/workflows/ci.yml",
        "      - uses: dtolnay/rust-toolchain@1.98.1\n"
        "      - uses: dtolnay/rust-toolchain@1.120.0\n"
        "      - uses: dtolnay/rust-toolchain@1.91.1\n"
        "      - uses: dtolnay/rust-toolchain@master\n"
        "        run: cargo +1.91.1 check\n"
        "        run: cargo +1.90.0 check\n"
        "        run: cargo +$NIGHTLY miri test\n",
    )
    write(
        repo,
        ".github/workflows/pypi.yml",
        '          rust-toolchain: "1.97.0"\n          rust-toolchain: "1.98.1"\n',
    )
    git(repo, "add", ".")
    result = policy(repo, "tree", "--metadata", "none")
    assert found(result) == [
        "T2 .github/workflows/ci.yml:2",
        "T2 .github/workflows/ci.yml:6",
        "T2 .github/workflows/pypi.yml:1",
    ]


def test_t3_private_directory_is_never_tracked_or_referenced(repo: Path) -> None:
    write(repo, ".claude/notes.md", "private\n")
    write(
        repo,
        "crates/grafeo-core/src/doc.rs",
        "//! See .claude/ARCHITECTURE.md\n//! Desktop reads ~/.claude.json\n",
    )
    write(repo, ".gitignore", ".claude/\n")
    git(repo, "add", "-f", ".")
    result = policy(repo, "tree", "--metadata", "none")
    assert found(result) == [
        "T3 .claude/notes.md",
        "T3 crates/grafeo-core/src/doc.rs:1",
    ]


def test_t4_public_text_has_no_dashes(repo: Path) -> None:
    write(repo, "docs/index.md", f"fine\nbad {EM} line\n")
    write(repo, "CHANGELOG.md", f"- fixed {EN} thing\n")
    write(repo, ".github/PULL_REQUEST_TEMPLATE.md", "fine\n")
    write(repo, "crates/grafeo-core/src/x.rs", f"// {EM} in code is D2, not T4\n")
    git(repo, "add", ".")
    result = policy(repo, "tree", "--metadata", "none")
    assert found(result) == ["T4 CHANGELOG.md:1", "T4 docs/index.md:2"]


def test_tree_passes_on_a_clean_repository(repo: Path) -> None:
    result = policy(repo, "tree", "--metadata", "none")
    assert (result.returncode, found(result)) == (0, [])


# --------------------------------------------------------------- T1 boundaries


def metadata(edges: dict[str, list[tuple[str, str | None]]]) -> dict:
    """Minimal `cargo metadata` JSON: package name -> [(dependency, kind)]."""
    names = set(edges) | {dep for deps in edges.values() for dep, _ in deps}

    def pkg_id(name: str) -> str:
        source = (
            "path+file:///ws"
            if name.startswith("grafeo")
            else "registry+https://github.com/rust-lang/crates.io-index"
        )
        return f"{source}#{name}@1.0.0"

    return {
        "packages": [
            {
                "id": pkg_id(n),
                "name": n,
                "source": None if n.startswith("grafeo") else "registry",
            }
            for n in sorted(names)
        ],
        "workspace_members": [
            pkg_id(n) for n in sorted(names) if n.startswith("grafeo")
        ],
        "resolve": {
            "nodes": [
                {
                    "id": pkg_id(n),
                    "deps": [
                        {
                            "name": d.replace("-", "_"),
                            "pkg": pkg_id(d),
                            "dep_kinds": [{"kind": k, "target": None}],
                        }
                        for d, k in edges.get(n, [])
                    ],
                }
                for n in sorted(names)
            ]
        },
    }


CLEAN = {
    "grafeo-common": [("thiserror", None)],
    "grafeo-core": [
        ("grafeo-common", None),
        ("parquet", None),
        ("crc32fast", None),
        ("tempfile", "dev"),
    ],
    "grafeo-storage": [("grafeo-common", None), ("memmap2", None), ("tokio", None)],
    "grafeo-adapters": [("grafeo-common", None), ("grafeo-core", None)],
    "grafeo-engine": [
        ("grafeo-core", None),
        ("grafeo-storage", None),
        ("grafeo-adapters", None),
        ("tokio", None),
    ],
    "parquet": [("snap", None)],
    "tempfile": [("fs2", None)],
}


def boundary(edges: dict) -> list[str]:
    return [
        f"{f.rule} {f.message}" for f in check_policy.boundary_findings(metadata(edges))
    ]


def test_t1_clean_graph() -> None:
    assert boundary(CLEAN) == []


def test_t1_forbidden_internal_edge() -> None:
    edges = {
        **CLEAN,
        "grafeo-storage": [*CLEAN["grafeo-storage"], ("grafeo-core", None)],
    }
    assert boundary(edges) == [
        "T1 grafeo-storage depends on grafeo-core; it may only depend on grafeo-common",
    ]


def test_t1_transitive_io_dependency_names_the_path() -> None:
    edges = {**CLEAN, "parquet": [("snap", None), ("tokio", None)]}
    assert boundary(edges) == [
        "T1 grafeo-core reaches tokio through grafeo-core -> parquet -> tokio",
        "T1 grafeo-adapters reaches tokio through grafeo-adapters -> grafeo-core -> parquet -> tokio",
    ]


def test_t1_build_dependencies_count_but_dev_dependencies_do_not() -> None:
    edges = {
        **CLEAN,
        "grafeo-core": [*CLEAN["grafeo-core"], ("memmap2", "build"), ("fs2", "dev")],
    }
    assert boundary(edges) == [
        "T1 grafeo-core reaches memmap2 through grafeo-core -> memmap2",
        "T1 grafeo-adapters reaches memmap2 through grafeo-adapters -> grafeo-core -> memmap2",
    ]


def test_t1_adapters_direct_storage_dependency() -> None:
    edges = {
        **CLEAN,
        "grafeo-adapters": [*CLEAN["grafeo-adapters"], ("crc32fast", None)],
    }
    assert boundary(edges) == [
        "T1 grafeo-adapters depends directly on crc32fast (storage I/O belongs in grafeo-storage)"
    ]


def test_t1_missing_crate_is_reported() -> None:
    edges = {k: v for k, v in CLEAN.items() if k != "grafeo-storage"}
    edges["grafeo-engine"] = [
        d for d in edges["grafeo-engine"] if d[0] != "grafeo-storage"
    ]
    assert boundary(edges) == [
        "T1 grafeo-storage is missing from cargo metadata; update the boundary rules"
    ]


# ------------------------------------------------------------------ commit-msg


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "fix: x\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n",
            ["P3 COMMIT_EDITMSG:3"],
        ),
        (
            "fix: x\n\nGenerated with [Claude Code](https://claude.com/claude-code)\n",
            ["P3 COMMIT_EDITMSG:3"],
        ),
        (
            "fix: x\n\nCo-authored-by: Copilot <175728472+Copilot@users.noreply.github.com>\n",
            ["P3 COMMIT_EDITMSG:3"],
        ),
        (
            "feat: y\n\nCo-authored-by: cursoragent <cursoragent@cursor.com>\n",
            ["P3 COMMIT_EDITMSG:3"],
        ),
        ("fix: x\n\nCo-authored-by: Jules Winnfield <jules@example.org>\n", []),
        ("fix: x\n\nCo-authored-by: Mia Wallace <mia@example.org>\nFixes #511\n", []),
        ("Merge pr-370\n# Co-Authored-By: Claude <noreply@anthropic.com>\n", []),
    ],
)
def test_commit_message(tmp_path: Path, message: str, expected: list[str]) -> None:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(message, encoding="utf-8")
    result = policy(tmp_path, "commit-msg", str(path))
    assert found(result) == expected
    assert result.returncode == (1 if expected else 0)
