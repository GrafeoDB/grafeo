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


def test_d7_text_addressed_to_the_private_workflow(repo: Path) -> None:
    lines = [
        # Each a finding: a session role by name, the inbox, a private note.
        "// openCypher List4; the Architect's ruling, 2026-11-05",
        "/// confirmed by the Engineer against each algorithm",
        "/// groups as a literal (Architect, 2026-11-06)",
        "// a sweep (Review, 2026-11-06)",
        "//! answers (the review's oracle)",
        "// Review's sweep of the procedure errors",
        "// the Orchestrator asked for it",
        "// fails today (inbox, 2026-11-06)",
        "// the table in entity-values-design.md",
        "// see decisions.md",
        # None a finding: ordinary words.
        "fn review_threads() {}",
        "// Decided (#728 review): the change goes through",
        "// Review the plan before it runs",
        "// the architecture of the planner",
        "// an engineer reading the code",
        "// a code review found it",
    ]
    write(repo, "crates/grafeo-core/src/notes.rs", "\n".join(lines) + "\n")
    write(
        repo,
        "tests/spec/lpg/gql/notes.gtest",
        "# openCypher List4, the Architect's ruling\n",
    )
    write(repo, "docs/guide.md", "The plan is in the inbox.\n")
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == [
        "D7 crates/grafeo-core/src/notes.rs:1",
        "D7 crates/grafeo-core/src/notes.rs:2",
        "D7 crates/grafeo-core/src/notes.rs:3",
        "D7 crates/grafeo-core/src/notes.rs:4",
        "D7 crates/grafeo-core/src/notes.rs:5",
        "D7 crates/grafeo-core/src/notes.rs:6",
        "D7 crates/grafeo-core/src/notes.rs:7",
        "D7 crates/grafeo-core/src/notes.rs:8",
        "D7 crates/grafeo-core/src/notes.rs:9",
        "D7 crates/grafeo-core/src/notes.rs:10",
        "D7 docs/guide.md:1",
        "D7 tests/spec/lpg/gql/notes.gtest:1",
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


def test_d5_leaves_test_module_files_alone(repo: Path) -> None:
    write(
        repo,
        "crates/grafeo-core/src/execution/operators/writer_recording_tests.rs",
        "impl GraphBackend for Refusing {}\nimpl GraphStore for Refusing {}\n",
    )
    write(
        repo,
        "crates/grafeo-engine/src/database/row_group_sections/tests.rs",
        "impl GraphStore for Fake {}\n",
    )
    write(
        repo,
        "crates/grafeo-engine/src/database/contests.rs",
        "impl GraphBackend for Logged {}\n",
    )
    result = policy(repo, "diff", "--base", "HEAD")
    assert found(result) == ["D5 crates/grafeo-engine/src/database/contests.rs:1"]


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
        "impl GraphStoreMut for super::rowgroup::RowGroupStore {}\n"
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


def test_t5_every_published_crate_declares_the_msrv(tmp_path: Path) -> None:
    write(tmp_path, "Cargo.toml", '[workspace.package]\nrust-version = "1.99.0"\n')

    def package(name: str, rust_version: str | None, **extra: object) -> dict:
        return {
            "id": f"path+file:///ws#{name}@1.0.0",
            "name": name,
            "source": None,
            "rust_version": rust_version,
            "manifest_path": str(tmp_path / "crates" / name / "Cargo.toml"),
            **extra,
        }

    packages = [
        package("grafeo", None),
        package("grafeo-core", "1.99.0"),
        package("grafeo-engine", "1.91.1"),
        package("grafeo-spec-tests", None, publish=[]),
        {
            "id": "registry+https://github.com/rust-lang/crates.io-index#serde@1.0.0",
            "name": "serde",
            "source": "registry",
            "rust_version": "1.61",
        },
    ]
    meta = {
        "packages": packages,
        "workspace_members": [p["id"] for p in packages if p["source"] is None],
    }
    findings = check_policy.rust_version_findings(tmp_path, meta)
    assert [(f.rule, f.path, f.message) for f in findings] == [
        (
            "T5",
            "crates/grafeo/Cargo.toml",
            (
                "grafeo declares rust-version none, not the workspace's 1.99.0; "
                "add `rust-version.workspace = true` to the crate's [package]"
            ),
        ),
        (
            "T5",
            "crates/grafeo-engine/Cargo.toml",
            (
                "grafeo-engine declares rust-version 1.91.1, not the workspace's 1.99.0; "
                "add `rust-version.workspace = true` to the crate's [package]"
            ),
        ),
    ]


def test_t5_holds_for_this_repository() -> None:
    root = SCRIPT.parents[1]
    assert (
        check_policy.rust_version_findings(root, check_policy.cargo_metadata(root))
        == []
    )


# ------------------------------------------------------------------ T7 test binaries


def test_t7_every_test_file_is_built_once_for_a_stated_reason(tmp_path: Path) -> None:
    manifest = """[package]
name = "grafeo-core"
autotests = false

[[test]]
name = "main"
path = "tests/main.rs"

# Counts allocations.
[[test]]
name = "counted"
path = "tests/counted.rs"

[[test]]
name = "unexplained"
path = "tests/unexplained.rs"

# Declared in main.rs too.
[[test]]
name = "twice"
path = "tests/twice.rs"
"""
    files = {
        "crates/grafeo-core/Cargo.toml": manifest,
        "crates/grafeo-core/tests/main.rs": "//! Tests.\n\nmod a_file;\nmod twice;\n",
        "crates/grafeo-core/tests/a_file.rs": "",
        "crates/grafeo-core/tests/counted.rs": "",
        "crates/grafeo-core/tests/forgotten.rs": "",
        "crates/grafeo-core/tests/twice.rs": "",
        "crates/grafeo-core/tests/unexplained.rs": "",
        "crates/grafeo-core/tests/common/helper.rs": "",
        # Without `autotests = false` Cargo finds every test file itself.
        "crates/grafeo-wasm/Cargo.toml": '[package]\nname = "grafeo-wasm"\n',
        "crates/grafeo-wasm/tests/web.rs": "",
    }
    for rel, text in files.items():
        write(tmp_path, rel, text)
    found = check_policy.test_binary_findings(tmp_path, sorted(files))
    assert all(f.rule == "T7" for f in found)
    assert [(f.path, f.line, f.message.split(";")[0]) for f in found] == [
        (
            "crates/grafeo-core/tests/forgotten.rs",
            None,
            "this test file is never built",
        ),
        ("crates/grafeo-core/tests/twice.rs", None, "this test file is built twice"),
        (
            "crates/grafeo-core/Cargo.toml",
            14,
            'name = "unexplained" has no comment saying why it is separate',
        ),
    ]


def write_lf(root: Path, rel: str, text: str) -> None:
    """Writes `text` as the repository holds it, with line feeds on every platform."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())


def t7_found(tmp_path: Path, files: dict[str, str]) -> list[tuple[str | None, str]]:
    for rel, text in files.items():
        write_lf(tmp_path, rel, text)
    found = check_policy.test_binary_findings(tmp_path, sorted(files))
    return [(f.path, f.message.split(";")[0]) for f in found]


MAIN_ONLY = """[package]
name = "grafeo-core"
autotests = false

[[test]]
name = "main"
path = "tests/main.rs"
"""


@pytest.mark.parametrize(
    "declaration",
    [
        "pub mod a_file;",
        "pub(crate) mod a_file;",
        "  mod  a_file ;",
        "pub  mod\ta_file;",
    ],
)
def test_t7_a_module_may_be_written_any_way_rust_takes(
    tmp_path: Path, declaration: str
) -> None:
    assert (
        t7_found(
            tmp_path,
            {
                "crates/grafeo-core/Cargo.toml": MAIN_ONLY,
                "crates/grafeo-core/tests/main.rs": f"{declaration}\n",
                "crates/grafeo-core/tests/a_file.rs": "",
            },
        )
        == []
    )


def test_t7_a_module_in_a_comment_builds_nothing(tmp_path: Path) -> None:
    assert t7_found(
        tmp_path,
        {
            "crates/grafeo-core/Cargo.toml": MAIN_ONLY,
            "crates/grafeo-core/tests/main.rs": "// mod a_file;\n",
            "crates/grafeo-core/tests/a_file.rs": "",
        },
    ) == [("crates/grafeo-core/tests/a_file.rs", "this test file is never built")]


@pytest.mark.parametrize(
    "commented",
    [
        "/*\nmod a_file;\n*/\n",
        "/* outer\n/* inner */\nmod a_file;\n*/\n",
    ],
    ids=["block", "nested"],
)
def test_t7_a_module_in_a_block_comment_builds_nothing(
    tmp_path: Path, commented: str
) -> None:
    assert t7_found(
        tmp_path,
        {
            "crates/grafeo-core/Cargo.toml": MAIN_ONLY,
            "crates/grafeo-core/tests/main.rs": commented,
            "crates/grafeo-core/tests/a_file.rs": "",
        },
    ) == [("crates/grafeo-core/tests/a_file.rs", "this test file is never built")]


def test_t7_a_module_after_a_comment_is_built(tmp_path: Path) -> None:
    assert (
        t7_found(
            tmp_path,
            {
                "crates/grafeo-core/Cargo.toml": MAIN_ONLY,
                "crates/grafeo-core/tests/main.rs": (
                    "// see /* here\n/* a note */\nmod a_file;\n"
                ),
                "crates/grafeo-core/tests/a_file.rs": "",
            },
        )
        == []
    )


def test_t7_a_target_with_no_path_is_tests_of_its_name(tmp_path: Path) -> None:
    manifest = MAIN_ONLY + '\n# Counts allocations.\n[[test]]\nname = "counted"\n'
    assert (
        t7_found(
            tmp_path,
            {
                "crates/grafeo-core/Cargo.toml": manifest,
                "crates/grafeo-core/tests/main.rs": "",
                "crates/grafeo-core/tests/counted.rs": "",
            },
        )
        == []
    )


def test_t7_a_target_elsewhere_does_not_build_a_test_file_of_its_stem(
    tmp_path: Path,
) -> None:
    manifest = (
        MAIN_ONLY + '\n# Elsewhere.\n[[test]]\nname = "other"\npath = "src/foo.rs"\n'
    )
    assert t7_found(
        tmp_path,
        {
            "crates/grafeo-core/Cargo.toml": manifest,
            "crates/grafeo-core/src/foo.rs": "",
            "crates/grafeo-core/tests/main.rs": "",
            "crates/grafeo-core/tests/foo.rs": "",
        },
    ) == [("crates/grafeo-core/tests/foo.rs", "this test file is never built")]


def test_t7_a_main_no_target_builds_is_never_built(tmp_path: Path) -> None:
    manifest = '[package]\nname = "grafeo-core"\nautotests = false\n'
    assert t7_found(
        tmp_path,
        {
            "crates/grafeo-core/Cargo.toml": manifest,
            "crates/grafeo-core/tests/main.rs": "",
        },
    ) == [("crates/grafeo-core/tests/main.rs", "this test file is never built")]


@pytest.mark.parametrize(
    "entry",
    [
        '[[test]]\npath = "tests/main.rs"\nname = "main"\n',
        "[[test]]\nname = 'main'\npath = 'tests/main.rs'\n",
        '[[test]]\nharness = true\nname = "main"\npath = "tests/main.rs"\n',
    ],
)
def test_t7_the_main_target_needs_no_comment_in_any_order(
    tmp_path: Path, entry: str
) -> None:
    manifest = f'[package]\nname = "grafeo-core"\nautotests = false\n\n{entry}'
    assert (
        t7_found(
            tmp_path,
            {
                "crates/grafeo-core/Cargo.toml": manifest,
                "crates/grafeo-core/tests/main.rs": "",
            },
        )
        == []
    )


def test_t7_a_separate_target_needs_its_comment_in_any_order(tmp_path: Path) -> None:
    manifest = MAIN_ONLY + '\n[[test]]\npath = "tests/counted.rs"\nname = "counted"\n'
    found = t7_found(
        tmp_path,
        {
            "crates/grafeo-core/Cargo.toml": manifest,
            "crates/grafeo-core/tests/main.rs": "",
            "crates/grafeo-core/tests/counted.rs": "",
        },
    )
    assert len(found) == 1
    assert "has no comment saying why it is separate" in found[0][1]


def test_t7_holds_for_this_repository() -> None:
    root = SCRIPT.parents[1]
    files = check_policy.git(root, "ls-files").splitlines()
    assert check_policy.test_binary_findings(root, files) == []


# ------------------------------------------------------------------ T6 stability markers

STABILITY_DOC = """# API Stability

## Experimental APIs

| API | Why it is experimental |
| --- | --- |
| `Db::run_streaming` and `Stream`, Python `stream()` | New |
| The `grafeo::probe` module and `probe::read_probe` | New |

## Stable APIs

- `Db::run` is stable
"""

FACADE = """//! The facade.
//!
//! ## Stability
//!
//! Stable unless marked.

pub use grafeo_engine::probe;
pub use grafeo_engine::session;

/// Its own module.
///
/// # Stability
///
/// Stable.
pub mod store {}
"""

SESSION = """//! Sessions.
//!
//! # Stability
//!
//! Stable unless marked.

pub struct Db;

impl Db {
    /// Runs a query.
    pub fn run(&self) {}

    /// Runs a query as a stream.
    ///
    /// # Stability: Experimental
    ///
    /// New.
    #[must_use]
    pub fn run_streaming(&self) -> Stream {
        Stream
    }
}

/// A stream of rows.
///
/// # Stability: Experimental
///
/// New.
pub struct Stream;
"""

PROBE = """//! Probes.
//!
//! # Stability: Experimental
//!
//! New.

/// Reads a probe.
///
/// # Stability: Experimental
pub fn read_probe() {}
"""


def stability_tree(root: Path, **overrides: str) -> list[str]:
    """The T6 findings of a facade with an engine, as `path:line message` lines."""
    files = {
        "docs/stability.md": STABILITY_DOC,
        "crates/grafeo/src/lib.rs": FACADE,
        "crates/grafeo-engine/src/session/mod.rs": SESSION,
        "crates/grafeo-engine/src/probe.rs": PROBE,
    }
    files.update(overrides)
    for rel, text in files.items():
        write(root, rel, text)
    found = check_policy.stability_findings(root, sorted(files))
    assert all(f.rule == "T6" for f in found)
    return [f"{f.path}:{f.line} {f.message.split(';')[0]}" for f in found]


def test_t6_the_table_and_the_markers_agree(tmp_path: Path) -> None:
    assert stability_tree(tmp_path) == []


def test_t6_a_field_after_an_experimental_field_keeps_its_own_docs(
    tmp_path: Path,
) -> None:
    write(
        tmp_path,
        "crates/grafeo-engine/src/result.rs",
        "/// A result.\npub struct Result {\n"
        "    /// Its profile.\n    ///\n    /// # Stability: Experimental\n"
        "    pub profile: Option<u8>,\n"
        "    /// What it holds.\n    pub kind: u8,\n"
        "    #[serde(default)]\n    /// Tagged.\n    ///\n    /// # Stability: Experimental\n"
        "    #[allow(\n        clippy::all,\n    )]\n    pub tagged: u8,\n}\n",
    )
    items, _ = check_policy.rust_items(tmp_path, ["crates/grafeo-engine/src/result.rs"])
    marked = {item.name: item.experimental for item in items if item.kind == "field"}
    assert marked == {"profile": True, "kind": False, "tagged": True}


def test_t6_an_undocumented_field_after_an_experimental_field_is_stable(
    tmp_path: Path,
) -> None:
    write(
        tmp_path,
        "crates/grafeo-engine/src/result.rs",
        "/// A result.\npub struct Result {\n"
        "    /// Its profile.\n    ///\n    /// # Stability: Experimental\n"
        "    pub profile: Option<u8>,\n    pub kind: u8,\n}\n",
    )
    items, _ = check_policy.rust_items(tmp_path, ["crates/grafeo-engine/src/result.rs"])
    marked = {item.name: item.experimental for item in items if item.kind == "field"}
    assert marked == {"profile": True, "kind": False}


def test_t6_a_table_entry_needs_its_marker(tmp_path: Path) -> None:
    unmarked = SESSION.replace(
        "/// A stream of rows.\n///\n/// # Stability: Experimental\n///\n/// New.\n",
        "/// A stream of rows.\n",
    )
    assert stability_tree(
        tmp_path, **{"crates/grafeo-engine/src/session/mod.rs": unmarked}
    ) == [
        (
            "crates/grafeo-engine/src/session/mod.rs:25 `Stream` is in the Experimental table "
            "but has no `# Stability: Experimental` section"
        )
    ]


def test_t6_a_marker_needs_its_table_entry(tmp_path: Path) -> None:
    marked = SESSION.replace(
        "    /// Runs a query.\n",
        "    /// Runs a query.\n    ///\n    /// # Stability: Experimental\n",
    )
    assert stability_tree(
        tmp_path, **{"crates/grafeo-engine/src/session/mod.rs": marked}
    ) == [
        (
            "crates/grafeo-engine/src/session/mod.rs:13 `run` is marked experimental but not in "
            "the Experimental table of docs/stability.md"
        )
    ]


def test_t6_a_table_entry_must_exist(tmp_path: Path) -> None:
    doc = STABILITY_DOC.replace(
        "`Db::run_streaming` and", "`Db::run_streaming`, `Db::run_gone` and"
    )
    assert stability_tree(tmp_path, **{"docs/stability.md": doc}) == [
        "docs/stability.md:7 `Db::run_gone` is not defined in the facade's crates"
    ]


def test_t6_the_member_of_another_type_does_not_count(tmp_path: Path) -> None:
    other = SESSION.replace("impl Db {", "impl Other {")
    found = stability_tree(
        tmp_path, **{"crates/grafeo-engine/src/session/mod.rs": other}
    )
    assert (
        "docs/stability.md:7 `Db::run_streaming` is not defined in the facade's crates"
        in found
    )


def test_t6_an_experimental_module_needs_its_table_entry(tmp_path: Path) -> None:
    doc = STABILITY_DOC.replace(
        "| The `grafeo::probe` module and `probe::read_probe` | New |\n", ""
    )
    found = stability_tree(tmp_path, **{"docs/stability.md": doc})
    assert found == [
        (
            "crates/grafeo-engine/src/probe.rs:10 `read_probe` is marked experimental but not "
            "in the Experimental table of docs/stability.md"
        ),
        (
            "crates/grafeo-engine/src/probe.rs:3 module `probe` is marked experimental but not "
            "in the Experimental table"
        ),
    ]


def test_t6_the_facade_and_its_modules_say_what_is_stable(tmp_path: Path) -> None:
    facade = FACADE.replace(
        "//!\n//! ## Stability\n//!\n//! Stable unless marked.\n", ""
    ).replace("///\n/// # Stability\n///\n/// Stable.\n", "")
    session = SESSION.replace(
        "//!\n//! # Stability\n//!\n//! Stable unless marked.\n", ""
    )
    found = stability_tree(
        tmp_path,
        **{
            "crates/grafeo/src/lib.rs": facade,
            "crates/grafeo-engine/src/session/mod.rs": session,
        },
    )
    assert found == [
        "crates/grafeo/src/lib.rs:1 the crate documentation has no `# Stability` section",
        "crates/grafeo/src/lib.rs:4 re-exported module `session` has no `# Stability` section",
        "crates/grafeo/src/lib.rs:7 module `store` has no `# Stability` section",
    ]


def test_t6_holds_for_this_repository() -> None:
    root = SCRIPT.parents[1]
    files = check_policy.git(root, "ls-files").splitlines()
    assert check_policy.stability_findings(root, files) == []


# ------------------------------------------------------------------ T8 Node.js package

NODE_TARGETS = {
    "x86_64-apple-darwin": "darwin-x64",
    "aarch64-apple-darwin": "darwin-arm64",
    "x86_64-pc-windows-msvc": "win32-x64-msvc",
    "x86_64-unknown-linux-gnu": "linux-x64-gnu",
    "aarch64-unknown-linux-gnu": "linux-arm64-gnu",
    "aarch64-unknown-linux-musl": "linux-arm64-musl",
}


def node_package(
    root: Path,
    targets: list[str] | None = None,
    built: list[str] | None = None,
    folders: dict[str, str] | None = None,
    **manifest: object,
) -> list[str]:
    """Writes a Node.js package and its release job; returns the files, as the rule gets them."""
    targets = list(NODE_TARGETS) if targets is None else targets
    built = list(NODE_TARGETS) if built is None else built
    folders = (
        {folder: f"@grafeo-db/js-{folder}" for folder in NODE_TARGETS.values()}
        if folders is None
        else folders
    )
    files = ["crates/bindings/node/package.json", ".github/workflows/release.yml"]
    write_lf(
        root,
        files[0],
        json.dumps({"name": "@grafeo-db/js", "napi": {"targets": targets}, **manifest}),
    )
    matrix = "".join(f"          - os: any\n            target: {t}\n" for t in built)
    write_lf(
        root,
        files[1],
        "jobs:\n  build-node-native:\n    strategy:\n      matrix:\n        include:\n"
        f"{matrix}    steps:\n      - run: echo\n"
        "  build-other:\n    strategy:\n      matrix:\n        include:\n"
        "          - target: wasm32-unknown-unknown\n",
    )
    for folder, name in folders.items():
        rel = f"crates/bindings/node/npm/{folder}/package.json"
        write_lf(root, rel, json.dumps({"name": name}))
        files.append(rel)
    return files


def node_findings(root: Path, files: list[str]) -> list[tuple[str, str]]:
    found = check_policy.node_package_findings(root, sorted(files))
    assert all(f.rule == "T8" for f in found)
    return [(f.path, f.message.split(";")[0]) for f in found]


def test_t8_a_repository_with_no_node_package_reports_nothing(tmp_path: Path) -> None:
    assert check_policy.node_package_findings(tmp_path, []) == []


def test_t8_a_package_in_step_reports_nothing(tmp_path: Path) -> None:
    assert node_findings(tmp_path, node_package(tmp_path)) == []


def test_t8_the_manifest_lists_no_platform_packages(tmp_path: Path) -> None:
    files = node_package(
        tmp_path, optionalDependencies={"@grafeo-db/js-darwin-x64": "1.2.3"}
    )
    assert node_findings(tmp_path, files) == [
        ("crates/bindings/node/package.json", "lists optional dependencies")
    ]


def test_t8_a_target_the_release_does_not_build(tmp_path: Path) -> None:
    built = [t for t in NODE_TARGETS if t != "aarch64-unknown-linux-musl"]
    files = node_package(tmp_path, built=built)
    assert node_findings(tmp_path, files) == [
        (".github/workflows/release.yml", "does not build aarch64-unknown-linux-musl")
    ]


def test_t8_a_build_with_no_target_in_the_manifest(tmp_path: Path) -> None:
    targets = [t for t in NODE_TARGETS if t != "aarch64-apple-darwin"]
    folders = {
        folder: f"@grafeo-db/js-{folder}"
        for target, folder in NODE_TARGETS.items()
        if target != "aarch64-apple-darwin"
    }
    files = node_package(tmp_path, targets=targets, folders=folders)
    assert node_findings(tmp_path, files) == [
        ("crates/bindings/node/package.json", "has no napi target aarch64-apple-darwin")
    ]


def test_t8_a_target_with_no_known_folder(tmp_path: Path) -> None:
    extra = "riscv64gc-unknown-linux-gnu"
    files = node_package(
        tmp_path, targets=[*NODE_TARGETS, extra], built=[*NODE_TARGETS, extra]
    )
    assert node_findings(tmp_path, files) == [
        ("crates/bindings/node/package.json", f"no known npm folder for {extra}")
    ]


def test_t8_a_target_with_no_folder(tmp_path: Path) -> None:
    folders = {
        folder: f"@grafeo-db/js-{folder}"
        for folder in NODE_TARGETS.values()
        if folder != "linux-x64-gnu"
    }
    assert node_findings(tmp_path, node_package(tmp_path, folders=folders)) == [
        ("crates/bindings/node/npm/linux-x64-gnu", "this folder is missing")
    ]


def test_t8_a_folder_no_target_builds(tmp_path: Path) -> None:
    folders = {folder: f"@grafeo-db/js-{folder}" for folder in NODE_TARGETS.values()}
    folders["linux-riscv64-gnu"] = "@grafeo-db/js-linux-riscv64-gnu"
    assert node_findings(tmp_path, node_package(tmp_path, folders=folders)) == [
        (
            "crates/bindings/node/npm/linux-riscv64-gnu",
            "no napi target builds this folder",
        )
    ]


def test_t8_a_folder_with_another_name(tmp_path: Path) -> None:
    folders = {folder: f"@grafeo-db/js-{folder}" for folder in NODE_TARGETS.values()}
    folders["linux-x64-gnu"] = "@grafeo-db/js-linux-x64"
    assert node_findings(tmp_path, node_package(tmp_path, folders=folders)) == [
        (
            "crates/bindings/node/npm/linux-x64-gnu/package.json",
            "its name is not @grafeo-db/js-linux-x64-gnu",
        )
    ]


def test_t8_a_release_with_no_build_job_builds_nothing(tmp_path: Path) -> None:
    files = node_package(tmp_path)
    write_lf(
        tmp_path, ".github/workflows/release.yml", "jobs:\n  other:\n    steps: []\n"
    )
    found = node_findings(tmp_path, files)
    assert len(found) == len(NODE_TARGETS)
    assert all(message.startswith("does not build") for _, message in found)


def test_t8_only_the_node_build_job_counts(tmp_path: Path) -> None:
    # The other job's wasm32 target in `node_package` is not a Node.js target.
    assert node_findings(tmp_path, node_package(tmp_path)) == []


def test_t8_a_file_deeper_in_a_folder_is_not_a_folder(tmp_path: Path) -> None:
    files = node_package(tmp_path)
    rel = "crates/bindings/node/npm/linux-x64-gnu/sub/package.json"
    write_lf(tmp_path, rel, "{}")
    assert node_findings(tmp_path, [*files, rel]) == []


def test_t8_the_repository_is_in_step() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "crates/bindings/node/npm",
            "crates/bindings/node/package.json",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    files = result.stdout.splitlines()
    assert check_policy.node_package_findings(root, files) == []


def test_t8_a_folder_with_no_manifest_is_a_folder(tmp_path: Path) -> None:
    files = node_package(tmp_path)
    rel = "crates/bindings/node/npm/linux-riscv64-gnu/README.md"
    write_lf(tmp_path, rel, "")
    assert node_findings(tmp_path, [*files, rel]) == [
        (
            "crates/bindings/node/npm/linux-riscv64-gnu",
            "no napi target builds this folder",
        )
    ]
